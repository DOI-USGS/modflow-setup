import sys
import os
import time
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import flopy
mf6 = flopy.mf6
from .discretization import (make_idomain, deactivate_idomain_above,
                             find_remove_isolated_cells,
                             create_vertical_pass_through_cells)
from .fileio import (load, dump, load_cfg,
                     flopy_mfsimulation_load)
from .gis import get_values_at_points
from .grid import write_bbox_shapefile, get_point_on_national_hydrogeologic_grid
from .tdis import setup_perioddata
from .utils import update, get_input_arguments, flatten
from .wells import setup_wel_data
from .mfmodel import MFsetupMixin


class MF6model(MFsetupMixin, mf6.ModflowGwf):
    """Class representing a MODFLOW-6 model.
    """

    source_path = os.path.split(__file__)[0]

    def __init__(self, simulation, parent=None, cfg=None,
                 modelname='model', exe_name='mf6',
                 version='mf6', **kwargs):
        mf6.ModflowGwf.__init__(self, simulation,
                                modelname, exe_name=exe_name, version=version,
                                **kwargs)
        MFsetupMixin.__init__(self, parent=parent)

        # default configuration
        self._package_setup_order = ['tdis', 'dis', 'ic', 'npf', 'sto', 'rch', 'oc',
                                     'ghb', 'lak', 'sfr',
                                     'wel', 'maw', 'gag', 'ims']
        self.cfg = load(self.source_path + '/mf6_defaults.yml')
        self.cfg['filename'] = self.source_path + '/mf6_defaults.yml'
        self._load_cfg(cfg)  # update configuration dict with values in cfg
        self.relative_external_paths = self.cfg['model'].get('relative_external_paths', True)
        self.model_ws = self._get_model_ws()

        # property attributes
        self._idomain = None

        # other attributes
        self._features = {} # dictionary for caching shapefile datasets in memory
        self._drop_thin_cells = self.cfg['dis'].get('drop_thin_cells', True)

        # arrays remade during this session
        self.updated_arrays = set()

    def __repr__(self):
        return MFsetupMixin.__repr__(self)

    def __str__(self):
        return MFsetupMixin.__repr__(self)

    @property
    def nlay(self):
        return self.cfg['dis']['dimensions'].get('nlay', 1)

    @property
    def length_units(self):
        return self.cfg['dis']['options']['length_units']

    @property
    def time_units(self):
        return self.cfg['tdis']['options']['time_units']

    @property
    def idomain(self):
        """3D array indicating which cells will be included in the simulation.
        Made a property so that it can be easily updated when any packages
        it depends on change.
        """
        if self._idomain is None and 'DIS' in self.get_package_list():
            self._set_idomain()
        return self._idomain

    def _load_cfg(self, cfg):
        """Load configuration file; update cfg dictionary."""
        if isinstance(cfg, str):
            assert os.path.exists(cfg), "config file {} not found".format(cfg)
            updates = load(cfg)
            updates['filename'] = cfg
        elif isinstance(cfg, dict):
            updates = cfg
        elif cfg is None:
            return
        else:
            raise TypeError("unrecognized input for cfg")

        # make sure empty variables get initialized as dicts
        for k, v in self.cfg.items():
            if v is None:
                cfg[k] = {}
        for k, v in updates.items():
            if v is None:
                cfg[k] = {}
        update(self.cfg, updates)

        # setup or load the simulation
        kwargs = self.cfg['simulation'].copy()
        if os.path.exists('{}.nam'.format(kwargs['sim_name'])):
            try:
                kwargs = get_input_arguments(kwargs, mf6.MFSimulation.load, warn=False)
                self._sim = mf6.MFSimulation.load(**kwargs)
            except:
                # create simulation
                kwargs = get_input_arguments(kwargs, mf6.MFSimulation, warn=False)
                self._sim = mf6.MFSimulation(**kwargs)

        # make sure that the output paths exist
        #self.external_path = self.cfg['model']['external_path']
        #output_paths = [self.cfg['intermediate_data']['output_folder'],
        #                self.cfg['simulation']['sim_ws'],
        #                ]
        output_paths = list(self.cfg['postprocessing']['output_folders'].values())
        for folder in output_paths:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # absolute path to config file
        self._config_path = os.path.split(os.path.abspath(self.cfg['filename']))[0]

        # set package keys to default dicts
        for pkg in self._package_setup_order:
            self.cfg[pkg] = defaultdict(dict, self.cfg.get(pkg, {}))

        # other variables
        self.cfg['external_files'] = {}

    def _set_idomain(self):
        """Remake the idomain array from the source data,
        no data values in the top and bottom arrays, and
        so that cells above SFR reaches are inactive."""
        idomain_from_layer_elevations = make_idomain(self.dis.top.array,
                                                     self.dis.botm.array,
                                                     nodata=self._nodata_value,
                                                     minimum_layer_thickness=self.cfg['dis'].get('minimum_layer_thickness', 1),
                                                     drop_thin_cells=True, tol=1e-4)
        # include cells that are active in the existing idomain array
        # and cells inactivated on the basis of layer elevations
        idomain = (self.dis.idomain.array == 1) & (idomain_from_layer_elevations == 1)
        idomain = idomain.astype(int)

        # remove cells that are above stream cells
        if 'SFR' in self.get_package_list():
            idomain = deactivate_idomain_above(idomain, self.sfr.packagedata)

        # inactivate any isolated cells that could cause problems with the solution
        idomain = find_remove_isolated_cells(idomain, minimum_cluster_size=20)

        # create pass-through cells in layers that have an inactive cell above and below
        # by setting these cells to -1
        idomain = create_vertical_pass_through_cells(idomain)

        self._idomain = idomain

        # re-write the input files
        self._setup_array('dis', 'idomain',
                          data={i: arr for i, arr in enumerate(idomain)},
                          by_layer=True, write_fmt='%d', dtype=int)
        self.dis.idomain = self.cfg['dis']['griddata']['idomain']

    def _set_perioddata(self):
        """Sets up the perioddata DataFrame."""
        self._perioddata = setup_perioddata(self.cfg, self.time_units)

    def get_flopy_external_file_input(self, var):
        """Repath intermediate external file input to the
        external file path that MODFLOW will use. Copy the
        file because MF6 flopy reads and writes to the same location.

        Parameters
        ----------
        var : str
            key in self.cfg['intermediate_data'] dict

        Returns
        -------
        input : dict or list of dicts
            MODFLOW6 external file input format
            {'filename': <filename>}
        """
        pass
        #intermediate_paths = self.cfg['intermediate_data'][var]
        #if isinstance(intermediate_paths, str):
        #    intermediate_paths = [intermediate_paths]
        #external_path = os.path.basename(os.path.normpath(self.external_path))
        #input = []
        #for f in intermediate_paths:
        #    outf = os.path.join(external_path, os.path.split(f)[1])
        #    input.append({'filename': outf})
        #    shutil.copy(f, os.path.normpath(self.external_path))
        #if len(input) == 1:
        #    input = input[0]
        #return input

    def get_package_list(self):
        """Replicate this method in flopy.modflow.Modflow.
        """
        # TODO: this should reference namfile dict
        return [p.name[0].upper() for p in self.packagelist]

    def get_raster_values_at_cell_centers(self, raster, out_of_bounds_errors='coerce'):
        """Sample raster values at centroids
        of model grid cells."""
        values = get_values_at_points(raster,
                                      x=self.modelgrid.xcellcenters.ravel(),
                                      y=self.modelgrid.ycellcenters.ravel(),
                                      out_of_bounds_errors=out_of_bounds_errors)
        if self.modelgrid.grid_type == 'structured':
            values = np.reshape(values, (self.nrow, self.ncol))
        return values

    def get_raster_statistics_for_cells(self, top, stat='mean'):
        """Compute zonal statics for raster pixels within
        each model cell.
        """
        raise NotImplementedError()

    def setup_grid(self, write_shapefile=True):
        """set the grid info dict
        (grid object will be updated automatically)"""
        grid = self.cfg['setup_grid'].copy()
        grid_file = grid.pop('grid_file').format(self.name)

        # arguments supplied to DIS have priority over those supplied to setup_grid
        for param in ['nrow', 'ncol']:
            grid.update({param: self.cfg['dis']['dimensions'][param]})
        for param in ['delr', 'delc']:
            grid.update({param: self.cfg['dis']['griddata'][param]})

        # optionally align grid with national hydrologic grid
        if grid['snap_to_NHG']:
            x, y = get_point_on_national_hydrogeologic_grid(grid['xoff'],
                                                            grid['yoff']
                                                            )
            grid['xoff'] = x
            grid['yoff'] = y
        dump(grid_file.format(self.name), grid)
        self.cfg['grid'] = grid
        if write_shapefile:
            write_bbox_shapefile(self.modelgrid,
                                 os.path.join(self.cfg['postprocessing']['output_folders']['shapefiles'],
                                              '{}_bbox.shp'.format(self.name)))

    def setup_dis(self):
        """"""
        package = 'dis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # resample the top from the DEM
        if self.cfg['dis']['remake_top']:
            self._setup_array(package, 'top', write_fmt='%.2f')

        # make the botm array
        self._setup_array(package, 'botm', by_layer=True, write_fmt='%.2f')

        # initial idomain input for creating a dis package instance
        self._setup_array(package, 'idomain', by_layer=True, write_fmt='%d',
                          dtype=int)

        # put together keyword arguments for dis package
        kwargs = self.cfg['grid'].copy() # nrow, ncol, delr, delc
        kwargs.update(self.cfg['dis']['dimensions']) # nper, nlay, etc.
        kwargs.update(self.cfg['dis']['griddata'])
        kwargs.update(self.cfg['dis'])

        # modelgrid: dis arguments
        remaps = {'xoff': 'xorigin',
                  'yoff': 'yorigin',
                  'rotation': 'angrot'}

        for k, v in remaps.items():
            kwargs[v] = kwargs.pop(k)
        kwargs['length_units'] = self.length_units
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfdis)
        dis = mf6.ModflowGwfdis(model=self, **kwargs)
        self._perioddata = None  # reset perioddata
        self._modelgrid = None  # override DIS package grid setup
        self._reset_bc_arrays()
        self._set_idomain()
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    def setup_tdis(self):
        """
        Sets up the TDIS package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'tdis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        perioddata = mf6.ModflowTdis.perioddata.empty(self, self.nper)
        for col in ['perlen', 'nstp', 'tsmult']:
            perioddata[col] = self.perioddata[col].values
        kwargs = self.cfg['tdis']['options']
        kwargs['nper'] = self.nper
        kwargs['perioddata'] = perioddata
        kwargs = get_input_arguments(kwargs, mf6.ModflowTdis)
        tdis = mf6.ModflowTdis(self.simulation, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return tdis

    def setup_ic(self):
        """
        Sets up the IC package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'ic'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the k array
        self._setup_array(package, 'strt', by_layer=True, write_fmt='%.2f')

        kwargs = self.cfg[package]['griddata'].copy()
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfic)
        ic = mf6.ModflowGwfic(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return ic

    def setup_npf(self):
        """
        Sets up the NPF package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'npf'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the k array
        self._setup_array(package, 'k', by_layer=True, write_fmt='%.6e')

        # make the k33 array (kv)
        self._setup_array(package, 'k33', by_layer=True, write_fmt='%.6e')

        kwargs = self.cfg[package]['options'].copy()
        kwargs.update(self.cfg[package]['griddata'].copy())
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfnpf)
        npf = mf6.ModflowGwfnpf(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return npf

    def setup_sto(self):
        """
        Sets up the STO package.

        Parameters
        ----------

        Notes
        -----

        """

        if np.all(self.perioddata['steady']):
            print('Skipping STO package, no transient stress periods...')
            return

        package = 'sto'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the sy array
        self._setup_array(package, 'sy', by_layer=True, write_fmt='%.6e')

        # make the ss array
        self._setup_array(package, 'ss', by_layer=True, write_fmt='%.6e')

        kwargs = self.cfg[package]['options'].copy()
        kwargs.update(self.cfg[package]['griddata'].copy())
        kwargs['steady_state'] = {k: v for k, v in self.cfg['sto']['steady'].items() if v}
        kwargs['transient'] = {k: True for k, v in self.cfg['sto']['steady'].items() if not v}
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfsto)
        sto = mf6.ModflowGwfsto(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return sto

    def setup_rch(self):
        """
        Sets up the RCH package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'rch'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the irch array
        # TODO: ich
        pass

        # make the rech array
        self._setup_array(package, 'recharge', by_layer=False,
                          resample_method='linear', write_fmt='%.6e',
                          write_nodata=0.)

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfrcha)
        rch = mf6.ModflowGwfrcha(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return rch

    def setup_wel(self):
        """
        Sets up the WEL package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'wel'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # munge well package input
        # returns dataframe with information to populate stress_period_data
        df = setup_wel_data(self)

        # set up stress_period_data
        spd = {}
        period_groups = df.groupby('per')
        for kper in range(self.nper):
            if kper in period_groups.groups:
                group = period_groups.get_group(kper)
                kspd = mf6.ModflowGwfwel.stress_period_data.empty(self,
                                                                  len(group),
                                                                  boundnames=True)[0]
                kspd['cellid'] = list(zip(group.k, group.i, group.j))
                kspd['q'] = group['flux']
                kspd['boundnames'] = group['comments']
                spd[kper] = kspd
            else:
                spd[kper] = None
        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        kwargs['stress_period_data'] = spd
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfwel)
        wel = mf6.ModflowGwfwel(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return wel

    def setup_oc(self):
        """
        Sets up the OC package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'oc'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        kwargs = self.cfg[package]
        kwargs['budget_filerecord'] = self.cfg[package]['budget_fileout_fmt'].format(self.name)
        kwargs['head_filerecord'] = self.cfg[package]['head_fileout_fmt'].format(self.name)
        for rec in ['printrecord', 'saverecord']:
            if rec in kwargs:
                data = kwargs[rec]
                mf6_input = {}
                for kper, words in data.items():
                    mf6_input[kper] = []
                    for var, instruction in words.items():
                        mf6_input[kper].append((var, instruction))
                kwargs[rec] = mf6_input
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfoc)
        oc = mf6.ModflowGwfoc(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return oc

    def setup_ims(self):
        """
        Sets up the IMS package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'ims'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        kwargs = flatten(self.cfg[package])
        kwargs = get_input_arguments(kwargs, mf6.ModflowIms)
        ims = mf6.ModflowIms(self.simulation, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return ims

    def write_input(self):
        """Same syntax as MODFLOW-2005 flopy
        """
        self.simulation.write_simulation()

    @staticmethod
    def _parse_modflowgwf_kwargs(cfg):

        if isinstance(cfg['simulation'], dict):
            # create simulation
            sim = flopy.mf6.MFSimulation(**cfg['simulation'])
            cfg['model']['simulation'] = sim
            sim_ws = cfg['simulation']['sim_ws']
        elif isinstance(cfg['simulation'], mf6.MFSimulation):
            sim_ws = cfg['simulation'].sim_ws
        else:
            raise TypeError('unrecognized configuration input for simulation.')

        # listing file
        cfg['model']['list'] = os.path.join(sim_ws,
                                            cfg['model']['list_filename_fmt']
                                            .format(cfg['model']['modelname']))

        # newton options
        if cfg['model']['options'].get('newton', False):
            cfg['model']['options']['newtonoptions'] = ['']
        if cfg['model']['options'].get('newton_under_relaxation', False):
            cfg['model']['options']['newtonoptions'] = ['under_relaxation']
        cfg['model'].update(cfg['model']['options'])
        return cfg

    @classmethod
    def setup_from_yaml(cls, yamlfile, verbose=False):
        """Make a model from scratch, using information in a yamlfile.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with inset setup information.

        Returns
        -------
        m : MF6model.MF6model instance
        """

        cfg = cls.load_cfg(yamlfile, verbose=verbose)
        print('\nSetting up {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        cfg = cls._parse_modflowgwf_kwargs(cfg)
        kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
        m = cls(cfg=cfg, **kwargs)

        if 'grid' not in m.cfg.keys():
            m.setup_grid()

        # set up tdis package
        m.setup_tdis()

        # set up all of the packages specified in the config file
        package_list = m.package_list #['sfr'] #m.package_list # ['tdis', 'dis', 'npf', 'oc']
        for pkg in package_list:
            package_setup = getattr(cls, 'setup_{}'.format(pkg.strip('6')))
            if not callable(package_setup):
                package_setup = getattr(MFsetupMixin, 'setup_{}'.format(pkg.strip('6')))
            package_setup(m)


        print('finished setting up model in {:.2f}s'.format(time.time() - t0))
        print('\n{}'.format(m))
        # Export a grid outline shapefile.
        #write_bbox_shapefile(m.sr, '../gis/model_bounds.shp')
        #print('wrote bounding box shapefile')
        return m

    @staticmethod
    def load_cfg(yamlfile, verbose=False):
        """Load model configuration info, adjusting paths to model_ws."""
        return load_cfg(yamlfile, default_file='/mf6_defaults.yml')

    @classmethod
    def load(cls, yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = cls.load_cfg(yamlfile, verbose=verbose)
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        cfg = cls._parse_modflowgwf_kwargs(cfg)
        kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
        m = cls(cfg=cfg, **kwargs)

        if 'grid' not in m.cfg.keys():
            # apply model name if grid_file includes format string
            grid_file = cfg['setup_grid']['grid_file'].format(m.name)
            m.cfg['setup_grid']['grid_file'] = grid_file
            if os.path.exists(grid_file):
                print('Loading model grid definition from {}'.format(grid_file))
                m.cfg['grid'] = load(grid_file)
            else:
                m.setup_grid()

        # execute the flopy load code on the pre-defined simulation and model instances
        # (so that the end result is a MFsetup.MF6model instance)
        # (kludgy)
        sim = flopy_mfsimulation_load(cfg['model']['simulation'], m)
        m = sim.get_model(model_name=m.name)
        print('finished loading model in {:.2f}s'.format(time.time() - t0))
        return m