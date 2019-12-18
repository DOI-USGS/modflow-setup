import sys
import os
import time
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
mf6 = flopy.mf6
from gisutils import get_values_at_points
from .discretization import (make_idomain, deactivate_idomain_above,
                             find_remove_isolated_cells,
                             create_vertical_pass_through_cells)
from .fileio import (load, dump, load_cfg,
                     flopy_mfsimulation_load)
from .grid import setup_structured_grid
from .mf5to6 import get_package_name
from .obs import setup_head_observations
from .tdis import setup_perioddata, parse_perioddata_groups
from .tmr import Tmr
from .units import lenuni_text, itmuni_text
from .utils import update, get_input_arguments, flatten, get_packages
from .wells import setup_wel_data
from .mfmodel import MFsetupMixin


class MF6model(MFsetupMixin, mf6.ModflowGwf):
    """Class representing a MODFLOW-6 model.
    """
    default_file = '/mf6_defaults.yml'

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
                                     'wel', 'maw', 'obs', 'ims']
        self.cfg = load(self.source_path + self.default_file) #'/mf6_defaults.yml')
        self.cfg['filename'] = self.source_path + self.default_file #'/mf6_defaults.yml'
        self._set_cfg(cfg)   # set up the model configuration dictionary
        self.relative_external_paths = self.cfg.get('model', {}).get('relative_external_paths', True)
        self.model_ws = self._get_model_ws()

        # property attributes
        self._idomain = None

        # other attributes
        self._features = {} # dictionary for caching shapefile datasets in memory
        self._drop_thin_cells = self.cfg.get('dis', {}).get('drop_thin_cells', True)

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

    def _set_idomain(self):
        """Remake the idomain array from the source data,
        no data values in the top and bottom arrays, and
        so that cells above SFR reaches are inactive."""
        idomain_from_layer_elevations = make_idomain(self.dis.top.array,
                                                     self.dis.botm.array,
                                                     nodata=self._nodata_value,
                                                     minimum_layer_thickness=self.cfg['dis'].get('minimum_layer_thickness', 1),
                                                     drop_thin_cells=self._drop_thin_cells,
                                                     tol=1e-4)
        # include cells that are active in the existing idomain array
        # and cells inactivated on the basis of layer elevations
        idomain = (self.dis.idomain.array == 1) & (idomain_from_layer_elevations == 1)
        idomain = idomain.astype(int)

        # remove cells that are above stream cells
        if 'SFR' in self.get_package_list():
            idomain = deactivate_idomain_above(idomain, self.sfr.packagedata)

        # inactivate any isolated cells that could cause problems with the solution
        idomain = find_remove_isolated_cells(idomain, minimum_cluster_size=20)

        # create pass-through cells in inactive cells that have an active cell above and below
        # by setting these cells to -1
        idomain = create_vertical_pass_through_cells(idomain)

        self._idomain = idomain

        # re-write the input files
        self._setup_array('dis', 'idomain',
                          data={i: arr for i, arr in enumerate(idomain)},
                          datatype='array3d', write_fmt='%d', dtype=int)
        self.dis.idomain = self.cfg['dis']['griddata']['idomain']

    def _set_parent(self):
        """Set attributes related to a parent or source model
        if one is specified."""

        if self.cfg['parent']['version'] == 'mf6':
            raise NotImplementedError("MODFLOW-6 parent models")

        kwargs = self.cfg['parent'].copy()
        if kwargs is not None:
            kwargs = kwargs.copy()
            kwargs['f'] = kwargs.pop('namefile')

            # load only specified packages that the parent model has
            packages_in_parent_namefile = get_packages(os.path.join(kwargs['model_ws'],
                                                                    kwargs['f']))
            specified_packages = set(self.cfg['model'].get('packages', set()))
            # get equivalent packages to load if parent is another MODFLOW version;
            # then flatten (a package may have more than one equivalent)
            parent_packages = [get_package_name(p, kwargs['version'])
                               for p in specified_packages]
            parent_packages = {item for subset in parent_packages for item in subset}
            load_only = list(set(packages_in_parent_namefile).intersection(parent_packages))
            kwargs['load_only'] = load_only
            kwargs = get_input_arguments(kwargs, fm.Modflow.load, warn=False)

            print('loading parent model {}...'.format(os.path.join(kwargs['model_ws'],
                                                                   kwargs['f'])))
            t0 = time.time()
            self._parent = fm.Modflow.load(**kwargs)
            print("finished in {:.2f}s\n".format(time.time() - t0))

            # parent model units
            if 'length_units' not in self.cfg['parent']:
                self.cfg['parent']['length_units'] = lenuni_text[self.parent.dis.lenuni]
            if 'time_units' not in self.cfg['parent']:
                self.cfg['parent']['time_units'] = itmuni_text[self.parent.dis.itmuni]

            # set the parent model grid from mg_kwargs if not None
            # otherwise, convert parent model grid to MFsetupGrid
            mg_kwargs = self.cfg['parent'].get('SpatialReference',
                                          self.cfg['parent'].get('modelgrid', None))
            self._set_parent_modelgrid(mg_kwargs)

            # default_source_data, where omitted configuration input is
            # obtained from parent model by default
            if self.cfg['parent'].get('default_source_data'):
                self._parent_default_source_data = True
                if self.cfg['dis']['dimensions'].get('nlay') is None:
                    self.cfg['dis']['dimensions']['nlay'] = self.parent.dis.nlay
                if self.cfg['tdis'].get('start_date_time') is None:
                    self.cfg['tdis']['start_date_time'] = self.cfg['parent']['start_date_time']
                # only get time dis information from parent if
                # no periodata groups are specified, and nper is not specified under dimensions
                has_perioddata_groups = any([isinstance(k, dict)
                                             for k in self.cfg['tdis']['perioddata'].values()])
                if not has_perioddata_groups:
                    if self.cfg['tdis']['dimensions'].get('nper') is None:
                        self.cfg['dis']['nper'] = self.parent.dis.nper
                    for var in ['perlen', 'nstp', 'tsmult']:
                        if self.cfg['dis']['perioddata'].get(var) is None:
                            self.cfg['dis']['perioddata'][var] = self.parent.dis.__dict__[var].array
                    if self.cfg['sto'].get('steady') is None:
                        self.cfg['sto']['steady'] = self.parent.dis.steady.array

    def _update_grid_configuration_with_dis(self):
        """Update grid configuration with any information supplied to dis package
        (so that settings specified for DIS package have priority). This method
        is called by MFsetupMixin.setup_grid.
        """
        for param in ['nrow', 'ncol']:
            if param in self.cfg['dis']['dimensions']:
                self.cfg['setup_grid'][param] = self.cfg['dis']['dimensions'][param]
        for param in ['delr', 'delc']:
            if param in self.cfg['dis']['griddata']:
                self.cfg['setup_grid'][param] = self.cfg['dis']['griddata'][param]

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

    def setup_dis(self):
        """"""
        package = 'dis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # resample the top from the DEM
        if self.cfg['dis']['remake_top']:
            self._setup_array(package, 'top', datatype='array2d', write_fmt='%.2f')

        # make the botm array
        self._setup_array(package, 'botm', datatype='array3d', write_fmt='%.2f')

        # initial idomain input for creating a dis package instance
        self._setup_array(package, 'idomain', datatype='array3d', write_fmt='%d',
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
            if v not in kwargs:
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
        self._setup_array(package, 'strt', datatype='array3d', write_fmt='%.2f')

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
        hiKlakes_value = float(self.cfg['parent'].get('hiKlakes_value', 1e4))

        # make the k array
        self._setup_array(package, 'k', vmin=0, vmax=hiKlakes_value,
                          datatype='array3d', write_fmt='%.6e')

        # make the k33 array (kv)
        self._setup_array(package, 'k33', vmin=0, vmax=hiKlakes_value,
                          datatype='array3d', write_fmt='%.6e')

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
        self._setup_array(package, 'sy', datatype='array3d', write_fmt='%.6e')

        # make the ss array
        self._setup_array(package, 'ss', datatype='array3d', write_fmt='%.6e')

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
        self._setup_array(package, 'recharge', datatype='transient2d',
                          resample_method='nearest', write_fmt='%.6e',
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

    def setup_obs(self):
        """
        Sets up the OBS utility.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'obs'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # munge the observation data
        df = setup_head_observations(self,
                                     format=package,
                                     obsname_column='obsname')

        # reformat to flopy input format
        obsdata = df[['obsname', 'obstype', 'id']].to_records(index=False)
        filename = self.cfg[package]['filename_fmt'].format(self.name)
        obsdata = {filename: obsdata}

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        kwargs['continuous'] = obsdata
        kwargs = get_input_arguments(kwargs, mf6.ModflowUtlobs)
        obs = mf6.ModflowUtlobs(self,  **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return obs

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
        # parse both flopy and mf6-style input into flopy input
        for rec in ['printrecord', 'saverecord']:
            if rec in kwargs:
                data = kwargs[rec]
                mf6_input = {}
                for kper, words in data.items():
                    mf6_input[kper] = []
                    for var, instruction in words.items():
                        mf6_input[kper].append((var, instruction))
                kwargs[rec] = mf6_input
            elif 'period_options' in kwargs:
                mf6_input = defaultdict(list)
                for kper, options in kwargs['period_options'].items():
                    for words in options:
                        type, var, instruction = words.split()
                        if type == rec.replace('record', ''):
                            mf6_input[kper].append((var, instruction))
                if len(mf6_input) > 0:
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

    def setup_perimeter_boundary(self):
        """Set up constant head package for perimeter boundary.
        TODO: integrate perimeter boundary with wel package setup
        """
        print('setting up specified head perimeter boundary with CHD package...')
        t0 = time.time()

        tmr = Tmr(self.parent, self,
                  parent_head_file=self.cfg['parent']['headfile'],
                  inset_parent_layer_mapping=self.parent_layers,
                  copy_stress_periods=self.cfg['parent']['copy_stress_periods'])

        df = tmr.get_inset_boundary_heads()

        spd = {}
        by_period = df.groupby('per')
        tmp = mf6.ModflowGwfchd.stress_period_data.empty(self, maxbound=len(by_period.get_group(0)))[0]
        for per, df_per in by_period:
            spd[per] = tmp.copy() # need to make a copy otherwise they'll all be the same!!
            spd[per]['cellid'] = list(zip(df_per['k'], df_per['i'], df_per['j']))
            spd[per]['head'] = df_per['bhead']

        kwargs = flatten(self.cfg['chd'])
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfchd)
        kwargs['stress_period_data'] = spd
        chd = mf6.ModflowGwfchd(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return chd

    def write_input(self):
        """Same syntax as MODFLOW-2005 flopy
        """
        self.simulation.write_simulation()

    @staticmethod
    def _parse_model_kwargs(cfg):

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
    def load(cls, yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = load_cfg(yamlfile, verbose=verbose, default_file=cls.default_file) # '/mf6_defaults.yml')
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        cfg = cls._parse_model_kwargs(cfg)
        kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
        m = cls(cfg=cfg, **kwargs)

        if 'grid' not in m.cfg.keys():
            # apply model name if grid_file includes format string
            grid_file = cfg['setup_grid']['output_files']['grid_file'].format(m.name)
            m.cfg['setup_grid']['output_files']['grid_file'] = grid_file
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