import sys
import os
import time
import shutil
from collections import defaultdict
import numpy as np
import flopy
mf6 = flopy.mf6
from .discretization import (fix_model_layer_conflicts, verify_minimum_layer_thickness,
                             fill_layers, make_idomain, deactivate_idomain_above)
from .fileio import load, dump, check_source_files, load_array, save_array, load_cfg
from .interpolate import regrid
from .gis import get_values_at_points
from .grid import write_bbox_shapefile, get_grid_bounding_box, get_point_on_national_hydrogeologic_grid
from .tdis import setup_perioddata
from .utils import update, get_input_arguments
from .units import convert_length_units, convert_time_units
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

        # property attributes
        self._idomain = None

        # other attributes
        self._features = {} # dictionary for caching shapefile datasets in memory
        self._drop_thin_cells = self.cfg['dis'].get('drop_thin_cells', True)

        # arrays remade during this session
        self.updated_arrays = set()

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
            idomain = make_idomain(self.dis.top.array, self.dis.botm.array,
                                   nodata=self._nodata_value,
                                   minimum_layer_thickness=self.cfg['dis'].get('minimum_layer_thickness', 1),
                                   drop_thin_cells=True, tol=1e-4)
            # remove cells that are above stream cells
            if 'SFR' in self.get_package_list():
                idomain = deactivate_idomain_above(idomain, self.sfr.reach_data)
            self._idomain = idomain
            self.dis.idomain = idomain
        return self._idomain

    def _load_cfg(self, cfg):
        """Load configuration file; update cfg dictionary."""
        if isinstance(cfg, str):
            assert os.path.exists(cfg), "config file {} not found".format(cfg)
            updates = load(cfg)
            updates['filename'] = cfg
        elif isinstance(cfg, dict):
            updates = cfg
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
            kwargs = get_input_arguments(kwargs, mf6.MFSimulation.load, warn=False)
            self._sim = mf6.MFSimulation.load(**kwargs)
        else:
            # create simulation
            kwargs = get_input_arguments(kwargs, mf6.MFSimulation, warn=False)
            self._sim = mf6.MFSimulation(**kwargs)

        # make sure that the output paths exist
        self.external_path = self.cfg['external_path']
        output_paths = [self.cfg['intermediate_data']['output_folder'],
                        self.cfg['simulation']['sim_ws'],
                        os.path.join(self.cfg['simulation']['sim_ws'], self.external_path)]
        output_paths += list(self.cfg['postprocessing']['output_folders'].values())
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

    def _set_perioddata(self):
        """Sets up the perioddata DataFrame."""
        perioddata = self.cfg['tdis']['perioddata'].copy()
        if perioddata.get('perlen_units') is None:
            perioddata['model_time_units'] = self.time_units
        perioddata.update({'nper': self.cfg['tdis']['dimensions']['nper'],
                           'steady': self.cfg['sto']['steady'],
                           'oc': self.cfg['oc']['saverecord']})
        self._perioddata = setup_perioddata(self.cfg['tdis']['options']['start_date_time'],
                                            self.cfg['tdis']['options'].get('end_date_time'),
                                            **perioddata)

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
        intermediate_paths = self.cfg['intermediate_data'][var]
        if isinstance(intermediate_paths, str):
            intermediate_paths = [intermediate_paths]
        external_path = os.path.basename(os.path.normpath(self.external_path))
        input = []
        for f in intermediate_paths:
            outf = os.path.join(external_path, os.path.split(f)[1])
            input.append({'filename': outf})
            shutil.copy(f, os.path.normpath(self.external_path))
        if len(input) == 1:
            input = input[0]
        return input

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
        dump(grid_file, grid)
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

        # make the idomain array
        self._setup_array(package, 'idomain', by_layer=True, write_fmt='%d')

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
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfdis)
        dis = mf6.ModflowGwfdis(model=self, **kwargs)
        self._perioddata = None  # reset perioddata
        self._modelgrid = None  # override DIS package grid setup
        self._isbc = None  # reset BC property arrays
        self._idomain = None
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    def setup_dis_old(self):
        """
        Sets up the DIS package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'dis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        minimum_layer_thickness = self.cfg['dis']['minimum_layer_thickness']
        remake_arrays = self.cfg['dis']['remake_arrays']
        regrid_top_from_dem = self.cfg['dis']['regrid_top_from_dem']

        # source data
        top_file = self.cfg['dis']['source_data']['top']
        botm_files = self.cfg['dis']['source_data']['botm']
        elevation_units = self.cfg['dis']['source_data']['elevation_units']

        # set paths to intermediate files and external files
        self.setup_external_filepaths(package, 'top', self.cfg['dis']['top_filename'],
                                      nfiles=1)
        self.setup_external_filepaths(package, 'botm', self.cfg['dis']['botm_filename_fmt'],
                                      nfiles=self.nlay)
        self.setup_external_filepaths(package, 'idomain', self.cfg['dis']['idomain_filename_fmt'],
                                      nfiles=self.nlay)

        if remake_arrays:
            check_source_files(botm_files.values())

            # resample the top from the DEM
            if regrid_top_from_dem:
                check_source_files(top_file)
                elevation_units_mult = convert_length_units(elevation_units, self.length_units)
                #top = self.get_raster_statistics_for_cells(top)
                top = self.get_raster_values_at_cell_centers(top_file)
                # save out the model top
                top *= elevation_units_mult
                save_array(self.cfg['intermediate_data']['top'][0], top, fmt='%.2f')
                self.updated_arrays.add('top')
            else:
                top = self.load_array(self.cfg['intermediate_data']['top'])

            # sample botm elevations from rasters
            sorted_keys = sorted(botm_files.keys())
            assert sorted_keys[-1] == self.nlay - 1, \
                "Max bottom layer specified is {}: {}; " \
                "need to provide surface for model botm".format(sorted_keys[-1],
                                                                botm_files[sorted_keys[-1]])
            all_surfaces = np.zeros((self.nlay+1, self.nrow, self.ncol), dtype=float) * np.nan
            all_surfaces[0] = top
            for k, rasterfile in botm_files.items():
                all_surfaces[k+1] = self.get_raster_values_at_cell_centers(botm_files[k])

            # for layers without a surface, set bottom elevation
            # so that layer thicknesses between raster surfaces are equal
            all_surfaces = fill_layers(all_surfaces)

            # fix any layering conflicts and save out botm files
            botm = np.array(all_surfaces[1:])
            botm = fix_model_layer_conflicts(top, botm,
                                             minimum_thickness=minimum_layer_thickness)
            for i, f in enumerate(self.cfg['intermediate_data']['botm']):
                save_array(f, botm[i], fmt='%.2f')
            self.updated_arrays.add('botm')

            # setup idomain from invalid botm elevations
            idomain = np.ones((self.nlay, self.nrow, self.ncol))
            idomain[np.isnan(botm)] = 0
            idomain = idomain.astype(int)
            self._idomain = idomain.astype(int)

            for i, f in enumerate(self.cfg['intermediate_data']['idomain']):
                save_array(f, self.idomain[i], fmt='%d')
            self.updated_arrays.add('idomain')
            self._isbc = None # reset the isbc property

        else: # check bottom of layer 1 to confirm that it is the right shape
            self.load_array(self.cfg['intermediate_data']['botm'][0])

        # put together keyword arguments for dis package
        kwargs = self.cfg['grid'].copy() # nrow, ncol, delr, delc
        kwargs.update(self.cfg['dis']['dimensions']) # nper, nlay, etc.
        kwargs.update(self.cfg['dis']['griddata'])

        # we need flopy to read the intermediate files
        # (it will write the files listed under [<package>][<variable>] names in cfg)
        kwargs.update({'top': self.get_flopy_external_file_input('top'),
                       'botm': self.get_flopy_external_file_input('botm'),
                       'idomain': self.get_flopy_external_file_input('idomain')
                      })

        # modelgrid: dis arguments
        remaps = {'xoff': 'xorigin',
                  'yoff': 'yorigin',
                  'rotation': 'angrot'}

        for modelgridk, disk in remaps.items():
            kwargs[disk] = kwargs.pop(modelgridk)
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfdis)

        # create the dis package
        dis = mf6.ModflowGwfdis(model=self, **kwargs)

        print('Checking layer thicknesses...')
        # get copies of arrays as they are in the DIS package
        # so that arrays only have to be fetched from MFArray instances once
        # if array's weren't recreated, flopy will load them from external files
        #top = self.dis.top.array.copy()
        #botm = self.dis.botm.array.copy()
        #idomain = self.dis.idomain.array.copy()
        # MF6 array access is too slow; read arrays directly for now
        if not remake_arrays:
            top = self.load_array(self.cfg['intermediate_data']['top'])
            botm = self.load_array(self.cfg['intermediate_data']['botm'])
            idomain = self.load_array(self.cfg['intermediate_data']['idomain'])
            self._idomain = idomain.astype(int)

        # check layer thicknesses (in case bad files were read in)
        isvalid = verify_minimum_layer_thickness(top, botm, idomain,
                                                 minimum_layer_thickness)
        if not isvalid:
            print('Fixing cell thicknesses less than {} {}'.format(minimum_layer_thickness, self.length_units))
            botm = fix_model_layer_conflicts(top,
                                             botm,
                                             idomain,
                                             minimum_thickness=minimum_layer_thickness)
            for i, f in enumerate(sorted(self.cfg['intermediate_data']['botm'])):
                save_array(f, botm[i], fmt='%.2f')

            # horrible kludge to deal with flopy external file handling
            kwargs.update({'botm': self.get_flopy_external_file_input('botm'),
                           })
            dis = mf6.ModflowGwfdis(model=self, **kwargs)

        print("{} setup finished in {:.2f}s\n".format(package.upper(), time.time() - t0))
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
                          resample_method='linear', write_fmt='%.6e')

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfrcha)
        rch = mf6.ModflowGwfrcha(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return rch

    def setup_sfr(self):
        """
        Sets up the SFR package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'sfr'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        sfr = None
        self._idomain = None
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return sfr

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

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
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
                    for var, instruction in words.items():
                        mf6_input[kper] = [(var, instruction)]
                kwargs[rec] = mf6_input
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfoc)
        oc = mf6.ModflowGwfoc(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return oc

    @staticmethod
    def setup_from_yaml(yamlfile, verbose=False):
        """Make a model from scratch, using information in a yamlfile.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with inset setup information.

        Returns
        -------
        m : MF6model.MF6model instance
        """

        cfg = MF6model.load_cfg(yamlfile, verbose=verbose)
        print('\nSetting up {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        # create simulation
        sim = flopy.mf6.MFSimulation(**cfg['simulation'])
        cfg['model']['simulation'] = sim

        kwargs = get_input_arguments(cfg['model'], MF6model)
        m = MF6model(cfg=cfg, **kwargs)

        if 'grid' not in m.cfg.keys():
            m.setup_grid()

        # set up all of the packages specified in the config file
        package_list = ['tdis', 'dis', 'npf', 'oc']  #m.package_list
        for pkg in package_list:
            package_setup = getattr(MF6model, 'setup_{}'.format(pkg.strip('6')))
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
