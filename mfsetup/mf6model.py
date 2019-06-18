import sys
import os
import time
import shutil
import numpy as np
import flopy
mf6 = flopy.mf6
from flopy.discretization.structuredgrid import StructuredGrid
from .discretization import fix_model_layer_conflicts, verify_minimum_layer_thickness, deactivate_idomain_above, fill_layers
from .fileio import load, dump, check_source_files, load_array, save_array, setup_external_filepaths
from .interpolate import regrid
from .gis import get_values_at_points
from .grid import write_bbox_shapefile, get_grid_bounding_box
from .tdis import setup_perioddata
from .utils import update, get_input_arguments
from .units import convert_length_units, convert_time_units


class MF6model(mf6.ModflowGwf):
    """Class representing a MODFLOW-6 model.
    """

    source_path = os.path.split(__file__)[0]
    # default configuration
    cfg = load(source_path + '/mf6_defaults.yml')
    cfg['filename'] = source_path + '/mf6_defaults.yml'

    def __init__(self, simulation, cfg=None,
                 modelname='model', exe_name='mf6',
                 version='mf6', **kwargs):
        mf6.ModflowGwf.__init__(self, simulation,
                                modelname, exe_name=exe_name, version=version,
                                **kwargs)

        # property attributes
        self._modelgrid = None
        self._idomain = None
        self._isbc = None
        self._nper = None
        self._perioddata = None

        self._features = {} # dictionary for caching shapefile datasets in memory

        self._load_cfg(cfg)  # update configuration dict with values in cfg

        # arrays remade during this session
        self.updated_arrays = set()

    @property
    def nper(self):
        if self._nper is None:
            self._set_period_data()
            self._nper = len(self.perioddata)
        return self._nper

    @property
    def nlay(self):
        return self.cfg['dis']['dimensions'].get('nlay', 1)

    @property
    def nrow(self):
        if self.modelgrid.grid_type == 'structured':
            return self.modelgrid.nrow

    @property
    def ncol(self):
        if self.modelgrid.grid_type == 'structured':
            return self.modelgrid.ncol

    @property
    def length_units(self):
        return self.cfg['dis']['options']['length_units']

    @property
    def time_units(self):
        return self.cfg['tdis']['options']['time_units']

    @property
    def modelgrid(self):
        if self._modelgrid is None:
            kwargs = self.cfg.get('grid').copy()
            if kwargs is not None:
                if np.isscalar(kwargs['delr']):
                    kwargs['delr'] = np.ones(kwargs['ncol'], dtype=float) * kwargs['delr']
                if np.isscalar(kwargs['delc']):
                    kwargs['delc'] = np.ones(kwargs['nrow'], dtype=float) * kwargs['delc']
                kwargs.pop('nrow')
                kwargs.pop('ncol')
                kwargs['angrot'] = kwargs.pop('rotation')
                self._modelgrid = StructuredGrid(**kwargs)
        return self._modelgrid

    @property
    def bbox(self):
        if self._bbox is None and self.sr is not None:
            self._bbox = get_grid_bounding_box(self.modelgrid)
        return self._bbox

    @property
    def perioddata(self):
        """DataFrame summarizing stress period information.
        Columns:
          start_datetime : pandas datetimes; start date/time of each stress period
          (does not include steady-state periods)
          end_datetime : pandas datetimes; end date/time of each stress period
          (does not include steady-state periods)
          time : float; cumulative MODFLOW time (includes steady-state periods)
          per : zero-based stress period
          perlen : stress period length in model time units
          nstp : number of timesteps in the stress period
          tsmult : timestep multiplier for stress period
          steady : True=steady-state, False=Transient
          oc : MODFLOW-6 output control options
        """
        if self._perioddata is None:
            self._set_period_data()
        return self._perioddata

    @property
    def parent(self):
        return self._parent

    @property
    def parent_layers(self):
        """Mapping between layers in source model and
        layers in destination model."""
        self._parent_layers = None
        if self._parent_layers is None:
            parent_layers = self.cfg['dis'].get('source_data', {}).get('botm', {}).get('from_parent')
            if parent_layers is None:
                parent_layers = dict(zip(range(self.parent.nlay), range(self.parent.nlay)))
            self._parent_layers = parent_layers
        return self._parent_layers

    @property
    def package_list(self):
        return [p for p in self._package_setup_order
                if p in self.cfg['model']['packages']]

    @property
    def perimeter_bc_type(self):
        """Dictates how perimeter boundaries are set up.

        if 'head'; a constant head package is created
            from the parent model starting heads
        if 'flux'; a specified flux boundary is created
            from parent model cell by cell flow output
            """
        perimeter_boundary_type = self.cfg['model'].get('perimeter_boundary_type')
        if perimeter_boundary_type is not None:
            if 'head' in perimeter_boundary_type:
                return 'head'
            if 'flux' in perimeter_boundary_type:
                return 'flux'

    @property
    def tmpdir(self):
        return self.cfg['intermediate_data']['tmpdir']

    @property
    def idomain(self):
        """3D array indicating which cells will be included in the simulation.
        Made a property so that it can be easily updated when any packages
        it depends on change.
        """
        if self._idomain is None and 'DIS' in self.packagelist:
            idomain = np.abs(~np.isnan(self.dis.botm.array).astype(int))
            # remove cells that are above stream cells
            if 'SFR' in self.get_package_list():
                idomain = deactivate_idomain_above(idomain, self.sfr.reach_data)
            self._idomain = idomain
        return self._idomain

    @property
    def isbc(self):
        """3D array summarizing the boundary conditions in each model cell.
       -1 : constant head
        0 : no boundary conditions
        1 : sfr
        2 : well
        3 : ghb
        """
        if 'DIS' not in self.get_package_list():
            return None
        elif self._isbc is None:
            isbc = np.zeros((self.nlay, self.nrow, self.ncol))
            isbc[0] = self._isbc2d

            lake_botm_elevations = self.dis.top.array - self.lake_bathymetry
            layer_tops = np.concatenate([[self.dis.top.array], self.dis.botm.array[:-1]])
            # lakes must be at least 10% into a layer to get simulated in that layer
            below = layer_tops > lake_botm_elevations + 0.1
            for i, ibelow in enumerate(below[1:]):
                if np.any(ibelow):
                    isbc[i+1][ibelow] = self._isbc2d[ibelow]
            # add other bcs
            if 'SFR' in self.get_package_list():
                k, i, j = self.sfr.reach_data['k'], \
                          self.sfr.reach_data['i'], \
                          self.sfr.reach_data['j']
                isbc[k, i, j][isbc[k, i, j] != 1] = 3
            if 'WEL' in self.get_package_list():
                k, i, j = self.wel.stress_period_data[0]['k'], \
                          self.wel.stress_period_data[0]['i'], \
                          self.wel.stress_period_data[0]['j']
                isbc[k, i, j][isbc[k, i, j] == 0] = -1
            self._isbc = isbc
            self._lakarr = None
        return self._isbc

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
                k[v] = {}
        for k, v in updates.items():
            if v is None:
                k[v] = {}
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
        output_paths = [self.cfg['intermediate_data']['tmpdir'],
                       self.cfg['simulation']['sim_ws'],
                       self.external_path]
        output_paths += list(self.cfg['postprocessing']['output_folders'].values())
        for folder in output_paths:
            if not os.path.exists(folder):
                os.makedirs(folder)
        # other variables
        self.cfg['external_files'] = {}

    def _set_period_data(self):
        """Sets up the perioddata DataFrame."""
        perioddata = self.cfg['tdis']['perioddata'].copy()
        if perioddata.get('perlen_units') is None:
            perioddata['model_time_units'] = self.time_units
        perioddata.update({'nper': self.cfg['tdis']['dimensions']['nper'],
                           'steady': self.cfg['sto']['steady'],
                           'oc': self.cfg['oc']['period_options']})
        self._perioddata = setup_perioddata(self.cfg['tdis']['options']['start_date_time'],
                                            self.cfg['tdis']['options'].get('end_date_time'),
                                            **perioddata)

    def load_array(self, filename):
        """Load an array and check the shape.
        """
        if isinstance(filename, list):
            arrays = []
            for f in filename:
                arrays.append(load_array(f, shape=(self.nrow, self.ncol)))
            return np.array(arrays)
        return load_array(filename, shape=(self.nrow, self.ncol))

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

        self.cfg['grid'] = grid
        dump(grid_file, self.cfg['grid'])
        if write_shapefile:
            write_bbox_shapefile(self.modelgrid,
                                 os.path.join(self.cfg['postprocessing']['output_folders']['shapefiles'],
                                              '{}_bbox.shp'.format(self.name)))

    def setup_external_filepaths(self, package, variable_name,
                                 filename_format, nfiles=1):
        """Set up external file paths for a MODFLOW package variable. Sets paths
        for intermediate files, which are written from the (processed) source data.
        Intermediate files are supplied to Flopy as external files for a given package
        variable. Flopy writes external files to a specified location when the MODFLOW
        package file is written. This method gets the external file paths that
        will be written by FloPy, and puts them in the configuration dictionary
        under their respective variables.

        Parameters
        ----------
        package : str
            Three-letter package abreviation (e.g. 'DIS' for discretization)
        variable_name : str
            FloPy name of variable represented by external files (e.g. 'top' or 'botm')
        filename_format : str
            File path to the external file(s). Can be a string representing a single file
            (e.g. 'top.dat'), or for variables where a file is written for each layer or
            stress period, a format string that will be formated with the zero-based layer
            number (e.g. 'botm{}.dat') for files botm0.dat, botm1.dat, ...
        nfiles : int
            Number of external files for the variable (e.g. nlay or nper)

        Returns
        -------
        Adds intermediated file paths to model.cfg[<package>]['intermediate_data']
        Adds external file paths to model.cfg[<package>][<variable_name>]
        """
        setup_external_filepaths(self, package, variable_name,
                                 filename_format, nfiles=nfiles)


    def setup_dis(self):
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
        tdis = mf6.ModflowTdis(self.simulation,
                               time_units=self.time_units,
                               start_date_time=self.perioddata['start_datetime'][0].strftime('%Y-%m-%d'),
                               nper=self.nper,
                               perioddata=perioddata)
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
        ic = None
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
        npf = None
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
        package = 'sto'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        kwargs = self.cfg['sto']['options'].copy()
        kwargs.update(self.cfg['sto']['griddata'].copy())
        kwargs['steady_state'] = {k: v for k, v in self.cfg['sto']['steady'].items() if v}
        kwargs['transient'] = {k: True for k, v in self.cfg['sto']['steady'].items() if not v}
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfsto)
        #sto = mf6.ModflowGwfsto(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return #sto

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
        rch = None
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
        wel = None
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
        oc = None
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

        m = MF6model(cfg=cfg, **cfg['model'])

        if 'grid' not in m.cfg.keys():
            m.setup_grid()

        # set up all of the packages specified in the config file
        package_list = m.cfg['nam']['packages']
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
        """Load model configuration info, adjusting paths to model_ws.
        """

        source_path = os.path.split(__file__)[0]
        # default configuration
        cfg = load(source_path + '/mf6_defaults.yml')
        cfg['filename'] = source_path + '/mf6_defaults.yml'

        # recursively update defaults with information from yamlfile
        update(cfg, load(yamlfile))
        cfg['model'].update({'verbose': verbose})
        cfg['simulation']['sim_ws'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
                                                cfg['simulation']['sim_ws'])

        def set_path(path):
            return os.path.join(cfg['simulation']['sim_ws'],
                                path)

        cfg['intermediate_data']['tmpdir'] = set_path(cfg['intermediate_data']['tmpdir'])
        cfg['external_path'] = set_path(cfg['external_path'])
        cfg['setup_grid']['grid_file'] = set_path(os.path.split(cfg['setup_grid']['grid_file'])[-1])
        mapping = {'shapefiles': 'shps'}
        for f in cfg['postprocessing']['output_folders']:
            path = mapping.get(f, f)
            cfg['postprocessing']['output_folders'][f] = set_path(path)
        return cfg