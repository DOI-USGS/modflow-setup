import os
import time
import numpy as np
import pandas as pd
from .gis import shp2df, get_values_at_points, intersect, project, get_proj4
from .grid import MFsetupGrid, get_ij, write_bbox_shapefile
from .fileio import load, dump, load_array, save_array, check_source_files, flopy_mf2005_load, \
    load_cfg, setup_external_filepaths
from .utils import update, get_input_arguments
from .interpolate import interp_weights, interpolate, regrid, get_source_dest_model_xys
from .lakes import make_lakarr2d, make_bdlknc_zones, make_bdlknc2d
from .utils import update, get_input_arguments
from .sourcedata import ArraySourceData, MFArrayData, TabularSourceData, setup_array
from .units import convert_length_units, convert_time_units, convert_flux_units, lenuni_text, itmuni_text
from sfrmaker import lines
from sfrmaker.utils import assign_layers


class MFsetupMixin():
    """Mixin class for shared functionality between MF6model and MFnwtModel.
    Meant to be inherited by both those classes and not be called directly.

    https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
    """

    def __init__(self, parent):

        # property attributes
        self._nper = None
        self._perioddata = None
        self._sr = None
        self._modelgrid = None
        self._bbox = None
        self._parent = parent
        self._parent_layers = None
        self._parent_mask = None
        self._lakarr_2d = None
        self._isbc_2d = None
        self._lakarr = None
        self._isbc = None
        self._lake_bathymetry = None
        self._precipitation = None
        self._evaporation = None
        self._lake_recharge = None
        self._nodata_value = -9999

        # flopy settings
        self._mg_resync = False

        self._features = {}  # dictionary for caching shapefile datasets in memory

        # arrays remade during this session
        self.updated_arrays = set()

        # cache of interpolation weights to speed up regridding
        self._interp_weights = None

    def __repr__(self):
        txt = '{} model:\n'.format(self.name)
        if self.parent is not None:
            txt += 'Parent model: {}/{}\n'.format(self.parent.model_ws, self.parent.name)
        if self.modelgrid is not None:
            txt += 'CRS: {}\n'.format(self.modelgrid.proj4)
            if self.modelgrid.epsg is not None:
                txt += '(epsg: {})\n'.format(self.modelgrid.epsg)
            txt += 'Bounds: {}\n'.format(self.modelgrid.extent)
            txt += 'Grid spacing: {:.2f} {}\n'.format(self.modelgrid.delr[0],
                                                      self.modelgrid.units)
            txt += '{:d} layer(s), {:d} row(s), {:d} column(s), {:d} stress period(s)\n'\
                .format(self.nlay, self.nrow, self.ncol, self.nper)
        txt += 'Packages:'
        for pkg in self.get_package_list():
            txt += ' {}'.format(pkg.lower())
        txt += '\n'
        txt += '{:d} LAKE package lakes'.format(self.nlakes)
        txt += '\n'
        return txt

    def __eq__(self, other):
        """Test for equality to another model object."""
        if not isinstance(other, self.__class__):
            return False
        if other.get_package_list() != self.get_package_list():
            return False
        if other.modelgrid != self.modelgrid:
            return False
        if other.nlay != self.nlay:
            return False
        if not np.array_equal(other.perioddata, self.perioddata):
            return False
        #  TODO: add checks of actual array values and other parameters
        return True

    @property
    def nper(self):
        if self.perioddata is not None:
            return len(self.perioddata)

    @property
    def nrow(self):
        if self.modelgrid.grid_type == 'structured':
            return self.modelgrid.nrow

    @property
    def ncol(self):
        if self.modelgrid.grid_type == 'structured':
            return self.modelgrid.ncol

    @property
    def modelgrid(self):
        if self._modelgrid is None:
            kwargs = self.cfg.get('grid').copy()
            if kwargs is not None:
                if np.isscalar(kwargs['delr']):
                    kwargs['delr'] = np.ones(kwargs['ncol'], dtype=float) * kwargs['delr']
                if np.isscalar(kwargs['delc']):
                    kwargs['delc'] = np.ones(kwargs['nrow'], dtype=float) * kwargs['delc']
                kwargs['lenuni'] = 2 # use units of meters for model grid
                renames = {'rotation': 'angrot'}
                for k, v in renames.items():
                    if k in kwargs:
                        kwargs[v] = kwargs.pop(k)
                kwargs = get_input_arguments(kwargs, MFsetupGrid)
                self._modelgrid = MFsetupGrid(**kwargs)
        return self._modelgrid

    @property
    def bbox(self):
        if self._bbox is None and self.modelgrid is not None:
            self._bbox = self.modelgrid.bbox
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
            self._set_perioddata()
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
        """Definitive list of packages. Get from namefile input first
        (as in mf6 input), then look under model input.
        """
        packages = self.cfg.get('nam', {}).get('packages', [])
        if len(packages) == 0:
            packages = self.cfg['model'].get('packages', [])
        return [p for p in self._package_setup_order
                if p in packages]

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
        return self.cfg['intermediate_data']['output_folder']

    @property
    def interp_weights(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to inset."""
        if self._interp_weights is None:
            parent_xy, inset_xy = get_source_dest_model_xys(self.parent,
                                                                        self)
            self._interp_weights = interp_weights(parent_xy, inset_xy)
        return self._interp_weights

    @property
    def parent_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the inset model domain. Used to speed up interpolation
        of parent grid values onto inset grid."""
        if self._parent_mask is None:
            x, y = np.squeeze(self.bbox.exterior.coords.xy)
            pi, pj = get_ij(self.parent.modelgrid, x, y)
            pad = 2
            i0 = np.max([pi.min() - pad, 0])
            i1 = np.min([pi.max() + pad, self.parent.nrow])
            j0 = np.max([pj.min() - pad, 0])
            j1 = np.min([pj.max() + pad, self.parent.ncol])
            mask = np.zeros((self.parent.nrow, self.parent.ncol), dtype=bool)
            mask[i0:i1, j0:j1] = True
            self._parent_mask = mask
        return self._parent_mask

    @property
    def nlakes(self):
        if self.lakarr is not None:
            return int(np.max(self.lakarr))
        else:
            return 0

    @property
    def _lakarr2d(self):
        """2-D array of areal extent of lakes. Non-zero values
        correspond to lak package IDs."""
        if self._lakarr_2d is None:
            self._set_lakarr2d()
        return self._lakarr_2d

    @property
    def lakarr(self):
        """3-D array of lake extents in each layer. Non-zero values
        correspond to lak package IDs. Extent of lake in
        each layer is based on bathymetry and model layer thickness.

        TODO : figure out how to handle lakes with MF6
        """
        if self._lakarr is None:
            self.setup_external_filepaths('lak', 'lakarr',
                                          self.cfg['lak']['{}_filename_fmt'.format('lakarr')],
                                          nfiles=self.nlay)
            if self.isbc is None:
                return None
            else:
                self._set_lakarr()
        return self._lakarr

    @property
    def _isbc2d(self):
        """2-D array indicating which cells have lakes.
        -1 : well
        0 : no lake
        1 : lak package lake (lakarr > 0)
        2 : high-k lake
        3 : sfr
        """
        if self._isbc_2d is None:
            self._set_isbc2d()
        return self._isbc_2d

    @property
    def isbc(self):
        """3D array indicating which cells have a lake in each layer.
        -1 : well
        0 : no lake
        1 : lak package lake (lakarr > 0)
        2 : high-k lake
        3 : sfr
        """
        if 'DIS' not in self.get_package_list():
            return None
        elif self._isbc is None:
            self._set_isbc()
        return self._isbc

    @property
    def lake_bathymetry(self):
        """Put lake bathymetry setup logic here instead of DIS package.
        """

        if self._lake_bathymetry is None:
            self._set_lake_bathymetry()
        return self._lake_bathymetry

    @property
    def precipitation(self):
        """Lake precipitation at each stress period, in model units.
        """
        if self._precipitation is None or \
                len(self._precipitation) != self.nper:
            precip = self.cfg['lak'].get('precip', 0)
            # copy to all stress periods
            if np.isscalar(precip):
                precip = [precip] * self.nper
            elif len(precip) < self.nper:
                for i in range(self.nper - len(precip)):
                    precip.append(precip[-1])
            self._precipitation = np.array(precip)
        return self._precipitation

    @property
    def evaporation(self):
        """Lake evaporation at each stress period, in model units.
        """
        if self._evaporation is None or \
                len(self._evaporation) != self.nper:
            evap = self.cfg['lak'].get('evap', 0)
            # copy to all stress periods
            if np.isscalar(evap):
                evap = [evap] * self.nper
            elif len(evap) < self.nper:
                for i in range(self.nper - len(evap)):
                    evap.append(evap[-1])
            self._evaporation = np.array(evap)
        return self._evaporation

    @property
    def lake_recharge(self):
        """Recharge value to apply to high-K lakes, in model units.
        """
        if self.precipitation is not None and self.evaporation is not None:
            self._lake_recharge = self.precipitation - self.evaporation
            return self._lake_recharge

    def load_array(self, filename):
        if isinstance(filename, list):
            arrays = []
            for f in filename:
                arrays.append(load_array(f,
                                         shape=(self.nrow, self.ncol),
                                         nodata=self._nodata_value
                                         )
                              )
            return np.array(arrays)
        return load_array(filename, shape=(self.nrow, self.ncol))

    def load_features(self, filename, filter=None,
                      id_column=None, include_ids=None,
                      cache=True):
        """Load vector and attribute data from a shapefile;
        cache it to the _features dictionary.
        """
        if isinstance(filename, str):
            features_file = [filename]

        dfs_list = []
        for f in features_file:
            if f not in self._features.keys():
                if os.path.exists(f):
                    features_proj_str = get_proj4(f)
                    model_proj_str = "epsg:{}".format(self.cfg['setup_grid']['epsg'])
                    if filter is None:
                        if self.bbox is not None:
                            bbox = self.bbox
                        elif self.parent.modelgrid is not None:
                            bbox = self.parent.modelgrid.bbox
                            model_proj_str = self.parent.modelgrid.proj_str
                            assert model_proj_str is not None

                        if features_proj_str.lower() != model_proj_str:
                            filter = project(bbox, model_proj_str, features_proj_str).bounds
                        else:
                            filter = bbox.bounds

                    df = shp2df(f, filter=filter)
                    df.columns = [c.lower() for c in df.columns]
                    if features_proj_str.lower() != model_proj_str:
                        df['geometry'] = project(df['geometry'], features_proj_str, model_proj_str)
                    if cache:
                        print('caching data in {}...'.format(f))
                        self._features[f] = df
                else:
                    print('feature input file {} not found'.format(f))
                    return
            else:
                df = self._features[f]
            if id_column is not None and include_ids is not None:
                id_column = id_column.lower()
                df = df.loc[df[id_column].isin(include_ids)]
            dfs_list.append(df)
        df = pd.concat(dfs_list)
        return df

    def get_boundary_cells(self):
        """Get the i, j locations of cells along the model perimeter.

        Returns
        -------
        k, i, j : 1D numpy arrays of ints
            zero-based layer, row, column locations of boundary cells
        """
        # top row, right side, left side, bottom row
        i_top = [0] * self.ncol
        j_top = list(range(self.ncol))
        i_left = list(range(1, self.nrow - 1))
        j_left = [0] * (self.nrow - 2)
        i_right = i_left
        j_right = [self.ncol - 1] * (self.nrow - 2)
        i_botm = [self.nrow - 1] * self.ncol
        j_botm = j_top
        i = i_top + i_left + i_right + i_botm
        j = j_top + j_left + j_right + j_botm

        assert len(i) == 2 * self.nrow + 2 * self.ncol - 4
        nlaycells = len(i)
        k = sorted(list(range(self.nlay)) * len(i))
        i = i * self.nlay
        j = j * self.nlay
        assert np.sum(k[nlaycells:nlaycells * 2]) == nlaycells
        return k, i, j

    def regrid_from_parent(self, parent_array,
                           mask=None,
                           method='linear'):
        """Interpolate values in parent array onto
        the inset model grid, using SpatialReference instances
        attached to the parent and inset models.

        Parameters
        ----------
        parent_array : ndarray
            Values from parent model to be interpolated to inset grid.
            1 or 2-D numpy array of same sizes as a
            layer of the parent model.
        mask : ndarray (bool)
            1 or 2-D numpy array of same sizes as a
            layer of the parent model. True values
            indicate cells to include in interpolation,
            False values indicate cells that will be
            dropped.
        method : str ('linear', 'nearest')
            Interpolation method.
        """
        if mask is not None:
            return regrid(parent_array, self.parent.modelgrid, self.modelgrid,
                          mask1=mask,
                          method=method)
        if method == 'linear':
            parent_values = parent_array.flatten()[self.parent_mask.flatten()]
            regridded = interpolate(parent_values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = regrid(parent_array, self.parent.modelgrid, self.modelgrid,
                               method='nearest')
        regridded = np.reshape(regridded, (self.nrow, self.ncol))
        return regridded

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
        filepaths : list
            List of external file paths

        Adds intermediated file paths to model.cfg[<package>]['intermediate_data']
        Adds external file paths to model.cfg[<package>][<variable_name>]
        """
        return setup_external_filepaths(self, package, variable_name,
                                        filename_format, nfiles=nfiles)

    def _reset_bc_arrays(self):
        """Reset the boundary condition property arrays in order.
        _lakarr2d
        _isbc2d  (depends on _lakarr2d)
        _lake_bathymetry (depends on _isbc2d)
        _isbc  (depends on _isbc2d)
        _lakarr  (depends on _isbc and _lakarr2d)
        """
        self._set_lakarr2d() # calls self._set_isbc2d(), which calls self._set_lake_bathymetry()
        self._set_isbc() # calls self._set_lakarr()

    def _set_isbc2d(self):
            isbc = np.zeros((self.nrow, self.ncol))
            lakesdata = None
            lakes_shapefile = self.cfg['lak'].get('source_data', {}).get('lakes_shapefile')
            if lakes_shapefile is not None:
                if isinstance(lakes_shapefile, str):
                    lakes_shapefile = {'filename': lakes_shapefile}
                kwargs = get_input_arguments(lakes_shapefile, self.load_features)
                kwargs.pop('include_ids')  # load all lakes in shapefile
                lakesdata = self.load_features(**kwargs)
            if lakesdata is not None:
                isanylake = intersect(lakesdata, self.modelgrid)
                isbc[isanylake > 0] = 2
                isbc[self._lakarr2d > 0] = 1
            if 'SFR' in self.get_package_list():
                i, j = self.sfr.reach_data['i'], \
                          self.sfr.reach_data['j']
                isbc[i, j][isbc[i, j] != 1] = 3
            if 'WEL' in self.get_package_list():
                i, j = self.wel.stress_period_data[0]['i'], \
                       self.wel.stress_period_data[0]['j']
                isbc[i, j][isbc[i, j] == 0] = -1
            self._isbc_2d = isbc
            self._set_lake_bathymetry()

    def _set_isbc(self):
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
            self._set_lakarr()

    def _set_lakarr2d(self):
            lakarr2d = np.zeros((self.nrow, self.ncol))
            if 'lak' in self.package_list:
                lakes_shapefile = self.cfg['lak'].get('source_data', {}).get('lakes_shapefile', {}).copy()
                lakesdata = self.load_features(**lakes_shapefile)  # caches loaded features
                lakes_shapefile['lakesdata'] = lakesdata
                lakes_shapefile.pop('filename')
                lakarr2d = make_lakarr2d(self.modelgrid, **lakes_shapefile)
            self._lakarr_2d = lakarr2d
            self._set_isbc2d()

    def _set_lakarr(self):
        self.setup_external_filepaths('lak', 'lakarr',
                                      self.cfg['lak']['{}_filename_fmt'.format('lakarr')],
                                      nfiles=self.nlay)
        # assign lakarr values from 3D isbc array
        lakarr = np.zeros((self.nlay, self.nrow, self.ncol), dtype=int)
        for k in range(self.nlay):
            lakarr[k][self.isbc[k] == 1] = self._lakarr2d[self.isbc[k] == 1]
        for k, ilakarr in enumerate(lakarr):
            save_array(self.cfg['intermediate_data']['lakarr'][0][k], ilakarr, fmt='%d')
        self._lakarr = lakarr

    def _set_lake_bathymetry(self):
        bathymetry_file = self.cfg.get('lak', {}).get('source_data', {}).get('bathymetry_raster')
        default_lake_depth = self.cfg['model'].get('default_lake_depth', 2)
        if bathymetry_file is not None:
            lmult = 1.0
            if isinstance(bathymetry_file, dict):
                lmult = convert_length_units(bathymetry_file.get('length_units', 0),
                                             self.length_units)
                bathymetry_file = bathymetry_file['filename']

            # sample pre-made bathymetry at grid points
            bathy = get_values_at_points(bathymetry_file,
                                         x=self.modelgrid.xcellcenters.ravel(),
                                         y=self.modelgrid.ycellcenters.ravel(),
                                         out_of_bounds_errors='coerce')
            bathy = np.reshape(bathy, (self.nrow, self.ncol)) * lmult
            bathy[(bathy < 0) | np.isnan(bathy)] = 0

            # fill bathymetry grid in remaining lake cells with default lake depth
            # also ensure that all non lake cells have bathy=0
            fill = (bathy == 0) & (self._isbc2d > 0) & (self._isbc2d < 3)
            bathy[fill] = default_lake_depth
            bathy[(self._isbc2d > 1) & (self._isbc2d > 2)] = 0
        else:
            bathy = np.zeros((self.nrow, self.ncol))
        self._lake_bathymetry = bathy

    def _setup_array(self, package, var, vmin=-1e30, vmax=1e30,
                      source_model=None, source_package=None,
                      **kwargs):
        return setup_array(self, package, var, vmin=vmin, vmax=vmax,
                           source_model=source_model, source_package=source_package,
                           **kwargs)

    def load_grid(self, gridfile=None):
        """Load model grid information from a json or yml file."""
        if gridfile is None:
            if os.path.exists(self.cfg['setup_grid']['grid_file']):
                gridfile = self.cfg['setup_grid']['grid_file']
        print('Loading model grid information from {}'.format(gridfile))
        self.cfg['grid'] = load(gridfile)

    def setup_sfr(self):
        package = 'sfr'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # input
        flowlines = self.cfg['sfr'].get('source_data', {}).get('flowlines')
        if flowlines is not None:
            if 'nhdplus_paths' in flowlines.keys():
                nhdplus_paths = flowlines['nhdplus_paths']
                for f in nhdplus_paths:
                    if not os.path.exists(f):
                        print('SFR setup: missing input file: {}'.format(f))
                        nhdplus_paths.remove(f)
                if len(nhdplus_paths) == 0:
                    return

                # create an sfrmaker.lines instance
                filter = project(self.bbox, self.modelgrid.proj_str, 'epsg:4269').bounds
                lns = lines.from_NHDPlus_v2(NHDPlus_paths=nhdplus_paths,
                                            filter=filter)
            else:
                for key in ['filename', 'filenames']:
                    if key in flowlines:
                        kwargs = flowlines.copy()
                        kwargs['shapefile'] = kwargs.pop(key)
                        check_source_files(kwargs['shapefile'])
                        if 'epsg' not in kwargs:
                            kwargs['proj4'] = get_proj4(kwargs['shapefile'])
                        else:
                            kwargs['proj4'] = 'epsg:{}'.format(kwargs['epsg'])

                        filter = self.bbox.bounds
                        if kwargs['proj4'] != self.modelgrid.proj_str:
                            filter = project(self.bbox, self.modelgrid.proj_str, kwargs['proj4']).bounds
                        kwargs['filter'] = filter
                        # create an sfrmaker.lines instance
                        kwargs = get_input_arguments(kwargs, lines.from_shapefile)
                        lns = lines.from_shapefile(**kwargs)
                        break
        else:
            return

        # output
        output_path = self.cfg['sfr'].get('output_path')
        if output_path is not None:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        else:
            output_path = self.cfg['postprocessing']['output_folders']['shps']
            self.cfg['sfr']['output_path'] = output_path

        # create isfr array (where SFR cells will be populated)
        if self.version == 'mf6':
            active_cells = self.idomain.sum(axis=0) > 0
        else:
            active_cells = self.ibound.sum(axis=0) > 0
        isfr = active_cells & (self._isbc2d == 0)
        #  kludge to get sfrmaker to work with modelgrid
        self.modelgrid.model_length_units = self.length_units

        # create an sfrmaker.sfrdata instance from the lines instance
        from flopy.utils.reference import SpatialReference
        sr = SpatialReference(delr=self.modelgrid.delr, delc=self.modelgrid.delc,
                              xll=self.modelgrid.xoffset, yll=self.modelgrid.yoffset,
                              rotation=self.modelgrid.angrot, epsg=self.modelgrid.epsg,
                              proj4_str=self.modelgrid.proj_str)
        sfr = lns.to_sfr(sr=sr,
                         isfr=isfr
                         )
        if self.cfg['sfr']['set_streambed_top_elevations_from_dem']:
            sfr.set_streambed_top_elevations_from_dem(self.cfg['source_data']['dem'],
                                                      dem_z_units=self.cfg['source_data']['elevation_units'])
        else:
            sfr.reach_data['strtop'] = sfr.interpolate_to_reaches('elevup', 'elevdn')

        # assign layers to the sfr reaches
        botm = self.dis.botm.array.copy()
        layers, new_botm = assign_layers(sfr.reach_data, botm_array=botm)
        sfr.reach_data['k'] = layers
        if new_botm is not None:
            if self.cfg['intermediate_data'].get('botm') is None:
                f = os.path.normpath(os.path.join(self.model_ws,
                                                  self.external_path,
                                                  self.cfg['dis']['botm_filename_fmt'].format(self.nlay - 1)
                                                  ))
            else:
                f = self.cfg['intermediate_data']['botm'][-1]
            save_array(f, new_botm, fmt='%.2f')
            print('(new model botm after assigning SFR reaches to layers)')
            botm[-1] = new_botm
            self.dis.botm = botm

        # write reach and segment data tables
        sfr.write_tables('{}/{}'.format(output_path, self.name))

        # export shapefiles of lines, routing, cell polygons, inlets and outlets
        sfr.export_cells('{}/{}_sfr_cells.shp'.format(output_path, self.name))
        sfr.export_outlets('{}/{}_sfr_outlets.shp'.format(output_path, self.name))
        sfr.export_transient_variable('flow', '{}/{}_sfr_inlets.shp'.format(output_path, self.name))
        sfr.export_lines('{}/{}_sfr_lines.shp'.format(output_path, self.name))
        sfr.export_routing('{}/{}_sfr_routing.shp'.format(output_path, self.name))

        # attach the sfrmaker.sfrdata instance as an attribute
        self.sfrdata = sfr

        # create the flopy SFR package instance
        if self.version != 'mf6':
            sfr.create_ModflowSfr2(model=self, istcb2=223)
            sfr_package = sfr.ModflowSfr2
        else:
            sfr_package = sfr.create_mf6sfr(model=self)
            # monkey patch ModflowGwfsfr instance to behave like ModflowSfr2
            sfr_package.reach_data = sfr.ModflowSfr2.reach_data

        # reset dependent arrays
        self._reset_bc_arrays()
        if self.version == 'mf6':
            self._set_idomain()
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return sfr_package
