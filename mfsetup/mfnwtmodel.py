import sys
sys.path.append('..')
sys.path.append('/Users/aleaf/Documents/GitHub/sfrmaker')
import os
import time
import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd
from shapely.geometry import MultiPolygon, Point
import flopy
fm = flopy.modflow
from flopy.modflow import Modflow
from flopy.utils import binaryfile as bf
from .discretization import fix_model_layer_conflicts, verify_minimum_layer_thickness, fill_layers
from .tdis import setup_perioddata
from .gis import shp2df, get_values_at_points, intersect, project, get_proj4
from .grid import MFsetupGrid, get_ij, write_bbox_shapefile
from .fileio import load, dump, load_array, save_array, check_source_files, flopy_mf2005_load, \
    load_cfg, setup_external_filepaths
from .utils import update, get_input_arguments
from .interpolate import interp_weights, interpolate, regrid, get_source_dest_model_xys
from .lakes import make_lakarr2d, make_bdlknc_zones, make_bdlknc2d
from .utils import update, get_packages
from .obs import read_observation_data
from .sourcedata import ArraySourceData, MFArrayData, TabularSourceData, setup_array
from .tmr import Tmr
from .units import convert_length_units, convert_time_units, convert_flux_units, lenuni_text, itmuni_text
from .wateruse import get_mean_pumping_rates, resample_pumping_rates
from sfrmaker import lines
from sfrmaker.utils import assign_layers


class MFnwtModel(Modflow):
    """Class representing a MODFLOW-NWT model"""

    source_path = os.path.split(__file__)[0]

    def __init__(self, parent=None, cfg=None,
                 modelname='model', exe_name='mfnwt',
                 version='mfnwt', model_ws='.',
                 external_path='external/', **kwargs):

        Modflow.__init__(self, modelname, exe_name=exe_name, version=version,
                         model_ws=model_ws, external_path=external_path,
                         **kwargs)

        # default configuration
        self._package_setup_order = ['dis', 'bas6', 'upw', 'rch', 'oc',
                                     'ghb', 'lak', 'sfr',
                                     'wel', 'mnw2', 'gag', 'hyd', 'nwt']
        self.cfg = load(self.source_path + '/mfnwt_defaults.yml')
        self.cfg['filename'] = self.source_path + '/mfnwt_defaults.yml'

        # property attributes
        self._nper = None
        self._perioddata = None
        self._sr = None
        self._modelgrid = None
        self._bbox = None
        self._parent = parent
        self.__parent_mask = None
        self.__lakarr2d = None
        self.__isbc2d = None
        self._ibound = None
        self._lakarr = None
        self._isbc = None
        self._lake_bathymetry = None
        self._precipitation = None
        self._evaporation = None
        self._lake_recharge = None

        # flopy settings
        self._mg_resync = False

        self._features = {} # dictionary for caching shapefile datasets in memory

        self._load_cfg(cfg)  # update configuration dict with values in cfg

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
        txt += '{} LAKE package lakes'.format(self.nlakes)
        txt += '\n'
        return txt

    def __eq__(self, other):
        """Test for equality to another model object."""
        if not isinstance(other, MFnwtModel):
            return False
        if other.cfg != self.cfg:
            return False
        if other.get_package_list() != self.get_package_list():
            return False
        #if other.sr != self.sr:
        #    return False
        return True


    @property
    def nper(self):
        if self.perioddata is not None:
            return len(self.perioddata)

    @property
    def nlay(self):
        return self.cfg['dis'].get('nlay', 1)

    @property
    def nrow(self):
        return self.modelgrid.nrow

    @property
    def ncol(self):
        return self.modelgrid.ncol

    @property
    def length_units(self):
        return lenuni_text[self.cfg['dis']['lenuni']]

    @property
    def time_units(self):
        return itmuni_text[self.cfg['dis']['itmuni']]

    @property
    def sr(self):
        if self._sr is None:
            pass
            #kwargs = self.cfg.get('grid').copy()
            #if kwargs is not None:
            #    if np.isscalar(kwargs['delr']):
            #        kwargs['delr'] = np.ones(kwargs['ncol'], dtype=float) * kwargs['delr']
            #    if np.isscalar(kwargs['delc']):
            #        kwargs['delc'] = np.ones(kwargs['nrow'], dtype=float) * kwargs['delc']
            #    kwargs.pop('nrow')
            #    kwargs.pop('ncol')
            #    self._sr = SpatialReference(**kwargs)
        return self._sr

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

        if 'head' in self.cfg['model']['perimeter_boundary_type']:
            return 'head'
        if 'flux' in self.cfg['model']['perimeter_boundary_type']:
            return 'flux'

    @property
    def ipakcb(self):
        """By default write everything to one cell budget file."""
        return self.cfg['upw'].get('ipakcb', 53)

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
    def _parent_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the inset model domain. Used to speed up interpolation
        of parent grid values onto inset grid."""
        if self.__parent_mask is None:
            x, y = np.squeeze(self.bbox.exterior.coords.xy)
            pi, pj = get_ij(self.parent.modelgrid, x, y)
            pad = 2
            i0, i1 = pi.min() - pad, pi.max() + pad
            j0, j1 = pj.min() - pad, pj.max() + pad
            mask = np.zeros((self.parent.nrow, self.parent.ncol), dtype=bool)
            mask[i0:i1, j0:j1] = True
            self.__parent_mask = mask
        return self.__parent_mask

    @property
    def nlakes(self):
        lakes = self.cfg['lak'].get('include_lakes')
        if lakes is None:
            return 0
        else:
            return len(lakes)

    @property
    def _lakarr2d(self):
        """2-D array of areal extent of lakes. Non-zero values
        correspond to lak package IDs."""
        if self.__lakarr2d is None:
            lakarr2d = np.zeros((self.nrow, self.ncol))
            if 'lak' in self.package_list:
                lakes_shapefile = self.cfg['lak']['source_data'].get('lakes_shapefile')
                if lakes_shapefile is not None:
                    shapefile = lakes_shapefile['filename']
                    id_column = lakes_shapefile.get('id_column')
                    include_lakes = lakes_shapefile.get('include_lakes')
                lakesdata = self.load_features(shapefile)
                if lakesdata is not None:
                    if include_lakes is not None:
                        lakarr2d = make_lakarr2d(self.modelgrid,
                                                 lakesdata,
                                                 include_hydroids=include_lakes,
                                                 id_column=id_column)
            self.__lakarr2d = lakarr2d
            self.__isbc2d = None
        return self.__lakarr2d

    @property
    def lakarr(self):
        """3-D array of lake extents in each layer. Non-zero values
        correspond to lak package IDs. Extent of lake in
        each layer is based on bathymetry and model layer thickness.
        """
        if self._lakarr is None:
            lakarr_file_fmt = self.cfg['lak']['lakarr_filename_fmt']
            intermediate_lakarrfiles = ['{}/{}'.format(self.tmpdir,
                                                       lakarr_file_fmt.format(i))
                                        for i in range(self.nlay)]
            self.cfg['intermediate_data']['lakarr'] = intermediate_lakarrfiles
            self.cfg['lak']['lakarr'] = [os.path.join(self.model_ws,
                                                      self.external_path,
                                                      lakarr_file_fmt.format(i))
                                         for i in range(self.nlay)]
            if self.isbc is None:
                return None
            else:
                # assign lakarr values from 3D isbc array
                lakarr = np.zeros((self.nlay, self.nrow, self.ncol))
                for k in range(self.nlay):
                    lakarr[k][self.isbc[k] == 1] = self._lakarr2d[self.isbc[k] == 1]
            for k, ilakarr in enumerate(lakarr):
                save_array(intermediate_lakarrfiles[k], ilakarr, fmt='%d')
            self._lakarr = lakarr
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
        if self.__isbc2d is None:
            isbc = np.zeros((self.nrow, self.ncol))
            lakesdata = None
            lakes_shapefile = self.cfg['lak']['source_data'].get('lakes_shapefile')
            if lakes_shapefile is not None:
                if isinstance(lakes_shapefile, dict):
                    lakes_shapefile = lakes_shapefile['filename']
                lakesdata = self.load_features(lakes_shapefile)
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
            self.__isbc2d = isbc
            self._lake_bathymetry = None
        return self.__isbc2d

    @property
    def ibound(self):
        """3D array indicating which cells will be included in the simulation.
        Made a property so that it can be easily updated when any packages
        it depends on change.

        TODO: move setup code from setup_bas6
        """
        pass
        #if self._ibound is None and 'DIS' in self.package_list:
        #    ibound = np.abs(~np.isnan(self.dis.botm.array).astype(int))
        #    # remove cells that are above stream cells
        #    if 'SFR' in self.get_package_list():
        #        ibound = deactivate_idomain_above(ibound, self.sfr.reach_data)
        #    self._ibound = ibound
        #return self._ibound

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

    @property
    def lake_bathymetry(self):
        """Put lake bathymetry setup logic here instead of DIS package.
        """
        default_lake_depth = self.cfg['model'].get('default_lake_depth', 2)
        if self._lake_bathymetry is None:
            bathymetry_file = self.cfg['lak']['source_data'].get('bathymetry_raster')
            lmult = 1.0
            if isinstance(bathymetry_file, dict):
                lmult = convert_length_units(bathymetry_file.get('length_units', 0),
                                             self.length_units)
                bathymetry_file = bathymetry_file['filename']
            if bathymetry_file is None:
                bathy = np.zeros((self.nrow, self.ncol))
            else:
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
            self._lake_bathymetry = bathy
        return self._lake_bathymetry

    @property
    def precipitation(self):
        """Lake precipitation at each stress period, in model units.
        """
        if self._precipitation is None or \
                len(self._precipitation) != self.nper:
            precip = self.cfg['lak']['precip']
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
            evap = self.cfg['lak']['evap']
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
        if self._lake_recharge is None:
            if self.precipitation is not None and self.evaporation is not None:
                self._lake_recharge = self.precipitation - self.evaporation
        return self._lake_recharge

    def _load_cfg(self, cfg):
        """Load configuration file; update dictionary.
        TODO: need to figure out what goes here and what goes in load_cfg static method. Or maybe this should be called set_cfg
        """
        if isinstance(cfg, str):
            assert os.path.exists(cfg), "config file {} not found".format(cfg)
            updates = load(cfg)
            updates['filename'] = cfg
        elif isinstance(cfg, dict):
            updates = cfg.copy()
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

        # load the parent model
        if 'namefile' in self.cfg.get('parent', {}).keys():
            kwargs = self.cfg['parent'].copy()
            kwargs['f'] = kwargs.pop('namefile')
            # load only specified packages that the parent model has
            packages_in_parent_namefile = get_packages(os.path.join(kwargs['model_ws'],
                                                                    kwargs['f']))
            load_only = list(set(packages_in_parent_namefile).intersection(
                set(self.cfg['model'].get('packages', set()))))
            kwargs['load_only'] = load_only
            kwargs = get_input_arguments(kwargs, fm.Modflow.load, warn=False)

            print('loading parent model...')
            t0 = time.time()
            self._parent = fm.Modflow.load(**kwargs)
            print("finished in {:.2f}s\n".format(time.time() - t0))

            mg_kwargs = self.cfg['parent'].get('SpatialReference',
                                               self.cfg['parent'].get('modelgrid', None))
        # set the parent model grid from mg_kwargs if not None
        # otherwise, convert parent model grid to MFsetupGrid
        self._set_parent_modelgrid(mg_kwargs)

        # make sure that the output paths exist
        output_paths = [self.cfg['intermediate_data']['output_folder'],
                        self.cfg['model']['model_ws'],
                        os.path.join(self.cfg['model']['model_ws'], self.cfg['model']['external_path'])
                        ]
        output_paths += list(self.cfg['postprocessing']['output_folders'].values())
        for folder in output_paths:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # absolute path to config file
        self._config_path = os.path.split(os.path.abspath(self.cfg['filename']))[0]

        # other variables
        self.cfg['external_files'] = {}
        # TODO: extend to multiple source models
        # TODO: allow for mf6 parent
        if 'length_units' not in self.cfg['parent']:
            self.cfg['parent']['length_units'] = lenuni_text[self.parent.dis.lenuni]
        if 'time_units' not in self.cfg['parent']:
            self.cfg['parent']['time_units'] = itmuni_text[self.parent.dis.itmuni]

    def _set_parent_modelgrid(self, mg_kwargs=None):
        """Reset the parent model grid from keyword arguments
        or existing modelgrid, and DIS package.
        """
        if mg_kwargs is not None:
            kwargs = mg_kwargs.copy()
        else:
            kwargs = {'xoff': self.parent.modelgrid.xoffset,
                      'yoff': self.parent.modelgrid.yoffset,
                      'angrot': self.parent.modelgrid.angrot,
                      'epsg': self.parent.modelgrid.epsg,
                      'proj4': self.parent.modelgrid.proj4,
                      }
        parent_lenuni = self.parent.dis.lenuni
        if 'lenuni' in self.cfg['parent']:
            parent_lenuni = self.cfg['parent']['lenuni']
        elif 'length_units' in self.cfg['parent']:
            parent_lenuni = lenuni_text[self.cfg['parent']['length_units']]

        self.parent.dis.lenuni = parent_lenuni
        lmult = convert_length_units(parent_lenuni, 'meters')
        kwargs['delr'] = self.parent.dis.delr.array * lmult
        kwargs['delc'] = self.parent.dis.delc.array * lmult
        kwargs['lenuni'] = 2  # parent modelgrid in same CRS as inset modelgrid
        kwargs = get_input_arguments(kwargs, MFsetupGrid, warn=False)
        self._parent._mg_resync = False
        self._parent._modelgrid = MFsetupGrid(**kwargs)

    def _set_perioddata(self):
        """Sets up the perioddata DataFrame.

        Needs some work to be more general.
        """
        parent_sp = self.cfg['model']['parent_stress_periods']

        # use all stress periods from parent model
        if isinstance(parent_sp, str) and parent_sp.lower() == 'all':
            nper = self.cfg['dis'].get('nper')
            if nper is None or nper < self.parent.nper:
                nper = self.parent.nper
            parent_sp = list(range(self.parent.nper))
            self.cfg['dis']['perlen'] = None # set from parent model
            self.cfg['dis']['steady'] = None

        # use only specified stress periods from parent model
        elif isinstance(parent_sp, list):
            # limit parent stress periods to include
            # to those in parent model and nper specified for inset
            nper = self.cfg['dis'].get('nper', len(parent_sp))

            parent_sp = [0]
            perlen = [self.parent.dis.perlen.array[0]]
            for i, p in enumerate(self.cfg['model']['parent_stress_periods']):
                if i == nper:
                    break
                if p == self.parent.nper:
                    break
                if p > 0:
                    parent_sp.append(p)
                    perlen.append(self.parent.dis.perlen.array[p])
            if nper < len(parent_sp):
                nper = len(parent_sp)
            else:
                n_parent_per = len(parent_sp)
                for i in range(nper - n_parent_per):
                    parent_sp.append(parent_sp[-1])

        # no parent stress periods specified, # default to just first stress period
        else:
            nper = self.cfg['dis'].get('nper', 1)
            parent_sp = [0]
            for i in range(nper - 1):
                parent_sp.append(parent_sp[-1])

        assert len(parent_sp) == nper
        self.cfg['dis']['nper'] = nper
        self.cfg['model']['parent_stress_periods'] = parent_sp

        #if self.cfg['dis'].get('steady') is None:
        #    self.cfg['dis']['steady'] = [True] + [False] * (nper)
        if self.cfg['dis'].get('steady') is not None:
            self.cfg['dis']['steady'] = np.array(self.cfg['dis']['steady']).astype(bool).tolist()

        for var in ['perlen', 'nstp', 'tsmult', 'steady']:
            arg = self.cfg['dis'][var]
            if arg is not None and not np.isscalar(arg):
                assert len(arg) == nper, \
                    "Variable {} must be a scalar or have {} entries (one for each stress period).\n" \
                    "Or leave as None to set from parent model".format(var, nper)
            elif np.isscalar(arg):
                self.cfg['dis'][var] = [arg] * nper
            else:
                self.cfg['dis'][var] = getattr(self.parent.dis, var)[self.cfg['model']['parent_stress_periods']]

        steady = {kper: issteady for kper, issteady in enumerate(self.cfg['dis']['steady'])}

        perioddata = setup_perioddata(self.cfg['model']['start_date_time'],
                                      self.cfg['model'].get('end_date_time'),
                                      nper=nper,
                                      perlen=self.cfg['dis']['perlen'],
                                      model_time_units=self.time_units,
                                      freq=self.cfg['dis'].get('freq'),
                                      steady=steady,
                                      nstp=self.cfg['dis']['nstp'],
                                      tsmult=self.cfg['dis']['tsmult'],
                                      oc=self.cfg['oc']['period_options'],
                                      )
        perioddata['parent_sp'] = parent_sp
        assert np.array_equal(perioddata['per'].values, np.arange(len(perioddata)))
        self._perioddata = perioddata
        self._nper = None

    def load_array(self, filename):
        return load_array(filename, shape=(self.nrow, self.ncol))

    def load_features(self, filename, filter=None,
                      id_column=None, include_ids=None,
                      cache=True):
        """Load vector and attribute data from a shapefile;
        cache it to the _features dictionary.
        """
        if filename not in self._features.keys():
            if os.path.exists(filename):
                features_proj_str = get_proj4(filename)
                model_proj_str = "+init=epsg:{}".format(self.cfg['setup_grid']['epsg'])
                if filter is None:
                    if self.bbox is not None:
                        bbox = self.bbox
                    elif self.parent.modelgrid is not None:
                        bbox = self.parent.modelgrid.bbox
                        model_proj_str = self.parent.modelgrid.proj4
                        assert model_proj_str is not None

                    if features_proj_str.lower() != model_proj_str:
                        filter = project(bbox, model_proj_str, features_proj_str).bounds
                    else:
                        filter = bbox.bounds

                df = shp2df(filename, filter=filter)
                if id_column is not None and include_ids is not None:
                    df = df.loc[df[id_column].isin(include_ids)]
                if features_proj_str.lower() != model_proj_str:
                    df['geometry'] = project(df['geometry'], features_proj_str, model_proj_str)
                if cache:
                    print('caching data in {}...'.format(filename))
                    self._features[filename] = df
            else:
                return None
        else:
            df = self._features[filename]
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
            parent_values = parent_array.flatten()[self._parent_mask.flatten()]
            regridded = interpolate(parent_values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = regrid(parent_array, self.parent.modelgrid, self.modelgrid,
                               method='nearest')
        regridded = np.reshape(regridded, (self.nrow, self.ncol))
        return regridded

    def setup_grid(self, features=None,
                   features_shapefile=None,
                   id_column='HYDROID', include_ids=[],
                   buffer=1000, dxy=20,
                   xoff=None, yoff=None,
                   epsg=None,
                   remake=True,
                   variable_mappings={},
                   grid_file='grid.yml',
                   write_shapefile=True):
        """

        Parameters
        ----------
        parent : flopy.modflow.Modflow instance

        features : list of shapely.geometry objects

        Returns
        -------

        """
        print('setting up model grid...')
        t0 = time.time()

        # conversions for inset/parent model units to meters
        inset_lmult = convert_length_units(self.length_units, 'meters')
        parent_lmult = convert_length_units(self.parent.dis.lenuni, 'meters')
        dxy_m = np.round(dxy * inset_lmult, 4) # dxy is specified in inset model units

        # set up the parent modelgrid if it isn't
        parent_delr_m = np.round(self.parent.dis.delr.array[0] * parent_lmult, 4)
        parent_delc_m = np.round(self.parent.dis.delc.array[0] * parent_lmult, 4)

        if epsg is None:
            epsg = self.parent.modelgrid.epsg

        if not remake:
            try:
                model_info = load(grid_file)
            except:
                remake = True

        if remake:
            # make grid from xoff, yoff, dxy and nrow, ncol
            if xoff is not None and yoff is not None:

                assert 'nrow' in self.cfg['dis'] and 'ncol' in self.cfg['dis'], \
                "Need to specify nrow and ncol if specifying xoffset and yoffset."
                height_m = np.round(dxy_m * self.cfg['dis']['nrow'], 4)
                width_m = np.round(dxy_m * self.cfg['dis']['ncol'], 4)
                xul = xoff
                yul = yoff + height_m
                pass
            # make grid using buffered feature bounding box
            else:
                if features is None and features_shapefile is not None:

                    df = self.load_features(features_shapefile,
                                            id_column=id_column, include_ids=include_ids,
                                            filter=self.parent.modelgrid.bbox.bounds,
                                            cache=False)
                    rows = df.loc[df[id_column].isin(include_ids)]
                    features = rows.geometry.tolist()
                if isinstance(features, list):
                    if len(features) > 1:
                        features = MultiPolygon(features)
                    else:
                        features = features[0]

                x1, y1, x2, y2 = features.bounds

                L = buffer  # characteristic length or buffer around chain of lakes, in m
                xul = x1 - L
                yul = y2 + L
                height_m = np.round(yul - (y1 - L), 4) # initial model height from buffer distance
                width_m = np.round((x2 + L) - xul, 4)

            # get location of coinciding cell in parent model for upper left
            pi, pj = self.parent.modelgrid.intersect(xul, yul)
            verts = np.array(self.parent.modelgrid.get_cell_vertices(pi, pj))
            xul, yul = verts[:, 0].min(), verts[:, 1].max()

            def roundup(number, increment):
                return int(np.ceil(number / increment) * increment)


            self.height = roundup(height_m, parent_delr_m)
            self.width = roundup(width_m, parent_delc_m)

            # update nrow, ncol after snapping to parent grid
            nrow = int(self.height / dxy_m) # h is in meters
            ncol = int(self.width / dxy_m)
            self.cfg['dis']['nrow'] = nrow
            self.cfg['dis']['ncol'] = ncol
            #lenuni = self.cfg['dis'].get('lenuni', 2)

            # set the grid info dict
            # spacing is in meters (consistent with projected CRS)
            # (modelgrid object will be updated automatically)
            xll = xul
            yll = yul - self.height
            self.cfg['grid'] = {'nrow': nrow, 'ncol': ncol,
                                'delr': dxy_m, 'delc': dxy_m,
                                'xoff': xll, 'yoff': yll,
                                'xul': xul, 'yul': yul,
                                'epsg': epsg,
                                'lenuni': 2
                                }

            #grid_file = os.path.join(self.cfg['model']['model_ws'], os.path.split(grid_file)[1])
            dump(grid_file, self.cfg['grid'])
            if write_shapefile:
                write_bbox_shapefile(self.modelgrid,
                                     os.path.join(self.cfg['postprocessing']['output_folders']['shapefiles'],
                                                  '{}_bbox.shp'.format(self.name)))
        print("finished in {:.2f}s\n".format(time.time() - t0))

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

    def _setup_array(self, variable, intermediate_files,
                     varname, multiplier=1, regrid_from_parent=False,
                     regrid_method='nearest',
                     fmt='%.6e'):
        """Set up an array variable that may be entered as
        a scalar, external file, or a list that may be a
        mix of scalars and external files.

        Parameters
        ----------
        variable : scalar, external file, or a list that may be a mix of scalars and external files
            MODFLOW/Flopy array input (Util2D-like)
        intermediate_files : list of external file paths
            External files that will be supplied to Flopy as input for variable.
        varname : str; Flopy variable name (e.g. 'hk' or 'ss'
            Name of the variable. Used to assign appropriate values for areas with lakes,
            keep track of what arrays have been created, and for exception reporting
            in the case of unrecognized input.
        multiplier : scalar
            Multiply array values by this amount.
        regrid_from_parent : bool
            If True, variable is a 3D numpy array (nper x nrow x ncol) or (nlay x nrow x ncol)
            from parent model. Interpolate values onto inset grid.
        regrid_method : str; ('nearest' or 'linear')
            How inset values are sampled from parent model array.
        fmt : str (numpy.savetxt-style output format, default='%.6f')
        """
        print('setting up {}...'.format(varname))
        kh_highk_lakes = float(self.cfg['parent'].get('hiKlakes_value', 1e4))

        var = variable

        if isinstance(var, str) or np.isscalar(var):
            var = [var] * len(intermediate_files)
        knt = 0  # separate counter for inset layers (if var is from parent model)
        for i, ivar in enumerate(var):
            if isinstance(ivar, str):
                # sample "source_data" that may not be on same grid
                if ivar.endswith(".asc") or ivar.endswith(".tif"):
                    ivar = get_values_at_points(ivar,
                                                self.modelgrid.xcellcenters.ravel(),
                                                self.modelgrid.ycellcenters.ravel())
                    ivar = np.reshape(ivar, (self.nrow, self.ncol))
                # read numpy array on same grid
                else:
                    ivar = self.load_array(ivar)
            # convert scalar to numpy array
            elif np.isscalar(ivar):
                ivar = np.ones((self.nrow, self.ncol)) * ivar
            # interpolate variable from parent model arrays to inset model grid
            elif regrid_from_parent:

                # exclude high-k lake values in interpolation from parent model
                if varname in ['sy', 'ss']:
                    mask = self._parent_mask & (ivar < 1.)
                elif varname == 'hk':
                    mask = self._parent_mask & (ivar < kh_highk_lakes)
                elif varname == 'vka':
                    mask = self._parent_mask & (ivar < kh_highk_lakes)
                elif varname == 'rech':
                    mask = self._parent_mask #& (ivar > 0.)
                else:
                    mask = None
                ivar = self.regrid_from_parent(ivar, mask=mask,
                                               method=regrid_method)
            elif isinstance(ivar, np.ndarray) and ivar.size == self.nrow * self.ncol:
                if ivar.shape != (self.nrow, self.ncol):
                    ivar = np.reshape(ivar, (self.nrow, self.ncol))
            else:
                raise ValueError("unrecognized input for {}: {}".format(varname, ivar))

            # handle lakes
            if varname == 'rech':
                # assign high-k lake recharge for stress period
                # apply in same units as source recharge array
                ivar[self.isbc[0] == 2] = self.lake_recharge[i] * 1/self.cfg['rch']['unit_conversion']
                # zero-values to lak package lakes
                ivar[self.isbc[0] == 1] = 0.

            # convert to inset layering by iterating thru
            # each inset layer within the parent layer
            # TODO: need to generalize layer setup similar to MF6model
            nsublayers = 1
            if self.nlay - self.parent.nlay == 1 and \
                    len(var) == 4 and knt == 0 and \
                    varname in ['strt', 'hk', 'vka', 'sy', 'ss']:
                nsublayers = 2
            for ii in range(nsublayers):
                iivar = ivar.copy()
                # handle lakes
                if varname == 'hk':
                    iivar[self.isbc[knt] == 2] = kh_highk_lakes
                elif varname == 'vka':
                    pass # not sure what if anything needs to be done for vka
                elif varname == 'sy':
                    iivar[self.isbc[knt] == 2] = 1.0
                elif varname == 'ss':
                    iivar[self.isbc[knt] == 2] = 1.0
                save_array(intermediate_files[knt], iivar * multiplier, fmt=fmt)
                knt += 1
        self.updated_arrays.add(varname)

    def _setup_array2(self, package, var, vmin=-1e30, vmax=1e30,
                      source_model=None, source_package=None,
                      **kwargs):
        return setup_array(self, package, var, vmin=vmin, vmax=vmax,
                           source_model=source_model, source_package=source_package,
                           **kwargs)

    def setup_dis(self):
        """"""
        package = 'dis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # resample the top from the DEM
        if self.cfg['dis']['remake_top']:
            self._setup_array2(package, 'top')

        # make the botm array
        self._setup_array2(package, 'botm', by_layer=True)

        # put together keyword arguments for dis package
        kwargs = self.cfg['grid'].copy() # nrow, ncol, delr, delc
        kwargs.update(self.cfg['dis']) # nper, nlay, etc.
        kwargs = get_input_arguments(kwargs, fm.ModflowDis)
        # we need flopy to read the intermediate files
        # (it will write the files in cfg)
        lmult = convert_length_units('meters', self.length_units)
        kwargs.update({'top': self.cfg['intermediate_data']['top'][0],
                       'botm': self.cfg['intermediate_data']['botm'],
                       'nper': self.nper,
                       'delc': self.modelgrid.delc * lmult,
                       'delr': self.modelgrid.delr * lmult
                      })
        for arg in ['perlen', 'nstp', 'tsmult', 'steady']:
            kwargs[arg] = self.perioddata[arg].values

        dis = fm.ModflowDis(model=self, **kwargs)
        self._perioddata = None  # reset perioddata
        self._modelgrid = None  # override DIS package grid setup
        self._isbc = None  # reset BC property arrays
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    def setup_bas6(self):
        """"""
        package = 'bas6'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # source data
        #headfile = self.cfg['parent']['headfile']
        #headfile_stress_period = self.cfg['model']['parent_stress_periods'][0]
        #check_source_files([self.cfg['parent']['headfile']])

        # make the strt array
        self._setup_array2(package, 'strt', by_layer=True)

        kwargs = {}
        kwargs = get_input_arguments(kwargs, fm.ModflowBas)
        bas = fm.ModflowBas(model=self,
                             hnoflo=self.cfg['bas']['hnoflo'],
                             ibound=intermediate_iboundfiles,
                             strt=intermediate_strtfiles)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return bas

    def setup_chd(self):
        """Set up constant head package for perimeter boundary.
        """
        print('setting up CHD package...')
        t0 = time.time()
        # source data
        headfile = self.cfg['parent']['headfile']
        check_source_files([headfile])
        hdsobj = bf.HeadFile(headfile, precision='single')
        all_kstpkper = hdsobj.get_kstpkper()

        # get the last timestep in each stress period if there are more than one
        kstpkper = []
        unique_kper = []
        for (kstp, kper) in all_kstpkper:
            if kper not in unique_kper:
                kstpkper.append((kstp, kper))
                unique_kper.append(kper)

        assert len(unique_kper) == len(set(self.cfg['model']['parent_stress_periods'])), \
        "read {} from {},\nexpected stress periods: {}".format(kstpkper,
                                                               headfile,
                                                               sorted(list(set(self.cfg['model']['parent_stress_periods'])))
                                                               )
        k, i, j = self.get_boundary_cells()

        # get heads from parent model
        dfs = []
        for inset_per, parent_kstpkper in enumerate(kstpkper):
            hds = hdsobj.get_data(kstpkper=parent_kstpkper)

            regridded = np.zeros((self.nlay, self.nrow, self.ncol))
            for layer, khds in enumerate(hds):
                if layer > 0 and self.nlay - self.parent.nlay == 1:
                    layer += 1
                regridded[layer] = self.regrid_from_parent(khds, method='linear')
            if self.nlay - self.parent.nlay == 1:
                regridded[1] = regridded[[0, 2]].mean(axis=0)
            df = pd.DataFrame({'per': inset_per,
                               'k': k,
                               'i': i,
                               'j': j,
                               'bhead': regridded[k, i, j]})
            dfs.append(df)
        tmp = fm.ModflowChd.get_empty(len(df))
        spd = {}
        for per in range(len(dfs)):
            spd[per] = tmp.copy() # need to make a copy otherwise they'll all be the same!!
            spd[per]['k'] = df['k']
            spd[per]['i'] = df['i']
            spd[per]['j'] = df['j']
            # assign starting and ending head values for each period
            # starting chd is parent values for previous period
            # ending chd is parent values for that period
            if per == 0:
                spd[per]['shead'] = dfs[per]['bhead']
                spd[per]['ehead'] = dfs[per]['bhead']
            else:
                spd[per]['shead'] = dfs[per - 1]['bhead']
                spd[per]['ehead'] = dfs[per]['bhead']

        chd = fm.ModflowChd(self, stress_period_data=spd)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return chd

    def setup_oc(self):

        print('setting up OC package...')
        t0 = time.time()
        words = self.cfg['oc']['period_options']
        stress_period_data = {}
        for kper in range(self.nper):
            last = self.dis.nstp.array[kper] - 1
            stress_period_data[(kper, last)] = words.get(kper, words[0])
            #for kstp in range(self.dis.nstp.array[kper]):
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return fm.ModflowOc(self, stress_period_data=stress_period_data)

    def setup_rch(self):
        package = 'rch'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        rech_file_fmt = self.cfg['rch']['rech_filename_fmt']

        # recharge specified directly as 'rech' variable
        rech = self.cfg['rch'].get('rech')
        if rech is not None:
            rech = MFArrayData(values=rech,
                               multiplier=self.cfg['rch'].get('mult', 1.),
                               length_units=self.cfg['rch'].get('rech_length_units', 'unknown'),
                               time_units=self.cfg['rch'].get('rech_time_units', 'unknown'),
                               dest_model=self
                               )
            data = rech.get_data()

        # recharge specified as source_data
        else:
            # recharge input from raster or array data
            if 'source_data' in self.cfg['rch']:
                rech = ArraySourceData.from_config(self.cfg['rch']['source_data']['infiltration'],
                                                   dest_model=self)
                data = rech.get_data()

            # recharge regridded from parent model
            elif self.parent is not None:
                if 'RCH' not in self.parent.get_package_list():
                    rchfile = '{}/{}.rch'.format(self.parent.model_ws,
                                                 self.parent.name)
                    uzffile = rchfile[:-4] + '.uzf'
                    if os.path.exists(rchfile):
                        parent_rch = fm.ModflowRch.load(rchfile,
                                                        model=self.parent)
                        nper = self.parent.rch.rech.array.shape[0]
                        inf_array = self.parent.rch.rech.array[:, 0]
                    elif os.path.exists(uzffile):
                        parent_uzf = fm.ModflowUzf1.load(uzffile,
                                                         model=self.parent)
                        nper = self.parent.uzf.finf.array.shape[0]
                        inf_array = self.parent.uzf.finf.array[:, 0]
                rech = ArraySourceData(dest_model=self,
                                       source_modelgrid=self.parent.modelgrid,
                                       source_array=inf_array)
                data = rech.get_data()

            else:
                return

        # intermediate data
        # set paths to intermediate files and external files
        self.setup_external_filepaths(package, 'rech', rech_file_fmt,
                                      nfiles=len(data))

        # write out array data to intermediate files
        # assign lake recharge values (water balance surplus) for any high-K lakes
        for i, arr in data.items():
            arr[self.isbc[0] == 2] = self.lake_recharge[i]
            # zero-values to lak package lakes
            arr[self.isbc[0] == 1] = 0.
            np.savetxt(self.cfg['intermediate_data']['rech'][i], arr, fmt='%.6e')

        # create flopy package instance
        rch = fm.ModflowRch(self,
                            rech={i: f for i, f in
                                  enumerate(self.cfg['intermediate_data']['rech'])},
                            ipakcb=self.ipakcb)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return rch

    def setup_upw(self):
        """
        """
        package = 'upw'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        hiKlakes_value = float(self.cfg['parent'].get('hiKlakes_value', 1e4))

        # copy transient variables if they were included in config file
        # defaults are hard coded to arrays in parent model priority
        # over config file values, in the case that ss and sy weren't entered
        hk = self.cfg['upw'].get('hk')
        vka = self.cfg['upw'].get('vka')
        default_sy = 0.1
        default_ss = 1e-6

        # Determine which hk, vka to use
        # load parent upw if it's needed and not loaded
        source_package = package
        if np.any(np.array([hk, vka]) == None) and \
                'UPW' not in self.parent.get_package_list() and \
                'LPF' not in self.parent.get_package_list():
            for ext, pckgcls in {'upw': fm.ModflowUpw,
                                 'lpf': fm.ModflowLpf,
                                 }.items():
                pckgfile = '{}/{}.{}'.format(self.parent.model_ws, self.parent.name, package)
                if os.path.exists(pckgfile):
                    upw = pckgcls.load(pckgfile, self.parent)
                    source_package = ext
                    break

        self._setup_array2(package, 'hk', vmin=0, vmax=hiKlakes_value,
                           source_package=source_package, by_layer=True)
        self._setup_array2(package, 'vka', vmin=0, vmax=hiKlakes_value,
                           source_package=source_package, by_layer=True)
        if np.any(~self.dis.steady.array):
            self._setup_array2(package, 'sy', vmin=0, vmax=1,
                               source_package=source_package, by_layer=True)
            self._setup_array2(package, 'ss', vmin=0, vmax=1,
                               source_package=source_package, by_layer=True)
            sy = self.cfg['intermediate_data']['sy']
            ss = self.cfg['intermediate_data']['ss']
        else:
            sy = default_sy
            ss = default_ss

        upw = fm.ModflowUpw(self, hk=self.cfg['intermediate_data']['hk'],
                            vka=self.cfg['intermediate_data']['vka'],
                            sy=sy,
                            ss=ss,
                            layvka=self.cfg['upw']['layvka'],
                            laytyp=self.cfg['upw']['laytyp'],
                            hdry=self.cfg['upw']['hdry'],
                            ipakcb=self.cfg['upw']['ipakcb'])
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return upw

    def setup_wel(self):
        """
        Setup the WEL package, including boundary fluxes and any pumping.

        This will need some additional customization if pumping is mixed with
        the perimeter fluxes.

        """

        print('setting up WEL package...')
        t0 = time.time()
        # master dataframe for stress period data
        df = pd.DataFrame(columns=['per', 'k', 'i', 'j', 'flux', 'comments'])

        # Get steady-state pumping rates
        check_source_files([self.cfg['source_data']['water_use'],
                            self.cfg['source_data']['water_use_points']])

        # fill out period stats
        period_stats = self.cfg['wel']['period_stats']
        if isinstance(period_stats, str):
            period_stats = {kper: period_stats for kper in range(self.nper)}

        # separate out stress periods with period mean statistics vs.
        # those to be resampled based on start/end dates
        resampled_periods = {k:v for k, v in period_stats.items()
                             if v == 'resample'}
        periods_with_dataset_means = {k: v for k, v in period_stats.items()
                                      if k not in resampled_periods}

        if len(periods_with_dataset_means) > 0:
            wu_means = get_mean_pumping_rates(self.cfg['source_data']['water_use'],
                                      self.cfg['source_data']['water_use_points'],
                                      period_stats=periods_with_dataset_means,
                                      model=self)
            df = df.append(wu_means)
        if len(resampled_periods) > 0:
            wu_resampled = resample_pumping_rates(self.cfg['source_data']['water_use'],
                                      self.cfg['source_data']['water_use_points'],
                                      model=self)
            df = df.append(wu_resampled)

        # boundary fluxes
        if self.perimeter_bc_type == 'flux':
            assert self.parent is not None, "need parent model for TMR cut"

            # boundary fluxes
            kstpkper = [(0, 0)]
            tmr = Tmr(self.parent, self)

            # parent periods to copy over
            kstpkper = [(0, per) for per in self.cfg['model']['parent_stress_periods']]
            bfluxes = tmr.get_inset_boundary_fluxes(kstpkper=kstpkper)
            bfluxes['comments'] = 'boundary_flux'
            df = df.append(bfluxes)

        # added wells
        added_wells = self.cfg['wel'].get('added_wells')
        if added_wells is not None:
            if isinstance(added_wells, str):
                aw = pd.read_csv(added_wells)
                aw.rename(columns={'name': 'comments'}, inplace=True)
            elif isinstance(added_wells, dict):
                added_wells = {k: v for k, v in added_wells.items() if v is not None}
                if len(added_wells) > 0:
                    aw = pd.DataFrame(added_wells).T
                    aw['comments'] = aw.index
                else:
                    aw = None
            elif isinstance(added_wells, pd.DataFrame):
                aw = added_wells
                aw['comments'] = aw.index
            else:
                raise IOError('unrecognized added_wells input')

            if aw is not None:
                if 'x' in aw.columns and 'y' in aw.columns:
                    aw['i'], aw['j'] = self.modelgrid.intersect(aw['x'].values,
                                                      aw['y'].values)

                aw['per'] = aw['per'].astype(int)
                aw['k'] = aw['k'].astype(int)
                df = df.append(aw)

        df['per'] = df['per'].astype(int)
        df['k'] = df['k'].astype(int)
        # Copy fluxes to subsequent stress periods as necessary
        # so that fluxes aren't unintentionally shut off;
        # for example if water use is only specified for period 0,
        # but the added well pumps in period 1, copy water use
        # fluxes to period 1.
        last_specified_per = int(df.per.max())
        copied_fluxes = [df]
        for i in range(last_specified_per):
            # only copied fluxes of a given stress period once
            # then evaluate copied fluxes (with new stress periods) and copy those once
            # after specified per-1, all non-zero fluxes should be propegated
            # to last stress period
            # copy non-zero fluxes that are not already in subsequent stress periods
            if i < len(copied_fluxes):
                in_subsequent_periods = copied_fluxes[i].comments.duplicated(keep=False)
                # (copied_fluxes[i].per < last_specified_per) & \
                tocopy = (copied_fluxes[i].flux != 0) & \
                         ~in_subsequent_periods
                if np.any(tocopy):
                    copied = copied_fluxes[i].loc[tocopy].copy()

                    # make sure that wells to be copied aren't in subsequent stress periods
                    duplicated = np.array([r.comments in df.loc[df.per > i, 'comments']
                                           for idx, r in copied.iterrows()])
                    copied = copied.loc[~duplicated]
                    copied['per'] += 1
                    copied_fluxes.append(copied)
        df = pd.concat(copied_fluxes, axis=0)
        wel_lookup_file = os.path.join(self.model_ws, os.path.split(self.cfg['wel']['lookup_file'])[1])
        self.cfg['wel']['lookup_file'] = wel_lookup_file

        # save a lookup file with well site numbers/categories
        df[['per', 'k', 'i', 'j', 'flux', 'comments']].to_csv(wel_lookup_file, index=False)

        # extend spd dtype to include comments
        dtype = fm.ModflowWel.get_default_dtype()

        # setup stress period data
        groups = df.groupby('per')
        spd = {}
        for per, perdf in groups:
            ra = np.recarray(len(perdf), dtype=dtype)
            for c in ['k', 'i', 'j', 'flux']:
                ra[c] = perdf[c]
            spd[per] = ra

        wel = fm.ModflowWel(self, ipakcb=self.ipakcb,
                            options=self.cfg['wel']['options'],
                            stress_period_data=spd)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return wel

    def setup_mnw2(self):

        print('setting up MNW2 package...')
        t0 = time.time()

        # added wells
        added_wells = self.cfg['mnw'].get('added_wells')
        if added_wells is not None:
            if isinstance(added_wells, str):
                aw = pd.read_csv(added_wells)
                aw.rename(columns={'name': 'comments'}, inplace=True)
            elif isinstance(added_wells, dict):
                added_wells = {k: v for k, v in added_wells.items() if v is not None}
                if len(added_wells) > 0:
                    aw = pd.DataFrame(added_wells).T
                    aw['comments'] = aw.index
                else:
                    aw = None
            elif isinstance(added_wells, pd.DataFrame):
                aw = added_wells
                aw['comments'] = aw.index
            else:
                raise IOError('unrecognized added_wells input')

            k, ztop, zbotm = 0, 0, 0
            zpump = None

            wells = aw.groupby('comments').first()
            periods = aw
            if 'x' in wells.columns and 'y' in wells.columns:
                wells['i'], wells['j'] = self.modelgrid.intersect(wells['x'].values,
                                                     wells['y'].values)
            if 'depth' in wells.columns:
                wellhead_elevations = self.dis.top.array[wells.i, wells.j]
                ztop = wellhead_elevations - (5*.3048) # 5 ft casing
                zbotm = wellhead_elevations - wells.depth
                zpump = zbotm + 1 # 1 meter off bottom
            elif 'ztop' in wells.columns and 'zbotm' in wells.columns:
                ztop = wells.ztop
                zbotm = wells.zbotm
                zpump = zbotm + 1
            if 'k' in wells.columns:
                k = wells.k

            for var in ['losstype', 'pumploc', 'rw', 'rskin', 'kskin']:
                if var not in wells.columns:
                    wells[var] = self.cfg['mnw']['defaults'][var]

            nd = fm.ModflowMnw2.get_empty_node_data(len(wells))
            nd['k'] = k
            nd['i'] = wells.i
            nd['j'] = wells.j
            nd['ztop'] = ztop
            nd['zbotm'] = zbotm
            nd['wellid'] = wells.index
            nd['losstype'] = wells.losstype
            nd['pumploc'] = wells.pumploc
            nd['rw'] = wells.rw
            nd['rskin'] = wells.rskin
            nd['kskin'] = wells.kskin
            if zpump is not None:
                nd['zpump'] = zpump

            spd = {}
            for per, group in periods.groupby('per'):
                spd_per = fm.ModflowMnw2.get_empty_stress_period_data(len(group))
                spd_per['wellid'] = group.comments
                spd_per['qdes'] = group.flux
                spd[per] = spd_per
            itmp = []
            for per in range(self.nper):
                if per in spd.keys():
                    itmp.append(len(spd[per]))
                else:
                    itmp.append(0)

            mnw = fm.ModflowMnw2(self, mnwmax=len(wells), ipakcb=self.ipakcb,
                                 mnwprnt=1,
                                 node_data=nd, stress_period_data=spd,
                                 itmp=itmp
                                 )
            print("finished in {:.2f}s\n".format(time.time() - t0))
            return mnw
        else:
            print('No wells specified in configuration file!\n')
            return None

    def setup_sfr(self):

        print('setting up SFR package...')
        t0 = time.time()

        # input
        nhdplus_paths = self.cfg['sfr'].get('nhdplus_paths')
        if nhdplus_paths is not None:
            for f in nhdplus_paths:
                if not os.path.exists(f):
                    print('SFR setup: missing input file: {}'.format(f))
                    nhdplus_paths.remove(f)
            if len(nhdplus_paths) == 0:
                return

            # create an sfrmaker.lines instance
            filter = project(self.bbox, self.modelgrid.proj_str, '+init=epsg:4269').bounds
            lns = lines.from_NHDPlus_v2(NHDPlus_paths=nhdplus_paths,
                                        filter=filter)

        elif self.cfg['sfr'].get('flowlines_file') is not None:
            kwargs = self.cfg['sfr']['flowlines_file']
            if 'epsg' not in kwargs:
                kwargs['proj4'] = get_proj4(kwargs['shapefile'])
            else:
                kwargs['proj4'] = '+init=epsg:{}'.format(kwargs['epsg'])

            filter = self.bbox.bounds
            if kwargs['proj4'] != self.modelgrid.proj_str:
                filter = project(self.bbox, self.modelgrid.proj_str, kwargs['proj4']).bounds
            kwargs['filter'] = filter
            # create an sfrmaker.lines instance
            kwargs = get_input_arguments(kwargs, lines.from_shapefile)
            lns = lines.from_shapefile(**kwargs)

        else:
            print('no SFR input')
            self.cfg['model']['packages'].remove('sfr')
            return

        # output
        output_path = self.cfg['sfr']['output_path']
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # create isfr array (where SFR cells will be populated)
        isfr = (self.bas6.ibound.array[0] == 1) & (self._isbc2d != 1)
        #  kludge to get sfrmaker to work with modelgrid
        self.modelgrid.model_length_units = lenuni_text[self.dis.lenuni]

        # create an sfrmaker.sfrdata instance from the lines instance
        sfr = lns.to_sfr(sr=self.modelgrid,
                         isfr=isfr
                         )

        sfr.reach_data['strtop'] = sfr.interpolate_to_reaches('elevup', 'elevdn')

        # assign layers to the sfr reaches
        botm = self.dis.botm.array.copy()
        layers, new_botm = assign_layers(sfr.reach_data, botm_array=botm)
        sfr.reach_data['k'] = layers
        if new_botm is not None:
            self.dis.botm = new_botm
            for f in [self.cfg['intermediate_data']['botm'][-1],
                      self.cfg['dis']['botm'][-1]]:
                if isinstance(f, str):
                    np.savetxt(f, botm.shape[0], new_botm, fmt='%.2f')

        # write reach and segment data tables
        sfr.write_tables('{}/{}'.format(output_path, self.name))

        # create the SFR package
        sfr.create_ModflowSfr2(model=self, istcb2=223)
        self.__isbc2d = None # reset BCs arrays
        self._isbc = None

        # export shapefiles of lines, routing, cell polygons, inlets and outlets
        sfr.export_cells('{}/{}_sfr_cells.shp'.format(output_path, self.name))
        sfr.export_outlets('{}/{}_sfr_outlets.shp'.format(output_path, self.name))
        sfr.export_transient_variable('flow', '{}/{}_sfr_inlets.shp'.format(output_path, self.name))
        sfr.export_lines('{}/{}_sfr_lines.shp'.format(output_path, self.name))
        sfr.export_routing('{}/{}_sfr_routing.shp'.format(output_path, self.name))
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return self.sfr

    def setup_lak(self):

        print('setting up LAKE package...')
        t0 = time.time()
        if self.lakarr.sum() == 0:
            print("lakes_shapefile not specified, or no lakes in model area")
            return
        # source data
        nlakes = len(self.cfg['lak']['include_lakes'])  # number of lakes
        lakesdata = self.load_features(self.cfg['source_data'].get('lakes_shapefile'))

        # lake package settings
        theta = self.cfg['lak']['theta']
        nssitr = self.cfg['lak']['nssitr']
        sscncr = self.cfg['lak']['sscncr']
        surfdep = self.cfg['lak']['surfdep']
        littoral_leakance = self.cfg['lak']['littoral_leakance']
        profundal_leakance = self.cfg['lak']['profundal_leakance']
        stage_area_volume_file = self.cfg['lak'].get('stage_area_volume')
        tab_files_argument = self.cfg['lak'].get('tab_files')  # configured below if None
        tab_units = self.cfg['lak'].get('tab_units')
        start_tab_units_at = 150  # default starting number for iunittab

        # intermediate files
        intermediate_lakzones = os.path.join(self.tmpdir,
                                           os.path.split(self.cfg['lak']['lakzones'])[-1])
        bdlknc_file_fmt = self.cfg['lak']['bdlknc_filename_fmt']
        intermediate_bdlkncfiles = ['{}/{}'.format(self.tmpdir,
                                                   bdlknc_file_fmt.format(i))
                                    for i in range(self.nlay)]
        self.cfg['intermediate_data']['bdlknc'] = intermediate_bdlkncfiles

        # external arrays read by MODFLOW
        # (set to reflect expected locations where flopy will save them)
        self.cfg['lak']['bdlknc'] = [os.path.join(self.model_ws,
                                                  self.external_path,
                                                  bdlknc_file_fmt.format(i))
                                     for i in range(self.nlay)]

        # set up the tab files, if any
        if stage_area_volume_file is not None:
            print('setting up tabfiles...')
            df = pd.read_csv(stage_area_volume_file)
            if self.cfg['lak'].get('stage_area_volume_column_mappings') is not None:
                df.rename(columns=self.cfg['lak'].get('stage_area_volume_column_mappings'),
                          inplace=True)
            df.columns = [c.lower() for c in df.columns]
            lakes = df.groupby('hydroid')
            n_included_lakes = len(set(self.cfg['lak']['include_lakes']).\
                                   intersection(set(lakes.groups.keys())))
            assert n_included_lakes == nlakes, "stage_area_volume (tableinput) option" \
                                               " requires info for each lake, " \
                                               "only these HYDROIDs found:\n{}".format(df.hydroid.tolist())
            tab_files = []
            tab_units = []
            for i, hydroid in enumerate(self.cfg['lak']['include_lakes']):
                dfl = lakes.get_group(hydroid)
                assert len(dfl) == 151, "151 values required for each lake; " \
                                        "only {} for HYDROID {} in {}".format(len(dfl), hydroid, stage_area_volume_file)
                tabfilename = '{}/{}/{}_stage_area_volume.dat'.format(self.model_ws,
                                                                      self.external_path,
                                                                      hydroid)
                dfl[['stage', 'volume', 'area']].to_csv(tabfilename, index=False, header=False,
                                                        sep=' ', float_format='%.5e')
                print('wrote {}'.format(tabfilename))
                tab_files.append(tabfilename)
                tab_units.append(start_tab_units_at + i)

            # tabfiles aren't rewritten by flopy on package write
            self.cfg['lak']['tab_files'] = tab_files
            # kludge to deal with ugliness of lake package external file handling
            # (need to give path relative to model_ws, not folder that flopy is working in)
            tab_files_argument = [f.replace(self.model_ws, '').strip('/') for f in tab_files]

        # make the arrays or load them
        if 'lakzones' not in self.updated_arrays:
            lakzones = make_bdlknc_zones(self.modelgrid, lakesdata,
                                         include_hydroids=self.cfg['lak']['include_lakes'])
            save_array(intermediate_lakzones, lakzones, fmt='%d')
            self.updated_arrays.add('lakzones')
        else:
            lakzones = load_array(intermediate_lakzones)

        if 'bdlknc' not in self.updated_arrays:
            bdlknc = np.zeros((self.nlay, self.nrow, self.ncol))
            # make the areal footprint of lakebed leakance from the zones (layer 1)
            bdlknc[0] = make_bdlknc2d(lakzones, littoral_leakance, profundal_leakance)
            for k in range(self.nlay):
                if k > 0:
                    # for each underlying layer, assign profundal leakance to cells were isbc == 1
                    bdlknc[k][self.isbc[k] == 1] = profundal_leakance
                save_array(intermediate_bdlkncfiles[k], bdlknc[k], fmt='%.6e')
            self.updated_arrays.add('bdlknc')

        # save a lookup file mapping lake ids to hydroids
        lak_lookup_file = os.path.join(self.model_ws, os.path.split(self.cfg['lak']['lookup_file'])[1])
        self.cfg['lak']['lookup_file'] = lak_lookup_file
        df = pd.DataFrame({'lakid': np.arange(1, nlakes+1),
                           'hydroid': self.cfg['lak']['include_lakes']})
        df.to_csv(lak_lookup_file, index=False)

        # get estimates of stage from model top, for specifying ranges
        stages = []
        for lakid in range(1, nlakes+1):
            loc = self.lakarr[0] == lakid
            est_stage = self.dis.top.array[loc].min()
            stages.append(est_stage)
        stages = np.array(stages)

        # setup stress period data
        tol = 5  # specify lake stage range as +/- this value
        ssmn, ssmx = stages - tol, stages + tol
        stage_range = list(zip(ssmn, ssmx))
        lakarr_spd = {0: self.cfg['intermediate_data']['lakarr']}
        bdlknc_spd = {0: self.cfg['intermediate_data']['bdlknc']}

        # set up dataset 9
        flux_data = {}
        for i, steady in enumerate(self.dis.steady.array):
            if i > 0 and steady:
                flux_data_i = []
                for lake_ssmn, lake_ssmx in zip(ssmn, ssmx):
                    flux_data_i.append([self.precipitation[i], self.evaporation[i], 0, 0, lake_ssmn, lake_ssmx])
            else:
                flux_data_i = [[self.precipitation[i], self.evaporation[i], 0, 0]] * nlakes
            flux_data[i] = flux_data_i
        options = ['tableinput'] if tab_files_argument is not None else None

        lak = fm.mflak.ModflowLak(self,
                                  nlakes=nlakes,
                                  theta=theta,
                                  nssitr=nssitr,
                                  sscncr=sscncr,
                                  surfdep=surfdep,
                                  stages=stages,
                                  stage_range=stage_range,
                                  lakarr=lakarr_spd,
                                  bdlknc=bdlknc_spd,
                                  flux_data=flux_data,
                                  tab_files=tab_files_argument, #This needs to be in the order of the lake IDs!
                                  tab_units=tab_units,
                                  options=options,
                                  ipakcb=self.ipakcb,
                                  lwrt=0
                                  )
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return lak

    def setup_nwt(self):

        print('setting up NWT package...')
        t0 = time.time()
        use_existing_file = self.cfg['nwt']['use_existing_file']
        kwargs = self.cfg['nwt']
        if use_existing_file is not None:
            #set use_existing_file relative to source path
            filepath = os.path.join(self._config_path,
                                    use_existing_file)

            assert os.path.exists(filepath), "Couldn't find {}, need a path to a NWT file".format(filepath)
            nwt = fm.ModflowNwt.load(filepath, model=self)
        else:
            kwargs = get_input_arguments(kwargs, fm.ModflowNwt)
            nwt = fm.ModflowNwt(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return nwt

    def setup_hyd(self):

        print('setting up HYDMOD package...')
        t0 = time.time()
        obs_info_files = self.cfg['hyd'].get('observation_data')
        if obs_info_files is None:
            print("No observation data for hydmod.")
            return

        # get obs_info_files into dictionary format
        # filename: dict of column names mappings
        if isinstance(obs_info_files, str):
            obs_info_files = [obs_info_files]
        if isinstance(obs_info_files, list):
            obs_info_files = {f: self.cfg['hyd']['default_columns']
                              for f in obs_info_files}
        elif isinstance(obs_info_files, dict):
            for k, v in obs_info_files.items():
                if v is None:
                    obs_info_files[k] = self.cfg['hyd']['default_columns']

        check_source_files(obs_info_files.keys())
        # dictionaries mapping from obstypes to hydmod input
        pckg = {'LK': 'BAS', # head package for high-K lakes; lake package lakes get dropped
                 'GW': 'BAS',
                'head': 'BAS',
                'lake': 'BAS',
                 'ST': 'SFR',
                'flux': 'SFR'
                 }
        arr = {'LK': 'HD',  # head package for high-K lakes; lake package lakes get dropped
               'GW': 'HD',
               'ST': 'SO',
               'flux': 'SO'
               }
        print('Reading observation files...')
        dfs = []
        for f, column_info in obs_info_files.items():
            print(f)
            df = read_observation_data(f, column_info,
                                       column_mappings=self.cfg['hyd'].get('column_mappings'))
            #df = pd.read_csv(f)
            #df.columns = [s.lower() for s in df.columns]
            #df['file'] = f
            #xcol = col_info.get('x_location_col')
            #ycol = col_info.get('y_location_col')
            #obstype_col = col_info.get('obstype_col')
            #if xcol is None or xcol.lower() not in df.columns:
            #    raise ValueError("Column {} not in {}; need to specify x_location_col in config file"
            #                     .format(xcol, f))
            #else:
            #    print('    x location col: {}'.format(xcol))
            #if ycol is None or ycol.lower() not in df.columns:
            #    raise ValueError("Column {} not in {}; need to specify y_location_col in config file"
            #                     .format(ycol, f))
            #else:
            #    print('    y location col: {}'.format(ycol))
            #rename = {xcol.lower(): 'x',
            #          ycol.lower(): 'y'
            #          }
            #if obstype_col is not None:
            #    rename.update({obstype_col.lower(): 'obs_type'})
            #    print('    observation type col: {}'.format(obstype_col))
            #else:
            #    print('    no observation type col specified; observations assumed to be heads')
            #column_mappings = self.cfg['hyd'].get('column_mappings')
            #if column_mappings is not None:
            #    for k, v in column_mappings.items():
            #        if not isinstance(v, list):
            #            v = [v]
            #        for vi in v:
            #            rename.update({vi.lower(): k.lower()})
            #            if vi in df.columns:
            #                print('    observation label column: {}'.format(vi))
            #df.rename(columns=rename, inplace=True)

            if 'obs_type' in df.columns and 'pckg' not in df.columns:
                df['pckg'] = [pckg.get(s, 'BAS') for s in df['obs_type']]
            elif 'pckg' not in df.columns:
                df['pckg'] = 'BAS' # default to getting heads
            if 'obs_type' in df.columns and 'intyp' not in df.columns:
                df['arr'] = [arr.get(s, 'HD') for s in df['obs_type']]
            elif 'arr' not in df.columns:
                df['arr'] = 'HD'
            df['intyp'] = ['I' if p == 'BAS' else 'C' for p in df['pckg']]
            df['hydlbl'] = df['hydlbl'].astype(str).str.lower()

            dfs.append(df[['pckg', 'arr', 'intyp', 'x', 'y', 'hydlbl', 'file']])
        df = pd.concat(dfs, axis=0)

        print('\nCulling observations to model area...')
        df['geometry'] = [Point(x, y) for x, y in zip(df.x, df.y)]
        within = [g.within(self.bbox) for g in df.geometry]
        df = df.loc[within].copy()

        print('Dropping head observations that coincide with Lake Package Lakes...')
        i, j = get_ij(self.modelgrid, df.x.values, df.y.values)
        islak = self.lakarr[0, i, j] != 0
        df = df.loc[~islak].copy()

        drop_obs = self.cfg['hyd'].get('drop_observations', [])
        if len(drop_obs) > 0:
            print('Dropping head observations specified in {}...'.format(self.cfg.get('filename', 'config file')))
            df = df.loc[~df.hydlbl.astype(str).isin(drop_obs)]

        duplicated = df.hydlbl.duplicated(keep=False)
        if duplicated.sum() > 0:
            print('Warning- {} duplicate observation names encountered. First instance of each name will be used.'.format(duplicated.sum()))
            print(df.loc[duplicated, ['hydlbl', 'file']])

        # make sure every head observation is in each layer
        non_heads = df.loc[df.arr != 'HD'].copy()
        heads = df.loc[df.arr == 'HD'].copy()
        heads0 = heads.groupby('hydlbl').first().reset_index()
        heads0['hydlbl'] = heads0['hydlbl'].astype(str)
        heads_all_layers = pd.concat([heads0] * self.nlay).sort_values(by='hydlbl')
        heads_all_layers['klay'] = list(range(self.nlay)) * len(heads0)
        df = pd.concat([heads_all_layers, non_heads], axis=0)

        # get model locations
        xl, yl = self.modelgrid.get_local_coords(df.x.values, df.y.values)
        df['xl'] = xl
        df['yl'] = yl

        # create observation data recarray
        obsdata = fm.ModflowHyd.get_empty(len(df))
        for c in obsdata.dtype.names:
            assert c in df.columns, "Missing observation data field: {}".format(c)
            obsdata[c] = df[c]
        nhyd = len(df)
        hyd = flopy.modflow.ModflowHyd(self, nhyd=nhyd, hydnoh=-999, obsdata=obsdata)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return hyd

    def setup_gag(self):

        print('setting up GAGE package...')
        t0 = time.time()
        # setup gage package output for all included lakes
        ngages = 0
        nlak_gages = 0
        if 'LAK' in self.get_package_list():
            nlak_gages = self.lak.nlakes
        if nlak_gages > 0:
            ngages += nlak_gages
            lak_gagelocs = list(np.arange(1, nlak_gages+1) * -1)
            lak_gagerch = [0] * nlak_gages # dummy list to maintain index position
            lak_outtype = [self.cfg['gag']['lak_outtype']] * nlak_gages
            lak_files = ['lak{}_{}.ggo'.format(i+1, hydroid)
                         for i, hydroid in enumerate(self.cfg['lak']['include_lakes'])]

        # need to add streams at some point
        nstream_gages = 0
        if 'SFR' in self.get_package_list():

            obs_info_files = self.cfg['gag']['observation_data']

            # get obs_info_files into dictionary format
            # filename: dict of column names mappings
            if isinstance(obs_info_files, str):
                obs_info_files = [obs_info_files]
            if isinstance(obs_info_files, list):
                obs_info_files = {f: self.cfg['gag']['default_columns']
                                  for f in obs_info_files}
            elif isinstance(obs_info_files, dict):
                for k, v in obs_info_files.items():
                    if v is None:
                        obs_info_files[k] = self.cfg['gag']['default_columns']

            check_source_files(obs_info_files.keys())

            print('Reading observation files...')
            dfs = []
            for f, column_info in obs_info_files.items():
                print(f)
                df = read_observation_data(f,
                                           column_info,
                                           column_mappings=self.cfg['hyd'].get('column_mappings'))
                dfs.append(df) # cull to cols that are needed
            df = pd.concat(dfs, axis=0)
            assert True
        ngages += nstream_gages
        stream_gageseg = []
        stream_gagerch = []
        stream_outtype = [self.cfg['gag']['sfr_outtype']] * nstream_gages
        stream_files = []

        if ngages == 0:
            print('No gage package input.')
            return

        # create flopy gage package object
        gage_data = fm.ModflowGage.get_empty(ncells=ngages)
        gage_data['gageloc'] = lak_gagelocs + stream_gageseg
        gage_data['gagerch'] = lak_gagerch + stream_gagerch
        gage_data['unit'] = np.arange(self.cfg['gag']['starting_unit_number'],
                                      self.cfg['gag']['starting_unit_number'] + ngages)
        gage_data['outtype'] = lak_outtype + stream_outtype
        if self.cfg['gag'].get('ggo_files') is None:
            self.cfg['gag']['ggo_files'] = lak_files + stream_files
        gag = fm.ModflowGage(self, numgage=len(gage_data),
                             gage_data=gage_data,
                             files=self.cfg['gag']['ggo_files'],
                             )
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return gag

    def load_grid(self, gridfile=None):
        """Load model grid information from a json or yml file."""
        if gridfile is None:
            if os.path.exists(self.cfg['setup_grid']['grid_file']):
                gridfile = self.cfg['setup_grid']['grid_file']
        print('Loading model grid information from {}'.format(gridfile))
        self.cfg['grid'] = load(gridfile)

    @staticmethod
    def setup_from_yaml(yamlfile, verbose=False):
        """Make a model from scratch, using information in a yamlfile.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with model setup information.

        Returns
        -------
        m : mfsetup.MFnwtModel model object
        """

        cfg = MFnwtModel.load_cfg(yamlfile, verbose=verbose)
        cfg['filename'] = yamlfile
        print('\nSetting up {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = MFnwtModel(cfg=cfg, **cfg['model'])
        assert m.exe_name != 'mf2005.exe'

        kwargs = m.cfg['setup_grid']
        rename = kwargs.get('variable_mappings', {})
        for k, v in rename.items():
            if k in kwargs:
                kwargs[v] = kwargs.pop(k)
        kwargs = get_input_arguments(kwargs, m.setup_grid)
        if 'grid' not in m.cfg.keys():
            m.setup_grid(**kwargs)

        # set up all of the packages specified in the config file
        for pkg in m.package_list:
            package_setup = getattr(MFnwtModel, 'setup_{}'.format(pkg))
            package_setup(m)

        if m.perimeter_bc_type == 'head':
            chd = m.setup_chd()
        print('finished setting up model in {:.2f}s'.format(time.time() - t0))
        print('\n{}'.format(m))
        #Export a grid outline shapefile.
        write_bbox_shapefile(m.modelgrid,
                             os.path.join(m.cfg['postprocessing']['output_folders']['shapefiles'],
                                          'model_bounds.shp'))
        print('wrote bounding box shapefile')
        return m

    @staticmethod
    def load_cfg(yamlfile, verbose=False):
        """Load model configuration info, adjusting paths to model_ws."""
        return load_cfg(yamlfile, default_file='/mfnwt_defaults.yml')
        #print('loading configuration file {}...'.format(yamlfile))
        #source_path = os.path.split(__file__)[0]
        ## default configuration
        #cfg = load(source_path + '/mfnwt_defaults.yml')
        #cfg['filename'] = source_path + '/mfnwt_defaults.yml'
#
        ## recursively update defaults with information from yamlfile
        #update(cfg, load(yamlfile))
        #cfg['model'].update({'verbose': verbose})
#
        #cfg = set_absolute_paths_to_location()
        ## set paths relative to config file
        ## TODO: need more general way to handle paths
        #cfg['model']['model_ws'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                        cfg['model']['model_ws'])
        #cfg['parent']['model_ws'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         cfg['parent']['model_ws'])
        #cfg['parent']['headfile'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         cfg['parent']['headfile'])
        #if cfg['setup_grid'].get('features_file') is not None:
        #    cfg['setup_grid']['features_file'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         cfg['setup_grid']['features_file'])
        #skip_keys = ['elevation_units']
        #for k, v in cfg.get('source_data', {}).items():
        #    if k not in skip_keys:
        #        cfg['source_data'][k] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         v)
        #if cfg.get('hyd', {}).get('observation_data') is not None:
        #    for i, f in enumerate(cfg['hyd']['observation_data']):
        #        cfg['hyd']['observation_data'][i] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                                         f)
        #if cfg['setup_grid'].get('features_file') is not None:
        #    cfg['setup_grid']['features_file'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         cfg['setup_grid']['features_file'])
        #if cfg.get('lak', {}).get('stage_area_volume') is not None:
        #    cfg['lak']['stage_area_volume'] = os.path.join(os.path.split(os.path.abspath(yamlfile))[0],
        #                                         cfg['lak']['stage_area_volume'])
        #def set_path(relative_path):
        #    return os.path.join(cfg['model']['model_ws'],
        #                        relative_path)
        #cfg['intermediate_data']['tmpdir'] = set_path(cfg['intermediate_data']['tmpdir'])
        ##cfg['model']['external_path'] = set_path(cfg['model']['external_path'])
        #cfg['setup_grid']['grid_file'] = set_path(os.path.split(cfg['setup_grid']['grid_file'])[-1])
        #mapping = {'shapefiles': 'shps'}
        #for k, v in cfg['postprocessing']['output_folders'].items():
        #    if k in mapping.keys():
        #        cfg['postprocessing']['output_folders'][mapping[k]] = cfg['postprocessing']['output_folders'].pop(k)
        #for k, v in cfg['postprocessing']['output_folders'].items():
        #    cfg['postprocessing']['output_folders'][k] = set_path(v)
        #return cfg

    @staticmethod
    def load(yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = MFnwtModel.load_cfg(yamlfile, verbose=verbose)
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = MFnwtModel(cfg=cfg, **cfg['model'])
        if 'grid' not in m.cfg.keys():
            if os.path.exists(cfg['setup_grid']['grid_file']):
                print('Loading model grid definition from {}'.format(cfg['setup_grid']['grid_file']))
                m.cfg['grid'] = load(m.cfg['setup_grid']['grid_file'])
            else:
                m.setup_grid(**m.cfg['setup_grid'])
        m = flopy_mf2005_load(m, load_only=load_only, forgive=forgive, check=check)
        print('finished loading model in {:.2f}s'.format(time.time() - t0))
        return m


class Inset(MFnwtModel):
    """Class representing a MODFLOW-NWT model that is an
    inset of a parent MODFLOW model."""
    def __init__(self, parent=None, cfg=None,
                 modelname='inset', exe_name='mfnwt',
                 version='mfnwt', model_ws='.',
                 external_path='external/', **kwargs):

        MFnwtModel.__init__(self,  parent=parent, cfg=cfg,
                            modelname=modelname, exe_name=exe_name,
                            version=version, model_ws=model_ws,
                            external_path=external_path, **kwargs)

        inset_cfg = load(self.source_path + '/inset_defaults.yml')
        update(self.cfg, inset_cfg)
        self.cfg['filename'] = self.source_path + '/inset_defaults.yml'

