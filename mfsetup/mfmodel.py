import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from packaging import version

fm = flopy.modflow
mf6 = flopy.mf6
import gisutils
import sfrmaker
from gisutils import get_shapefile_crs, get_values_at_points, project
from sfrmaker import Lines
from sfrmaker.utils import assign_layers

from mfsetup.bcs import (
    get_bc_package_cells,
    setup_basic_stress_data,
    setup_flopy_stress_period_data,
)
from mfsetup.config import validate_configuration
from mfsetup.fileio import (
    check_source_files,
    load,
    load_array,
    load_cfg,
    save_array,
    set_cfg_paths_to_absolute,
    setup_external_filepaths,
)
from mfsetup.grid import MFsetupGrid, get_ij, rasterize, setup_structured_grid
from mfsetup.interpolate import (
    get_source_dest_model_xys,
    interp_weights,
    interpolate,
    regrid,
)
from mfsetup.lakes import make_lakarr2d, setup_lake_fluxes, setup_lake_info
from mfsetup.mf5to6 import (
    get_model_length_units,
    get_model_time_units,
    get_package_name,
)
from mfsetup.model_version import get_versions
from mfsetup.sourcedata import TransientTabularSourceData, setup_array
from mfsetup.tdis import (
    concat_periodata_groups,
    get_parent_stress_periods,
    parse_perioddata_groups,
    setup_perioddata,
    setup_perioddata_group,
)
from mfsetup.tmr import Tmr
from mfsetup.units import convert_length_units, lenuni_text, lenuni_values
from mfsetup.utils import flatten, get_input_arguments, get_packages, update
from mfsetup.wells import setup_wel_data

if version.parse(gisutils.__version__) < version.parse('0.2.2'):
    warnings.warn('Automatic reprojection functionality requires gis-utils >= 0.2.2'
                  '\nPlease pip install --upgrade gis-utils')
if version.parse(sfrmaker.__version__) < version.parse('0.6'):
    warnings.warn('sfr: sfrmaker_options: add_outlet functionality requires sfrmaker >= 0.6'
                  '\nPlease pip install --upgrade sfrmaker')


class MFsetupMixin():
    """Mixin class for shared functionality between MF6model and MFnwtModel.
    Meant to be inherited by both those classes and not be called directly.

    https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
    """
    source_path = Path(__file__).parent
    """        -1 : well
        0 : no lake
        1 : lak package lake (lakarr > 0)
        2 : high-k lake
        3 : ghb
        4 : sfr"""
    # package variable name: number
    bc_numbers = {'wel': -1,
                  'lak': 1,
                  'high-k lake': 2,
                  'ghb': 3,
                  'sfr': 4,
                  'riv': 5
                  }
    model_type = "mfsetup"

    def __init__(self, parent):

        # property attributes
        self._cfg = None
        self._nper = None
        self._perioddata = None
        self._sr = None
        self._modelgrid = None
        self._bbox = None
        self._parent = parent
        self._parent_layers = None
        self._parent_default_source_data = False
        self._parent_mask = None
        self._lakarr_2d = None
        self._isbc_2d = None
        self._lakarr = None
        self._isbc = None
        self._lake_bathymetry = None
        self._high_k_lake_recharge = None
        self._nodata_value = -9999
        self._model_ws = None
        self._abs_model_ws = None
        self._model_version = None  # semantic version of model
        self._longname = None  # long name for model (short name is self.name)
        self._header = None  # header for files and repr
        self.inset = None  # dictionary of inset models attached to LGR parent
        self._is_lgr = False  # flag for lgr inset models
        self.lgr = None  # holds flopy Lgr utility object
        self._lgr_idomain2d = None # array of Lgr inset model locations within parent grid
        self.tmr = None  # holds TMR class instance for TMR-type perimeter boundaries
        self._load = False  # whether model is being made or loaded from existing files
        self.lake_info = None
        self.lake_fluxes = None

        # flopy settings
        self._mg_resync = False

        self._features = {}  # dictionary for caching shapefile datasets in memory

        # arrays remade during this session
        self.updated_arrays = set()

        # cache of interpolation weights to speed up regridding
        self._interp_weights = None


    def __repr__(self):
        header = f'{self.header}\n'
        txt = ''
        if self.parent is not None:
            txt += 'Parent model: {}/{}\n'.format(self.parent.model_ws, self.parent.name)
        if self._modelgrid is not None:
            txt += f'{self._modelgrid.__repr__()}'
        txt += 'Packages:'
        for pkg in self.get_package_list():
            txt += ' {}'.format(pkg.lower())
        txt += '\n'
        txt += f'{self.nper:d} period(s):\n'
        if self._perioddata is not None:
            cols = ['per', 'start_datetime', 'end_datetime', 'perlen', 'steady', 'nstp']
            txt += self.perioddata[cols].head(3).to_string(index=False)
            txt += '\n   ...\n'
            tail = self.perioddata[cols].tail(1).to_string(index=False)
            txt += tail.split('\n')[1]
        txt = header + txt
        return txt

    def __eq__(self, other):
        """Test for equality to another model object."""
        if not isinstance(other, self.__class__):
            return False
        # kludge: skip obs packages for now
        # - obs packages aren't read in with same name under which they were created
        # - also SFR_OBS package is handled by SFRmaker instead of Flopy;
        # a loaded version of a model might have SFR_OBS,
        # where a freshly made version may not (even though SFRmaker will write it)
        #
        all_packages = set(self.get_package_list()).union(other.get_package_list())
        exceptions = {p for p in all_packages if p.lower().startswith('obs')
                      or p.lower().endswith('obs')}
        other_packages = [s for s in sorted(other.get_package_list())
                          if s not in exceptions]
        packages = [s for s in sorted(self.get_package_list())
                    if s not in exceptions]
        if other_packages != packages:
            return False
        if other.modelgrid != self.modelgrid:
            return False
        if other.nlay != self.nlay:
            return False
        if not np.array_equal(other.perioddata, self.perioddata):
            return False
        #  TODO: add checks of actual array values and other parameters
        for k, v in self.__dict__.items():
            if k in ['cfg',
                     'sfrdata',
                     '_load',
                     '_packagelist',
                     '_package_paths',
                     'package_key_dict',
                     'package_type_dict',
                     'package_name_dict',
                     '_ftype_num_dict']:
                continue
            elif k not in other.__dict__:
                return False
            elif type(v) == bool:
                if not v == other.__dict__[k]:
                    return False
            elif k == 'cfg':
                continue
            elif type(v) in [str, int, float, dict, list]:
                if v != other.__dict__[k]:
                    pass
                continue
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
            self.setup_grid()
        # trap for instance where default (base) modelgrid
        # instance is attached to the flopy model
        # (because the grid hasn't been set up with)
        # self._modelgrid.nlay will error in this case
        # because of NotImplementedError in base class
        elif self._modelgrid.grid_type is None:
            pass
        # add layer tops and bottoms and idomain to the model grid
        # if they haven't been yet
        elif self._modelgrid.nlay is None and 'DIS' in self.get_package_list():
            self._modelgrid._top = self.dis.top.array
            self._modelgrid._botm = self.dis.botm.array
            if self.version == 'mf6':
                self._modelgrid._idomain = self.dis.idomain.array
            elif 'bas6' in self.get_package_list():
                self._modelgrid._idomain = self.bas6.ibound.array
            #self.setup_grid()
        return self._modelgrid

    @property
    def bbox(self):
        if self._bbox is None and self.modelgrid is not None:
            self._bbox = self.modelgrid.bbox
        return self._bbox

    #@property
    #def perioddata(self):
    #    """DataFrame summarizing stress period information.
#
    #    Columns:
#
    #      start_date_time : pandas datetimes; start date/time of each stress period
    #      (does not include steady-state periods)
    #      end_date_time : pandas datetimes; end date/time of each stress period
    #      (does not include steady-state periods)
    #      time : float; cumulative MODFLOW time (includes steady-state periods)
    #      per : zero-based stress period
    #      perlen : stress period length in model time units
    #      nstp : number of timesteps in the stress period
    #      tsmult : timestep multiplier for stress period
    #      steady : True=steady-state, False=Transient
    #      oc : MODFLOW-6 output control options
    #    """
    #    if self._perioddata is None:
    #        perioddata = setup_perioddata(self)
    #    return self._perioddata

    @property
    def parent(self):
        return self._parent

    @property
    def parent_layers(self):
        """Mapping between layers in source model and
        layers in destination model.

        Returns
        -------
        parent_layers : dict
            {inset layer : parent layer}
        """
        if self._parent_layers is None:
            parent_layers = None
            botm_source_data = self.cfg['dis'].get('source_data', {}).get('botm', {})
            nlay = self.modelgrid.nlay
            if nlay is None:
                nlay = self.cfg['dis']['dimensions']['nlay']
            if self.cfg['parent'].get('inset_layer_mapping') is not None:
                parent_layers = self.cfg['parent'].get('inset_layer_mapping')
            elif isinstance(botm_source_data, dict) and 'from_parent' in botm_source_data:
                parent_layers = botm_source_data.get('from_parent')
            elif self.parent is not None and (self.parent.modelgrid.nlay == nlay):
                parent_layers = dict(zip(range(self.parent.modelgrid.nlay),
                                         range(nlay)))
            else:
                #parent_layers = dict(zip(range(self.parent.modelgrid.nlay), range(self.parent.modelgrid.nlay)))
                parent_layers = None
            self._parent_layers = parent_layers
        return self._parent_layers

    @property
    def parent_stress_periods(self):
        """Mapping between stress periods in source model and
        stress periods in destination model.

        Returns
        -------
        parent_stress_periods : dict
            {inset stress period : parent stress period}
        """
        return dict(zip(self.perioddata['per'], self.perioddata['parent_sp']))

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
    def model_ws(self):
        if self._model_ws is None:
            self._model_ws = Path(self._get_model_ws())
        return self._model_ws

    @model_ws.setter
    def model_ws(self, model_ws):
        self._model_ws = model_ws
        self._abs_model_ws = os.path.normpath(os.path.abspath(model_ws))

    @property
    def model_version(self):
        """Semantic version of model, using a hacked version of the versioneer.
        Version is reported using git tags for the model repository
        or a start_version: key specified in the configuration file (default 0).
        The start_version or tag is then appended by the remaining information
        in a pep440-post style version tag (e.g. most recent git commit hash
        for the model repository + "dirty" if the model repository has uncommited changes)

        References
        ----------
        https://github.com/warner/python-versioneer
        https://github.com/warner/python-versioneer/blob/master/details.md
        """
        if self._model_version is None:
            self._model_version = get_versions(path=self.model_ws,
                                   start_version=str(self.cfg['metadata']['start_version']))
        return self._model_version

    @property
    def longname(self):
        if self._longname is None:
            longname = self.cfg['metadata'].get('longname')
            if longname is None:
                longname = f'{self.name} model'
            self._longname = longname
        return self._longname

    @property
    def header(self):
        if self._header is None:
            version_str = self.model_version['version']
            header = f'{self.longname} version {version_str}'
            self._header = header
        return self._header

    @property
    def tmpdir(self):
        #abspath = os.path.abspath(
        #        self.cfg['intermediate_data']['output_folder'])
        abspath = self.model_ws / 'original-arrays'
        self.cfg['intermediate_data']['output_folder'] = str(abspath)
        abspath.mkdir(exist_ok=True)
        #if not os.path.isdir(abspath):
        #    os.makedirs(abspath)
        tmpdir = abspath
        if self.relative_external_paths:
            #tmpdir = os.path.relpath(abspath)
            tmpdir = abspath.relative_to(self.model_ws)
        #else:
        #   do we need to normalize with Pathlib??
        #    tmpdir = os.path.normpath(abspath)
        return tmpdir

    @property
    def external_path(self):
        abspath = os.path.abspath(
            self.cfg.get('model', {}).get('external_path', 'external'))
        if not os.path.isdir(abspath):
            os.makedirs(abspath)
        if self.relative_external_paths:
            ext_path = os.path.relpath(abspath)
        else:
            ext_path = os.path.normpath(abspath)
        return ext_path

    @external_path.setter
    def external_path(self, x):
        pass # bypass any setting in parent class

    @property
    def interp_weights(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to pfl_nwt."""
        if self._interp_weights is None:
            parent_xy, inset_xy = get_source_dest_model_xys(self.parent,
                                                                        self)
            self._interp_weights = interp_weights(parent_xy, inset_xy)
        return self._interp_weights

    @property
    def parent_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the inset model domain, with a surrounding buffer.
        Used to speed up interpolation of parent grid values onto inset model grid."""
        if self._parent_mask is None:
            x, y = np.squeeze(self.bbox.exterior.coords.xy)
            pi, pj = get_ij(self.parent.modelgrid, x, y)
            pad = 3
            i0 = np.max([pi.min() - pad, 0])
            i1 = np.min([pi.max() + pad + 1, self.parent.modelgrid.nrow])
            j0 = np.max([pj.min() - pad, 0])
            j1 = np.min([pj.max() + pad + 1, self.parent.modelgrid.ncol])
            mask = np.zeros((self.parent.modelgrid.nrow, self.parent.modelgrid.ncol), dtype=bool)
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
        """
        if self._lakarr is None:
            self.setup_external_filepaths('lak', 'lakarr',
                                          self.cfg['lak']['{}_filename_fmt'.format('lakarr')],
                                          file_numbers=list(range(self.nlay)))
            if self.isbc is None:
                return None
            else:
                self._set_lakarr()
        return self._lakarr

    @property
    def _isbc2d(self):
        """2-D array indicating the i, j locations of
        boundary conditions.
        -1 : well
        0 : no lake
        1 : lak package lake (lakarr > 0)
        2 : high-k lake
        3 : ghb
        4 : sfr
        5 : riv

        see also the .bc_numbers attibute
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
        3 : ghb
        4 : sfr
        5 : riv

        see also the .bc_numbers attibute
        """
        # DIS package is needed to set up the isbc array
        # (to compare lake bottom elevations to layer bottoms)
        if self.get_package('dis') is None:
            return None
        if self._isbc is None:
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
    def high_k_lake_recharge(self):
        """Recharge value to apply to high-K lakes, in model units.
        """
        if self._high_k_lake_recharge is None and self.cfg['high_k_lakes']['simulate_high_k_lakes']:
            if self.lake_info is None:
                self.lake_info = setup_lake_info(self)
                if self.lake_info is not None:
                    self.lake_fluxes = setup_lake_fluxes(self, block='high_k_lakes')
                    self._high_k_lake_recharge = self.lake_fluxes.groupby('per').mean()['highk_lake_rech'].sort_index()
        return self._high_k_lake_recharge

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

    def load_features(self, filename, bbox_filter=None,
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
                    features_crs = get_shapefile_crs(f)
                    if bbox_filter is None:
                        if self.bbox is not None:
                            bbox = self.bbox
                        elif self.parent.modelgrid is not None:
                            bbox = self.parent.modelgrid.bbox
                            model_crs = self.parent.modelgrid.crs
                            assert model_crs is not None

                        if features_crs != self.modelgrid.crs:
                            bbox_filter = project(bbox, self.modelgrid.crs, features_crs).bounds
                        else:
                            bbox_filter = bbox.bounds

                    # implement automatic reprojection in gis-utils
                    # maintaining backwards compatibility
                    df = gpd.read_file(f)
                    df.to_crs(self.modelgrid.crs, inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    if cache:
                        print('caching data in {}...'.format(f))
                        self._features[f] = df
                else:
                    print('feature input file {} not found'.format(f))
                    return
            else:
                df = self._features[f]
            if id_column is not None:
                id_column = id_column.lower()
                # convert any floating point dtypes to integer
                if df[id_column].dtype == float:
                    df[id_column] = df[id_column].astype('int64')
                df.index = df[id_column]
            if include_ids is not None:
                df = df.loc[include_ids].copy()
            dfs_list.append(df)
        df = pd.concat(dfs_list)
        if len(df) == 0:
            warnings.warn('No features loaded from {}!'.format(filename))
        return df

    def get_boundary_cells(self, exclude_inactive=False):
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
        k = np.array(sorted(list(range(self.nlay)) * len(i)))
        i = np.array(i * self.nlay)
        j = np.array(j * self.nlay)
        assert np.sum(k[nlaycells:nlaycells * 2]) == nlaycells

        if exclude_inactive:
            if self.version == 'mf6':
                active_cells = self.idomain[k, i, j] >= 1
            else:
                active_cells = self.ibound[k, i, j] >= 1
            k = k[active_cells].copy()
            i = i[active_cells].copy()
            j = j[active_cells].copy()
        return k, i, j

    def regrid_from_parent(self, parent_array,
                           mask=None,
                           method='linear'):
        """Interpolate values in parent array onto
        the pfl_nwt model grid, using model grid instances
        attached to the parent and pfl_nwt models.

        Parameters
        ----------
        parent_array : ndarray
            Values from parent model to be interpolated to pfl_nwt grid.
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
            #parent_values = parent_array.flatten()[self.parent_mask.flatten()]
            parent_values = parent_array[self.parent_mask].flatten()
            regridded = interpolate(parent_values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = regrid(parent_array, self.parent.modelgrid, self.modelgrid,
                               method='nearest')
        regridded = np.reshape(regridded, (self.nrow, self.ncol))
        return regridded

    def setup_external_filepaths(self, package, variable_name,
                                 filename_format, file_numbers=None):
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
        file_numbers : list of ints
            List of numbers for the external files. Usually these represent zero-based
            layers or stress periods.
        relative_external_paths : bool
            If true, external paths will be specified relative to model_ws,
            otherwise, they will be absolute paths
        Returns
        -------
        filepaths : list
            List of external file paths

        Adds intermediated file paths to model.cfg[<package>]['intermediate_data']
        Adds external file paths to model.cfg[<package>][<variable_name>]
        """
        # for lgr models, add the model name to the external filename
        # if lgr parent or lgr inset
        if self.lgr or self._is_lgr:
            filename_format = '{}_{}'.format(self.name, filename_format)
        return setup_external_filepaths(self, package, variable_name,
                                        filename_format, file_numbers=file_numbers,
                                        relative_external_paths=self.relative_external_paths)

    def _get_model_ws(self, cfg=None):
        if cfg is None:
            cfg = self.cfg
        if self.version == 'mf6':
            abspath = os.path.abspath(cfg.get('simulation', {}).get('sim_ws', '.'))
        else:
            abspath = os.path.abspath(cfg.get('model', {}).get('model_ws', '.'))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        self._abs_model_ws = os.path.normpath(abspath)
        os.chdir(abspath)  # within a session, modflow-setup operates in the model_ws
        if self.relative_external_paths:
            model_ws = os.path.relpath(abspath)
        else:
            model_ws = os.path.normpath(abspath)
        return Path(model_ws)

    def _reset_bc_arrays(self):
        """Reset the boundary condition property arrays in order.
        _lakarr2d (depends on _lakarr_2d
        _isbc2d  (depends on _lakarr2d)
        _lake_bathymetry (depends on _isbc2d)
        _isbc  (depends on _isbc2d)
        _lakarr  (depends on _isbc and _lakarr2d)
        """
        self._lakarr_2d = None
        self._isbc_2d = None #  (depends on _lakarr2d)
        self._lake_bathymetry = None # (depends on _isbc2d)
        self._isbc = None #  (depends on _isbc2d)
        self._lakarr = None #
        #self._set_lakarr2d() # calls self._set_isbc2d(), which calls self._set_lake_bathymetry()
        #self._set_isbc() # calls self._set_lakarr()

    def _set_cfg(self, user_specified_cfg):
        """Load configuration file; update dictionary.
        """
        #self.cfg = defaultdict(dict)
        self.cfg = defaultdict(dict, self.cfg)

        if isinstance(user_specified_cfg, str) or \
                isinstance(user_specified_cfg, Path):
            raise ValueError("Configuration should have already been loaded")
            # convert to an absolute path
            #user_specified_cfg = Path(user_specified_cfg).resolve()
            #assert user_specified_cfg.exists(), \
            #    "config file {} not found".format(user_specified_cfg)
            #updates = load(user_specified_cfg)
            #updates['filename'] = user_specified_cfg
        elif isinstance(user_specified_cfg, dict):
            updates = user_specified_cfg.copy()
        elif user_specified_cfg is None:
            return
        else:
            raise TypeError("unrecognized input for cfg")

        # if the user specifies a complexity option for IMS or NWT,
        # don't import any defaults
        ims_cfg = updates.get('ims', {})
        if ims_cfg.get('options', {}).get('complexity'):
            # delete the defaults
            for default_block in 'nonlinear', 'linear':
                if default_block in self.cfg['ims']:
                    del self.cfg['ims'][default_block]
        nwt_cfg = updates.get('nwt', {})
        if nwt_cfg.get('options', 'specified').lower() != 'specified':
            keep_args = {'headtol', 'fluxtol', 'maxiterout',
                         'thickfact', 'linmeth', 'iprnwt', 'ibotav',
                         'Continue', 'use_existing_file'}
            self.cfg['nwt'] = {k: v for k, v in self.cfg['nwt'].items() if k in keep_args}

        update(self.cfg, updates)
        # make sure empty variables get initialized as dicts
        for k, v in self.cfg.items():
            if v is None:
                self.cfg[k] = {}

        if 'filename' in self.cfg:
            config_file_path = Path(self.cfg['filename'])
            if config_file_path.is_absolute():
                self.cfg = set_cfg_paths_to_absolute(self.cfg, config_file_path.parent)

        # mf6 models: set up or load the simulation
        if self.version == 'mf6':
            kwargs = self.cfg['simulation'].copy()
            kwargs.update(self.cfg['simulation']['options'])
            if os.path.exists('{}.nam'.format(kwargs['sim_name'])) and self._load:
                try:
                    kwargs = get_input_arguments(kwargs, mf6.MFSimulation.load, warn=False)
                    self._sim = mf6.MFSimulation.load(**kwargs)
                except:
                    # create simulation
                    kwargs = get_input_arguments(kwargs, mf6.MFSimulation, warn=False)
                    self._sim = mf6.MFSimulation(**kwargs)
            else:
                # create simulation
                kwargs = get_input_arguments(kwargs, mf6.MFSimulation, warn=False)
                self._sim = mf6.MFSimulation(**kwargs)

        # load the parent model (skip if already attached)
        if 'namefile' in self.cfg.get('parent', {}).keys():
            self._set_parent()

        output_paths = self.cfg['postprocessing']['output_folders']
        for name, folder_path in output_paths.items():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            setattr(self, '_{}_path'.format(name), folder_path)

        # absolute path to config file
        self._config_path = os.path.split(os.path.abspath(str(self.cfg['filename'])))[0]

        # set package keys to default dicts
        for pkg in self._package_setup_order:
            self.cfg[pkg] = defaultdict(dict, self.cfg.get(pkg, {}))

        # other variables
        self.cfg['external_files'] = {}

        # validate the configuration
        validate_configuration(self.cfg)

    def _get_high_k_lakes(self):
        """Get the i, j locations of any high-k lakes within the model grid.
        """
        lakesdata = None
        lakes_shapefile = self.cfg['high_k_lakes'].get('source_data', {}).get('lakes_shapefile')
        if lakes_shapefile is not None:
            if isinstance(lakes_shapefile, str):
                lakes_shapefile = {'filename': lakes_shapefile}
            kwargs = get_input_arguments(lakes_shapefile, self.load_features)
            if 'include_ids' in kwargs:  # load all lakes in shapefile
                kwargs.pop('include_ids')
            lakesdata = self.load_features(**kwargs)
        if lakesdata is not None:
            is_high_k_lake = rasterize(lakesdata, self.modelgrid)
            return is_high_k_lake > 0

    def _set_isbc2d(self):
        """Set up the _isbc2d array, that indicates the i,j locations
        of boundary conditions.
        """
        isbc = np.zeros((self.nrow, self.ncol), dtype=int)

        # high-k lakes
        if self.cfg['high_k_lakes']['simulate_high_k_lakes']:
            is_high_k_lake = self._get_high_k_lakes()
            if is_high_k_lake is not None:
                isbc[is_high_k_lake] = 2

        # lake package lakes
        isbc[self._lakarr2d > 0] = 1

        # add other bcs
        for packagename, bcnumber in self.bc_numbers.items():
            if 'lak' not in packagename:
                package = self.get_package(packagename)
                if package is not None:
                    # handle multiple instances of package
                    # (in MODFLOW-6)
                    if isinstance(package, flopy.pakbase.PackageInterface):
                        packages = [package]
                    else:
                        packages = package
                    for package in packages:
                        k, i, j = get_bc_package_cells(package)
                        not_a_lake = np.where(isbc[i, j] != 1)
                        i = i[not_a_lake]
                        j = j[not_a_lake]
                        isbc[i, j] = bcnumber
        self._isbc_2d = isbc
        self._set_lake_bathymetry()

    def _set_isbc(self):
            isbc = np.zeros((self.nlay, self.nrow, self.ncol), dtype=int)
            isbc[0] = self._isbc2d

            # in mf6 models, the model top is set to the lake botm
            # and any layers originally above the lake botm
            # are also reset to the lake botm (given zero-thickness)
            lake_botm_elevations = self.dis.top.array
            below = self.dis.botm.array >= lake_botm_elevations
            if not self.version == 'mf6':
                lake_botm_elevations = self.dis.top.array - self.lake_bathymetry
                layer_tops = np.concatenate([[self.dis.top.array], self.dis.botm.array[:-1]])
                # lakes must be at least 10% into a layer to get simulated in that layer
                below = layer_tops > lake_botm_elevations + 0.1
            for i, ibelow in enumerate(below[1:]):
                if np.any(ibelow):
                    isbc[i+1][ibelow] = self._isbc2d[ibelow]
            # add other bcs
            for packagename, bcnumber in self.bc_numbers.items():
                if 'lak' not in packagename:
                    package = self.get_package(packagename)
                    if package is not None:
                        # handle multiple instances of package
                        # (in MODFLOW-6)
                        if isinstance(package, flopy.pakbase.PackageInterface):
                            packages = [package]
                        else:
                            packages = package
                        for package in packages:
                            k, i, j = get_bc_package_cells(package)
                            not_a_lake = np.where(isbc[k, i, j] != 1)
                            k = k[not_a_lake]
                            i = i[not_a_lake]
                            j = j[not_a_lake]
                            isbc[k, i, j] = bcnumber
            self._isbc = isbc
            self._set_lakarr()

    def _set_lakarr2d(self):
            lakarr2d = np.zeros((self.nrow, self.ncol), dtype=int)
            if 'lak' in self.package_list:
                lakes_shapefile = self.cfg['lak'].get('source_data', {}).get('lakes_shapefile', {}).copy()
                if lakes_shapefile:
                    kwargs = get_input_arguments(lakes_shapefile, self.load_features)
                    lakesdata = self.load_features(**kwargs)  # caches loaded features
                    lakes_shapefile['lakesdata'] = lakesdata
                    lakes_shapefile.pop('filename')
                    kwargs = get_input_arguments(lakes_shapefile, make_lakarr2d)
                    lakarr2d = make_lakarr2d(self.modelgrid, **kwargs)
            self._lakarr_2d = lakarr2d
            self._set_isbc2d()

    def _set_lakarr(self):
        self.setup_external_filepaths('lak', 'lakarr',
                                      self.cfg['lak']['{}_filename_fmt'.format('lakarr')],
                                      file_numbers=list(range(self.nlay)))
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
                                         points_crs=self.modelgrid.crs,
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
                      'crs': self.parent.modelgrid.crs,
                      'epsg': self.parent.modelgrid.epsg,
                      #'proj4': self.parent.modelgrid.proj4,
                      }
        parent_units = get_model_length_units(self.parent)
        if 'lenuni' in self.cfg['parent']:
            parent_units = lenuni_text[self.cfg['parent']['lenuni']]
        elif 'length_units' in self.cfg['parent']:
            parent_units = self.cfg['parent']['length_units']

        if self.version == 'mf6':
            self.parent.dis.length_units = parent_units
        else:
            self.parent.dis.lenuni = lenuni_values[parent_units]

        # make sure crs is populated, then get CRS units for the grid
        from gisutils import get_authority_crs
        if kwargs.get('crs') is not None:
            kwargs['crs'] = get_authority_crs(kwargs['crs'])
        elif kwargs.get('epsg') is not None:
            kwargs['crs'] = get_authority_crs(kwargs['epsg'])
        # no parent CRS info, assume the parent model is in the same CRS
        elif self.cfg['setup_grid'].get('crs') is not None:
            kwargs['crs'] = get_authority_crs(self.cfg['setup_grid']['crs'])
        # no parent CRS info, assume the parent model is in the same CRS
        elif self.cfg['setup_grid'].get('epsg') is not None:
            kwargs['crs'] = get_authority_crs(self.cfg['setup_grid']['epsg'])
        else:
            raise ValueError('No coordinate reference input in setup_grid: or parent: '
                             'SpatialReference: blocks of configuration file. Supply '
                             'at least coordinate reference information to '
                             'setup_grid: crs: item.')

        parent_grid_units = kwargs['crs'].axis_info[0].unit_name

        if 'foot' in parent_grid_units.lower() or 'feet' in parent_grid_units.lower():
            parent_grid_units = 'feet'
        elif 'metre' in parent_grid_units.lower() or 'meter' in parent_grid_units.lower():
            parent_grid_units = 'meters'
        else:
            raise ValueError(f'unrecognized CRS units {parent_grid_units}: CRS must be projected in feet or meters')

        # assume that model grid is in a projected CRS of meters
        lmult = convert_length_units(parent_units, parent_grid_units)
        kwargs['delr'] = self.parent.dis.delr.array * lmult
        kwargs['delc'] = self.parent.dis.delc.array * lmult
        kwargs['top'] = self.parent.dis.top.array
        kwargs['botm'] = self.parent.dis.botm.array
        if hasattr(self.parent.dis, 'laycbd'):
            kwargs['laycbd'] = self.parent.dis.laycbd.array
        # renames for parent modelgrid
        renames = {'rotation': 'angrot'}
        for k, v in renames.items():
            if k in kwargs:
                kwargs[v] = kwargs.pop(k)

        kwargs = get_input_arguments(kwargs, MFsetupGrid, warn=False)
        self._parent._mg_resync = False
        self._parent._modelgrid = MFsetupGrid(**kwargs)

    def _set_parent(self):
        """Set attributes related to a parent or source model
        if one is specified.
        """

        # if it's an LGR model (where parent is also being created)
        # set up the parent DIS package
        if self._is_lgr and isinstance(self.parent, MFsetupMixin):
            if 'DIS' not in self.parent.get_package_list():
                dis = self.parent.setup_dis()

        kwargs = self.cfg['parent'].copy()
        if kwargs is not None:
            kwargs = kwargs.copy()

            # load MF6 or MF2005 parent
            if self.parent is None:
                print('loading parent model {}...'.format(os.path.join(kwargs['model_ws'],
                                                                 kwargs['namefile'])))
                t0 = time.time()

                # load only specified packages that the parent model has
                packages_in_parent_namefile = get_packages(os.path.join(kwargs['model_ws'],
                                                                        kwargs['namefile']))
                # load at least these packages
                # so that there is complete information on model time and space dis
                default_parent_packages = {'dis', 'tdis'}
                specified_packages = set(self.cfg['model'].get('packages', set()))
                specified_packages.update(default_parent_packages)

                # get equivalent packages to load if parent is another MODFLOW version;
                # then flatten (a package may have more than one equivalent)
                parent_packages = [get_package_name(p, kwargs['version'])
                                   for p in specified_packages]
                parent_packages = {item for subset in parent_packages for item in subset}
                if kwargs['version'] == 'mf6':
                    parent_packages.add('sto')
                load_only = list(set(packages_in_parent_namefile).intersection(parent_packages))
                if 'load_only' not in kwargs:
                    kwargs['load_only'] = load_only
                if 'skip_load' in kwargs:
                    kwargs['skip_load'] = [s.lower() for s in kwargs['skip_load']]
                    kwargs['load_only'] = [pckg for pckg in kwargs['load_only']
                                           if pckg not in kwargs['skip_load']]

                if self.cfg['parent']['version'] == 'mf6':
                    sim_kwargs = kwargs.copy()
                    if 'sim_name' not in kwargs:
                        sim_kwargs['sim_name'] = kwargs.get('simulation', 'mfsim')
                    if 'sim_ws' not in kwargs:
                        sim_kwargs['sim_ws'] = sim_kwargs.get('model_ws', '.')
                    sim_kwargs = get_input_arguments(sim_kwargs, mf6.MFSimulation.load, warn=False)
                    parent_sim = mf6.MFSimulation.load(**sim_kwargs)
                    modelname, _ = os.path.splitext(kwargs['namefile'])
                    self._parent = parent_sim.get_model(modelname)
                else:
                    kwargs['f'] = kwargs.pop('namefile')
                    kwargs = get_input_arguments(kwargs, fm.Modflow.load, warn=False)
                    self._parent = fm.Modflow.load(**kwargs)
                print("finished in {:.2f}s\n".format(time.time() - t0))

            # set parent model units in config if not entered
            if 'length_units' not in self.cfg['parent']:
                self.cfg['parent']['length_units'] = get_model_length_units(self.parent)
            if 'time_units' not in self.cfg['parent']:
                self.cfg['parent']['time_units'] = get_model_time_units(self.parent)

            # set the parent model grid from mg_kwargs if not None
            # otherwise, convert parent model grid to MFsetupGrid
            mg_kwargs = self.cfg['parent'].get('SpatialReference',
                                          self.cfg['parent'].get('modelgrid', None))
            # check configuration file input
            # for consistency with parent model DIS package input
            # (configuration file input may be different if an existing model
            # doesn't have a valid spatial reference in the DIS package)
            mf6_names = {
                'rotation': 'angrot',
                'xoff': 'xorigin',
                'yoff': 'yorigin'
            }
            if mg_kwargs is not None and (self.parent.version == 'mf6') and not\
                            mg_kwargs.get('override_dis_package_input', False):
                for variable, mf6_name in mf6_names.items():
                    if (variable in mg_kwargs) and\
                        ('DIS' in self.parent.get_package_list()):
                        dis_value = getattr(self.parent.dis, mf6_name).array
                        if not np.allclose(mg_kwargs[variable], dis_value):
                            raise ValueError(
                "Configuration file entry parent: SpatialReference: "
                f"{variable}: {mg_kwargs[variable]} does not match {mf6_name}={dis_value} "
                "specified in the parent model DIS package file. Either make "
                "these consistent or specify override_dis_package_input: True "
                "in the parent: SpatialReference: configuration block.")
            self._set_parent_modelgrid(mg_kwargs)

            # setup parent model perioddata table
            if getattr(self.parent, 'perioddata', None) is None:
                kwargs = self.cfg['parent'].copy()
                kwargs['model_time_units'] = self.cfg['parent']['time_units']
                if self.parent.version == 'mf6':
                    for var in ['perlen', 'nstp', 'tsmult']:
                        kwargs[var] = getattr(self.parent.modeltime, var)
                    kwargs['steady'] = self.parent.modeltime.steady_state
                    kwargs['nper'] = self.parent.simulation.tdis.nper.array
                else:
                    for var in ['perlen', 'steady', 'nstp', 'tsmult']:
                        kwargs[var] = self.parent.dis.__dict__[var].array
                    kwargs['nper'] = self.parent.dis.nper
                kwargs = get_input_arguments(kwargs, setup_perioddata_group)
                kwargs['oc_saverecord'] = {}
                if hasattr(self.parent, '_perioddata'):
                    self._parent._perioddata = setup_perioddata_group(**kwargs)
                else:
                    self._parent.perioddata = setup_perioddata_group(**kwargs)

            # default_source_data, where omitted configuration input is
            # obtained from parent model by default
            # Set default_source_data to True by default if it isn't specified
            if self.cfg['parent'].get('default_source_data') is None:
                self.cfg['parent']['default_source_data'] = True
            if self.cfg['parent'].get('default_source_data'):
                self._parent_default_source_data = True

                # set number of layers from parent if not specified
                if self.version == 'mf6' and self.cfg['dis']['dimensions'].get('nlay') is None:
                    self.cfg['dis']['dimensions']['nlay'] = getattr(self.parent.dis.nlay, 'array',
                                                                    self.parent.dis.nlay)
                elif self.cfg['dis'].get('nlay') is None:
                    self.cfg['dis']['nlay'] = getattr(self.parent.dis.nlay, 'array',
                                                      self.parent.dis.nlay)

                # set start date/time from parent if not specified
                if not self._is_lgr:
                    parent_start_date_time = self.cfg.get('parent', {}).get('start_date_time')
                    if self.version == 'mf6':
                        if self.cfg['tdis']['options'].get('start_date_time', '1970-01-01') == '1970-01-01' \
                                and parent_start_date_time is not None:
                            self.cfg['tdis']['options']['start_date_time'] = self.cfg['parent']['start_date_time']
                    else:
                        if self.cfg['dis'].get('start_date_time', '1970-01-01') == '1970-01-01' \
                                and parent_start_date_time is not None:
                            self.cfg['dis']['start_date_time'] = self.cfg['parent']['start_date_time']

                    # only get time dis information from parent if
                    # no periodata groups are specified, and nper is not specified under dimensions
                    tdis_package = 'tdis' if self.version == 'mf6' else 'dis'
                    # check if any item within perioddata block is a dictionary
                    # (groups are subblocks within perioddata block)
                    has_perioddata_groups = any([isinstance(k, dict)
                                                 for k in self.cfg[tdis_package]['perioddata'].values()])
                    # get the number of inset model periods
                    if not has_perioddata_groups:
                        if self.version == 'mf6':
                            if self.cfg['tdis']['dimensions'].get('nper') is None:
                                self.cfg['tdis']['dimensions']['nper'] = self.parent.modeltime.nper
                            nper = self.cfg['tdis']['dimensions']['nper']
                        else:
                            if self.cfg['dis']['nper'] is None:
                                self.cfg['dis']['nper'] = self.dis.nper
                            nper = self.cfg['dis']['nper']
                        # get the periods that are shared with the parent model
                        parent_periods = get_parent_stress_periods(self.parent, nper=nper,
                                                                   parent_stress_periods=self.cfg['parent'][
                                                                       'copy_stress_periods'])
                        # get time discretization info. from the parent model
                        if self.version == 'mf6':
                            for var in ['perlen', 'nstp', 'tsmult']:
                                if self.cfg['tdis']['perioddata'].get(var) is None:
                                    self.cfg['tdis']['perioddata'][var] = getattr(self.parent.modeltime, var)[
                                        parent_periods]
                                # 'steady' can be specified under sto package (as in MODFLOW-6)
                                # or within perioddata group blocks
                                # but not in the tdis perioddata block itset
                                if self.cfg['sto'].get('steady') is None:
                                    self.cfg['sto']['steady'] = self.parent.modeltime.steady_state[parent_periods]
                        else:
                            for var in ['perlen', 'nstp', 'tsmult', 'steady']:
                                if self.cfg['dis'].get(var) is None:
                                    self.cfg['dis'][var] = self.parent.dis.__dict__[var].array[parent_periods]

    def _setup_array(self, package, var, vmin=-1e30, vmax=1e30,
                      source_model=None, source_package=None,
                      **kwargs):
        return setup_array(self, package, var, vmin=vmin, vmax=vmax,
                           source_model=source_model, source_package=source_package,
                           **kwargs)

    def _setup_basic_stress_package(self, package, flopy_package_class,
                                    variable_columns, rivdata=None,
                                    **kwargs):
        print(f'\nSetting up {package.upper()} package...')
        t0 = time.time()

        # possible future support to
        # handle filenames of multiple packages
        # leave this out for now because of additional complexity
        # from multiple sets of external files
        #existing_packages = getattr(self, package, None)
        #filename = f"{self.name}.{package}"
        #if existing_packages is not None:
        #    try:
        #        len(existing_packages)
        #        suffix = len(existing_packages) + 1
        #    except:
        #        suffix = 1
        #    filename = f"{self.name}-{suffix}.{package}"

        # perimeter boundary (CHD or WEL)
        dfs = []
        if 'perimeter_boundary' in kwargs:
            perimeter_cfg = kwargs['perimeter_boundary']
            if package == 'chd':
                perimeter_cfg['boundary_type'] = 'head'
                boundname = 'perimeter-heads'
            elif package == 'wel':
                perimeter_cfg['boundary_type'] = 'flux'
                boundname = 'perimeter-fluxes'
            else:
                raise ValueError(f'Unsupported package for perimeter_boundary: {package.upper()}')
            if 'inset_parent_period_mapping' not in perimeter_cfg:
                perimeter_cfg['inset_parent_period_mapping'] = self.parent_stress_periods
            if 'parent_start_time' not in perimeter_cfg:
                perimeter_cfg['parent_start_date_time'] = self.parent.perioddata['start_datetime'][0]
            self.tmr = Tmr(self.parent, self, **perimeter_cfg)
            df = self.tmr.get_inset_boundary_values()

            # add boundname to allow boundary flux to be tracked as observation
            df['boundname'] = boundname
            dfs.append(df)

        # RIV package converted from SFR input
        elif rivdata is not None:
            if 'name' in rivdata.stress_period_data.columns:
                rivdata.stress_period_data['boundname'] = rivdata.stress_period_data['name']
            dfs.append(rivdata.stress_period_data)

        # set up package from user input
        df_sd = None
        if 'source_data' in kwargs:
            if package == 'wel':
                dropped_wells_file =\
                    kwargs.get('output_files', {})\
                    .get('dropped_wells_file', '{}_dropped_wells.csv').format(self.name)
                df_sd = setup_wel_data(self,
                                       source_data=kwargs['source_data'],
                                       dropped_wells_file=dropped_wells_file)
            else:
                df_sd = setup_basic_stress_data(self, **kwargs['source_data'], **kwargs.get('mfsetup_options', dict()))
            if df_sd is not None and len(df_sd) > 0:
                dfs.append(df_sd)
        # set up package from parent model
        elif self.cfg['parent'].get('default_source_data') and\
            hasattr(self.parent, package):
            if package == 'wel':
                dropped_wells_file =\
                    kwargs['output_files']['dropped_wells_file'].format(self.name)
                df_sd = setup_wel_data(self,
                                       dropped_wells_file=dropped_wells_file)
            else:
                print(f'Skipping setup of {package.upper()} Package from parent model-- not implemented.')
            if df_sd is not None and len(df_sd) > 0:
                dfs.append(df_sd)
        if len(dfs) == 0:
            print(f"{package.upper()} package:\n"
                  "No input specified or package configuration file input "
                  "not understood. See the Configuration "
                  "File Gallery in the online docs for example input "
                  "Note that direct input to basic stress period packages "
                  "is currently not supported.")
            return
        else:
            df = pd.concat(dfs, axis=0)

        # option to write stress_period_data to external files
        if self.version == 'mf6':
            external_files = self.cfg[package]['mfsetup_options'].get('external_files', True)
        else:
            # external list or tabular type files not supported for MODFLOW-NWT
            # adding support for this may require changes to Flopy
            external_files = False
        external_filename_fmt = self.cfg[package]['mfsetup_options']['external_filename_fmt']
        spd = setup_flopy_stress_period_data(self, package, df,
                                                 flopy_package_class=flopy_package_class,
                                                 variable_columns=variable_columns,
                                                 external_files=external_files,
                                                 external_filename_fmt=external_filename_fmt)

        kwargs = self.cfg[package]
        if isinstance(self.cfg[package]['options'], dict):
            kwargs.update(self.cfg[package]['options'])
        #kwargs['filename'] = filename
        # add observation for perimeter BCs
        # and any user input with a boundname col
        obslist = []
        obsfile = f'{self.name}.{package}.obs.output.csv'
        if 'perimeter_boundary' in kwargs:
            perimeter_btype = f"perimeter-{perimeter_cfg['boundary_type']}"
            obslist.append((perimeter_btype, package, perimeter_btype))
        if 'boundname' in df.columns:
            unique_boundnames = df['boundname'].unique()
            for bname in unique_boundnames:
                obslist.append((bname, package, bname))
        if len(obslist) > 0:
            kwargs['observations'] = {obsfile: obslist}
        kwargs = get_input_arguments(kwargs, flopy_package_class)
        if not external_files:
            kwargs['stress_period_data'] = spd
        pckg = flopy_package_class(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return pckg

    def setup_grid(self):
        """Set up the attached modelgrid instance from configuration input
        """
        if self.cfg['grid']:
            cfg = self.cfg['grid']
            cfg['rotation'] = self.cfg['grid']['angrot']
        else:
            cfg = self.cfg['setup_grid'] #.copy()
        # update grid configuration with any information supplied to dis package
        # (so that settings specified for DIS package have priority)
        self._update_grid_configuration_with_dis()
        if not cfg['structured']:
            raise NotImplementedError('Support for unstructured grids')
        features_shapefile = cfg.get('source_data', {}).get('features_shapefile')
        if features_shapefile is not None and 'features_shapefile' not in cfg:
            features_shapefile['features_shapefile'] = features_shapefile['filename']
            del features_shapefile['filename']
            cfg.update(features_shapefile)
        cfg['parent_model'] = self.parent
        cfg['model_length_units'] = self.length_units
        output_files = self.cfg['setup_grid']['output_files']
        cfg['grid_file'] = output_files['grid_file'].format(self.name)
        bbox_shapefile_name = Path(output_files['bbox_shapefile'].format(self.name)).name
        cfg['bbox_shapefile'] = Path(self._shapefiles_path) / bbox_shapefile_name
        if 'DIS' in self.get_package_list():
            cfg['top'] = self.dis.top.array
            cfg['botm'] = self.dis.botm.array

        # if model is an LGR inset with the default rotation=0
        # and the LGR parent is rotated
        # assume that the inset model rotation should == parent
        # (different LGR parent/inset rotations not allowed)
        if self._is_lgr and (cfg['rotation'] == 0) and\
            self.parent.modelgrid.angrot != 0:
                cfg['rotation'] = self.parent.modelgrid.angrot

        if os.path.exists(cfg['grid_file']) and self._load:
            print('Loading model grid definition from {}'.format(cfg['grid_file']))
            cfg.update(load(cfg['grid_file']))
            self.cfg['grid'] = cfg
            kwargs = get_input_arguments(self.cfg['grid'], MFsetupGrid)
            self._modelgrid = MFsetupGrid(**kwargs)
            self._modelgrid.cfg = self.cfg['grid']
        else:
            kwargs = get_input_arguments(cfg, setup_structured_grid)
            if not set(kwargs.keys()).intersection({
                'features_shapefile', 'features', 'xoff', 'yoff', 'xul', 'yul'}):
                raise ValueError(
                    "No features_shapefile or xoff, yoff supplied "
                    "to setup_grid: block. Check configuration file input, "
                    "including for accidental indentation of the setup_grid: block.")
            self._modelgrid = setup_structured_grid(**kwargs)
            self.cfg['grid'] = self._modelgrid.cfg
            # update DIS package configuration
            if self.version == 'mf6':
                self.cfg['dis']['dimensions']['nrow'] = self.cfg['grid']['nrow']
                self.cfg['dis']['dimensions']['ncol'] = self.cfg['grid']['ncol']
            else:
                self.cfg['dis']['nrow'] = self.cfg['grid']['nrow']
                self.cfg['dis']['ncol'] = self.cfg['grid']['ncol']

        self._reset_bc_arrays()

        # set up local grid refinement
        if 'lgr' in self.cfg['setup_grid'].keys():
            if self.version != 'mf6':
                raise TypeError('LGR only supported for MODFLOW-6 models.')
            if not self.lgr:
                self.lgr = True
            for key, cfg in self.cfg['setup_grid']['lgr'].items():
                existing_inset_models = set()
                if isinstance(self.inset, dict):
                    existing_inset_models = {k for k, v in self.inset.items()}
                if key not in existing_inset_models:
                    self.create_lgr_models()

    def load_grid(self, gridfile=None):
        """Load model grid information from a json or yml file."""
        if gridfile is None:
            if os.path.exists(self.cfg['setup_grid']['grid_file']):
                gridfile = self.cfg['setup_grid']['grid_file']
        print('Loading model grid information from {}'.format(gridfile))
        self.cfg['grid'] = load(gridfile)

    def setup_sfr(self, **kwargs):
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
                bbox_filter = project(self.bbox, self.modelgrid.crs, 'epsg:4269').bounds
                lines = Lines.from_nhdplus_v2(NHDPlus_paths=nhdplus_paths,
                                            bbox_filter=bbox_filter)
            else:
                for key in ['filename', 'filenames']:
                    if key in flowlines:
                        kwargs = flowlines.copy()
                        kwargs['shapefile'] = kwargs.pop(key)
                        check_source_files(kwargs['shapefile'])
                        if 'epsg' not in kwargs:
                            try:
                                from gisutils import get_shapefile_crs
                                shapefile_crs = get_shapefile_crs(kwargs['shapefile'])
                            except Exception as e:
                                print(e)
                                msg = ('Need gis-utils >= 0.2 to get crs'
                                       ' for shapefile: {}\nPlease pip install '
                                       '--upgrade gis-utils'.format(kwargs['shapefile']))
                                print(msg)
                        else:
                            shapefile_crs = pyproj.crs.CRS.from_epsg(kwargs['epsg'])
                        authority = shapefile_crs.to_authority()
                        if authority is not None:
                            shapefile_crs = pyproj.CRS.from_user_input(shapefile_crs.to_authority())

                        bbox_filter = self.bbox.bounds
                        if shapefile_crs != self.modelgrid.crs:
                            bbox_filter = project(self.bbox, self.modelgrid.crs, shapefile_crs).bounds
                        kwargs['bbox_filter'] = bbox_filter
                        # create an sfrmaker.lines instance
                        kwargs = get_input_arguments(kwargs, Lines.from_shapefile)
                        lines = Lines.from_shapefile(**kwargs)
                        break
        else:
            return

        # output
        output_path = self.cfg['sfr'].get('output_path')
        if output_path is not None:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        else:
            output_path = self.cfg['postprocessing']['output_folders']['shapefiles']
            self.cfg['sfr']['output_path'] = output_path

        # create isfr array (where SFR cells will be populated)
        if self.version == 'mf6':
            active_cells = np.sum(self.idomain >= 1, axis=0) > 0
            # For models with LGR, set the LGR area to isfr=0
            # to prevent SFR from being generated within the LGR area
            # needed for LGR models that only have refinement
            # in some layers (in other words, active parent model cells
            # below the LGR inset)
            if self.lgr:
                active_cells[self._lgr_idomain2d == 0] = 0
        else:
            active_cells = np.sum(self.ibound >= 1, axis=0) > 0
            #active_cells = self.ibound.sum(axis=0) > 0
        # only include active cells that don't have another boundary condition
        # (besides the wel package)
        isfr = active_cells & (self._isbc2d <= 0)

        #  kludge to get sfrmaker to work with modelgrid
        self.modelgrid.model_length_units = self.length_units

        # create an sfrmaker.sfrdata instance from the lines instance
        to_sfr_kwargs = self.cfg['sfr'].copy()
        if not self.cfg['sfr'].get('sfrmaker_options'):
            self.cfg['sfr']['sfrmaker_options'] = {}
        to_sfr_kwargs.update(self.cfg['sfr']['sfrmaker_options'])
        #to_sfr_kwargs = get_input_arguments(to_sfr_kwargs, Lines.to_sfr)
        sfr = lines.to_sfr(grid=self.modelgrid,
                         isfr=isfr,
                         model=self,
                         **to_sfr_kwargs)
        if self.cfg['sfr'].get('set_streambed_top_elevations_from_dem'):
            warnings.warn('sfr: set_streambed_top_elevations_from_dem option is now under sfr: sfrmaker_options',
                          DeprecationWarning)
            self.cfg['sfr']['sfrmaker_options']['set_streambed_top_elevations_from_dem'] = True
        if self.cfg['sfr']['sfrmaker_options'].get('set_streambed_top_elevations_from_dem'):
            dem_kwargs = self.cfg['sfr']['sfrmaker_options'].get('set_streambed_top_elevations_from_dem')
            if not isinstance(dem_kwargs, dict):
                dem_kwargs = {}
                error_msg = (
                    "If set_streambed_top_elevations_from_dem=True, "
                    "need a dem block in source_data for SFR package. "
                    "Otherwise set_streambed_top_elevations_from_dem should be"
                    "a block with arguments to "
                    "sfrmaker.SFRData.set_streambed_top_elevations_from_dem")
                assert 'dem' in self.cfg['sfr'].get('source_data', {}), error_msg
                dem_kwargs.update(self.cfg['sfr']['source_data']['dem'])
            sfr.set_streambed_top_elevations_from_dem(**dem_kwargs)
        else:
            sfr.reach_data['strtop'] = sfr.interpolate_to_reaches('elevup', 'elevdn')

        # assign layers to the sfr reaches
        botm = self.dis.botm.array.copy()
        if self.version == 'mf6':
            idomain = self.dis.idomain.array
        else:
            idomain = self.bas6.ibound.array
        layers, new_botm = assign_layers(sfr.reach_data,
                                         botm_array=botm,
                                         idomain=idomain)
        sfr.reach_data['k'] = layers
        if new_botm is not None:
            # run thru setup_array so that DIS input remains open/close
            self._setup_array('dis', 'botm',
                              data={i: arr for i, arr in enumerate(new_botm)},
                              datatype='array3d', write_fmt='%.2f', dtype=int)
            # reset the bottom array in flopy (and in memory)
            # is this necessary? =
            self.dis.botm = new_botm
            # set bottom array to external files
            if self.version == 'mf6':
                self.dis.botm = self.cfg['dis']['griddata']['botm']
            else:
                self.dis.botm = self.cfg['dis']['botm']
            print('\nModel cell bottom elevations adjusted after assigning '
                  'SFR reaches to layers\n(to accommodate SFR reach bottoms '
                  'below the previous model bottom)\n')

        # option to convert reaches to the River Package
        if self.cfg['sfr'].get('to_riv'):
            warnings.warn('sfr: to_riv option is now under sfr: sfrmaker_options',
                          DeprecationWarning)
            self.cfg['sfr']['sfrmaker_options']['to_riv'] = self.cfg['sfr'].get('to_riv')
        if self.cfg['sfr'].get('sfrmaker_options', {}).get('to_riv'):
            rivdata = sfr.to_riv(line_ids=self.cfg['sfr']['sfrmaker_options']['to_riv'],
                                 drop_in_sfr=True)
            # setup of RIV package from SFRmaker-derived RIVdata
            # and any user input
            # do this instead of 2 seperate packages
            # to avoid having two sets of external files
            self.setup_riv(rivdata, **self.cfg['riv'], **self.cfg['riv']['mfsetup_options'])
            rivdata_filename = self.cfg['riv']['output_files']['rivdata_file'].format(self.name)
            rivdata.write_table(os.path.join(self._tables_path, rivdata_filename))
            rivdata.write_shapefiles('{}/{}'.format(self._shapefiles_path, self.name))

        # optional routing input
        # (for a complete representation of a larger or more detailed
        #  stream network that may be culled in SFR package)
        sd = self.cfg['sfr'].get('source_data', {})
        routing_input_key = [k for k in sd.keys() if 'routing' in k]
        routing_input = None
        if len(routing_input_key) > 0:
            routing_input = sd.get(routing_input_key[0])
            routing = pd.read_csv(routing_input['filename'])
            routing = dict(zip(routing[routing_input['id_column']],
                               routing[routing_input['routing_column']]))
            # set any values (downstream lines) not in keys (upstream lines)
            # to 0 (outlet condition)
            routing = {k: v if v in routing.keys() else 0
                       for k, v in routing.items()}
        # use _original_routing attached to Lines instance as default
        else:
            routing = lines._original_routing

        # add inflows
        inflows_input = self.cfg['sfr'].get('source_data', {}).get('inflows')
        if inflows_input is not None:
            # resample inflows to model stress periods
            inflows_input['id_column'] = inflows_input['line_id_column']
            sd = TransientTabularSourceData.from_config(inflows_input,
                                                        dest_model=self)
            inflows_by_stress_period = sd.get_data()

            missing_sites = set(inflows_by_stress_period[inflows_input['id_column']]). \
                                difference(routing.keys())
            if any(missing_sites):
                # cast IDs to strings for compatibility with SFRmaker > 0.11.3
                # for now, assume IDs are numeric; future updates to SFRmaker
                # may eventually allow for alpha numeric IDs
                inflows_by_stress_period[inflows_input['id_column']] =\
                    inflows_by_stress_period[inflows_input['id_column']].astype(int).astype(str)

            # check if all inflow sites are included in sfr network
            missing_sites = set(inflows_by_stress_period[inflows_input['id_column']]). \
                                difference(routing.keys())
            # if there are missing sites, try using the supplied routing
            if any(missing_sites):
                raise KeyError(('inflow sites {} are not within the model sfr network. '
                                'Please supply an inflows_routing source_data block '
                                '(see shellmound example config file)'.format(missing_sites)))

            # add resampled inflows to SFR package
            inflows_input['data'] = inflows_by_stress_period
            inflows_input['flowline_routing'] = routing
            if self.version == 'mf6':
                inflows_input['variable'] = 'inflow'
                method = sfr.add_to_perioddata
            else:
                inflows_input['variable'] = 'flow'
                method = sfr.add_to_segment_data
            kwargs = get_input_arguments(inflows_input.copy(), method)
            method(**kwargs)

        # add runoff
        runoff_input = self.cfg['sfr'].get('source_data', {}).get('runoff')
        if runoff_input is not None:
            # resample inflows to model stress periods
            runoff_input['id_column'] = runoff_input['line_id_column']
            sd = TransientTabularSourceData.from_config(runoff_input,
                                                        dest_model=self)
            runoff_by_stress_period = sd.get_data()

            # check if all sites are included in sfr network
            missing_sites = set(runoff_by_stress_period[runoff_input['id_column']]). \
                                difference(routing.keys())
            if any(missing_sites):
                warnings.warn(('runoff sites {} are not within the model sfr network. '
                               'Please supply an inflows_routing source_data block '
                               '(see shellmound example config file)'.format(missing_sites)),
                               UserWarning)

            # add resampled inflows to SFR package
            runoff_input['data'] = runoff_by_stress_period
            runoff_input['flowline_routing'] = routing
            runoff_input['variable'] = 'runoff'
            runoff_input['distribute_flows_to_reaches'] = True
            if self.version == 'mf6':
                method = sfr.add_to_perioddata
            else:
                method = sfr.add_to_segment_data
            kwargs = get_input_arguments(runoff_input.copy(), method)
            method(**kwargs)

        # add observations
        observations_input = self.cfg['sfr'].get('source_data', {}).get('observations')
        if self.version != 'mf6':
            sfr.gage_starting_unit_number = self.cfg['gag']['starting_unit_number']
        if observations_input is not None:
            key = 'filename' if 'filename' in observations_input else 'filenames'
            observations_input['data'] = observations_input[key]
            kwargs = get_input_arguments(observations_input.copy(), sfr.add_observations)
            obsdata = sfr.add_observations(**kwargs)
            # resample observations to model stress periods; write to table

        # write reach and segment data tables
        sfr.write_tables('{}/{}'.format(self._tables_path, self.name))

        # export shapefiles of lines, routing, cell polygons, inlets and outlets
        sfr.write_shapefiles('{}/{}'.format(self._shapefiles_path, self.name))

        # create the flopy SFR package instance
        sfr.create_modflow_sfr2(model=self, istcb2=223)
        if self.version != 'mf6':
            sfr_package = sfr.modflow_sfr2
        else:
            # pass options kwargs through to mf6 constructor
            kwargs = flatten({k:v for k, v in self.cfg[package].items() if k not in
                              {'source_data', 'flowlines', 'inflows', 'observations',
                               'inflows_routing', 'dem', 'sfrmaker_options'}})
            kwargs = get_input_arguments(kwargs, mf6.ModflowGwfsfr)
            sfr_package = sfr.create_mf6sfr(model=self, **kwargs)
            # monkey patch ModflowGwfsfr instance to behave like ModflowSfr2
            sfr_package.reach_data = sfr.modflow_sfr2.reach_data

        # attach the sfrmaker.sfrdata instance as an attribute
        self.sfrdata = sfr

        # reset dependent arrays
        self._reset_bc_arrays()
        if self.version == 'mf6':
            self._set_idomain()
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return sfr_package

    def setup_solver(self):
        if self.version == 'mf6':
            solver_package = 'ims'
        else:
            solver_package = 'nwt'
        assert solver_package not in self.package_list
        setup_method_name = 'setup_{}'.format(solver_package)
        package_setup = getattr(self, setup_method_name, None)
        package_setup()

    def setup_packages(self, reset_existing=True):
        package_list = self.package_list #['sfr'] #m.package_list # ['tdis', 'dis', 'npf', 'oc']
        if not reset_existing:
            package_list = [p for p in package_list if p.upper() not in self.get_package_list()]
        for pkg in package_list:
            setup_method_name = f'setup_{pkg}'
            package_setup = getattr(self, setup_method_name, None)
            if package_setup is None:
                print('{} package not supported for MODFLOW version={}'.format(pkg.upper(), self.version))
                continue
            if not callable(package_setup):
                package_setup = getattr(MFsetupMixin, 'setup_{}'.format(pkg.strip('6')))
            # avoid multiple package instances for now, except for obs
            if self.version != 'mf6' or pkg == 'obs' or not hasattr(self, pkg):
                package_setup(**self.cfg[pkg], **self.cfg[pkg]['mfsetup_options'])


    @classmethod
    def load_cfg(cls, yamlfile, verbose=False):
        """Loads a configuration file, with default settings
        specific to the MFnwtModel or MF6model class.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with pfl_nwt setup information.
        verbose : bool

        Returns
        -------
        cfg : dict (configuration dictionary)
        """
        return load_cfg(yamlfile, verbose=verbose, default_file=cls.default_file)

    @classmethod
    def setup_from_yaml(cls, yamlfile, verbose=False):
        """Make a model from scratch, using information in a yamlfile.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with pfl_nwt setup information.
        verbose : bool

        Returns
        -------
        m : model instance
        """
        cfg = cls.load_cfg(yamlfile, verbose=verbose)
        return cls.setup_from_cfg(cfg, verbose=verbose)

    @classmethod
    def setup_from_cfg(cls, cfg, verbose=False):
        """Make a model from scratch, using information in a configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary, as produced by the model.load_cfg method.
        verbose : bool

        Returns
        -------
        m : model instance
        """
        cfg_filename = Path(cfg.get('filename', '')).name
        msg = f"\nSetting up {cfg['model']['modelname']} model"
        if len(cfg_filename) > 0:
            msg += f" from configuration in {cfg_filename}"
        print(msg)
        t0 = time.time()

        m = cls(cfg=cfg) #, **kwargs)

        # make a grid if one isn't already specified
        if 'grid' not in m.cfg.keys():
            m.setup_grid()

        # establish time discretization, including TDIS setup for MODFLOW-6
        m.setup_tdis()

        # set up the solver
        m.setup_solver()

        # set up all of the packages specified in the config file
        m.setup_packages(reset_existing=False)

        # LGR inset model(s)
        if m.inset is not None:
            for k, v in m.inset.items():
                if v._is_lgr:
                    v.setup_packages()
            m.setup_lgr_exchanges()

        print('finished setting up model in {:.2f}s'.format(time.time() - t0))
        print('\n{}'.format(m))
        return m
