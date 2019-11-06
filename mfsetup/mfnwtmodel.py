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
from .discretization import fix_model_layer_conflicts, verify_minimum_layer_thickness, fill_empty_layers
from .tdis import setup_perioddata_group
from .grid import MFsetupGrid, get_ij, write_bbox_shapefile
from .fileio import load, dump, load_array, save_array, check_source_files, flopy_mf2005_load, \
    load_cfg, setup_external_filepaths
from .utils import update, get_input_arguments
from .interpolate import interp_weights, interpolate, regrid, get_source_dest_model_xys
from .lakes import make_lakarr2d, make_bdlknc_zones, make_bdlknc2d
from .utils import update, get_packages
from .obs import read_observation_data
from .sourcedata import ArraySourceData, MFArrayData, TabularSourceData, setup_array
from .units import convert_length_units, convert_time_units, convert_flux_units, lenuni_text, itmuni_text
from .wells import setup_wel_data
from .mfmodel import MFsetupMixin


class MFnwtModel(MFsetupMixin, Modflow):
    """Class representing a MODFLOW-NWT model"""

    source_path = os.path.split(__file__)[0]

    def __init__(self, parent=None, cfg=None,
                 modelname='model', exe_name='mfnwt',
                 version='mfnwt', model_ws='.',
                 external_path='external/', **kwargs):

        Modflow.__init__(self, modelname, exe_name=exe_name, version=version,
                         model_ws=model_ws, external_path=external_path,
                         **kwargs)
        MFsetupMixin.__init__(self, parent=parent)

        # default configuration
        self._package_setup_order = ['dis', 'bas6', 'upw', 'rch', 'oc',
                                     'ghb', 'lak', 'sfr',
                                     'wel', 'mnw2', 'gag', 'hyd', 'nwt']
        self.cfg = load(self.source_path + '/mfnwt_defaults.yml')
        self.cfg['filename'] = self.source_path + '/mfnwt_defaults.yml'
        self._load_cfg(cfg)  # update configuration dict with values in cfg
        self.relative_external_paths = self.cfg['model'].get('relative_external_paths', True)
        self.model_ws = self._get_model_ws()

        # property arrays
        self._ibound = None

    def __repr__(self):
        return MFsetupMixin.__repr__(self)

    @property
    def nlay(self):
        return self.cfg['dis'].get('nlay', 1)

    @property
    def length_units(self):
        return lenuni_text[self.cfg['dis']['lenuni']]

    @property
    def time_units(self):
        return itmuni_text[self.cfg['dis']['itmuni']]

    @property
    def ipakcb(self):
        """By default write everything to one cell budget file."""
        return self.cfg['upw'].get('ipakcb', 53)

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
        elif cfg is None:
            return
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

            print('loading parent model {}...'.format(os.path.join(kwargs['model_ws'],
                                                                   kwargs['f'])))
            t0 = time.time()
            self._parent = fm.Modflow.load(**kwargs)
            print("finished in {:.2f}s\n".format(time.time() - t0))

            mg_kwargs = self.cfg['parent'].get('SpatialReference',
                                               self.cfg['parent'].get('modelgrid', None))
        # set the parent model grid from mg_kwargs if not None
        # otherwise, convert parent model grid to MFsetupGrid
        self._set_parent_modelgrid(mg_kwargs)

        # make sure that the output paths exist
        #output_paths = [self.cfg['intermediate_data']['output_folder'],
        #                self.cfg['model']['model_ws'],
        #                os.path.join(self.cfg['model']['model_ws'], self.cfg['model']['external_path'])
        #                ]
        output_paths = list(self.cfg['postprocessing']['output_folders'].values())
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

        perioddata = setup_perioddata_group(self.cfg['model']['start_date_time'],
                                            self.cfg['model'].get('end_date_time'),
                                            nper=nper,
                                            perlen=self.cfg['dis']['perlen'],
                                            model_time_units=self.time_units,
                                            freq=self.cfg['dis'].get('freq'),
                                            steady=steady,
                                            nstp=self.cfg['dis']['nstp'],
                                            tsmult=self.cfg['dis']['tsmult'],
                                            oc_saverecord=self.cfg['oc']['period_options'],
                                            )
        perioddata['parent_sp'] = parent_sp
        assert np.array_equal(perioddata['per'].values, np.arange(len(perioddata)))
        self._perioddata = perioddata
        self._nper = None

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

        id_column = id_column.lower()

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
                                            filter=self.parent.modelgrid.bbox.bounds)
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

        # make the strt array
        self._setup_array(package, 'strt', by_layer=True)
        
        # make the ibound array
        self._setup_array(package, 'ibound', by_layer=True, write_fmt='%d')

        kwargs = get_input_arguments(self.cfg['bas6'], fm.ModflowBas)
        bas = fm.ModflowBas(model=self, **kwargs)
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

        package = 'oc'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        stress_period_data = {}
        for i, r in self.perioddata.iterrows():
            stress_period_data[(r.per, r.nstp -1)] = r.oc

        kwargs = self.cfg['oc']
        kwargs['stress_period_data'] = stress_period_data
        kwargs = get_input_arguments(kwargs, fm.ModflowOc)
        oc = fm.ModflowOc(model=self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return oc

    def setup_rch(self):
        package = 'rch'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the rech array
        self._setup_array(package, 'rech')

        # create flopy package instance
        kwargs = self.cfg['rch']
        kwargs['ipakcb'] = self.ipakcb
        kwargs = get_input_arguments(kwargs, fm.ModflowRch)
        rch = fm.ModflowRch(model=self, **kwargs)
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

        self._setup_array(package, 'hk', vmin=0, vmax=hiKlakes_value,
                           source_package=source_package, by_layer=True)
        self._setup_array(package, 'vka', vmin=0, vmax=hiKlakes_value,
                           source_package=source_package, by_layer=True)
        if np.any(~self.dis.steady.array):
            self._setup_array(package, 'sy', vmin=0, vmax=1,
                               source_package=source_package, by_layer=True)
            self._setup_array(package, 'ss', vmin=0, vmax=1,
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


        TODO: generalize well package setup with specific input requirements


        """

        print('setting up WEL package...')
        t0 = time.time()

        df = setup_wel_data(self)

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

    def setup_lak(self):

        print('setting up LAKE package...')
        t0 = time.time()
        if self.lakarr.sum() == 0:
            print("lakes_shapefile not specified, or no lakes in model area")
            return

        # source data
        source_data = self.cfg['lak']['source_data']
        lakesdata = self.load_features(**source_data['lakes_shapefile'])
        id_column = source_data['lakes_shapefile']['id_column'].lower()
        nlakes = len(lakesdata)

        # lake package settings
        start_tab_units_at = 150  # default starting number for iunittab

        # set up the tab files, if any
        if 'stage_area_volume_file' in source_data:
            print('setting up tabfiles...')
            sd = TabularSourceData.from_config(source_data['stage_area_volume_file'])
            df = sd.get_data()

            lakes = df.groupby(id_column)
            n_included_lakes = len(set(lakesdata[id_column]).\
                                   intersection(set(lakes.groups.keys())))
            assert n_included_lakes == nlakes, "stage_area_volume (tableinput) option" \
                                               " requires info for each lake, " \
                                               "only these HYDROIDs found:\n{}".format(df[id_column].tolist())
            tab_files = []
            tab_units = []
            for i, id in enumerate(lakesdata[id_column].tolist()):
                dfl = lakes.get_group(id)
                assert len(dfl) == 151, "151 values required for each lake; " \
                                        "only {} for HYDROID {} in {}"\
                    .format(len(dfl), id, source_data['stage_area_volume_file'])
                tabfilename = '{}/{}/{}_stage_area_volume.dat'.format(self.model_ws,
                                                                      self.external_path,
                                                                      id)
                dfl[['stage', 'volume', 'area']].to_csv(tabfilename, index=False, header=False,
                                                        sep=' ', float_format='%.5e')
                print('wrote {}'.format(tabfilename))
                tab_files.append(tabfilename)
                tab_units.append(start_tab_units_at + i)

            # tabfiles aren't rewritten by flopy on package write
            self.cfg['lak']['tab_files'] = tab_files
            # kludge to deal with ugliness of lake package external file handling
            # (need to give path relative to model_ws, not folder that flopy is working in)
            tab_files_argument = [os.path.relpath(f) for f in tab_files]

        self.setup_external_filepaths('lak', 'lakzones',
                                      self.cfg['lak']['{}_filename_fmt'.format('lakzones')],
                                      nfiles=1)
        self.setup_external_filepaths('lak', 'bdlknc',
                                      self.cfg['lak']['{}_filename_fmt'.format('bdlknc')],
                                      nfiles=self.nlay)

        # make the arrays or load them
        lakzones = make_bdlknc_zones(self.modelgrid, lakesdata,
                                     include_ids=lakesdata[id_column])
        save_array(self.cfg['intermediate_data']['lakzones'][0], lakzones, fmt='%d')

        bdlknc = np.zeros((self.nlay, self.nrow, self.ncol))
        # make the areal footprint of lakebed leakance from the zones (layer 1)
        bdlknc[0] = make_bdlknc2d(lakzones,
                                  self.cfg['lak']['littoral_leakance'],
                                  self.cfg['lak']['profundal_leakance'])
        for k in range(self.nlay):
            if k > 0:
                # for each underlying layer, assign profundal leakance to cells were isbc == 1
                bdlknc[k][self.isbc[k] == 1] = self.cfg['lak']['profundal_leakance']
            save_array(self.cfg['intermediate_data']['bdlknc'][0][k], bdlknc[k], fmt='%.6e')

        # save a lookup file mapping lake ids to hydroids
        df = pd.DataFrame({'lakid': np.arange(1, nlakes+1),
                           'hydroid': lakesdata[id_column].values})
        df.to_csv(self.cfg['lak']['output_files']['lookup_file'],
                  index=False)

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

        kwargs = self.cfg['lak']
        kwargs['nlakes'] = len(lakesdata)
        kwargs['stages'] = stages
        kwargs['stage_range'] = stage_range
        kwargs['flux_data'] = flux_data
        kwargs['tab_files'] = tab_files_argument  #This needs to be in the order of the lake IDs!
        kwargs['tab_units'] = tab_units
        kwargs['options'] = options
        kwargs['ipakcb'] = self.ipakcb
        kwargs['lwrt'] = 0
        kwargs = get_input_arguments(kwargs, fm.mflak.ModflowLak)
        lak = fm.ModflowLak(self, **kwargs)
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
        """TODO: generalize hydmod setup with specific input requirements"""
        print('setting up HYDMOD package...')
        t0 = time.time()
        obs_info_files = self.cfg['hyd'].get('source_data', {}).get('filenames')
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
            # TODO: make private attribute to facilitate keeping track of lake IDs
            lak_files = ['lak{}_{}.ggo'.format(i+1, hydroid)
                         for i, hydroid in enumerate(self.cfg['lak']['source_data']['lakes_shapefile']['include_ids'])]

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

    @classmethod
    def setup_from_yaml(cls, yamlfile, verbose=False):
        """Make a model from scratch, using information in a yamlfile.

        Parameters
        ----------
        yamlfile : str (filepath)
            Configuration file in YAML format with model setup information.

        Returns
        -------
        m : mfsetup.MFnwtModel model object
        """

        cfg = cls.load_cfg(yamlfile, verbose=verbose)
        cfg['filename'] = yamlfile
        print('\nSetting up {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = cls(cfg=cfg, **cfg['model'])
        assert m.exe_name != 'mf2005.exe'

        kwargs = m.cfg['setup_grid']
        source_data = m.cfg['setup_grid'].get('source_data', {})
        if 'features_shapefile' in source_data:
            kwargs.update(source_data['features_shapefile'])
            kwargs['features_shapefile'] = kwargs.get('filename')
        rename = kwargs.get('variable_mappings', {})
        for k, v in rename.items():
            if k in kwargs:
                kwargs[v] = kwargs.pop(k)
        kwargs = get_input_arguments(kwargs, m.setup_grid)
        if 'grid' not in m.cfg.keys():
            m.setup_grid(**kwargs)

        # set up all of the packages specified in the config file
        for pkg in m.package_list:
            package_setup = getattr(cls, 'setup_{}'.format(pkg))
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

    @classmethod
    def load(cls, yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = cls.load_cfg(yamlfile, verbose=verbose)
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = cls(cfg=cfg, **cfg['model'])
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

