import sys
sys.path.append('..')
sys.path.append('/Users/aleaf/Documents/GitHub/sfrmaker')
import os
import time
import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd
from shapely.geometry import MultiPolygon
import flopy
fm = flopy.modflow
from flopy.modflow import Modflow
from flopy.utils import binaryfile as bf
from .discretization import (deactivate_idomain_above,
                             find_remove_isolated_cells)
from .tdis import setup_perioddata_group
from .grid import write_bbox_shapefile, setup_structured_grid
from .fileio import load, dump, load_array, save_array, check_source_files, flopy_mf2005_load, \
    load_cfg, setup_external_filepaths
from .lakes import make_bdlknc_zones, make_bdlknc2d, setup_lake_fluxes, setup_lake_info
from .utils import update, get_packages, get_input_arguments
from .obs import read_observation_data, setup_head_observations
from .sourcedata import ArraySourceData, MFArrayData, TabularSourceData, setup_array
from .units import convert_length_units, convert_time_units, convert_flux_units, lenuni_text, itmuni_text, lenuni_values
from .wells import setup_wel_data
from .mfmodel import MFsetupMixin


class MFnwtModel(MFsetupMixin, Modflow):
    """Class representing a MODFLOW-NWT model"""

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
        # default configuration (different for nwt vs mf6)
        self.cfg = load(self.source_path + '/mfnwt_defaults.yml')
        self.cfg['filename'] = self.source_path + '/mfnwt_defaults.yml'
        self._set_cfg(cfg)  # set up the model configuration dictionary
        self.relative_external_paths = self.cfg.get('model', {}).get('relative_external_paths', True)
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
        """
        if self._ibound is None and 'BAS6' in self.get_package_list():
            self._set_ibound()
        return self._ibound

    def _set_ibound(self):
        """Remake the idomain array from the source data,
        no data values in the top and bottom arrays, and
        so that cells above SFR reaches are inactive."""

        # include cells that are active in the existing idomain array
        # and cells inactivated on the basis of layer elevations
        ibound = (self.bas6.ibound.array == 1)
        ibound = ibound.astype(int)

        # remove cells that are above stream cells
        if 'SFR' in self.get_package_list():
            ibound = deactivate_idomain_above(ibound, self.sfr.reach_data)

        # inactivate any isolated cells that could cause problems with the solution
        ibound = find_remove_isolated_cells(ibound, minimum_cluster_size=20)

        self._ibound = ibound
        # re-write the input files
        self._setup_array('bas6', 'ibound',
                          data={i: arr for i, arr in enumerate(ibound)},
                          datatype='array3d', write_fmt='%d', dtype=int)
        self.bas6.ibound = self.cfg['bas6']['ibound']

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
            load_only = list(set(packages_in_parent_namefile).intersection(
                set(self.cfg['model'].get('packages', set()))))
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
                if self.cfg['dis'].get('nlay') is None:
                    self.cfg['dis']['nlay'] = self.parent.dis.nlay
                if self.cfg['dis'].get('start_date_time') is None:
                    self.cfg['dis']['start_date_time'] = self.cfg['parent']['start_date_time']
                if self.cfg['dis'].get('nper') is None:
                    self.cfg['dis']['nper'] = self.parent.dis.nper
                for var in ['nper', 'perlen', 'nstp', 'tsmult', 'steady']:
                    if self.cfg['dis'].get(var) is None:
                        self.cfg['dis'][var] = self.parent.dis.__dict__[var].array

    def _set_perioddata(self):
        """Sets up the perioddata DataFrame.

        Needs some work to be more general.
        """
        parent_sp = self.cfg['parent']['copy_stress_periods']

        # use all stress periods from parent model
        if isinstance(parent_sp, str) and parent_sp.lower() == 'all':
            nper = self.cfg['dis'].get('nper')
            parent_sp = list(range(self.parent.nper))
            if nper is None or nper < self.parent.nper:
                nper = self.parent.nper
            elif nper > self.parent.nper:
                for i in range(nper - self.parent.nper):
                    parent_sp.append(parent_sp[-1])
            #self.cfg['dis']['perlen'] = None # set from parent model
            #self.cfg['dis']['steady'] = None

        # use only specified stress periods from parent model
        elif isinstance(parent_sp, list):
            # limit parent stress periods to include
            # to those in parent model and nper specified for pfl_nwt
            nper = self.cfg['dis'].get('nper', len(parent_sp))

            parent_sp = [0]
            perlen = [self.parent.dis.perlen.array[0]]
            for i, p in enumerate(self.cfg['parent']['copy_stress_periods']):
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
        self.cfg['parent']['copy_stress_periods'] = parent_sp

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
                self.cfg['dis'][var] = getattr(self.parent.dis, var)[self.cfg['parent']['copy_stress_periods']]

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

    def _update_grid_configuration_with_dis(self):
        """Update grid configuration with any information supplied to dis package
        (so that settings specified for DIS package have priority). This method
        is called by MFsetupMixin.setup_grid.
        """
        for param in ['nrow', 'ncol', 'delr', 'delc']:
            if param in self.cfg['dis']:
                self.cfg['setup_grid'].update({param: self.cfg['dis'][param]})

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
        self._reset_bc_arrays()
        #self._isbc = None  # reset BC property arrays
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    def setup_bas6(self):
        """"""
        package = 'bas6'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # make the strt array
        self._setup_array(package, 'strt', datatype='array3d', write_fmt='%.2f')
        
        # initial ibound input for creating a bas6 package instance
        self._setup_array(package, 'ibound', datatype='array3d', write_fmt='%d',
                          dtype=int)

        kwargs = get_input_arguments(self.cfg['bas6'], fm.ModflowBas)
        bas = fm.ModflowBas(model=self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return bas

    def setup_chd(self):
        """Set up constant head package for perimeter boundary.
        Todo: create separate perimeter boundary setup method/input block
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

        assert len(unique_kper) == len(set(self.cfg['parent']['copy_stress_periods'])), \
        "read {} from {},\nexpected stress periods: {}".format(kstpkper,
                                                               headfile,
                                                               sorted(list(set(self.cfg['parent']['copy_stress_periods'])))
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
        self._setup_array(package, 'rech', datatype='transient2d',
                          resample_method='linear',
                          write_fmt='%.6e',
                          write_nodata=0.)

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
                           source_package=source_package, datatype='array3d', write_fmt='%.6e')
        self._setup_array(package, 'vka', vmin=0, vmax=hiKlakes_value,
                           source_package=source_package, datatype='array3d', write_fmt='%.6e')
        if np.any(~self.dis.steady.array):
            self._setup_array(package, 'sy', vmin=0, vmax=1,
                              source_package=source_package,
                              datatype='array3d', write_fmt='%.6e')
            self._setup_array(package, 'ss', vmin=0, vmax=1,
                              source_package=source_package,
                              datatype='array3d', write_fmt='%.6e')
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
        self.lake_info = setup_lake_info(self)
        nlakes = len(self.lake_info)

        # lake package settings
        start_tab_units_at = 150  # default starting number for iunittab

        # set up the tab files, if any
        tab_files_argument = None
        tab_units = None
        if 'stage_area_volume_file' in source_data:
            print('setting up tabfiles...')
            sd = TabularSourceData.from_config(source_data['stage_area_volume_file'])
            df = sd.get_data()

            lakes = df.groupby(sd.id_column)
            n_included_lakes = len(set(self.lake_info['feat_id']).\
                                   intersection(set(lakes.groups.keys())))
            assert n_included_lakes == nlakes, "stage_area_volume (tableinput) option" \
                                               " requires info for each lake, " \
                                               "only these feature IDs found:\n{}".format(df[sd.id_column].tolist())
            tab_files = []
            tab_units = []
            for i, id in enumerate(self.lake_info['feat_id'].tolist()):
                dfl = lakes.get_group(id)
                assert len(dfl) == 151, "151 values required for each lake; " \
                                        "only {} for feature id {} in {}"\
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
        lakzones = make_bdlknc_zones(self.modelgrid, self.lake_info,
                                     include_ids=self.lake_info['feat_id'])
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

        # get estimates of stage from model top, for specifying ranges
        stages = []
        for lakid in self.lake_info['lak_id']:
            loc = self.lakarr[0] == lakid
            est_stage = self.dis.top.array[loc].min()
            stages.append(est_stage)
        stages = np.array(stages)

        # setup stress period data
        tol = 5  # specify lake stage range as +/- this value
        ssmn, ssmx = stages - tol, stages + tol
        stage_range = list(zip(ssmn, ssmx))

        # set up dataset 9
        # ssmn and ssmx values only required for steady-state periods > 0
        self.lake_fluxes = setup_lake_fluxes(self)
        precip = self.lake_fluxes['precipitation'].tolist()
        evap = self.lake_fluxes['evaporation'].tolist()
        flux_data = {}
        for i, steady in enumerate(self.dis.steady.array):
            if i > 0 and steady:
                flux_data_i = []
                for lake_ssmn, lake_ssmx in zip(ssmn, ssmx):
                    flux_data_i.append([precip[i], evap[i], 0, 0, lake_ssmn, lake_ssmx])
            else:
                flux_data_i = [[precip[i], evap[i], 0, 0]] * nlakes
            flux_data[i] = flux_data_i
        options = ['tableinput'] if tab_files_argument is not None else None

        kwargs = self.cfg['lak']
        kwargs['nlakes'] = len(self.lake_info)
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
        use_existing_file = self.cfg['nwt'].get('use_existing_file')
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
        package = 'hyd'
        print('setting up HYDMOD package...')
        t0 = time.time()

        # munge the head observation data
        df = setup_head_observations(self, format=package,
                                     obsname_column='hydlbl')

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
        starting_unit_number = self.cfg['gag']['starting_unit_number']
        if 'LAK' in self.get_package_list():
            nlak_gages = self.lak.nlakes
        if nlak_gages > 0:
            ngages += nlak_gages
            lak_gagelocs = list(np.arange(1, nlak_gages+1) * -1)
            lak_gagerch = [0] * nlak_gages # dummy list to maintain index position
            lak_outtype = [self.cfg['gag']['lak_outtype']] * nlak_gages
            # need minus sign to tell MF to read outtype
            lake_unit = list(-np.arange(starting_unit_number,
                                        starting_unit_number + nlak_gages))
            # TODO: make private attribute to facilitate keeping track of lake IDs
            lak_files = ['lak{}_{}.ggo'.format(i+1, hydroid)
                         for i, hydroid in enumerate(self.cfg['lak']['source_data']['lakes_shapefile']['include_ids'])]

        # need to add streams at some point
        nstream_gages = 0
        if 'SFR' in self.get_package_list():

            obs_info_files = self.cfg['gag'].get('observation_data')
            if obs_info_files is not None:
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

                print('Reading observation files...')
                check_source_files(obs_info_files.keys())
                dfs = []
                for f, column_info in obs_info_files.items():
                    print(f)
                    df = read_observation_data(f,
                                               column_info,
                                               column_mappings=self.cfg['hyd'].get('column_mappings'))
                    dfs.append(df) # cull to cols that are needed
                df = pd.concat(dfs, axis=0)
        ngages += nstream_gages
        # TODO: stream gage setup
        stream_gageseg = []
        stream_gagerch = []
        stream_unit = []
        stream_outtype = [self.cfg['gag']['sfr_outtype']] * nstream_gages
        stream_files = []

        if ngages == 0:
            print('No gage package input.')
            return

        # create flopy gage package object
        gage_data = fm.ModflowGage.get_empty(ncells=ngages)
        gage_data['gageloc'] = lak_gagelocs + stream_gageseg
        gage_data['gagerch'] = lak_gagerch + stream_gagerch
        gage_data['unit'] = lake_unit + stream_unit
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

        cfg = load_cfg(yamlfile)
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

    @classmethod
    def load(cls, yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = load_cfg(yamlfile, verbose=verbose)
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = cls(cfg=cfg, **cfg['model'])
        if 'grid' not in m.cfg.keys():
            grid_file = cfg['setup_grid']['output_files']['grid_file']
            if os.path.exists(grid_file):
                print('Loading model grid definition from {}'.format(grid_file))
                m.cfg['grid'] = load(grid_file)
            else:
                m.setup_grid(**m.cfg['setup_grid'])
        m = flopy_mf2005_load(m, load_only=load_only, forgive=forgive, check=check)
        print('finished loading model in {:.2f}s'.format(time.time() - t0))
        return m


class Inset(MFnwtModel):
    """Class representing a MODFLOW-NWT model that is an
    inset of a parent MODFLOW model."""
    def __init__(self, parent=None, cfg=None,
                 modelname='pfl_nwt', exe_name='mfnwt',
                 version='mfnwt', model_ws='.',
                 external_path='external/', **kwargs):

        MFnwtModel.__init__(self,  parent=parent, cfg=cfg,
                            modelname=modelname, exe_name=exe_name,
                            version=version, model_ws=model_ws,
                            external_path=external_path, **kwargs)

        inset_cfg = load(self.source_path + '/inset_defaults.yml')
        update(self.cfg, inset_cfg)
        self.cfg['filename'] = self.source_path + '/inset_defaults.yml'

