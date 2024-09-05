import os
import shutil
import time
from pathlib import Path

import flopy
import numpy as np
import pandas as pd

fm = flopy.modflow
from flopy.modflow import Modflow

from mfsetup.bcs import remove_inactive_bcs
from mfsetup.discretization import (
    deactivate_idomain_above,
    find_remove_isolated_cells,
    make_ibound,
)
from mfsetup.fileio import (
    add_version_to_fileheader,
    flopy_mf2005_load,
    load,
    load_cfg,
    save_array,
)
from mfsetup.ic import setup_strt
from mfsetup.lakes import (
    make_bdlknc2d,
    make_bdlknc_zones,
    setup_lake_fluxes,
    setup_lake_info,
    setup_lake_tablefiles,
)
from mfsetup.mfmodel import MFsetupMixin
from mfsetup.obs import setup_head_observations
from mfsetup.oc import parse_oc_period_input
from mfsetup.tdis import (
    get_parent_stress_periods,
    setup_perioddata,
    setup_perioddata_group,
)
from mfsetup.units import convert_length_units, itmuni_text, lenuni_text
from mfsetup.utils import get_input_arguments, get_packages


class MFnwtModel(MFsetupMixin, Modflow):
    """Class representing a MODFLOW-NWT model"""
    default_file = 'mfnwt_defaults.yml'

    def __init__(self, parent=None, cfg=None,
                 modelname='model', exe_name='mfnwt',
                 version='mfnwt', model_ws='.',
                 external_path='external/', **kwargs):
        defaults = {'parent': parent,
                    'modelname': modelname,
                    'exe_name': exe_name,
                    'version': version,
                    'model_ws': model_ws,
                    'external_path': external_path,
                    }
        # load configuration, if supplied
        if cfg is not None:
            if not isinstance(cfg, dict):
                cfg = self.load_cfg(cfg)
            cfg = self._parse_model_kwargs(cfg)
            defaults.update(cfg['model'])
            kwargs = {k: v for k, v in kwargs.items() if k not in defaults}
        # otherwise, pass arguments on to flopy constructor
        args = get_input_arguments(defaults, Modflow,
                                   exclude='packages')
        Modflow.__init__(self, **args, **kwargs)
        #Modflow.__init__(self, modelname, exe_name=exe_name, version=version,
        #                 model_ws=model_ws, external_path=external_path,
        #                 **kwargs)
        MFsetupMixin.__init__(self, parent=parent)

        # default configuration
        self._package_setup_order = ['dis', 'bas6', 'upw', 'rch', 'oc',
                                     'chd', 'ghb', 'lak', 'sfr', 'riv', 'wel', 'mnw2',
                                     'gag', 'hyd']
        # set up the model configuration dictionary
        # start with the defaults
        self.cfg = load(self.source_path / self.default_file)  # 'mf6_defaults.yml')
        self.relative_external_paths = self.cfg.get('model', {}).get('relative_external_paths', True)
        # set the model workspace and change working directory to there
        self.model_ws = self._get_model_ws(cfg=cfg)
        # update defaults with user-specified config. (loaded above)
        # set up and validate the model configuration dictionary
        self._set_cfg(cfg)

        # set the list file path
        self.lst.file_name = [self.cfg['model']['list_filename_fmt'].format(self.name)]

        # the "drop thin cells" option is not available for MODFLOW-2005 models
        self._drop_thin_cells = False

        # property arrays
        self._ibound = None

        # delete the temporary 'original-files' folder
        # if it already exists, to avoid side effects from stale files
        shutil.rmtree(self.tmpdir, ignore_errors=True)

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
    def perioddata(self):
        """DataFrame summarizing stress period information.
        Columns:
        ============== =========================================
        start_datetime Start date of each model stress period
        end_datetime   End date of each model stress period
        time           MODFLOW elapsed time, in days*
        per            Model stress period number
        perlen         Stress period length (days)
        nstp           Number of timesteps in stress period
        tsmult         Timestep multiplier
        steady         Steady-state or transient
        oc             Output control setting for MODFLOW
        parent_sp      Corresponding parent model stress period
        ============== =========================================

        TODO: the code here might still need to be adapted to
        parallel the code in MF6model.perioddata, to work with
        parent models that are already loaded but have no configuration.
        """
        if self._perioddata is None:
            default_start_datetime = self.cfg['dis'].get('start_date_time', '1970-01-01')
            tdis_perioddata_config = self.cfg['dis']
            nper = self.cfg['dis'].get('nper')
            steady = self.cfg['dis'].get('steady')
            parent_stress_periods=None
            if self.parent is not None:
                parent_stress_periods = self.cfg['parent'].get('copy_stress_periods')
            perioddata = setup_perioddata(
                    self,
                    tdis_perioddata_config=tdis_perioddata_config,
                    default_start_datetime=default_start_datetime,
                    nper=nper, steady=steady, time_units=self.time_units,
                    parent_model=self.parent,
                    parent_stress_periods=parent_stress_periods,
                    )
            self._perioddata = perioddata
            # reset nper property so that it will reference perioddata table
            self._nper = None
            self._perioddata.to_csv(f'{self._tables_path}/stress_period_data.csv', index=False)
            # update the model configuration
            if 'parent_sp' in perioddata.columns:
                self.cfg['parent']['copy_stress_periods'] = perioddata['parent_sp'].tolist()

        return self._perioddata

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
        ibound_from_layer_elevations = make_ibound(self.dis.top.array,
                                                     self.dis.botm.array,
                                                     nodata=self._nodata_value,
                                                     minimum_layer_thickness=self.cfg['dis'].get(
                                                         'minimum_layer_thickness', 1),
                                                     #drop_thin_cells=self._drop_thin_cells,
                                                     tol=1e-4)

        # include cells that are active in the existing idomain array
        # and cells inactivated on the basis of layer elevations
        ibound = (self.bas6.ibound.array > 0) & (ibound_from_layer_elevations >= 1)
        ibound = ibound.astype(int)

        # remove cells that conincide with lakes
        ibound[self.isbc == 1] = 0.

        # remove cells that are above stream cells
        if self.get_package('sfr') is not None:
            ibound = deactivate_idomain_above(ibound, self.sfr.reach_data)
        # remove cells that are above ghb cells
        if self.get_package('ghb') is not None:
            ibound = deactivate_idomain_above(ibound, self.ghb.stress_period_data[0])

        # inactivate any isolated cells that could cause problems with the solution
        ibound = find_remove_isolated_cells(ibound, minimum_cluster_size=20)

        self._ibound = ibound
        # re-write the input files
        self._setup_array('bas6', 'ibound', resample_method='nearest',
                          data={i: arr for i, arr in enumerate(ibound)},
                          datatype='array3d', write_fmt='%d', dtype=int)
        self.bas6.ibound = self.cfg['bas6']['ibound']

    def _set_parent(self):
        """Set attributes related to a parent or source model
        if one is specified."""

        if self.cfg['parent'].get('version') == 'mf6':
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
            if 'load_only' not in kwargs:
                    kwargs['load_only'] = load_only
            if 'skip_load' in kwargs:
                kwargs['skip_load'] = [s.lower() for s in kwargs['skip_load']]
                kwargs['load_only'] = [pckg for pckg in kwargs['load_only']
                                        if pckg not in kwargs['skip_load']]
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

            # parent model perioddata
            if not hasattr(self.parent, 'perioddata'):
                kwargs = {}
                kwargs['start_date_time'] = self.cfg['parent'].get('start_date_time',
                                                                   self.cfg['model'].get('start_date_time',
                                                                                         '1970-01-01'))
                kwargs['nper'] = self.parent.nper
                kwargs['model_time_units'] = self.cfg['parent']['time_units']
                for var in ['perlen', 'steady', 'nstp', 'tsmult']:
                    kwargs[var] = self.parent.dis.__dict__[var].array
                kwargs = get_input_arguments(kwargs, setup_perioddata_group)
                kwargs['oc_saverecord'] = {}
                self._parent.perioddata = setup_perioddata_group(**kwargs)

            # default_source_data, where omitted configuration input is
            # obtained from parent model by default
            # Set default_source_data to True by default if it isn't specified
            if self.cfg['parent'].get('default_source_data') is None:
                self.cfg['parent']['default_source_data'] = True
            if self.cfg['parent'].get('default_source_data'):
                self._parent_default_source_data = True
                if self.cfg['dis'].get('nlay') is None:
                    self.cfg['dis']['nlay'] = self.parent.dis.nlay
                parent_start_date_time = self.cfg.get('parent', {}).get('start_date_time')
                if self.cfg['dis'].get('start_date_time', '1970-01-01') == '1970-01-01' and parent_start_date_time is not None:
                    self.cfg['dis']['start_date_time'] = self.cfg['parent']['start_date_time']
                if self.cfg['dis'].get('nper') is None:
                    self.cfg['dis']['nper'] = self.parent.dis.nper
                parent_periods = get_parent_stress_periods(self.parent, nper=self.cfg['dis']['nper'],
                                                           parent_stress_periods=self.cfg['parent']['copy_stress_periods'])
                for var in ['perlen', 'nstp', 'tsmult', 'steady']:
                    if self.cfg['dis'].get(var) is None:
                        self.cfg['dis'][var] = self.parent.dis.__dict__[var].array[parent_periods]

    def _update_grid_configuration_with_dis(self):
        """Update grid configuration with any information supplied to dis package
        (so that settings specified for DIS package have priority). This method
        is called by MFsetupMixin.setup_grid.
        """
        for param in ['nrow', 'ncol', 'delr', 'delc']:
            if param in self.cfg['dis']:
                self.cfg['setup_grid'][param] = self.cfg['dis'][param]

    def setup_dis(self, **kwargs):
        """"""
        package = 'dis'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        # resample the top from the DEM
        if self.cfg['dis']['remake_top']:
            self._setup_array(package, 'top', datatype='array2d',
                              resample_method='linear',
                              write_fmt='%.2f')

        # make the botm array
        self._setup_array(package, 'botm', datatype='array3d',
                          resample_method='linear',
                          write_fmt='%.2f')

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
        #if not isinstance(self._modelgrid, MFsetupGrid):
        #    self._modelgrid = None  # override DIS package grid setup
        self.setup_grid()  # reset the model grid
        self._reset_bc_arrays()
        #self._isbc = None  # reset BC property arrays
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    def setup_tdis(self, **kwargs):
        """Calls the _set_perioddata, to establish time discretization. Only purpose
        is to conform to same syntax as mf6 for MFsetupMixin.setup_from_yaml()
        """
        self.perioddata

    def setup_bas6(self, **kwargs):
        """"""
        package = 'bas6'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        kwargs = self.cfg[package]
        kwargs['source_data_config'] = kwargs['source_data']
        kwargs['filename_fmt'] = kwargs['strt_filename_fmt']
        kwargs['write_fmt'] = kwargs['strt_write_fmt']

        # make the starting heads array
        strt = setup_strt(self, package, **kwargs)

        # initial ibound input for creating a bas6 package instance
        self._setup_array(package, 'ibound', datatype='array3d', write_fmt='%d',
                          resample_method='nearest',
                          dtype=int)

        kwargs = get_input_arguments(self.cfg['bas6'], fm.ModflowBas)
        kwargs['strt'] = strt
        bas = fm.ModflowBas(model=self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        self._set_ibound()
        return bas

    def setup_oc(self, **kwargs):

        package = 'oc'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        #stress_period_data = {}
        #for i, r in self.perioddata.iterrows():
        #    stress_period_data[(r.per, r.nstp -1)] = r.oc

        # use stress_period_data if supplied
        # (instead of period_input defaults)
        if 'stress_period_data' in self.cfg['oc']:
            del self.cfg['oc']['period_options']
        kwargs = self.cfg['oc']
        period_input = parse_oc_period_input(kwargs, nstp=self.perioddata.nstp,
                                             output_fmt='mfnwt')
        kwargs.update(period_input)
        kwargs = get_input_arguments(kwargs, fm.ModflowOc)
        oc = fm.ModflowOc(model=self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return oc

    def setup_rch(self, **kwargs):
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

    def setup_upw(self, **kwargs):
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
        if None in [hk, vka] and \
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

        self._setup_array(package, 'hk', vmin=0, vmax=hiKlakes_value, resample_method='linear',
                           source_package=source_package, datatype='array3d', write_fmt='%.6e')
        self._setup_array(package, 'vka', vmin=0, vmax=hiKlakes_value, resample_method='linear',
                           source_package=source_package, datatype='array3d', write_fmt='%.6e')
        if np.any(~self.dis.steady.array):
            self._setup_array(package, 'sy', vmin=0, vmax=1, resample_method='linear',
                              source_package=source_package,
                              datatype='array3d', write_fmt='%.6e')
            self._setup_array(package, 'ss', vmin=0, vmax=1, resample_method='linear',
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

    def setup_mnw2(self, **kwargs):

        print('setting up MNW2 package...')
        t0 = time.time()

        # added wells
        # todo: generalize MNW2 source data input; add auto-reprojection
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

    def setup_lak(self, **kwargs):

        print('setting up LAKE package...')
        t0 = time.time()
        # if shapefile of lakes was included,
        # lakarr should be automatically built by property method
        if self.lakarr.sum() == 0:
            print("lakes_shapefile not specified, or no lakes in model area")
            return

        # source data
        source_data = self.cfg['lak']['source_data']
        self.lake_info = setup_lake_info(self)
        nlakes = len(self.lake_info)

        # set up the tab files, if any
        tab_files_argument = None
        tab_units = None
        start_tab_units_at = 150  # default starting number for iunittab
        if 'stage_area_volume_file' in source_data:

            tab_files = setup_lake_tablefiles(self, source_data['stage_area_volume_file'])
            tab_units = list(range(start_tab_units_at, start_tab_units_at + len(tab_files)))

            # tabfiles aren't rewritten by flopy on package write
            self.cfg['lak']['tab_files'] = tab_files
            # kludge to deal with ugliness of lake package external file handling
            # (need to give path relative to model_ws, not folder that flopy is working in)
            tab_files_argument = [os.path.relpath(f) for f in tab_files]

        self.setup_external_filepaths('lak', 'lakzones',
                                      self.cfg['lak']['{}_filename_fmt'.format('lakzones')])
        self.setup_external_filepaths('lak', 'bdlknc',
                                      self.cfg['lak']['{}_filename_fmt'.format('bdlknc')],
                                      file_numbers=list(range(self.nlay)))

        # make the arrays or load them
        lakzones = make_bdlknc_zones(self.modelgrid, self.lake_info,
                                     include_ids=self.lake_info['feat_id'],
                                     littoral_zone_buffer_width=source_data['littoral_zone_buffer_width'])
        save_array(self.cfg['intermediate_data']['lakzones'][0], lakzones, fmt='%d')

        bdlknc = np.zeros((self.nlay, self.nrow, self.ncol))
        # make the areal footprint of lakebed leakance from the zones (layer 1)
        bdlknc[0] = make_bdlknc2d(lakzones,
                                  self.cfg['lak']['source_data']['littoral_leakance'],
                                  self.cfg['lak']['source_data']['profundal_leakance'])
        for k in range(self.nlay):
            if k > 0:
                # for each underlying layer, assign profundal leakance to cells were isbc == 1
                bdlknc[k][self.isbc[k] == 1] = self.cfg['lak']['source_data']['profundal_leakance']
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

    def setup_chd(self, **kwargs):
        """Set up the CHD Package.
        """
        return self._setup_basic_stress_package(
            'chd', fm.ModflowChd, ['head'], **kwargs)

    def setup_drn(self, **kwargs):
        """Set up the Drain Package.
        """
        return self._setup_basic_stress_package(
            'drn', fm.ModflowDrn, ['elev', 'cond'], **kwargs)

    def setup_ghb(self, **kwargs):
        """Set up the General Head Boundary Package.
        """
        return self._setup_basic_stress_package(
            'ghb', fm.ModflowGhb, ['bhead', 'cond'], **kwargs)


    def setup_riv(self, rivdata=None, **kwargs):
        """Set up the River Package.
        """
        return self._setup_basic_stress_package(
            'riv', fm.ModflowRiv, ['stage', 'cond', 'rbot'],
            rivdata=rivdata, **kwargs)

    def setup_wel(self, **kwargs):
        """Set up the Well Package.
        """
        return self._setup_basic_stress_package(
            'wel', fm.ModflowWel, ['flux'], **kwargs)

    def setup_nwt(self, **kwargs):

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

    def setup_hyd(self, **kwargs):
        """TODO: generalize hydmod setup with specific input requirements"""
        package = 'hyd'
        print('setting up HYDMOD package...')
        t0 = time.time()

        iobs_domain = None
        if not kwargs['mfsetup_options']['allow_obs_in_bc_cells']:
            # for now, discard any head observations in same (i, j) column of cells
            # as a non-well boundary condition
            # including lake package lakes and non lake, non well BCs
            # (high-K lakes are excluded, since we may want head obs at those locations,
            #  to serve as pseudo lake stage observations)
            iobs_domain = (self.isbc == 1) | np.any(self.isbc > 2)

        # munge the observation data
        df = setup_head_observations(self,
                                     obs_package=package,
                                     obsname_column='hydlbl',
                                     iobs_domain=iobs_domain,
                                     **kwargs['source_data'],
                                     **kwargs['mfsetup_options'])

        # create observation data recarray
        obsdata = fm.ModflowHyd.get_empty(len(df))
        for c in obsdata.dtype.names:
            assert c in df.columns, "Missing observation data field: {}".format(c)
            obsdata[c] = df[c]
        nhyd = len(df)
        hyd = flopy.modflow.ModflowHyd(self, nhyd=nhyd, hydnoh=-999, obsdata=obsdata)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return hyd

    def setup_gag(self, **kwargs):

        print('setting up GAGE package...')
        t0 = time.time()
        # setup gage package output for all included lakes
        ngages = 0
        nlak_gages = 0
        starting_unit_number = self.cfg['gag']['starting_unit_number']
        if self.get_package('lak') is not None:
            nlak_gages = self.lak.nlakes
        if nlak_gages > 0:
            ngages += nlak_gages
            lak_gagelocs = list(np.arange(1, nlak_gages+1) * -1)
            lak_gagerch = [0] * nlak_gages # dummy list to maintain index position
            lak_outtype = [self.cfg['gag']['lak_outtype']] * nlak_gages
            # need minus sign to tell MF to read outtype
            lake_unit = list(-np.arange(starting_unit_number,
                                        starting_unit_number + nlak_gages, dtype=int))
            # TODO: make private attribute to facilitate keeping track of lake IDs
            lak_files = ['lak{}_{}.ggo'.format(i+1, hydroid)
                         for i, hydroid in enumerate(self.cfg['lak']['source_data']['lakes_shapefile']['include_ids'])]
            # update the starting unit number of avoid collisions with other gage packages
            starting_unit_number = np.max(np.abs(lake_unit)) + 1

        # need to add streams at some point
        nstream_gages = 0
        stream_gageseg = []
        stream_gagerch = []
        stream_unit = []
        stream_outtype = []
        stream_files = []
        if self.get_package('sfr') is not None:
            #observations_input = self.cfg['sfr'].get('source_data', {}).get('observations')
            #obs_info_files = self.cfg['gag'].get('observation_data')
            #if obs_info_files is not None:
            #    # get obs_info_files into dictionary format
            #    # filename: dict of column names mappings
            #    if isinstance(obs_info_files, str):
            #        obs_info_files = [obs_info_files]
            #    if isinstance(obs_info_files, list):
            #        obs_info_files = {f: self.cfg['gag']['default_columns']
            #                          for f in obs_info_files}
            #    elif isinstance(obs_info_files, dict):
            #        for k, v in obs_info_files.items():
            #            if v is None:
            #                obs_info_files[k] = self.cfg['gag']['default_columns']
#
            #    print('Reading observation files...')
            #    check_source_files(obs_info_files.keys())
            #    dfs = []
            #    for f, column_info in obs_info_files.items():
            #        print(f)
            #        df = read_observation_data(f,
            #                                   column_info,
            #                                   column_mappings=self.cfg['hyd'].get('column_mappings'))
            #        dfs.append(df) # cull to cols that are needed
            #    df = pd.concat(dfs, axis=0)
            df = self.sfrdata.observations
            nstream_gages = len(df)
            stream_files = ['{}.ggo'.format(site_no) for site_no in df.obsname]
            stream_gageseg = df.iseg.tolist()
            stream_gagerch = df.ireach.tolist()
            stream_unit = list(np.arange(starting_unit_number,
                                         starting_unit_number + nstream_gages, dtype=int))
            stream_outtype = [self.cfg['gag']['sfr_outtype']] * nstream_gages
        ngages += nstream_gages

        if ngages == 0:
            print('No gage package input.')
            return

        # create flopy gage package object
        gage_data = fm.ModflowGage.get_empty(ncells=ngages)
        gage_data['gageloc'] = lak_gagelocs + stream_gageseg
        gage_data['gagerch'] = lak_gagerch + stream_gagerch
        gage_data['unit'] = lake_unit + stream_unit
        gage_data['outtype'] = lak_outtype + stream_outtype
        if len(self.cfg['gag'].get('ggo_files', {})) == 0:
            self.cfg['gag']['ggo_files'] = lak_files + stream_files
        gag = fm.ModflowGage(self, numgage=len(gage_data),
                             gage_data=gage_data,
                             files=self.cfg['gag']['ggo_files'],
                             )
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return gag

    def write_input(self):
        """Write the model input.
        """
        # prior to writing output
        # remove any BCs in inactive cells
        pckgs = ['CHD']
        for pckg in pckgs:
            package_instance = getattr(self, pckg.lower(), None)
            if package_instance is not None:
                remove_inactive_bcs(package_instance)

        # write the model with flopy
        # but skip the sfr package
        # by monkey-patching the write method
        SelPackList = [p for p in self.get_package_list() if p != 'SFR']
        super().write_input(SelPackList=SelPackList)

        # write the sfr package with SFRmaker
        # gage package was already set-up and then written by Flopy
        if 'SFR' in self.get_package_list():
            self.sfrdata.write_package(write_observations_input=False)

        # add version info to file headers
        files = [self.namefile]
        files += [p.file_name[0] for p in self.packagelist]
        for f in files:
            # either flopy or modflow
            # doesn't allow headers for some packages
            ext = Path(f).suffix
            if ext in {'.hyd', '.gag', '.gage'}:
                continue
            add_version_to_fileheader(f, model_info=self.header)

        if not self.cfg['mfsetup_options']['keep_original_arrays']:
            tmpdir_path = self.tmpdir
            shutil.rmtree(tmpdir_path)

    @staticmethod
    def _parse_model_kwargs(cfg):
        return cfg

    @classmethod
    def load(cls, yamlfile, load_only=None, verbose=False, forgive=False, check=False):
        """Load a model from a config file and set of MODFLOW files.
        """
        cfg = load_cfg(yamlfile, verbose=verbose, default_file=cls.default_file) # 'mfnwt_defaults.yml')
        print('\nLoading {} model from data in {}\n'.format(cfg['model']['modelname'], yamlfile))
        t0 = time.time()

        m = cls(cfg=cfg, **cfg['model'])
        if 'grid' not in m.cfg.keys():
            m.setup_grid()
            #grid_file = cfg['setup_grid']['output_files']['grid_file']
            #if os.path.exists(grid_file):
            #    print('Loading model grid definition from {}'.format(grid_file))
            #    m.cfg['grid'] = load(grid_file)
            #else:
            #    m.setup_grid()

        m = flopy_mf2005_load(m, load_only=load_only, forgive=forgive, check=check)
        print('finished loading model in {:.2f}s'.format(time.time() - t0))
        return m
