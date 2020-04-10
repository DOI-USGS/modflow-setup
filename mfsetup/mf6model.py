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
from flopy.utils.lgrutil import Lgr
from gisutils import get_values_at_points
from .discretization import (make_idomain, make_lgr_idomain, deactivate_idomain_above,
                             find_remove_isolated_cells,
                             create_vertical_pass_through_cells)
from .fileio import (load, dump, load_cfg,
                     flopy_mfsimulation_load)
from .grid import MFsetupGrid
from .lakes import (setup_lake_connectiondata, setup_lake_info,
                    setup_lake_tablefiles, setup_lake_fluxes,
                    get_lakeperioddata, setup_mf6_lake_obs)
from .mf5to6 import get_package_name
from .obs import setup_head_observations
from .tdis import setup_perioddata_group, get_parent_stress_periods
from .tmr import Tmr
from .units import lenuni_text, itmuni_text, convert_length_units, convert_time_units
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
                                     'ghb', 'sfr', 'lak',
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
        # loop thru LGR models and inactivate area of parent grid for each one
        lgr_idomain = np.ones(self.dis.idomain.array.shape, dtype=int)
        if isinstance(self.lgr, dict):
            for k, v in self.lgr.items():
                lgr_idomain[v.idomain == 0] = 0
        idomain_from_layer_elevations = make_idomain(self.dis.top.array,
                                                     self.dis.botm.array,
                                                     nodata=self._nodata_value,
                                                     minimum_layer_thickness=self.cfg['dis'].get('minimum_layer_thickness', 1),
                                                     drop_thin_cells=self._drop_thin_cells,
                                                     tol=1e-4)
        # include cells that are active in the existing idomain array
        # and cells inactivated on the basis of layer elevations
        idomain = (self.dis.idomain.array == 1) & \
                  (idomain_from_layer_elevations == 1) & \
                  (lgr_idomain == 1)
        idomain = idomain.astype(int)

        # remove cells that conincide with lakes
        idomain[self.isbc == 1] = 0.

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
        self._mg_resync = False

    def _set_parent(self):
        """Set attributes related to a parent or source model
        if one is specified.
        todo: move this method to mfmodel mixin class
        """

        if self.cfg['parent']['version'] == 'mf6':
            if 'lgr' in self.parent.cfg['setup_grid'].keys() and isinstance(self.parent, MF6model):
                if 'DIS' not in self.parent.get_package_list():
                    dis = self.parent.setup_dis()
                return
            else:
                raise NotImplementedError("TMR from MODFLOW-6 parent models")

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

            # parent model perioddata
            if not hasattr(self.parent, 'perioddata'):
                kwargs = self.cfg['parent'].copy()
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
                if self.cfg['dis']['dimensions'].get('nlay') is None:
                    self.cfg['dis']['dimensions']['nlay'] = self.parent.dis.nlay
                parent_start_date_time = self.cfg.get('parent', {}).get('start_date_time')
                if self.cfg['tdis'].get('start_date_time', '1970-01-01') == '1970-01-01' \
                        and parent_start_date_time is not None:
                    self.cfg['tdis']['start_date_time'] = self.cfg['parent']['start_date_time']

                # only get time dis information from parent if
                # no periodata groups are specified, and nper is not specified under dimensions
                has_perioddata_groups = any([isinstance(k, dict)
                                             for k in self.cfg['tdis']['perioddata'].values()])
                if not has_perioddata_groups:
                    if self.cfg['tdis']['dimensions'].get('nper') is None:
                        self.cfg['dis']['nper'] = self.parent.dis.nper
                    parent_periods = get_parent_stress_periods(self.parent, nper=self.cfg['dis']['nper'],
                                                               parent_stress_periods=self.cfg['parent'][
                                                                   'copy_stress_periods'])
                    for var in ['perlen', 'nstp', 'tsmult']:
                        if self.cfg['dis']['perioddata'].get(var) is None:
                            self.cfg['dis']['perioddata'][var] = self.parent.dis.__dict__[var].array[parent_periods]
                    if self.cfg['sto'].get('steady') is None:
                        self.cfg['sto']['steady'] = self.parent.dis.steady.array[parent_periods]

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

    def create_lgr_models(self):
        for k, v in self.cfg['setup_grid']['lgr'].items():
            # load the config file for lgr inset model
            inset_cfg = load_cfg(v['filename'],
                                 default_file='/mf6_defaults.yml')
            # if lgr inset has already been created
            if inset_cfg['model']['modelname'] in self.simulation._models:
                return
            inset_cfg['model']['simulation'] = self.simulation
            if 'ims' in inset_cfg['model']['packages']:
                inset_cfg['model']['packages'].remove('ims')
            # set parent configuation dictionary here
            # (even though parent model is explicitly set below)
            # so that the LGR grid is snapped to the parent grid
            inset_cfg['parent'] = {'namefile': self.namefile,
                                   'model_ws': self.model_ws,
                                   'version': 'mf6',
                                   'hiKlakes_value': self.cfg['model']['hiKlakes_value'],
                                   'default_source_data': True,
                                   'length_units': self.length_units,
                                   'time_units': self.time_units
                                   }
            inset_cfg = MF6model._parse_model_kwargs(inset_cfg)
            kwargs = get_input_arguments(inset_cfg['model'], mf6.ModflowGwf,
                                         exclude='packages')
            kwargs['parent'] = self  # otherwise will try to load parent model
            inset_model = MF6model(cfg=inset_cfg, **kwargs)
            inset_model.setup_grid()
            del inset_model.cfg['ims']
            inset_model.cfg['tdis'] = self.cfg['tdis']
            if self.inset is None:
                self.inset = {}
                self.lgr = {}
            self.inset[inset_model.name] = inset_model
            self.inset[inset_model.name]._is_lgr = True

            # create idomain indicating area of parent grid that is LGR
            lgr_idomain = make_lgr_idomain(self.modelgrid, self.inset[inset_model.name].modelgrid)

            ncpp = int(self.modelgrid.delr[0]/self.inset[inset_model.name].modelgrid.delr[0])
            ncppl = v.get('layer_refinement', 1)
            self.lgr[inset_model.name] = Lgr(self.nlay, self.nrow, self.ncol,
                                               self.dis.delr.array, self.dis.delc.array,
                                               self.dis.top.array, self.dis.botm.array,
                                               lgr_idomain, ncpp, ncppl)
            inset_model._perioddata = self.perioddata
            self._set_idomain()

    def setup_lgr_exchanges(self):
        for inset_name, inset_model in self.inset.items():
            # get the exchange data
            exchangelist = self.lgr[inset_name].get_exchange_data(angldegx=True, cdist=True)

            # make a dataframe for concise unpacking of cellids
            columns = ['cellidm1', 'cellidm2', 'ihc', 'cl1', 'cl2', 'hwva', 'angldegx', 'cdist']
            exchangedf = pd.DataFrame(exchangelist, columns=columns)

            # unpack the cellids and get their respective ibound values
            k1, i1, j1 = zip(*exchangedf['cellidm1'])
            k2, i2, j2 = zip(*exchangedf['cellidm2'])
            active1 = self.idomain[k1, i1, j1] == 1
            active2 = inset_model.idomain[k2, i2, j2] == 1

            # screen out connections involving an inactive cell
            active_connections = active1 & active2
            nexg = active_connections.sum()
            active_exchangelist = [l for i, l in enumerate(exchangelist) if active_connections[i]]

            # set up the exchange package
            gwfe = mf6.ModflowGwfgwf(self.simulation, exgtype='gwf6-gwf6',
                                     exgmnamea=self.name, exgmnameb=inset_name,
                                     nexg=nexg, auxiliary=[('angldegx', 'cdist')],
                                     exchangedata=active_exchangelist)

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
        #if not isinstance(self._modelgrid, MFsetupGrid):
        #    self._modelgrid = None  # override DIS package grid setup
        self._mg_resync = False
        self.setup_grid()  # reset the model grid
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
        # get steady/transient info from perioddata table
        # which parses it from either DIS or STO input (to allow consistent input structure with mf2005)
        kwargs['steady_state'] = {k: v for k, v in zip(self.perioddata['per'], self.perioddata['steady'])}
        kwargs['transient'] = {k: not v for k, v in zip(self.perioddata['per'], self.perioddata['steady'])}
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
        if len(df) == 0:
            print('No wells in active model area')
            return

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
                kspd['boundname'] = group['comments']
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

    def setup_lak(self):
        """
        Sets up the Lake package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'lak'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        if self.lakarr.sum() == 0:
            print("lakes_shapefile not specified, or no lakes in model area")
            return

        # source data
        source_data = self.cfg['lak']['source_data']

        # munge lake package input
        # returns dataframe with information for each lake
        self.lake_info = setup_lake_info(self)

        # returns dataframe with connection information
        connectiondata = setup_lake_connectiondata(self)
        nlakeconn = connectiondata.groupby('lakeno').count().iconn.to_dict()
        self.lake_info['nlakeconn'] = [nlakeconn[id - 1] for id in self.lake_info['lak_id']]

        # set up the tab files
        if 'stage_area_volume_file' in source_data:
            tab_files = setup_lake_tablefiles(self, source_data['stage_area_volume_file'])

            # tabfiles aren't rewritten by flopy on package write
            self.cfg['lak']['tab_files'] = tab_files
            # kludge to deal with ugliness of lake package external file handling
            # (need to give path relative to model_ws, not folder that flopy is working in)
            tab_files_argument = [os.path.relpath(f) for f in tab_files]

        # todo: implement lake outlets with SFR

        # perioddata
        self.lake_fluxes = setup_lake_fluxes(self)
        lakeperioddata = get_lakeperioddata(self.lake_fluxes)

        # set up input arguments
        kwargs = self.cfg[package].copy()
        options = self.cfg[package]['options'].copy()
        renames = {'budget_fileout': 'budget_filerecord',
                   'stage_fileout': 'stage_filerecord'}
        for k, v in renames.items():
            if k in options:
                options[v] = options.pop(k)
        kwargs.update(self.cfg[package]['options'])
        kwargs['time_conversion'] = convert_time_units(self.time_units, 'seconds')
        kwargs['length_conversion'] = convert_time_units(self.length_units, 'meters')
        kwargs['nlakes'] = len(self.lake_info)
        kwargs['noutlets'] = 0  # not implemented
        # [lakeno, strt, nlakeconn, aux, boundname]
        packagedata_cols = ['lak_id', 'strt', 'nlakeconn']
        if kwargs.get('boundnames'):
            packagedata_cols.append('name')
        packagedata = self.lake_info[packagedata_cols]
        packagedata['lak_id'] -= 1  # convert to zero-based
        kwargs['packagedata'] = packagedata.values.tolist()
        connectiondata_cols = ['lakeno', 'iconn', 'cellid', 'claktype', 'bedleak',
                               'belev', 'telev', 'connlen', 'connwidth']
        kwargs['connectiondata'] = connectiondata[connectiondata_cols].values.tolist()
        kwargs['ntables'] = len(tab_files)
        kwargs['tables'] = [(i, f) for i, f in enumerate(tab_files)]
        kwargs['outlets'] = None  # not implemented
        #kwargs['outletperioddata'] = None  # not implemented
        kwargs['perioddata'] = lakeperioddata

        # observations
        kwargs['observations'] = setup_mf6_lake_obs(kwargs)

        kwargs = get_input_arguments(kwargs, mf6.ModflowGwflak)
        lak = mf6.ModflowGwflak(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return lak

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
        #self.simulation.register_ims_package(ims, [self.name])
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
                  inset_parent_period_mapping=self.parent_stress_periods)

        df = tmr.get_inset_boundary_heads()

        spd = {}
        by_period = df.groupby('per')
        for per, df_per in by_period:
            maxbound = len(df_per)
            spd[per] = mf6.ModflowGwfchd.stress_period_data.empty(self, maxbound=maxbound)[0]
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

        if isinstance(cfg['model']['simulation'], str):
            # assume that simulation for model
            # is the one simulation specified in configuration
            # (regardless of the name specified in model configuration)
            cfg['model']['simulation'] = cfg['simulation']
        if isinstance(cfg['model']['simulation'], dict):
            # create simulation from simulation block in config dict
            sim = flopy.mf6.MFSimulation(**cfg['simulation'])
            cfg['model']['simulation'] = sim
            sim_ws = cfg['simulation']['sim_ws']
        # if a simulation has already been created, get the path from the instance
        elif isinstance(cfg['model']['simulation'], mf6.MFSimulation):
            sim_ws = cfg['model']['simulation'].simulation_data.mfpath._sim_path
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