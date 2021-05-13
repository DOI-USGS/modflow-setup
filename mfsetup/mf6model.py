import os
import shutil
import time
from collections import defaultdict

import flopy
import numpy as np
import pandas as pd

fm = flopy.modflow
mf6 = flopy.mf6
from flopy.utils.lgrutil import Lgr
from gisutils import get_values_at_points

from mfsetup.bcs import remove_inactive_bcs, setup_flopy_stress_period_data
from mfsetup.discretization import (
    ModflowGwfdis,
    create_vertical_pass_through_cells,
    deactivate_idomain_above,
    find_remove_isolated_cells,
    make_idomain,
    make_irch,
    make_lgr_idomain,
)
from mfsetup.fileio import (
    add_version_to_fileheader,
    flopy_mfsimulation_load,
    load,
    load_cfg,
)
from mfsetup.ic import setup_strt
from mfsetup.lakes import (
    get_lakeperioddata,
    setup_lake_connectiondata,
    setup_lake_fluxes,
    setup_lake_info,
    setup_lake_tablefiles,
    setup_mf6_lake_obs,
)
from mfsetup.mfmodel import MFsetupMixin
from mfsetup.mover import get_mover_sfr_package_input
from mfsetup.obs import setup_head_observations
from mfsetup.oc import parse_oc_period_input
from mfsetup.tdis import add_date_comments_to_tdis
from mfsetup.tmr import TmrNew
from mfsetup.units import convert_time_units
from mfsetup.utils import flatten, get_input_arguments
from mfsetup.wells import setup_wel_data


class MF6model(MFsetupMixin, mf6.ModflowGwf):
    """Class representing a MODFLOW-6 model.
    """
    default_file = '/mf6_defaults.yml'

    def __init__(self, simulation=None, parent=None, cfg=None,
                 modelname='model', exe_name='mf6',
                 version='mf6', lgr=False, **kwargs):
        defaults = {'simulation': simulation,
                    'parent': parent,
                    'modelname': modelname,
                    'exe_name': exe_name,
                    'version': version,
                    'lgr': lgr}
        # load configuration, if supplied
        if cfg is not None:
            if not isinstance(cfg, dict):
                cfg = self.load_cfg(cfg)
            cfg = self._parse_model_kwargs(cfg)
            defaults.update(cfg['model'])
            kwargs = {k: v for k, v in kwargs.items() if k not in defaults}
        # otherwise, pass arguments on to flopy constructor
        args = get_input_arguments(defaults, mf6.ModflowGwf,
                                     exclude='packages')
        mf6.ModflowGwf.__init__(self, **args, **kwargs)
        #mf6.ModflowGwf.__init__(self, simulation,
        #                        modelname, exe_name=exe_name, version=version,
        #                        **kwargs)
        MFsetupMixin.__init__(self, parent=parent)

        self._is_lgr = lgr
        self._package_setup_order = ['tdis', 'dis', 'ic', 'npf', 'sto', 'rch', 'oc',
                                     'chd', 'ghb', 'sfr', 'lak', 'riv',
                                     'wel', 'maw', 'obs']
        # set up the model configuration dictionary
        # start with the defaults
        self.cfg = load(self.source_path + self.default_file) #'/mf6_defaults.yml')
        self.relative_external_paths = self.cfg.get('model', {}).get('relative_external_paths', True)
        # set the model workspace and change working directory to there
        self.model_ws = self._get_model_ws(cfg=cfg)
        # update defaults with user-specified config. (loaded above)
        # set up and validate the model configuration dictionary
        self._set_cfg(cfg)

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
        so that cells above SFR reaches are inactive.

        Also remakes irch for the recharge package"""
        print('(re)setting the idomain array...')
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
        idomain = (self.dis.idomain.array >= 1) & \
                  (idomain_from_layer_elevations >= 1) & \
                  (lgr_idomain >= 1)
        idomain = idomain.astype(int)

        # remove cells that conincide with lakes
        idomain[self.isbc == 1] = 0.

        # remove cells that are above stream cells
        if self.get_package('sfr') is not None:
            idomain = deactivate_idomain_above(idomain, self.sfr.packagedata)

        # inactivate any isolated cells that could cause problems with the solution
        idomain = find_remove_isolated_cells(idomain, minimum_cluster_size=20)

        # create pass-through cells in inactive cells that have an active cell above and below
        # by setting these cells to -1
        idomain = create_vertical_pass_through_cells(idomain)

        self._idomain = idomain

        # take the updated idomain array and set cells != 1 to np.nan in layer botm array
        # including lake cells
        # effect is that the layer thicknesses in these cells will be set to zero
        # fill_cells_vertically will be run in the setup_array routine,
        # to collapse the nan cells to zero-thickness
        # (assign their layer botm to the next valid layer botm above)
        botm = self.dis.botm.array.copy()
        botm[(idomain != 1)] = np.nan

        # re-write the input files
        # todo: integrate this better with setup_dis
        # to reduce the number of times the arrays need to be remade

        self._setup_array('dis', 'botm',
                        data={i: arr for i, arr in enumerate(botm)},
                        datatype='array3d', resample_method='linear',
                        write_fmt='%.2f', dtype=float)
        self.dis.botm = self.cfg['dis']['griddata']['botm']
        self._setup_array('dis', 'idomain',
                          data={i: arr for i, arr in enumerate(idomain)},
                          datatype='array3d', resample_method='nearest',
                          write_fmt='%d', dtype=int)
        self.dis.idomain = self.cfg['dis']['griddata']['idomain']
        self._mg_resync = False
        self.setup_grid()  # reset the model grid

        # rebuild irch to keep it in sync with idomain changes
        irch = make_irch(idomain)
        self._setup_array('rch', 'irch',
                                data={0: irch},
                                datatype='array2d',
                                write_fmt='%d', dtype=int)
        #self.dis.irch = self.cfg['dis']['irch']

    def _update_grid_configuration_with_dis(self):
        """Update grid configuration with any information supplied to dis package
        (so that settings specified for DIS package have priority). This method
        is called by MFsetupMixin.setup_grid.
        """
        for param in ['nlay', 'nrow', 'ncol']:
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
                                      points_crs=self.modelgrid.crs,
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
            inset_model = MF6model(cfg=inset_cfg, lgr=True, **kwargs)
            inset_model._load = self._load  # whether model is being made or loaded from existing files
            inset_model.setup_grid()
            del inset_model.cfg['ims']
            inset_model.cfg['tdis'] = self.cfg['tdis']
            if self.inset is None:
                self.inset = {}
                self.lgr = {}

            self.inset[inset_model.name] = inset_model
            #self.inset[inset_model.name]._is_lgr = True

            # create idomain indicating area of parent grid that is LGR
            lgr_idomain = make_lgr_idomain(self.modelgrid, self.inset[inset_model.name].modelgrid)

            ncpp = int(self.modelgrid.delr[0] / self.inset[inset_model.name].modelgrid.delr[0])
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
            active1 = self.idomain[k1, i1, j1] >= 1
            active2 = inset_model.idomain[k2, i2, j2] >= 1

            # screen out connections involving an inactive cell
            active_connections = active1 & active2
            nexg = active_connections.sum()
            active_exchangelist = [l for i, l in enumerate(exchangelist) if active_connections[i]]

            # arguments to ModflowGwfgwf
            kwargs = {'exgtype': 'gwf6-gwf6',
                      'exgmnamea': self.name,
                      'exgmnameb': inset_name,
                      'nexg': nexg,
                      'auxiliary': [('angldegx', 'cdist')],
                      'exchangedata': active_exchangelist
                      }
            # add water mover files if there's a mover package
            # not sure if one mover file can be used for multiple exchange files or not
            # if separate (simulation) water mover files are needed,
            # than this would have to be restructured
            if 'mvr' in self.simulation.package_key_dict:
                if inset_name in self.simulation.mvr.packages.array['mname']:
                    kwargs['mvr_filerecord'] = self.simulation.mvr.filename

            # set up the exchange package
            kwargs = get_input_arguments(kwargs, mf6.ModflowGwfgwf)
            gwfe = mf6.ModflowGwfgwf(self.simulation, **kwargs)

    def setup_dis(self):
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

        # set number of layers to length of the created bottom array
        # this needs to be set prior to setting up the idomain,
        # otherwise idomain may have wrong number of layers
        self.cfg['dis']['dimensions']['nlay'] = len(self.cfg['dis']['griddata']['botm'])

        # initial idomain input for creating a dis package instance
        self._setup_array(package, 'idomain', datatype='array3d', write_fmt='%d',
                          resample_method='nearest',
                          dtype=int)

        # put together keyword arguments for dis package
        kwargs = self.cfg['grid'].copy() # nrow, ncol, delr, delc
        kwargs.update(self.cfg['dis'])
        kwargs.update(self.cfg['dis']['dimensions']) # nper, nlay, etc.
        kwargs.update(self.cfg['dis']['griddata'])

        # modelgrid: dis arguments
        remaps = {'xoff': 'xorigin',
                  'yoff': 'yorigin',
                  'rotation': 'angrot'}

        for k, v in remaps.items():
            if v not in kwargs:
                kwargs[v] = kwargs.pop(k)
        kwargs['length_units'] = self.length_units
        # get the arguments for the flopy version of ModflowGwfdis
        # but instantiate with modflow-setup subclass of ModflowGwfdis
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfdis)
        dis = ModflowGwfdis(model=self, **kwargs)
        self._perioddata = None  # reset perioddata
        #if not isinstance(self._modelgrid, MFsetupGrid):
        #    self._modelgrid = None  # override DIS package grid setup
        self._mg_resync = False
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

        kwargs = self.cfg[package]
        kwargs.update(self.cfg[package]['griddata'])
        kwargs['source_data_config'] = kwargs['source_data']
        kwargs['filename_fmt'] = kwargs['strt_filename_fmt']

        # make the starting heads array
        strt = setup_strt(self, package, **kwargs)

        ic = mf6.ModflowGwfic(self, strt=strt)
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
                          resample_method='linear',
                          datatype='array3d', write_fmt='%.6e')

        # make the k33 array (kv)
        self._setup_array(package, 'k33', vmin=0, vmax=hiKlakes_value,
                          resample_method='linear',
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
        self._setup_array(package, 'sy', datatype='array3d', resample_method='linear',
                          write_fmt='%.6e')

        # make the ss array
        self._setup_array(package, 'ss', datatype='array3d', resample_method='linear',
                          write_fmt='%.6e')

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
        irch = make_irch(self.idomain)

        self._setup_array('rch', 'irch',
                          data={0: irch},
                          datatype='array2d',
                          write_fmt='%d', dtype=int)

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

        # option to write stress_period_data to external files
        external_files = self.cfg[package]['external_files']

        # munge well package input
        # returns dataframe with information to populate stress_period_data
        df = setup_wel_data(self, for_external_files=external_files)
        if len(df) == 0:
            print('No wells in active model area')
            return

        # set up stress_period_data
        if external_files:
            # get the file path (allowing for different external file locations, specified name format, etc.)
            filename_format = self.cfg[package]['external_filename_fmt']
            filepaths = self.setup_external_filepaths(package, 'stress_period_data',
                                                      filename_format=filename_format,
                                                      file_numbers=sorted(df.per.unique().tolist()))
        spd = {}
        period_groups = df.groupby('per')
        for kper in range(self.nper):
            if kper in period_groups.groups:
                group = period_groups.get_group(kper)
                group.drop('per', axis=1, inplace=True)
                if external_files:
                    group.to_csv(filepaths[kper]['filename'], index=False, sep=' ', float_format='%g')
                    # make a copy for the intermediate data folder, for consistency with mf-2005
                    shutil.copy(filepaths[kper]['filename'], self.cfg['intermediate_data']['output_folder'])
                else:
                    kspd = mf6.ModflowGwfwel.stress_period_data.empty(self,
                                                                      len(group),
                                                                      boundnames=True)[0]
                    kspd['cellid'] = list(zip(group.k, group.i, group.j))
                    kspd['q'] = group['q']
                    kspd['boundname'] = group['boundname']
                    spd[kper] = kspd
            else:
                pass  # spd[kper] = None
        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        if not external_files:
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

        # option to write connectiondata to external file
        external_files = self.cfg['lak']['external_files']
        horizontal_connections = self.cfg['lak']['horizontal_connections']


        # source data
        source_data = self.cfg['lak']['source_data']

        # munge lake package input
        # returns dataframe with information for each lake
        self.lake_info = setup_lake_info(self)

        # returns dataframe with connection information
        connectiondata = setup_lake_connectiondata(self, for_external_file=external_files,
                                                   include_horizontal_connections=horizontal_connections)
        # lakeno column will have # in front if for_external_file=True
        lakeno_col = [c for c in connectiondata.columns if 'lakeno' in c][0]
        nlakeconn = connectiondata.groupby(lakeno_col).count().iconn.to_dict()
        offset = 0 if external_files else 1
        self.lake_info['nlakeconn'] = [nlakeconn[id - offset] for id in self.lake_info['lak_id']]

        # set up the tab files
        if 'stage_area_volume_file' in source_data:
            tab_files = setup_lake_tablefiles(self, source_data['stage_area_volume_file'])

            # tabfiles aren't rewritten by flopy on package write
            self.cfg['lak']['tab_files'] = tab_files
            # kludge to deal with ugliness of lake package external file handling
            # (need to give path relative to model_ws, not folder that flopy is working in)
            tab_files_argument = [os.path.relpath(f) for f in tab_files]
        else:
            tab_files = None
        # todo: implement lake outlets with SFR

        # perioddata
        self.lake_fluxes = setup_lake_fluxes(self)
        lakeperioddata = get_lakeperioddata(self.lake_fluxes)

        # set up external files
        connectiondata_cols = [lakeno_col, 'iconn', 'k', 'i', 'j', 'claktype', 'bedleak',
                               'belev', 'telev', 'connlen', 'connwidth']
        if external_files:
            # get the file path (allowing for different external file locations, specified name format, etc.)
            filepath = self.setup_external_filepaths(package, 'connectiondata',
                                                     self.cfg[package]['connectiondata_filename_fmt'])
            connectiondata[connectiondata_cols].to_csv(filepath[0]['filename'], index=False, sep=' ')
            # make a copy for the intermediate data folder, for consistency with mf-2005
            shutil.copy(filepath[0]['filename'], self.cfg['intermediate_data']['output_folder'])
        else:
            connectiondata_cols = connectiondata_cols[:2] + ['cellid'] + connectiondata_cols[5:]
            self.cfg[package]['connectiondata'] = connectiondata[connectiondata_cols].values.tolist()

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
        if  tab_files != None:
            kwargs['ntables'] = len(tab_files)
            kwargs['tables'] = [(i, f, 'junk', 'junk') for i, f in enumerate(tab_files)]
        kwargs['outlets'] = None  # not implemented
        #kwargs['outletperioddata'] = None  # not implemented
        kwargs['perioddata'] = lakeperioddata

        # observations
        kwargs['observations'] = setup_mf6_lake_obs(kwargs)

        kwargs = get_input_arguments(kwargs, mf6.ModflowGwflak)
        lak = mf6.ModflowGwflak(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return lak

    def setup_riv(self, rivdata=None):
        """Set up River package.
        TODO: riv package input through configuration file
        """
        package = 'riv'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        if rivdata is None:
            raise NotImplementedError("River package input through configuration file;"
                                      "currently only supported through to_riv option"
                                      "in sfr configuration block.")
        df = rivdata.stress_period_data
        if len(df) == 0:
            print('No input specified or streams not in model.')
            return

        # option to write stress_period_data to external files
        external_files = self.cfg[package].get('external_files', True)

        if external_files:
            # get the file path (allowing for different external file locations, specified name format, etc.)
            filename_format = package + '_{:03d}.dat'  # stress period suffix
            filepaths = self.setup_external_filepaths(package, 'stress_period_data',
                                                      filename_format=filename_format,
                                                      file_numbers=sorted(df.per.unique().tolist()))

        spd = {}
        by_period = df.groupby('per')
        for kper, df_per in by_period:
            if external_files:
                df_per = df_per[['k', 'i', 'j', 'stage', 'cond', 'rbot']].copy()
                for col in 'k', 'i', 'j':
                    df_per[col] += 1
                df_per.rename(columns={'k': '#k'}, inplace=True)
                df_per.to_csv(filepaths[kper]['filename'], index=False, sep=' ')
                # make a copy for the intermediate data folder, for consistency with mf-2005
                shutil.copy(filepaths[kper]['filename'], self.cfg['intermediate_data']['output_folder'])
            else:
                maxbound = len(df_per)
                spd[kper] = mf6.ModflowGwfchd.stress_period_data.empty(self, maxbound=maxbound,
                                                                       boundnames=True)[0]
                spd[kper]['cellid'] = list(zip(df_per['k'], df_per['i'], df_per['j']))
                for col in 'cond', 'stage', 'rbot':
                    spd[kper][col] = df_per[col]
                spd[kper]['boundname'] = ["'{}'".format(s) for s in df_per['name']]

        kwargs = self.cfg['riv']
        # need default options from rivdata instance or cfg defaults
        kwargs.update(self.cfg['riv']['options'])
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfriv)
        if not external_files:
            kwargs['stress_period_data'] = spd
        riv = mf6.ModflowGwfriv(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return riv

    def setup_chd(self):
        """
        Sets up the CHD package.

        Parameters
        ----------

        Notes
        -----

        """
        package = 'chd'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        package_config = self.cfg[package]

        # option to write stress_period_data to external files
        external_files = package_config['external_files']

        # perimeter boundary
        if 'perimeter_boundary' in package_config:
            perimeter_cfg = package_config['perimeter_boundary']
            perimeter_cfg['boundary_type'] = 'head'
            if 'inset_parent_period_mapping' not in perimeter_cfg:
                perimeter_cfg['inset_parent_period_mapping'] = self.parent_stress_periods
            if 'parent_start_time' not in perimeter_cfg:
                perimeter_cfg['parent_start_date_time'] = self.parent.perioddata['start_datetime'][0]
            self.tmr = TmrNew(self.parent, self, **perimeter_cfg)
            perimeter_df = self.tmr.get_inset_boundary_values()

            # add boundname to allow boundary flux to be tracked as observation
            perimeter_df['boundname'] = 'perimeter-heads'

            # get the stress period data
            # this also sets up the external file paths
            spd = setup_flopy_stress_period_data(self, package, perimeter_df,
                                                 flopy_package_class=mf6.ModflowGwfchd,
                                                 variable_column='head',
                                                 external_files=external_files,
                                                 external_filename_fmt=package_config['external_filename_fmt'])

        # placeholder for setting up user-specified CHD cells from CSV data
        # todo: support for non-perimeter chd cells
        df = pd.DataFrame()  # insert function here to get csv data into dataframe
        if len(df) == 0:
            print('No other CHD input specified')
            if 'perimeter_boundary' not in package_config:
                return

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        if not external_files:
            kwargs['stress_period_data'] = spd

        # add observation for flux thru perimeter heads
        if 'perimeter_boundary' in package_config:
            kwargs['observations'] = {f'{self.name}.chd.obs.output.csv':
                                          [('perimeter-heads', 'chd', 'perimeter-heads')]}
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfchd)
        chd = mf6.ModflowGwfchd(self, **kwargs)
        print("setup of chd took {:.2f}s\n".format(time.time() - t0))
        return chd

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

        period_input = parse_oc_period_input(kwargs)
        kwargs.update(period_input)

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

    def setup_simulation_mover(self):
        """Set up the MODFLOW-6 water mover package at the simulation level.
        Automate set-up of the mover between SFR packages in LGR parent and inset models.
        todo: automate set-up of mover between SFR and lakes (within a model).

        Other uses of the water mover need to be configured manually using flopy.
        """
        package = 'mvr'
        print('\nSetting up the simulation water mover package...')
        t0 = time.time()

        perioddata_dfs = []
        if self.get_package('sfr') is not None:
            if self.inset is not None:
                for inset_name, inset in self.inset.items():
                    if inset.get_package('sfr'):
                        inset_perioddata = get_mover_sfr_package_input(self, inset)
                        perioddata_dfs.append(inset_perioddata)
        if len(perioddata_dfs) > 0:
            perioddata = pd.concat(perioddata_dfs)
            if len(perioddata) > 0:
                kwargs = flatten(self.cfg[package])
                # modelnames (boolean) keyword to indicate that all package names will
                # be preceded by the model name for the package. Model names are
                # required when the Mover Package is used with a GWF-GWF Exchange. The
                # MODELNAME keyword should not be used for a Mover Package that is for
                # a single GWF Model.
                # this argument will need to be adapted for implementing a mover package within a model
                # (between lakes and sfr)
                kwargs['modelnames'] = True
                kwargs['maxmvr'] = len(perioddata)  # assumes that input for period 0 applies to all periods
                packages = set(list(zip(perioddata.mname1, perioddata.pname1)) +
                               list(zip(perioddata.mname2, perioddata.pname2)))
                kwargs['maxpackages'] = len(packages)
                kwargs['packages'] = list(packages)
                kwargs['perioddata'] = {0: perioddata.values.tolist()}  # assumes that input for period 0 applies to all periods
                kwargs = get_input_arguments(kwargs, mf6.ModflowGwfmvr)
                mvr = mf6.ModflowMvr(self.simulation, **kwargs)
                print("finished in {:.2f}s\n".format(time.time() - t0))
                return mvr
        else:
            print("no packages with mover information\n")

    def write_input(self):
        """Write the model input.
        """
        # prior to writing output
        # remove any BCs in inactive cells
        pckgs = ['CHD']
        for pckg in pckgs:
            package_instance = getattr(self, pckg.lower(), None)
            if package_instance is not None:
                external_files = self.cfg[pckg.lower()]['stress_period_data']
                remove_inactive_bcs(package_instance,
                                    external_files=external_files)

        # write the model with flopy
        # but skip the sfr package
        # by monkey-patching the write method
        def skip_write(**kwargs):
            pass
        if hasattr(self, 'sfr'):
            self.sfr.write = skip_write
        self.simulation.write_simulation()

        # write the sfr package with SFRmaker
        if 'SFR' in ' '.join(self.get_package_list()):
            options = []
            for k, b in self.cfg['sfr']['options'].items():
                if k == 'mover':
                    if 'mvr' not in self.simulation.package_key_dict:
                        continue
                options.append(k)
            if 'save_flows' in options:
                budget_fileout = '{}.{}'.format(self.name,
                                                self.cfg['sfr']['budget_fileout'])
                stage_fileout = '{}.{}'.format(self.name,
                                               self.cfg['sfr']['stage_fileout'])
                options.append('budget fileout {}'.format(budget_fileout))
                options.append('stage fileout {}'.format(stage_fileout))
            if len(self.sfrdata.observations) > 0:
                options.append('obs6 filein {}.{}'.format(self.name,
                                                          self.cfg['sfr']['obs6_filein_fmt'])
                               )
            self.sfrdata.write_package(idomain=self.idomain,
                                       version='mf6',
                                       options=options,
                                       external_files_path=self.external_path
                                       )

        # label stress periods in tdis file with comments
        self.perioddata.sort_values(by='per', inplace=True)
        add_date_comments_to_tdis(self.simulation.tdis.filename,
                                  self.perioddata.start_datetime,
                                  self.perioddata.end_datetime
                                  )
        # add version info to file headers
        files = [self.namefile]
        files += [p.filename for p in self.packagelist]
        files += [p.filename for k, p in self.simulation.package_key_dict.items()]
        for f in files:
            add_version_to_fileheader(f, model_info=self.header)

    @staticmethod
    def _parse_model_kwargs(cfg):

        if isinstance(cfg['model']['simulation'], str):
            # assume that simulation for model
            # is the one simulation specified in configuration
            # (regardless of the name specified in model configuration)
            cfg['model']['simulation'] = cfg['simulation']
        if isinstance(cfg['model']['simulation'], dict):
            # create simulation from simulation block in config dict
            kwargs = cfg['simulation'].copy()
            kwargs.update(cfg['simulation']['options'])
            kwargs = get_input_arguments(kwargs, mf6.MFSimulation)
            sim = flopy.mf6.MFSimulation(**kwargs)
            cfg['model']['simulation'] = sim
            sim_ws = cfg['simulation']['sim_ws']
        # if a simulation has already been created, get the path from the instance
        elif isinstance(cfg['model']['simulation'], mf6.MFSimulation):
            sim_ws = cfg['model']['simulation'].simulation_data.mfpath._sim_path
        else:
            raise TypeError('unrecognized configuration input for simulation.')

        # listing file
        cfg['model']['list'] = os.path.join(cfg['model']['list_filename_fmt']
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
        print('\nLoading simulation in {}\n'.format(yamlfile))
        t0 = time.time()

        #cfg = load_cfg(yamlfile, verbose=verbose, default_file=cls.default_file) # '/mf6_defaults.yml')
        #cfg = cls._parse_model_kwargs(cfg)
        #kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
        #                             exclude='packages')
        #model = cls(cfg=cfg, **kwargs)
        model = cls(cfg=yamlfile)
        model._load = True
        if 'grid' not in model.cfg.keys():
            model.setup_grid()
        sim = model.cfg['model']['simulation']  # should be a flopy.mf6.MFSimulation instance
        models = [model]
        if isinstance(model.inset, dict):
            for inset_name, inset in model.inset.items():
                models.append(inset)

        # execute the flopy load code on the pre-defined simulation and model instances
        # (so that the end result is a MFsetup.MF6model instance)
        # (kludgy)
        sim = flopy_mfsimulation_load(sim, models, load_only=load_only)

        # just return the parent model (inset models should be attached through the inset attribute,
        # in addition to through the .simulation flopy attribute)
        m = sim.get_model(model_name=model.name)
        print('finished loading model in {:.2f}s'.format(time.time() - t0))
        return m
