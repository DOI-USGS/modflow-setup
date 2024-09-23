import copy
import os
import shutil
import time
from pathlib import Path

import flopy
import numpy as np
import pandas as pd

fm = flopy.modflow
mf6 = flopy.mf6
from flopy.utils.lgrutil import Lgr
from gisutils import get_values_at_points

from mfsetup.bcs import remove_inactive_bcs
from mfsetup.discretization import (
    ModflowGwfdis,
    create_vertical_pass_through_cells,
    deactivate_idomain_above,
    find_remove_isolated_cells,
    make_idomain,
    make_irch,
    make_lgr_idomain,
)
from mfsetup.fileio import add_version_to_fileheader, flopy_mfsimulation_load
from mfsetup.fileio import load as load_config
from mfsetup.fileio import load_cfg
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
from mfsetup.obs import remove_inactive_obs, setup_head_observations
from mfsetup.oc import parse_oc_period_input
from mfsetup.tdis import add_date_comments_to_tdis, setup_perioddata
from mfsetup.units import convert_time_units
from mfsetup.utils import flatten, get_input_arguments


class MF6model(MFsetupMixin, mf6.ModflowGwf):
    """Class representing a MODFLOW-6 model.
    """
    default_file = 'mf6_defaults.yml'

    def __init__(self, simulation=None, modelname='model', parent=None, cfg=None,
                 exe_name='mf6', load=False,
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
                                     'chd', 'drn', 'ghb', 'sfr', 'lak', 'riv',
                                     'wel', 'maw', 'obs']
        # set up the model configuration dictionary
        # start with the defaults
        self.cfg = load_config(self.source_path / self.default_file) #'mf6_defaults.yml')
        self.relative_external_paths = self.cfg.get('model', {}).get('relative_external_paths', True)
        # set the model workspace and change working directory to there
        self.model_ws = self._get_model_ws(cfg=cfg)
        # update defaults with user-specified config. (loaded above)
        # set up and validate the model configuration dictionary
        self._load = load  # whether the model is being created or loaded
        self._set_cfg(cfg)

        # property attributes
        self._idomain = None

        # other attributes
        self._features = {} # dictionary for caching shapefile datasets in memory
        self._drop_thin_cells = self.cfg.get('dis', {}).get('drop_thin_cells', True)

        # arrays remade during this session
        self.updated_arrays = set()

        # delete the temporary 'original-files' folder
        # if it already exists, to avoid side effects from stale files
        if not self._is_lgr:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

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
        """
        if self._perioddata is None:
            # check first for already loaded time discretization info
            try:
                tdis_perioddata_config = {col: getattr(self.modeltime, col)
                                          for col in ['perlen', 'nstp', 'tsmult']}
                nper = self.modeltime.nper
                steady = self.modeltime.steady_state
                default_start_datetime = self.modeltime.start_datetime
            except:
                tdis_perioddata_config = self.cfg['tdis']['perioddata']
                default_start_datetime = self.cfg['tdis']['options'].get('start_date_time',
                                                                         '1970-01-01')
                #tdis_dimensions_config = self.cfg['tdis']['dimensions']
                nper = self.cfg['tdis']['dimensions'].get('nper')
                # steady can be input in either the tdis or sto input blocks
                steady = self.cfg['tdis'].get('steady')
                if steady is None:
                    steady = self.cfg['sto'].get('steady')

            parent_stress_periods = self.cfg.get('parent').get('copy_stress_periods')
            perioddata = setup_perioddata(
                    self,
                    tdis_perioddata_config=tdis_perioddata_config,
                    default_start_datetime=default_start_datetime,
                    nper=nper, steady=steady,
                    time_units=self.time_units,
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
            self._lgr_idomain2d = lgr_idomain[0]
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
        # idomain[self.isbc == 1] = 0.

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
            if 'filename' in v:
                inset_cfg = load_cfg(v['filename'],
                                    default_file='mf6_defaults.yml')
            elif 'cfg' in v:
                inset_cfg = copy.deepcopy(v['cfg'])
            else:
                raise ValueError('Unrecognized input in subblock lgr: '
                                 'Supply either a configuration filename: '
                                 'or additional yaml configuration under cfg:'
                                 )
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
            inset_model = MF6model(cfg=inset_cfg, lgr=True, load=self._load, **kwargs)
            #inset_model._load = self._load  # whether model is being made or loaded from existing files
            inset_model.setup_grid()
            del inset_model.cfg['ims']
            inset_model.cfg['tdis'] = self.cfg['tdis']
            if self.inset is None:
                self.inset = {}
                self.lgr = {}

            self.inset[inset_model.name] = inset_model
            #self.inset[inset_model.name]._is_lgr = True

            # establish inset model layering within parent model
            parent_start_layer = v.get('parent_start_layer', 0)
            # parent_end_layer is specified as the last zero-based
            # parent layer that includes LGR refinement (not as a slice end)
            parent_end_layer = v.get('parent_end_layer', self.nlay - 1)
            # the layer refinement can be specified as an int, a list or a dict
            ncppl_input = v.get('layer_refinement', 1)
            if np.isscalar(ncppl_input):
                ncppl = np.array([0] * self.modelgrid.nlay)
                ncppl[parent_start_layer:parent_end_layer+1] = ncppl_input
            elif isinstance(ncppl_input, list):
                if not len(ncppl_input) == self.modelgrid.nlay:
                    raise ValueError(
                        "Configuration input: layer_refinement specified as"
                        "a list must include a value for every layer."
                    )
                ncppl = ncppl_input.copy()
            elif isinstance(ncppl_input, dict):
                ncppl = [ncppl_input.get(i, 0) for i in range(self.modelgrid.nlay)]
            else:
                raise ValueError("Configuration input: Unsupported input for "
                                 "layer_refinement: supply an int, list or dict.")

            # refined layers must be consecutive, starting from layer 1
            is_refined = (np.array(ncppl) > 0).astype(int)
            last_refined_layer = max(np.where(is_refined > 0)[0])
            consecutive = all(np.diff(is_refined)[:last_refined_layer] == 0)
            if (is_refined[0] != 1) | (not consecutive):
                raise ValueError("Configuration input: layer_refinement must "
                                 "include consecutive sequence of layers, "
                                 "starting with the top layer.")
            # check the specified DIS package input is consistent
            # with the specified layer_refinement
            specified_nlay_dis = inset_cfg['dis']['dimensions'].get('nlay')
            # skip this check if nlay hasn't been entered into the configuration file yet
            if specified_nlay_dis and (np.sum(ncppl) != specified_nlay_dis):
                raise ValueError(
                    f"Configuration input: layer_refinement  of {ncppl} "
                    f"implies {is_refined.sum()} inset model layers.\n"
                    f"{specified_nlay_dis} inset model layers specified in DIS package.")
            # mapping between parent and inset model layers
            # that is used for copying input from parent model
            inset_parent_layer_mapping = dict()
            inset_k = -1
            for parent_k, n_inset_lay in enumerate(ncppl):
                for i in range(n_inset_lay):
                    inset_k += 1
                    inset_parent_layer_mapping[inset_k] = parent_k
            self.inset[inset_model.name].cfg['parent']['inset_layer_mapping'] =\
                inset_parent_layer_mapping
            # create idomain indicating area of parent grid that is LGR
            lgr_idomain = make_lgr_idomain(self.modelgrid, self.inset[inset_model.name].modelgrid,
                                           ncppl)

            # inset model horizontal refinement from parent resolution
            refinement = self.modelgrid.delr[0] / self.inset[inset_model.name].modelgrid.delr[0]
            if not np.round(refinement, 4).is_integer():
                raise ValueError(f"LGR inset model spacing must be a factor of the parent model spacing.")
            ncpp = int(refinement)
            self.lgr[inset_model.name] = Lgr(self.nlay, self.nrow, self.ncol,
                                             self.dis.delr.array, self.dis.delc.array,
                                             self.dis.top.array, self.dis.botm.array,
                                             lgr_idomain, ncpp, ncppl)
            inset_model._perioddata = self.perioddata
            # set parent model top in LGR area to bottom of LGR area
            # this is an initial draft;
            # bottom elevations are readjusted in sourcedata.py
            # when inset model DIS package botm array is set up
            # (set to mean of inset model bottom elevations
            #  within each parent cell)
            # number of layers in parent model with LGR
            n_parent_lgr_layers = np.sum(np.array(ncppl) > 0)
            lgr_area = self.lgr[inset_model.name].idomain == 0
            self.dis.top[lgr_area[0]] =\
                self.lgr[inset_model.name].botmp[n_parent_lgr_layers -1][lgr_area[0]]
            # set parent model layers in LGR area to zero-thickness
            new_parent_botm = self.dis.botm.array.copy()
            for k in range(n_parent_lgr_layers):
                new_parent_botm[k][lgr_area[0]] = self.dis.top[lgr_area[0]]
            self.dis.botm = new_parent_botm
            self._update_top_botm_external_files()


    def _update_top_botm_external_files(self):
        """Update the external files after assigning new elevations to the
        Discretization Package top and botm arrays; adjust idomain as needed."""
        # reset the model top
        # (this step may not be needed if the "original top" functionality
        # is limited to cases where there is a lake package,
        # or if the "original top"/"lake bathymetry" functionality is eliminated
        # and we instead require the top to be pre-processed)
        original_top_file = Path(self.external_path,
                    f"{self.name}_{self.cfg['dis']['top_filename_fmt']}.original")
        original_top_file.unlink(missing_ok=True)
        self._setup_array('dis', 'top',
                            data={0: self.dis.top.array},
                datatype='array2d', resample_method='linear',
                write_fmt='%.2f', dtype=float)
        # _set_idomain() regerates external files for bottom array
        self._set_idomain()


    def setup_lgr_exchanges(self):

        for inset_name, inset_model in self.inset.items():

            # update cell information for computing any bottom exchanges
            self.lgr[inset_name].top = inset_model.dis.top.array
            self.lgr[inset_name].botm = inset_model.dis.botm.array
            # update only the layers of the parent model below the child model
            parent_top_below_child = np.sum(self.lgr[inset_name].ncppl > 0) -1
            self.lgr[inset_name].botmp[parent_top_below_child:] =\
                self.dis.botm.array[parent_top_below_child:]

            # get the exchange data
            exchangelist = self.lgr[inset_name].get_exchange_data(angldegx=True, cdist=True)

            # make a dataframe for concise unpacking of cellids
            columns = ['cellidm1', 'cellidm2', 'ihc', 'cl1', 'cl2', 'hwva', 'angldegx', 'cdist']
            exchangedf = pd.DataFrame(exchangelist, columns=columns)

            # unpack the cellids and get their respective ibound values
            k1, i1, j1 = zip(*exchangedf['cellidm1'])
            k2, i2, j2 = zip(*exchangedf['cellidm2'])
            # limit connections to
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
            kwargs = get_input_arguments(kwargs, mf6.ModflowGwfgwf)

            # set up the exchange package
            gwfgwf = mf6.ModflowGwfgwf(self.simulation, **kwargs)

            # set up a Mover Package if needed
            self.setup_simulation_mover(gwfgwf)


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
        self._mg_resync = False
        self._reset_bc_arrays()
        self._set_idomain()
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return dis

    #def setup_tdis(self):
    def setup_tdis(self, **kwargs):
        """
        Sets up the TDIS package.
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

    def setup_ic(self, **kwargs):
        """
        Sets up the IC package.
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

    def setup_npf(self, **kwargs):
        """
        Sets up the NPF package.
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

    def setup_sto(self, **kwargs):
        """
        Sets up the STO package.
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
        kwargs['steady_state'] = {k: v for k, v in zip(self.perioddata['per'], self.perioddata['steady']) if v}
        kwargs['transient'] = {k: not v for k, v in zip(self.perioddata['per'], self.perioddata['steady'])}
        kwargs = get_input_arguments(kwargs, mf6.ModflowGwfsto)
        sto = mf6.ModflowGwfsto(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return sto

    def setup_rch(self, **kwargs):
        """
        Sets up the RCH package.
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

    def setup_lak(self, **kwargs):
        """
        Sets up the Lake package.
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
            kwargs['tables'] = [(i, f)  #, 'junk', 'junk')
                                for i, f in enumerate(tab_files)]
        kwargs['outlets'] = None  # not implemented
        #kwargs['outletperioddata'] = None  # not implemented
        kwargs['perioddata'] = lakeperioddata

        # observations
        kwargs['observations'] = setup_mf6_lake_obs(kwargs)

        kwargs = get_input_arguments(kwargs, mf6.ModflowGwflak)
        lak = mf6.ModflowGwflak(self, **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return lak


    def setup_chd(self, **kwargs):
        """Set up the CHD Package.
        """
        return self._setup_basic_stress_package(
            'chd', mf6.ModflowGwfchd, ['head'], **kwargs)


    def setup_drn(self, **kwargs):
        """Set up the Drain Package.
        """
        return self._setup_basic_stress_package(
            'drn', mf6.ModflowGwfdrn, ['elev', 'cond'], **kwargs)


    def setup_ghb(self, **kwargs):
        """Set up the General Head Boundary Package.
        """
        return self._setup_basic_stress_package(
            'ghb', mf6.ModflowGwfghb, ['bhead', 'cond'], **kwargs)


    def setup_riv(self, rivdata=None, **kwargs):
        """Set up the River Package.
        """
        return self._setup_basic_stress_package(
            'riv', mf6.ModflowGwfriv, ['stage', 'cond', 'rbot'],
            rivdata=rivdata, **kwargs)


    def setup_wel(self, **kwargs):
        """Set up the Well Package.
        """
        return self._setup_basic_stress_package(
            'wel', mf6.ModflowGwfwel, ['q'], **kwargs)


    def setup_obs(self, **kwargs):
        """
        Sets up the OBS utility.
        """
        package = 'obs'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()

        iobs_domain = None
        if not kwargs['mfsetup_options']['allow_obs_in_bc_cells']:
            # for now, discard any head observations in same (i, j) column of cells
            # as a non-well boundary condition
            # including lake package lakes and non lake, non well BCs
            # (high-K lakes are excluded, since we may want head obs at those locations,
            #  to serve as pseudo lake stage observations)
            iobs_domain = ~((self.isbc == 1) | np.any(self.isbc > 2, axis=0))

        # munge the observation data
        df = setup_head_observations(self,
                                     obs_package=package,
                                     obsname_column='obsname',
                                     iobs_domain=iobs_domain,
                                     **kwargs['source_data'],
                                     **kwargs['mfsetup_options'])

        # reformat to flopy input format
        obsdata = df[['obsname', 'obstype', 'id']].to_records(index=False)
        filename = self.cfg[package]['mfsetup_options']['filename_fmt'].format(self.name)
        obsdata = {filename: obsdata}

        kwargs = self.cfg[package].copy()
        kwargs.update(self.cfg[package]['options'])
        kwargs['continuous'] = obsdata
        kwargs = get_input_arguments(kwargs, mf6.ModflowUtlobs)
        obs = mf6.ModflowUtlobs(self,  **kwargs)
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return obs

    def setup_oc(self, **kwargs):
        """
        Sets up the OC package.
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
        """
        package = 'ims'
        print('\nSetting up {} package...'.format(package.upper()))
        t0 = time.time()
        kwargs = flatten(self.cfg[package])
        # renames to cover difference between mf6: flopy input
        renames = {'csv_outer_output': 'csv_outer_output_filerecord',
                   'csv_inner_output': 'csv_outer_inner_filerecord'
                   }
        for k, v in renames.items():
            if k in kwargs:
                kwargs[v] = kwargs[k]
        kwargs = get_input_arguments(kwargs, mf6.ModflowIms)
        ims = mf6.ModflowIms(self.simulation, **kwargs)
        #self.simulation.register_ims_package(ims, [self.name])
        print("finished in {:.2f}s\n".format(time.time() - t0))
        return ims

    def setup_simulation_mover(self, gwfgwf):
        """Set up the MODFLOW-6 water mover package at the simulation level.
        Automate set-up of the mover between SFR packages in LGR parent and inset models.
        todo: automate set-up of mover between SFR and lakes (within a model).

        Parameters
        ----------
        gwfgwf : Flopy :class:`~flopy.mf6.modflow.mfgwfgwf.ModflowGwfgwf` package instance

        Notes
        ------
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
                        inset_perioddata = get_mover_sfr_package_input(
                            self, inset, gwfgwf.exchangedata.array)
                        perioddata_dfs.append(inset_perioddata)
                        # for each SFR reach with a connection
                        # to a reach in another model
                        # set the SFR Package downstream connection to 0
                        for i, r in inset_perioddata.iterrows():
                            rd = self.simulation.get_model(r['mname1']).sfrdata.reach_data
                            rd.loc[rd['rno'] == r['id1']+1, 'outreach'] = 0
                            # fix flopy connectiondata as well
                            sfr_package = self.simulation.get_model(r['mname1']).sfr
                            cd = sfr_package.connectiondata.array.tolist()
                            # there should be no downstream reaches
                            # (indicated by negative numbers)
                            cd[r['id1']] = tuple(v for v in cd[r['id1']] if v > 0)
                            sfr_package.connectiondata = cd
                        # re-write the shapefile exports with corrected routing
                        inset.sfrdata.write_shapefiles(f'{inset._shapefiles_path}/{inset_name}')

                self.sfrdata.write_shapefiles(f'{self._shapefiles_path}/{self.name}')


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
                mvr = mf6.ModflowMvr(gwfgwf, **kwargs)
                print("finished in {:.2f}s\n".format(time.time() - t0))
                return mvr
        else:
            print("no packages with mover information\n")

    def write_input(self):
        """Write the model input.
        """
        # prior to writing output
        # remove any BCs in inactive cells
        # handle cases of single model or multi-model LGR simulation
        # by working with the simulation-level model dictionary
        for model_name, model in self.simulation.model_dict.items():
            pckgs = ['chd', 'drn', 'ghb', 'riv', 'wel']
            for pckg in pckgs:
                package_instance = getattr(model, pckg.lower(), None)
                if package_instance is not None:
                    external_files = model.cfg[pckg.lower()]['stress_period_data']
                    remove_inactive_bcs(package_instance,
                                        external_files=external_files)
            if hasattr(model, 'obs'):
                # handle case of single obs package, in which case model.obs
                # will be a ModflowUtlobs package instance
                try:
                    len(model.obs)
                    obs_packages = model.obs
                except:
                    obs_packages = [model.obs]
                for obs_package_instance in obs_packages:
                    remove_inactive_obs(obs_package_instance)

            # write the model with flopy
            # but skip the sfr package
            # by monkey-patching the write method
            def skip_write(**kwargs):
                pass
            if hasattr(model, 'sfr'):
                model.sfr.write = skip_write
        self.simulation.write_simulation()

        # post-flopy write actions
        for model_name, model in self.simulation.model_dict.items():
            # write the sfr package with SFRmaker
            if 'SFR' in ' '.join(model.get_package_list()):
                options = []
                for k, b in model.cfg['sfr']['options'].items():
                    options.append(k)
                if 'save_flows' in options:
                    budget_fileout = '{}.{}'.format(model_name,
                                                    model.cfg['sfr']['budget_fileout'])
                    stage_fileout = '{}.{}'.format(model_name,
                                                model.cfg['sfr']['stage_fileout'])
                    options.append('budget fileout {}'.format(budget_fileout))
                    options.append('stage fileout {}'.format(stage_fileout))
                if len(model.sfrdata.observations) > 0:
                    options.append('obs6 filein {}.{}'.format(model_name,
                                                            model.cfg['sfr']['obs6_filein_fmt'])
                                )
                model.sfrdata.write_package(idomain=model.idomain,
                                        version='mf6',
                                        options=options,
                                        external_files_path=model.external_path
                                        )
            # add version info to package file headers
            files = [model.namefile]
            files += [p.filename for p in model.packagelist]
            files += [p[0].filename for k, p in model.simulation.package_key_dict.items()]
            for f in files:
                add_version_to_fileheader(f, model_info=model.header)

            if not model.cfg['mfsetup_options']['keep_original_arrays']:
                shutil.rmtree(model.tmpdir)

        # label stress periods in tdis file with comments
        self.perioddata.sort_values(by='per', inplace=True)
        add_date_comments_to_tdis(self.simulation.tdis.filename,
                                  self.perioddata.start_datetime,
                                  self.perioddata.end_datetime
                                  )



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
    def load_from_config(cls, yamlfile, load_only=None):
        """Load a model from a configuration file and set of MODFLOW files.

        Parameters
        ----------
        yamlfile : pathlike
            Modflow setup YAML format configuration file
        load_only : list
            List of package abbreviations or package names corresponding to
            packages that flopy will load. default is None, which loads all
            packages. the discretization packages will load regardless of this
            setting. subpackages, like time series and observations, will also
            load regardless of this setting.
            example list: ['ic', 'maw', 'npf', 'oc', 'ims', 'gwf6-gwf6']

        Returns
        -------
        m : mfsetup.MF6model instance
        """
        print('\nLoading simulation in {}\n'.format(yamlfile))
        t0 = time.time()

        #cfg = load_cfg(yamlfile, verbose=verbose, default_file=cls.default_file) # 'mf6_defaults.yml')
        #cfg = cls._parse_model_kwargs(cfg)
        #kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
        #                             exclude='packages')
        #model = cls(cfg=cfg, **kwargs)
        model = cls(cfg=yamlfile, load=True)
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
