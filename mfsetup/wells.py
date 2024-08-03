import os
import warnings

import numpy as np
import pandas as pd
from gisutils import df2shp, project
from shapely.geometry import Point

from mfsetup.fileio import append_csv, check_source_files
from mfsetup.grid import get_ij
from mfsetup.sourcedata import TransientTabularSourceData
from mfsetup.wateruse import get_mean_pumping_rates, resample_pumping_rates


def setup_wel_data(model, source_data=None, #for_external_files=True,
                   dropped_wells_file='dropped_wells.csv'):
    """Performs the part of well package setup that is independent of
    MODFLOW version. Returns a DataFrame with the information
    needed to set up stress_period_data.
    """
    # default options for distributing fluxes vertically
    vfd_defaults = {'across_layers': False,
                    'distribute_by': 'transmissivity',
                    'screen_top_col': 'screen_top',
                    'screen_botm_col': 'screen_botm',
                    'minimum_layer_thickness': model.cfg['wel'].get('minimum_layer_thickness', 2.)
                    }

    # master dataframe for stress period data
    columns = ['per', 'k', 'i', 'j', 'q', 'boundname']
    df = pd.DataFrame(columns=columns)

    # check for source data
    datasets = source_data

    # delete the dropped wells file if it exists, to avoid confusion
    if os.path.exists(dropped_wells_file):
        os.remove(dropped_wells_file)

    # get well package input from source (parent) model in lieu of source data
    # todo: fetching correct well package from mf6 parent model
    if datasets is None and model.cfg['parent'].get('default_source_data') \
        and hasattr(model.parent, 'wel'):

        # get well stress period data from mfnwt or mf6 model
        parent = model.parent
        spd = get_package_stress_period_data(parent, package_name='wel')
        # map the parent stress period data to inset stress periods
        periods = spd.groupby('per')
        dfs = []
        for inset_per, parent_per in model.parent_stress_periods.items():
            if parent_per in periods.groups:
                period = periods.get_group(parent_per)
                if len(dfs) > 0 and period.drop('per', axis=1).equals(dfs[-1].drop('per', axis=1)):
                    continue
                else:
                    dfs.append(period)
        spd = pd.concat(dfs)

        parent_well_i = spd.i.copy()
        parent_well_j = spd.j.copy()
        parent_well_k = spd.k.copy()

        # set boundnames based on well locations in parent model
        parent_name = parent.name
        spd['boundname'] = ['{}_{}-{}-{}'.format(parent_name, pk, pi, pj)
                           for pk, pi, pj in zip(parent_well_k, parent_well_i, parent_well_j)]

        parent_well_x = parent.modelgrid.xcellcenters[parent_well_i, parent_well_j]
        parent_well_y = parent.modelgrid.ycellcenters[parent_well_i, parent_well_j]
        coords = project((parent_well_x, parent_well_y),
                          model.modelgrid.proj_str,
                          parent.modelgrid.proj_str)
        geoms = [Point(x, y) for x, y in zip(*coords)]
        bounds = model.modelgrid.bbox
        within = [g.within(bounds) for g in geoms]
        i, j = get_ij(model.modelgrid,
                      parent_well_x[within],
                      parent_well_y[within])
        spd = spd.loc[within].copy()
        spd['i'] = i
        spd['j'] = j

        # map wells to inset model layers if different from parent
        to_inset_layers = {v:k for k, v in model.parent_layers.items()}
        spd['k'] = [to_inset_layers.get(k, -9999) for k in spd['k']]
        spd = spd.loc[spd['k'] >= 0].copy()

        df = pd.concat([df, spd], axis=0)


    # read source data and map onto model space and time discretization
    # multiple types of source data can be submitted
    elif datasets is not None:
        for k, v in datasets.items():

            # determine the format
            if 'csvfile' in k.lower():  # generic csv
                #  read csv file and aggregate flow rates to model stress periods
                #  sum well fluxes co-located in a cell
                sd = TransientTabularSourceData.from_config(v,
                                                            resolve_duplicates_with='sum',
                                                            dest_model=model)
                csvdata = sd.get_data()
                csvdata.rename(columns={v['data_column']: 'q',
                                        v['id_column']: 'boundname'}, inplace=True)
                if 'k' not in csvdata.columns:
                    if model.nlay > 1:
                        vfd = vfd_defaults.copy()
                        vfd.update(v.get('vertical_flux_distribution', {}))
                        csvdata = assign_layers_from_screen_top_botm(csvdata,
                                                                     model,
                                                                     **vfd)
                    else:
                        csvdata['k'] = 0
                df = pd.concat([df, csvdata[columns]], axis=0)

            elif k.lower() == 'wells':  # generic dict
                added_wells = {k: v for k, v in v.items() if v is not None}
                if len(added_wells) > 0:
                    aw = pd.DataFrame(added_wells).T
                    aw['boundname'] = aw.index
                else:
                    aw = None
                if aw is not None:
                    if 'x' in aw.columns and 'y' in aw.columns:
                        aw['i'], aw['j'] = get_ij(model.modelgrid,
                                                  aw['x'].values,
                                                  aw['y'].values)
                    aw['per'] = aw['per'].astype(int)
                    if 'k' not in aw.columns:
                        if model.nlay > 1:
                            vfd = vfd_defaults.copy()
                            vfd.update(v.get('vertical_flux_distribution', {}))
                            aw = assign_layers_from_screen_top_botm(aw,
                                                                    model,
                                                                    **vfd)
                        else:
                            aw['k'] = 0
                    aw['k'] = aw['k'].astype(int)
                    df = pd.concat([df, aw], axis=0)

            elif k.lower() == 'wdnr_dataset':  # custom input format for WI DNR
                # Get steady-state pumping rates
                check_source_files([v['water_use'],
                                    v['water_use_points']])

                # fill out period stats
                period_stats = v['period_stats']
                if isinstance(period_stats, str):
                    period_stats = {kper: period_stats for kper in range(model.nper)}

                # separate out stress periods with period mean statistics vs.
                # those to be resampled based on start/end dates
                resampled_periods = {k: v for k, v in period_stats.items()
                                     if v == 'resample'}
                periods_with_dataset_means = {k: v for k, v in period_stats.items()
                                              if k not in resampled_periods}

                if len(periods_with_dataset_means) > 0:
                    wu_means = get_mean_pumping_rates(v['water_use'],
                                                      v['water_use_points'],
                                                      period_stats=periods_with_dataset_means,
                                                      drop_ids=v.get('drop_ids'),
                                                      model=model)
                    df = pd.concat([df, wu_means], axis=0)
                if len(resampled_periods) > 0:
                    wu_resampled = resample_pumping_rates(v['water_use'],
                                                          v['water_use_points'],
                                                          drop_ids=v.get('drop_ids'),
                                                          exclude_steady_state=True,
                                                          model=model)
                    df = pd.concat([df, wu_resampled], axis=0)

    for col in ['per', 'k', 'i', 'j']:
        df[col] = df[col].astype(int)

    # drop any k, i, j locations that are inactive
    if model.version == 'mf6':
        inactive = model.dis.idomain.array[df.k.values,
                                           df.i.values,
                                           df.j.values] < 1
    else:
        inactive = model.bas6.ibound.array[df.k.values,
                                           df.i.values,
                                           df.j.values] < 1

    # record dropped wells in csv file
    # (which might contain wells dropped by other routines)
    if np.any(inactive):
        # try moving the wells to the closest active layer first
        df['cellid'] = list(zip(df['k'], df['i'], df['j']))
        idm = model.idomain
        new_layers = {}
        for i, r in df.loc[inactive].iterrows():
            cellid = (r['k'], r['i'], r['j'])
            if cellid not in new_layers:
                k2 = move_to_active_layer(r['k'], r['i'], r['j'], idm)
                new_layers[cellid] = (k2, r['i'], r['j'])
                df.loc[df['cellid'] == cellid, 'k'] = k2

        # get inactive cells again after moving layers
        if model.version == 'mf6':
            inactive = model.dis.idomain.array[df.k.values,
                                            df.i.values,
                                            df.j.values] < 1
        else:
            inactive = model.bas6.ibound.array[df.k.values,
                                            df.i.values,
                                            df.j.values] < 1

        dropped = df.loc[inactive].copy()
        dropped = dropped.groupby(['k', 'i', 'j']).first().reset_index()
        dropped['reason'] = 'in inactive cell'
        dropped['routine'] = __name__ + '.setup_wel_data'
        append_csv(dropped_wells_file, dropped, index=False, float_format='%g')  # append to existing file if it exists
    df = df.loc[~inactive].copy()

    copy_fluxes_to_subsequent_periods = False
    if copy_fluxes_to_subsequent_periods and len(df) > 0:
        df = copy_fluxes_to_subsequent_periods(df)

    wel_lookup_file = model.cfg['wel']['output_files']['lookup_file'].format(model.name)
    wel_lookup_file = os.path.join(model._tables_path, os.path.split(wel_lookup_file)[1])
    model.cfg['wel']['output_files']['lookup_file'] = wel_lookup_file

    # verify that all wells have a boundname
    if df.boundname.isna().any():
        no_name = df.boundname.isna()
        k, i, j = df.loc[no_name, ['k', 'i', 'j']].T.values
        names = ['wel_{}-{}-{}'.format(k, i, j) for k, i, j in zip(k, i, j)]
        df.loc[no_name, 'boundname'] = names
    assert not df.boundname.isna().any()

    # if boundname is all ints (or can be casted as such)
    # convert to strings, otherwise MODFLOW may mistake
    # the boundnames in any observation files as cellids
    try:
        [int(s) for s in df['boundname']]
        df['boundname'] = [f"wel_{bn}" for bn in df['boundname']]
    except:
        pass

    # save a lookup file with well site numbers/categories
    df.sort_values(by=['boundname', 'per'], inplace=True)
    if model.version == 'mf6':
        cols = ['per', 'k', 'i', 'j', 'q', 'boundname']
    else:
        cols = ['per', 'k', 'i', 'j', 'flux', 'boundname']
        df.rename(columns={'q': 'flux'}, inplace=True)
    df[cols].to_csv(wel_lookup_file, index=False)

    # convert to one-based and comment out header if df will be written straight to external file
    #if for_external_files:
    #    df.rename(columns={'k': '#k'}, inplace=True)
    #    df['#k'] += 1
    #    df['i'] += 1
    #    df['j'] += 1
    return df


def assign_layers_from_screen_top_botm(data, model,
                                       flux_col='q',
                                       screen_top_col='screen_top',
                                       screen_botm_col='screen_botm',
                                       label_col='site_no',
                                       across_layers=False,
                                       distribute_by='transmissivity',
                                       minimum_layer_thickness=2.):
    """Assign model layers to pumping flux data based on
    open interval. Fluxes are assigned to either the thickest or
    highest transmissivity layer intersection with the well open interval. In
    the case of multiple intersections of identical thickness or transmissivity,
    the deepest (highest) thickness or transmissivity intersection is selected.

    Parameters
    ----------
    data : dataframe of well info
        Must have i, j or x, y locations
    model : mfsetup.MF6model or mfsetup.MFnwtModel instance
        Must have dis, and optionally, attached MFsetupGrid instance
    flux_col : column in data with well fluxes
    screen_top_col : column in data with screen top elevations
    screen_botm_col : column in data with screen bottom elevations
    label_col : column with well names (optional; default site_no)
    across_layers : bool
        True to distribute fluxes to multipler layers intersected by open interval
    distribute_by : str ('thickness' or 'transmissivity')
        Distribute fluxes to layers based on thickness or transmissivity of
        intersected open intervals.

    Returns
    -------
    data : dataframe of well info, modified so that each row represents
        pumping in a single model layer (with fluxes modified proportional
        to the amount of open interval in that layer).
    """
    # inactive cells in either MODFLOW version
    if model.version == 'mf6':
        idomain = model.idomain
    else:
        idomain = model.bas6.ibound.array

    # 'boundname' column is used by wel setup for identifying wells
    if label_col in data.columns:
        data['boundname'] = data[label_col]
    if across_layers:
        raise NotImplemented('Distributing fluxes to multiple layers')
    else:
        if distribute_by in {'thickness', 'transmissivity'}:
            i, j, x, y, screen_botm, screen_top = None, None, None, None, None, None
            if 'i' in data.columns and 'j' in data.columns:
                i, j = data['i'].values, data['j'].values
            elif 'x' in data.columns and 'y' in data.columns:
                raise NotImplementedError('Assigning well layers with just x, y')
                x, y = data['x'].values, data['y'].values
            if screen_top_col not in data.columns:
                raise ValueError(f"No screen top column ('{screen_top_col}') in input data!")
            screen_top = data[screen_top_col].values
            if screen_botm_col not in data.columns:
                raise ValueError(f"No screen bottom column ('{screen_botm_col}') in input data!")
            screen_botm = data[screen_botm_col].values

            # get starting heads if available
            no_strt_msg = (f'Well setup: distribute_by: {distribute_by} selected '
                           'but model has no {} package for computing sat. '
                           'thickness.\nUsing full layer thickness.')
            strt3D = None
            if model.version == 'mf6':
                strt_package = 'IC'
            else:
                strt_package = 'BAS6'

            if strt_package not in model.get_package_list():
                warnings.warn(no_strt_msg.format(strt_package), UserWarning)
                strt2D = None
                strt3D = None
            else:
                strt = getattr(getattr(model, strt_package.lower()), 'strt')
                strt3D = strt.array
                strt2D = strt3D[:, i, j]

            thicknesses = get_open_interval_thickness(model,
                                                      heads=strt2D,
                                                      i=i, j=j, x=x, y=y,
                                                      screen_top=screen_top,
                                                      screen_botm=screen_botm)
            hk = np.ones_like(thicknesses)
            if distribute_by == 'transmissivity':
                no_k_msg = ('Well setup: distribute_by: transmissivity selected '
                            'but model has no {} package.\nFalling back to'
                            'distributing wells by layer thickness.')
                if model.version == 'mf6':
                    hk_package = 'NPF'
                    hk_var = 'k'
                elif model.version == 'mfnwt':
                    hk_package = 'UPW'
                    hk_var = 'hk'
                else:
                    hk_package = 'LPF'
                    hk_var = 'hk'

                if hk_package not in model.get_package_list():
                    warnings.warn(no_k_msg.format(hk_package), UserWarning)
                    hk = np.ones_like(thicknesses)
                else:
                    hk = getattr(getattr(model, hk_package.lower()), hk_var)
                    hk = hk.array[:, i, j]

            # for each i, j location with a well,
            # get the layer with highest transmissivity in the open interval
            # if distribute_by == 'thickness' or no hk array,
            # T == thicknesses
            # round to avoid erratic floating point behavior
            # for (nearly) equal quantities
            T = np.round(thicknesses * hk, 2)

            # to get the deepest occurance of a max value
            # (argmax will result in the first, or shallowest)
            # take the argmax on the reversed view of the array
            # data['k'] = np.argmax(T, axis=0)
            T_r = T[::-1]
            data['k'] = len(T_r) - np.argmax(T_r, axis=0) - 1

            outfile = model.cfg['wel']['output_files']['dropped_wells_file'].format(model.name)
            bad_wells = pd.DataFrame()
            # for LGR parent models, remove wells with >50% of their open interval within the LGR area
            # (these should be represented in the LGR child model)
            if model.lgr:
                data['model_top'] = model.dis.top.array[data['i'], data['j']]
                data['frac_in_model'] = (data['model_top'] - data['screen_botm'])/\
                    (data['screen_top'] - data['screen_botm'])
                in_model = data['frac_in_model'] > 0.5
                bad_wells = pd.concat([bad_wells, data.loc[~in_model].copy()])
                bad_wells['category'] = 'dropped'
                bad_wells['reason'] =\
                    ">50%% of well in LGR area (should be represented in LGR child model)"
                data = data.loc[in_model].copy()
            # for LGR child models, remove wells with <50% of their open interval within the LGR area
            if model._is_lgr:
                data['model_botm'] = model.dis.botm.array[-1, data['i'], data['j']]
                data['frac_in_model'] = (data['screen_top'] - data['model_botm'])/\
                    (data['screen_top'] - data['screen_botm'])
                in_model = data['frac_in_model'] > 0.5
                bad_wells = pd.concat([bad_wells, data.loc[~in_model].copy()])
                bad_wells['category'] = 'dropped'
                bad_wells['reason'] = (
                    ">50%% of well below LGR area (should be represented in "
                    "underlying parent model, or model bottom and open interval "
                    "should be checked)."
                )
                data = data.loc[in_model].copy()


            # get thicknesses for all layers
            # (including portions of layers outside open interval)
            k, i, j = data['k'].values, data['i'].values, data['j'].values
            all_layers = np.zeros((model.nlay + 1, model.nrow, model.ncol))
            all_layers[0] = model.dis.top.array
            all_layers[1:] = model.dis.botm.array
            all_layer_thicknesses = np.abs(np.diff(all_layers, axis=0))
            layer_thicknesses = -np.diff(all_layers[:, i, j], axis=0)

            # only include thicknesses for valid layers
            # reset thicknesses to sat. thickness
            if strt3D is not None:
                sat_thickness = strt3D - model.dis.botm.array
                # cells where the head is above the layer top
                no_unsat = sat_thickness > all_layer_thicknesses
                sat_thickness[no_unsat] = all_layer_thicknesses[no_unsat]
                # cells where the head is below the cell bottom
                sat_thickness[sat_thickness < 0] = 0
                layer_thicknesses = sat_thickness[:, i, j]

            # set inactive cells to 0 thickness for the purpose or relocating wells
            layer_thicknesses[idomain[:, i, j] < 1] = 0
            data['idomain'] = idomain[k, i, j]
            data['laythick'] = layer_thicknesses[k, list(range(layer_thicknesses.shape[1]))]
            # flag layers that are too thin or inactive
            inactive = idomain[data.k, data.i, data.j] < 1
            invalid_open_interval = (data['laythick'] < minimum_layer_thickness) | inactive

            if any(invalid_open_interval):

                # move wells that are still in a thin layer to the thickest active layer
                data['orig_layer'] = data['k']
                # get T for all layers
                T_all_layers = np.round(layer_thicknesses * hk, 2)

                # to get the deepest occurance of a max value
                # (argmax will result in the first, or shallowest)
                # take the argmax on the reversed view of the array
                # Tmax_layer = np.argmax(T_all_layers, axis=0)
                T_all_layers_r = T_all_layers[::-1]
                Tmax_layer = len(T_all_layers_r) - np.argmax(T_all_layers_r, axis=0) - 1

                data.loc[invalid_open_interval, 'k'] = Tmax_layer[invalid_open_interval]
                data['laythick'] = layer_thicknesses[data['k'].values,
                                                     list(range(layer_thicknesses.shape[1]))]
                data['idomain'] = idomain[data['k'], i, j]

                # record which wells were moved or dropped, and why
                wells_in_too_thin_layers = data.loc[invalid_open_interval].copy()
                wells_in_too_thin_layers['category'] = 'moved'
                wells_in_too_thin_layers['reason'] = (f'longest open interval thickness < {minimum_layer_thickness} '
                                      f'{model.length_units} minimum '
                                      'or open interval placed well in inactive layer.'
                                      )
                msg = ('Warning: {} of {} wells in layers less than '
                       'specified minimum thickness of {} {}\n'
                       'were moved to the thickest layer at their i, j locations.\n'.format(invalid_open_interval.sum(),
                                                                        len(data),
                                                                        minimum_layer_thickness,
                                                                        model.length_units))
                still_below_minimum = wells_in_too_thin_layers['laythick'] < minimum_layer_thickness
                wells_in_too_thin_layers.loc[still_below_minimum, 'category'] = 'dropped'
                wells_in_too_thin_layers.loc[still_below_minimum, 'reason'] = 'no layer above minimum thickness of {} {}'.format(minimum_layer_thickness,
                                                                                          model.length_units)
                n_below = np.sum(still_below_minimum)
                if n_below > 0:
                    msg += ('Out of these, {} of {} total wells remaining in layers less than '
                            'specified minimum thickness of {} {}'
                            ''.format(n_below,
                                      len(data),
                                      minimum_layer_thickness,
                                      model.length_units))
                    if flux_col in data.columns:
                        pct_flux_below = 100 * wells_in_too_thin_layers.loc[still_below_minimum, flux_col].sum()/data[flux_col].sum()
                        msg +=  ', \nrepresenting {:.2f} %% of total flux,'.format(pct_flux_below)

                    msg += '\nwere dropped. See {} for details.'.format(outfile)
                    print(msg)

                # write shapefile and CSV output for wells that were dropped
                bad_wells = pd.concat([bad_wells, wells_in_too_thin_layers])
                bad_wells['routine'] = __name__ + '.assign_layers_from_screen_top_botm'
                cols = ['k', 'i', 'j', 'boundname',
                        'category', 'laythick', 'idomain', 'reason', 'routine', 'x', 'y']
                cols = [c for c in cols if c in bad_wells.columns]
                if flux_col in data.columns:
                    cols.insert(3, flux_col)
                flux_below = bad_wells.groupby(['k', 'i', 'j']).first().reset_index()[cols]
                append_csv(outfile, flux_below, index=False, float_format='%g')
                if 'x' in flux_below.columns and 'y' in flux_below.columns:
                    flux_below['geometry'] = [Point(xi, yi) for xi, yi in zip(flux_below.x, flux_below.y)]
                    df2shp(flux_below, outfile[:-4] + '.shp', epsg=model.modelgrid.epsg)

                # cull the wells that are still below the min. layer thickness
                data = data.loc[data['laythick'] > minimum_layer_thickness].copy()
        else:
            raise ValueError(f'Unrecognized argument for distribute_by: {distribute_by}')
    return data


def move_to_active_layer(k, i, j, idomain):
    """Given a k, i, j location, check that idomain is
    > 0 (active). If it is < 1 (inactive), try the cells
    above and below at the same i, j location, returning
    the first active cell encountered.

    Parameters
    ----------
    k : layer index
    i : row index
    j : column index
    idomain : array indicating active/inactive cells

    Returns
    -------
    k or k2 : new layer if an active cell is encountered,
    the original layer if not.
    """
    if idomain[k, i, j] < 1:
        for increment in range(idomain.shape[0] -1):
            # lock at next layer below, and above
            for sign in -1, 1:
                k2 = k + increment * sign
                if (k2 < idomain.shape[0]) and (k2 > -1):
                    if idomain[k2, i, j] > 0:
                        return k2
    # if no other layers are active, return the original layer
    return k




def get_open_interval_thickness(m,
                                heads=None,
                                i=None, j=None, x=None, y=None,
                                screen_top=None, screen_botm=None, nodata=-999):
    """
    Gets the thicknesses of each model layer at specified locations and
    open intervals. If heads are supplied, a saturated thickness is determined
    for each row, column or x, y location; otherwise, total layer thickness is used.
    Returned thicknesses are limited to open intervals (screen_top, screen_botm)
    if included, otherwise the layer tops and bottoms and (optionally) the water table
    are used.

    Parameters
    ----------
    m : mfsetup.MF6model or mfsetup.MFnwtModel instance
        Must have dis, and optionally, attached MFsetupGrid instance
    heads : 2D array OR 3D array (optional)
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D).
    i : 1D array-like of ints, of length n locations
        zero-based row indices (optional; alternately specify x, y)
    j : 1D array-like of ints, of length n locations
        zero-based column indices (optional; alternately specify x, y)
    x : 1D array-like of floats, of length n locations
        x locations in real world coordinates (optional)
    y : 1D array-like of floats, of length n locations
        y locations in real world coordinates (optional)
    screen_top : 1D array-like of floats, of length n locations
        open interval tops (optional; default is model top)
    screen_botm : 1D array-like of floats, of length n locations
        open interval bottoms (optional; default is model bottom)
    nodata : numeric
        optional; locations where heads=nodata will be assigned T=0

    Returns
    -------
    T : 2D array of same shape as heads (nlay x n locations)
        Transmissivities in each layer at each location

    """
    if i is not None and j is not None:
        pass
    elif x is not None and y is not None:
        # get row, col for observation locations
        i, j = get_ij(m.modelgrid, x, y)
    else:
        raise ValueError('Must specify row, column or x, y locations.')

    botm = m.dis.botm.array[:, i, j]

    if heads is None:
        # use model top elevations; expand to nlay x n locations
        heads = np.repeat(m.dis.top.array[np.newaxis, i, j], m.nlay, axis=0)
    if heads.shape == (m.nlay, m.nrow, m.ncol):
        heads = heads[:, i, j]

    msg = 'Shape of heads array must be nlay x n locations'
    assert heads.shape == botm.shape, msg

    # set open interval tops/bottoms to model top/bottom if None
    if screen_top is None:
        screen_top = m.dis.top.array[i, j]
    if screen_botm is None:
        screen_botm = m.dis.botm.array[-1, i, j]

    # make an array of layer tops
    tops = np.empty_like(botm, dtype=float)
    tops[0, :] = m.dis.top.array[i, j]
    tops[1:, :] = botm[:-1]

    # expand top and bottom arrays to be same shape as botm, thickness, etc.
    # (so we have an open interval value for each layer)
    sctoparr = np.zeros(botm.shape)
    sctoparr[:] = screen_top
    scbotarr = np.zeros(botm.shape)
    scbotarr[:] = screen_botm

    # start with layer tops
    # set tops above heads to heads
    # set tops above screen top to screen top
    # (we only care about the saturated open interval)
    openinvtop = tops.copy()
    openinvtop[openinvtop > heads] = heads[openinvtop > heads]
    openinvtop[openinvtop > sctoparr] = sctoparr[openinvtop > screen_top]

    # start with layer bottoms
    # set bottoms below screened interval to screened interval bottom
    # set screen bottoms below bottoms to layer bottoms
    openinvbotm = botm.copy()
    openinvbotm[openinvbotm < scbotarr] = scbotarr[openinvbotm < screen_botm]
    openinvbotm[scbotarr < botm] = botm[scbotarr < botm]

    # compute thickness of open interval in each layer
    thick = openinvtop - openinvbotm

    # assign open intervals above or below model to closest cell in column
    not_in_layer = np.sum(thick < 0, axis=0)
    not_in_any_layer = not_in_layer == thick.shape[0]
    for i, n in enumerate(not_in_any_layer):
        if n:
            closest = np.argmax(thick[:, i])
            thick[closest, i] = 1.
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells
    return thick


def copy_fluxes_to_subsequent_periods(df):
    """Copy fluxes to subsequent stress periods as necessary
    so that fluxes aren't unintentionally shut off;
    for example if water use is only specified for period 0,
    but the added well pumps in period 1, copy water use
    fluxes to period 1. This goes against the paradigm of
    MODFLOW 6, where wells not specified in a subsequent stress period
    are shut off.
    """
    last_specified_per = int(df.per.max())
    copied_fluxes = [df]
    for i in range(last_specified_per):
        # only copied fluxes of a given stress period once
        # then evaluate copied fluxes (with new stress periods) and copy those once
        # after specified per-1, all non-zero fluxes should be propegated
        # to last stress period
        # copy non-zero fluxes that are not already in subsequent stress periods
        if i < len(copied_fluxes):
            in_subsequent_periods = copied_fluxes[i].boundname.duplicated(keep=False)
            # (copied_fluxes[i].per < last_specified_per) & \
            tocopy = (copied_fluxes[i].flux != 0) & \
                     ~in_subsequent_periods
            if np.any(tocopy):
                copied = copied_fluxes[i].loc[tocopy].copy()

                # make sure that wells to be copied aren't in subsequent stress periods
                duplicated = np.array([r.boundname in df.loc[df.per > i, 'boundname']
                                       for idx, r in copied.iterrows()])
                copied = copied.loc[~duplicated]
                copied['per'] += 1
                copied_fluxes.append(copied)
    df = pd.concat(copied_fluxes, axis=0)
    return df


def get_package_stress_period_data(model, package_name, skip_packages=None):

    wel_packages = [p for p in model.get_package_list() if package_name in p.lower()]
    if skip_packages is not None:
        wel_packages = [p for p in wel_packages if p not in skip_packages]

    dfs = []
    for packagename in wel_packages:
        package = model.get_package(packagename)
        stress_period_data = package.stress_period_data
        for kper, spd in stress_period_data.data.items():
            spd = pd.DataFrame(spd)
            spd['per'] = kper
            dfs.append(spd)
    df = pd.concat(dfs)
    if model.version == 'mf6':
        k, i, j = zip(*df['cellid'])
        df.drop(['cellid'], axis=1, inplace=True)
        df['k'], df['i'], df['j'] = k, i, j
    # use MODFLOW-6 variable
    df.rename(columns={'flux': 'q'}, inplace=True)
    return df
