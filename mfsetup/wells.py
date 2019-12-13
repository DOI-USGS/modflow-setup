import os
import numpy as np
import pandas as pd
import flopy
from shapely.geometry import Point
from gisutils import project
from .fileio import check_source_files
from .grid import get_ij
from .sourcedata import TransientTabularSourceData
from .wateruse import get_mean_pumping_rates, resample_pumping_rates


def setup_wel_data(model):
    """Performs the part of well package setup that is independent of
    MODFLOW version. Returns a DataFrame with the information
    needed to set up stress_period_data.
    """
    # default options for distributing fluxes vertically
    vfd_defaults = {'across_layers': False,
                    'distribute_by': 'thickness',
                    'screen_top_col': 'screen_top',
                    'screen_botm_col': 'screen_botm',
                    'minimum_layer_thickness': model.cfg['wel'].get('minimum_layer_thickness', 2.)
                    }

    # master dataframe for stress period data
    columns = ['per', 'k', 'i', 'j', 'flux', 'comments']
    df = pd.DataFrame(columns=columns)

    # check for source data
    datasets = model.cfg['wel'].get('source_data')

    # get well package input from source (parent) model in lieu of source data
    # todo: fetching correct well package from mf6 parent model
    if datasets is None and model.cfg['parent'].get('default_source_data') \
        and 'WEL' in model.parent.get_package_list():

        # get well stress period data from mfnwt or mf6 model
        renames = {'q': 'flux',
                   'boundnames': 'comments'
                   }
        parent = model.parent
        spd = get_package_stress_period_data(parent, package_name='wel')
        parent_well_x = parent.modelgrid.xcellcenters[spd.i, spd.j]
        parent_well_y = parent.modelgrid.ycellcenters[spd.i, spd.j]
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
        spd.rename(columns=renames, inplace=True)
        df = df.append(spd)


    # read source data and map onto model space and time discretization
    # multiple types of source data can be submitted
    elif datasets is not None:
        for k, v in datasets.items():

            # determine the format
            if 'csvfile' in k.lower():  # generic csv
                sd = TransientTabularSourceData.from_config(v,
                                                            dest_model=model)
                csvdata = sd.get_data()
                csvdata.rename(columns={v['data_column']: 'flux',
                                        v['id_column']: 'comments'}, inplace=True)
                if 'k' not in csvdata.columns:
                    if model.nlay > 1:
                        vfd = vfd_defaults.copy()
                        vfd.update(v.get('vertical_flux_distribution', {}))
                        csvdata = assign_layers_from_screen_top_botm(csvdata,
                                                                     model,
                                                                     **vfd)
                    else:
                        csvdata['k'] = 0
                df = df.append(csvdata[columns])

            elif k.lower() == 'wells':  # generic dict
                added_wells = {k: v for k, v in v.items() if v is not None}
                if len(added_wells) > 0:
                    aw = pd.DataFrame(added_wells).T
                    aw['comments'] = aw.index
                else:
                    aw = None
                if aw is not None:
                    if 'x' in aw.columns and 'y' in aw.columns:
                        aw['i'], aw['j'] = get_ij(model.modelgrid,
                                                  aw['x'].values,
                                                  aw['y'].values)
                    aw['per'] = aw['per'].astype(int)
                    aw['k'] = aw['k'].astype(int)
                    df = df.append(aw)

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
                                                      model=model)
                    df = df.append(wu_means)
                if len(resampled_periods) > 0:
                    wu_resampled = resample_pumping_rates(model.cfg['source_data']['water_use'],
                                                          model.cfg['source_data']['water_use_points'],
                                                          model=model)
                    df = df.append(wu_resampled)

    # boundary fluxes from parent model
    if model.perimeter_bc_type == 'flux':
        assert model.parent is not None, "need parent model for TMR cut"

        # boundary fluxes
        kstpkper = [(0, 0)]
        tmr = Tmr(model.parent, model)

        # parent periods to copy over
        kstpkper = [(0, per) for per in model.cfg['model']['parent_stress_periods']]
        bfluxes = tmr.get_inset_boundary_fluxes(kstpkper=kstpkper)
        bfluxes['comments'] = 'boundary_flux'
        df = df.append(bfluxes)

    for col in ['per', 'k', 'i', 'j']:
        df[col] = df[col].astype(int)

    # drop any k, i, j locations that are inactive
    if model.version == 'mf6':
        inactive = model.dis.idomain.array[df.k.values,
                                           df.i.values,
                                           df.j.values] != 1
    else:
        inactive = model.bas6.ibound.array[df.k.values,
                                           df.i.values,
                                           df.j.values] != 1
    df = df.loc[~inactive].copy()

    copy_fluxes_to_subsequent_periods = False
    if copy_fluxes_to_subsequent_periods:
        df = copy_fluxes_to_subsequent_periods(df)

    wel_lookup_file = model.cfg['wel']['output_files']['lookup_file']
    wel_lookup_file = os.path.join(model.model_ws, os.path.split(wel_lookup_file)[1])
    model.cfg['wel']['output_files']['lookup_file'] = wel_lookup_file

    # save a lookup file with well site numbers/categories
    df[['per', 'k', 'i', 'j', 'flux', 'comments']].to_csv(wel_lookup_file, index=False)
    return df


def assign_layers_from_screen_top_botm(data, model,
                                       flux_col='flux',
                                       screen_top_col='screen_top',
                                       screen_botm_col='screen_botm',
                                       across_layers=False,
                                       distribute_by='thickness',
                                       minimum_layer_thickness=2.):
    """Assign model layers to pumping flux data based on
    open interval. Fluxes are applied to each layer proportional
    to the fraction of open interval in that layer.

    Parameters
    ----------
    data : dataframe of well info
        Must have i, j or x, y locations
    model : mfsetup.MF6model or mfsetup.MFnwtModel instance
        Must have dis, and optionally, attached MFsetupGrid instance
    flux_col : column in data with well fluxes
    screen_top_col : column in data with screen top elevations
    screen_botm_col : column in data with screen bottom elevations
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
    if across_layers:
        raise NotImplemented('Distributing fluxes to multiple layers')
    else:
        if distribute_by == 'thickness':
            i, j, x, y, screen_botm, screen_top = None, None, None, None, None, None
            if 'i' in data.columns and 'y' in data.columns:
                i, j = data['i'].values, data['j'].values
            elif 'x' in data.columns and 'y' in data.columns:
                x, y = data['x'].values, data['y'].values
            if screen_top_col in data.columns:
                screen_top = data[screen_top_col].values
            if screen_botm_col in data.columns:
                screen_botm = data[screen_botm_col].values
            thicknesses = get_open_interval_thickness(model,
                                                      i=i, j=j, x=x, y=y,
                                                      screen_top=screen_top,
                                                      screen_botm=screen_botm)
            # for each i, j location with a well,
            # get the layer with highest thickness in the open interval
            data['k'] = np.argmax(thicknesses, axis=0)
            # get the thickness for those layers
            all_layers = np.zeros((model.nlay + 1, model.nrow, model.ncol))
            all_layers[0] = model.dis.top.array
            all_layers[1:] = model.dis.botm.array
            layer_thicknesses = -np.diff(all_layers[:, i, j], axis=0)
            k_well_thickness = layer_thicknesses[data['k'].values,
                                                 list(range(layer_thicknesses.shape[1]))]
            below_minimum = k_well_thickness < minimum_layer_thickness
            n_below = np.sum(below_minimum)
            if n_below > 0:
                outpath = os.path.split(model.cfg['wel']['output_files']['lookup_file'])[0]
                outfile = os.path.join(outpath, 'dropped_wells.csv')
                flux_below = data.loc[below_minimum]
                pct_flux_below = 100*flux_below[flux_col].sum()/data[flux_col].sum()
                print('Warning: {} wells in layers less than '
                      'specified minimum thickness of {} {},'
                      'representing {:.2f} %% of total flux.\n'
                      'See {} for details'.format(n_below,
                                                  minimum_layer_thickness,
                                                  model.length_units,
                                                  pct_flux_below,
                                                  outfile))
                flux_below = flux_below.groupby(['k', 'i', 'j']).first().reset_index()[['k', 'i', 'j', flux_col, 'comments']]
                flux_below.to_csv(outfile, index=False)
                data = data.loc[~below_minimum].copy()

        elif distribute_by == 'tranmissivity':
            raise NotImplemented('Distributing well fluxes by layer transmissivity')

        else:
            raise ValueError('Unrecognized argument for distribute_by: {}'.format(distribute_by))
    return data


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
    return df


def get_package_stress_period_data(model, package_name, skip_packages=None):

    wel_packages = [p for p in model.get_package_list() if package_name in p.lower()]
    if skip_packages is not None:
        wel_packages = [p for p in wel_packages if p not in skip_packages]

    dfs = []
    for packagename in wel_packages:
        package = model.get_package(packagename)
        stress_period_data = package.stress_period_data
        # monkey patch the mf6 version to behave like the mf2005 version
        if isinstance(stress_period_data, flopy.mf6.data.mfdatalist.MFTransientList):
            stress_period_data.data = {per: ra for per, ra in enumerate(stress_period_data.array)}

        for kper, spd in stress_period_data.data.items():
            spd = pd.DataFrame(spd)
            spd['per'] = kper
            dfs.append(spd)
    df = pd.concat(dfs)
    if model.version == 'mf6':
        k, i, j = zip(*df['cellid'])
        df.drop(['cellid'], axis=1, inplace=True)
        df['k'], df['i'], df['j'] = k, i, j
    return df