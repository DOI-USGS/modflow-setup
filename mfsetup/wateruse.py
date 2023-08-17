import calendar
import time

import numpy as np
import pandas as pd
from gisutils import shp2df
from shapely.geometry import MultiPolygon, Polygon

from mfsetup import wells as wells
from mfsetup.discretization import get_layer, get_layer_thicknesses
from mfsetup.grid import get_ij
from mfsetup.mf5to6 import get_model_length_units
from mfsetup.units import convert_volume_units
from mfsetup.utils import get_input_arguments

months = {v.lower(): k for k, v in enumerate(calendar.month_name) if k > 0}


def read_wdnr_monthly_water_use(wu_file, wu_points, model,
                                active_area=None,
                                drop_ids=None,
                                minimum_layer_thickness=2
                                ):
    """Read water use data from a master file generated from
    WDNR_wu_data.ipynb. Cull data to area of model. Reshape
    to one month-year-site value per row.

    Parameters
    ----------
    wu_file : csv file
        Water use data ouput from the WDNR_wu_data.ipynb.
    wu_points : point shapefile
        Water use locations, generated in the WDNR_wu_data.ipynb
        Must be in same CRS as sr.
    model : flopy.modflow.Modflow instance
        Must have a valid attached .sr attribute defining the model grid.
        Only wells within the bounds of the sr will be retained.
        Sr is also used for row/column lookup.
        Must be in same CRS as wu_points.
    active_area : str (shapefile path) or shapely.geometry.Polygon
        Polygon denoting active area of the model. If specified,
        wells are culled to this area instead of the model bounding box.
        (default None)
    minimum_layer_thickness : scalar
        Minimum layer thickness to have pumping.

    Returns
    -------
    monthly_data : DataFrame

    """
    col_fmt = '{}_wdrl_gpm_amt'
    data_renames = {'site_seq_no': 'site_no',
                    'wdrl_year': 'year'}
    df = pd.read_csv(wu_file)
    drop_cols = [c for c in df.columns if 'unnamed' in c.lower()]
    drop_cols += ['objectid']
    df.drop(drop_cols, axis=1, inplace=True, errors='ignore')
    df.rename(columns=data_renames, inplace=True)
    if drop_ids is not None:
        df = df.loc[~df.site_no.isin(drop_ids)].copy()

    # implement automatic reprojection in gis-utils
    # maintaining backwards compatibility
    kwargs = {'dest_crs': model.modelgrid.crs}
    kwargs = get_input_arguments(kwargs, shp2df)
    locs = shp2df(wu_points, **kwargs)
    site_seq_col = [c for c in locs if 'site_se' in c.lower()]
    locs_renames = {c: 'site_no' for c in site_seq_col}
    locs.rename(columns=locs_renames, inplace=True)
    if drop_ids is not None:
        locs = locs.loc[~locs.site_no.isin(drop_ids)].copy()

    if active_area is None:
        # cull the data to the model bounds
        features = model.modelgrid.bbox
        txt = "No wells are inside the model bounds of {}"\
            .format(model.modelgrid.extent)
    elif isinstance(active_area, str):
        # implement automatic reprojection in gis-utils
        # maintaining backwards compatibility
        kwargs = {'dest_crs': model.modelgrid.crs}
        kwargs = get_input_arguments(kwargs, shp2df)
        features = shp2df(active_area, **kwargs).geometry.tolist()
        if len(features) > 1:
            features = MultiPolygon(features)
        else:
            features = Polygon(features[0])
        txt = "No wells are inside the area of {}"\
            .format(active_area)
    elif isinstance(active_area, Polygon):
        features = active_area

    within = [g.within(features) for g in locs.geometry]
    assert len(within) > 0, txt
    locs = locs.loc[within].copy()
    if len(locs) == 0:
        print('No wells within model area:\n{}\n{}'.format(wu_file, wu_points))
        return None, None
    df = df.loc[df.site_no.isin(locs.site_no)]
    df.sort_values(by=['site_no', 'year'], inplace=True)

    # create seperate dataframe with well info
    well_info = df[['site_no',
                    'well_radius_mm',
                    'borehole_radius_mm',
                    'well_depth_m',
                    'elev_open_int_top_m',
                    'elev_open_int_bot_m',
                    'screen_length_m',
                    'screen_midpoint_elev_m']].copy()
    # groupby site number to cull duplicate information
    well_info = well_info.groupby('site_no').first()
    well_info['site_no'] = well_info.index

    # add top elevation, screen midpoint elev, row, column and layer
    points = dict(zip(locs['site_no'], locs.geometry))
    well_info['x'] = [points[sn].x for sn in well_info.site_no]
    well_info['y'] = [points[sn].y for sn in well_info.site_no]

    # have to do a loop because modelgrid.rasterize currently only works with scalars
    print('intersecting wells with model grid...')
    t0 = time.time()
    #i, j = [], []
    #for x, y in zip(well_info.x.values, well_info.y.values):
    #    iy, jx = model.modelgrid.rasterize(x, y)
    #    i.append(iy)
    #    j.append(jx)
    i, j = get_ij(model.modelgrid, well_info.x.values, well_info.y.values)
    print("took {:.2f}s\n".format(time.time() - t0))

    top = model.dis.top.array
    botm = model.dis.botm.array
    thickness = get_layer_thicknesses(top, botm)
    well_info['i'] = i
    well_info['j'] = j
    well_info['elv_m'] = top[i, j]
    well_info['elv_top_m'] = well_info.elev_open_int_top_m
    well_info['elv_botm_m'] = well_info.elev_open_int_bot_m
    well_info['elv_mdpt_m'] = well_info.screen_midpoint_elev_m
    well_info['k'] = get_layer(botm, i, j, elev=well_info['elv_mdpt_m'].values)
    well_info['laythick'] = thickness[well_info.k.values, i, j]
    well_info['ktop'] = get_layer(botm, i, j, elev=well_info['elv_top_m'].values)
    well_info['kbotm'] = get_layer(botm, i, j, elev=well_info['elv_botm_m'].values)

    # for wells in a layer below minimum thickness
    # move to layer with screen top, then screen botm,
    # put remainder in layer 1 and hope for the best
    well_info = wells.assign_layers_from_screen_top_botm(well_info, model,
                                       flux_col='q',
                                       screen_top_col='elv_top_m',
                                       screen_botm_col='elv_botm_m',
                                       across_layers=False,
                                       distribute_by='transmissivity',
                                       minimum_layer_thickness=2.)
    isthin = well_info.laythick < minimum_layer_thickness
    assert not np.any(isthin)

    # make a datetime column
    monthlyQ_cols = [col_fmt.format(calendar.month_abbr[i]).lower()
                     for i in range(1, 13)]
    monthly_data = df[['site_no', 'year'] + monthlyQ_cols]
    monthly_data.columns = ['site_no', 'year'] + np.arange(1, 13).tolist()

    # stack the data
    # so that each row is a site number, year, month
    # reset the index to move multi-index levels back out to columns
    stacked = monthly_data.set_index(['site_no', 'year']).stack().reset_index()
    stacked.columns = ['site_no', 'year', 'month', 'gallons']
    stacked['datetime'] = pd.to_datetime(['{}-{:02d}'.format(y, m)
                                          for y, m in zip(stacked.year, stacked.month)])
    monthly_data = stacked
    return well_info, monthly_data


def get_mean_pumping_rates(wu_file, wu_points, model,
                           start_date='2012-01-01', end_date='2018-12-31',
                           period_stats={0: 'mean'},
                           active_area=None,
                           drop_ids=None,
                           minimum_layer_thickness=2):
    """Read water use data from a master file generated from
    WDNR_wu_data.ipynb. Cull data to area of model. Convert
    from monthly gallons to daily averages in m3/d
    for model stress periods.

    Parameters
    ----------
    wu_file : csv file
        Water use data ouput from the WDNR_wu_data.ipynb.
    wu_points : point shapefile
        Water use locations, generated in the WDNR_wu_data.ipynb
        Must be in same CRS as sr.
    model : flopy.modflow.Modflow instance
        Must have a valid attached .sr attribute defining the model grid.
        Only wells within the bounds of the sr will be retained.
        Sr is also used for row/column lookup.
        Must be in same CRS as wu_points.
    start_date : str (YYYY-MM-dd)
        Start date of time period to average.
    end_date : str (YYYY-MM-dd)
        End date of time period to average.
    period_stats : dict
        Dictionary of stats keyed by stress period. Stats include zero values, unless noted.
        keys : 0, 1, 2 ...
        values: str; indicate statistic to apply for each stress period
            'mean': mean pumping for period defined by start_date and end_date
            '<month>': average for a month of the year (e.g. 'august'),
            for the for period defined by start_date and end_date
    minimum_layer_thickness : scalar
        Minimum layer thickness to have pumping.

    Returns
    -------
    wu_data : DataFrame

    """
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
    well_info, monthly_data = read_wdnr_monthly_water_use(wu_file, wu_points, model,
                                                          active_area=active_area,
                                                          drop_ids=drop_ids,
                                                          minimum_layer_thickness=minimum_layer_thickness)
    if well_info is None:
        return
    # determine period for computing average pumping
    # make a dataframe for each stress period listed
    wel_data = []
    for per, stat in period_stats.items():

        if isinstance(stat, str):
            stat = stat.lower()
        elif isinstance(stat, list):
            stat, start_date, end_date = stat
            start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
            stat = stat.lower()
        # slice the monthly values to the period of start_date, end_date
        # aggregate to mean values in m3/d
        # (this section will need some work for generalized transient run setup)
        is_inperiod = (monthly_data.datetime > start_date) & (monthly_data.datetime < end_date)
        inperiod = monthly_data.loc[is_inperiod].copy()

        # compute average daily flux using the sum and number of days for each site
        # (otherwise each month is weighted equally)
        # convert units from monthly gallons to daily gallons
        inperiod['days'] = inperiod.datetime.dt.daysinmonth

        if stat == 'mean':
            period_data = inperiod.copy()
        # mean for given month (e.g. august mean)
        elif stat in months.keys() or stat in months.values():
            period_data = inperiod.loc[inperiod.month == months.get(stat, stat)].copy()
        else:
            raise ValueError('Unrecognized input for stat: {}'.format(stat))

        site_means = period_data.groupby('site_no').mean(numeric_only=True)
        site_sums = period_data.groupby('site_no').sum(numeric_only=True)
        site_means['gal_d'] = site_sums['gallons'] / site_sums['days']
        # conversion to model units is based on lenuni variable in DIS package
        gal_to_model_units = convert_volume_units('gal', get_model_length_units(model))
        site_means['q'] = site_means.gal_d * gal_to_model_units
        site_means['per'] = per

        wel_data.append(well_info[['k', 'i', 'j']].join(site_means[['q', 'per']], how='inner'))

    wel_data = pd.concat(wel_data, axis=0)
    # water use fluxes should be negative
    if not wel_data.q.max() <= 0:
        wel_data.loc[wel_data.q.abs() != 0., 'q'] *= -1
    wel_data['boundname'] = ['site{:d}'.format(s) for s in wel_data.index]
    assert not np.any(wel_data.isna()), "Nans in Well Data"
    return wel_data


def resample_pumping_rates(wu_file, wu_points, model,
                           active_area=None,
                           minimum_layer_thickness=2,
                           drop_ids=None,
                           exclude_steady_state=True,
                           dropna=False, na_fill_value=0.,
                           verbose=False):
    """Read water use data from a master file generated from
    WDNR_wu_data.ipynb. Cull data to area of model. Convert
    from monthly gallons to daily averages in m3/d
    for model stress periods.

    Parameters
    ----------
    wu_file : csv file
        Water use data ouput from the WDNR_wu_data.ipynb.
    wu_points : point shapefile
        Water use locations, generated in the WDNR_wu_data.ipynb
        Must be in same CRS as sr.
    model : flopy.modflow.Modflow instance
        Must have a valid attached .sr attribute defining the model grid.
        Only wells within the bounds of the sr will be retained.
        Sr is also used for row/column lookup.
        Must be in same CRS as wu_points.
    active_area : str (shapefile path) or shapely.geometry.Polygon
        Polygon denoting active area of the model. If specified,
        wells are culled to this area instead of the model bounding box.
        (default None)
    exclude_steady_state : bool
        Exclude steady-state stress periods from resampled output.
        (default True)
    minimum_layer_thickness : scalar
        Minimum layer thickness to have pumping.
    dropna : bool
        Flag to drop times (stress periods) where there is no data for a well
    na_fill_value : float
        If dropna == False, fill missing times (stress periods) with this value.

    Returns
    -------
    wu_data : DataFrame

    """
    assert not np.isnan(na_fill_value), "na_fill_value must be a number!"

    well_info, monthly_data = read_wdnr_monthly_water_use(wu_file,
                                                          wu_points,
                                                          model,
                                                          drop_ids=drop_ids,
                                                          active_area=active_area,
                                                          minimum_layer_thickness=minimum_layer_thickness)
    print('\nResampling pumping rates in {} to model stress periods...'.format(wu_file))
    if dropna:
        print('    wells with no data for a stress period will be dropped from that stress period.')
    else:
        print('    wells with no data for a stress period will be assigned {} pumping rates.'.format(na_fill_value))
    if exclude_steady_state:
        perioddata = model.perioddata.loc[~model.perioddata.steady].copy()
    else:
        perioddata = model.perioddata.copy()

    t0 = time.time()
    # reindex the record at each site to the model stress periods
    dfs = []
    for site, sitedata in monthly_data.groupby('site_no'):
        if site not in well_info.index:
            continue
        sitedata.index = sitedata.datetime
        assert not sitedata.index.duplicated().any()

        if dropna:
            site_period_data = sitedata.reindex(perioddata.start_datetime).dropna(axis=1)
        else:
            site_period_data = sitedata.reindex(perioddata.start_datetime)
            site_period_data.fillna(0, inplace=True)
            isna = site_period_data['site_no'] == 0.
            if np.any(isna):
                if verbose:
                    years = set(site_period_data.loc[isna, 'year'])
                    years = ', '.join(list(years))
                    print('Site {} has {} times with nans (in years {})- filling with {}s'.format(site,
                                                                                    np.sum(isna),
                                                                                    years,
                                                                                    na_fill_value))
            site_period_data['site_no'] = site
            site_period_data['year'] = site_period_data.index.year
            site_period_data['month'] = site_period_data.index.month
            site_period_data['datetime'] = site_period_data.index
        assert not site_period_data.isna().any().any()
        site_period_data.index = perioddata.index

        # copy stress periods and lengths from master stress period table
        for col in ['perlen', 'per']:
            site_period_data[col] = perioddata[col]

        # convert units from monthly gallon totals to daily model length units
        site_period_data['gal_d'] = site_period_data['gallons'] / site_period_data['perlen']
        gal_to_model_units = convert_volume_units('gal', get_model_length_units(model))#model.dis.lenuni]
        site_period_data['q'] = site_period_data.gal_d * gal_to_model_units
        for col in ['i', 'j', 'k']:
            site_period_data[col] = well_info.loc[site, col]
        site_period_data.index = [site] * len(site_period_data)
        dfs.append(site_period_data[['k', 'i', 'j', 'q', 'per']])
    wel_data = pd.concat(dfs)
    # water use fluxes should be negative
    if not wel_data.q.max() <= 0:
        wel_data.loc[wel_data.q.abs() != 0., 'q'] *= -1
    wel_data['boundname'] = ['site{:d}'.format(s) for s in wel_data.index]
    assert not np.any(wel_data.isna()), "Nans in Well Data"
    print("took {:.2f}s\n".format(time.time() - t0))
    return wel_data
