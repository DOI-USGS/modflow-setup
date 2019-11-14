import calendar
import time
from shapely.geometry import box
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from gisutils import shp2df
from .grid import get_ij

# conversions from gallons model length units
conversions = {1: 7.48052, # gallons per cubic foot
               2: 264.172} # gallons per cubic meter

months = {v.lower(): k for k, v in enumerate(calendar.month_name) if k > 0}


def read_wdnr_monthly_water_use(wu_file, wu_points, model,
                                active_area=None,
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
    df = pd.read_csv(wu_file)
    locs = shp2df(wu_points)


    if active_area is None:
        # cull the data to the model bounds
        features = model.modelgrid.bbox
        txt = "No wells are inside the model bounds of {}"\
            .format(model.modelgrid.extent)
    elif isinstance(active_area, str):
        features = shp2df(active_area).geometry.tolist()
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
    df = df.loc[df.site_no.isin(locs.site_seq0)]
    df.sort_values(by=['site_no', 'year'], inplace=True)

    # create seperate dataframe with well info
    well_info = df[['site_no',
                    'well_radius_mm',
                    'borehole_radius_mm',
                    'well_depth_m',
                    'depth_open_int_top_m',
                    'depth_open_int_bot_m',
                    'screen_length_m',
                    'screen_midpoint_m']].copy()
    # groupby site number to cull duplicate information
    well_info = well_info.groupby('site_no').first()
    well_info['site_no'] = well_info.index

    # add top elevation, screen midpoint elev, row, column and layer
    points = dict(zip(locs['site_seq0'], locs.geometry))
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

    well_info['i'] = i
    well_info['j'] = j
    well_info['elv_m'] = model.dis.top.array[i, j]
    well_info['elv_top_m'] = well_info.elv_m - well_info.depth_open_int_top_m
    well_info['elv_botm_m'] = well_info.elv_m - well_info.depth_open_int_top_m
    well_info['elv_mdpt_m'] = well_info.elv_m - well_info.screen_midpoint_m
    well_info['k'] = model.dis.get_layer(i, j, elev=well_info['elv_mdpt_m'].values)
    well_info['laythick'] = model.dis.thickness.array[well_info.k.values, i, j]
    well_info['ktop'] = model.dis.get_layer(i, j, elev=well_info['elv_top_m'].values)
    well_info['kbotm'] = model.dis.get_layer(i, j, elev=well_info['elv_botm_m'].values)

    # for wells in a layer below minimum thickness
    # move to layer with screen top, then screen botm,
    # put remainder in layer 1 and hope for the best
    isthin = well_info.laythick < minimum_layer_thickness
    well_info.loc[isthin, 'k'] = well_info.loc[isthin, 'ktop'].values
    well_info.loc[isthin, 'laythick'] = model.dis.thickness.array[well_info.k[isthin].values,
                                                                  well_info.i[isthin].values,
                                                                  well_info.j[isthin].values]
    isthin = well_info.laythick < minimum_layer_thickness
    well_info.loc[isthin, 'k'] = well_info.loc[isthin, 'kbotm'].values
    well_info.loc[isthin, 'laythick'] = model.dis.thickness.array[well_info.k[isthin].values,
                                                                  well_info.i[isthin].values,
                                                                  well_info.j[isthin].values]
    isthin = well_info.laythick < minimum_layer_thickness
    well_info.loc[isthin, 'k'] = 1
    well_info.loc[isthin, 'laythick'] = model.dis.thickness.array[well_info.k[isthin].values,
                                                                  well_info.i[isthin].values,
                                                                  well_info.j[isthin].values]
    isthin = well_info.laythick < minimum_layer_thickness
    assert not np.any(isthin)

    # make a datetime column
    monthlyQ_cols = ['Jan_wdrl_total_gallons',
                     'Feb_wdrl_total_gallons',
                     'Mar_wdrl_total_gallons',
                     'Apr_wdrl_total_gallons',
                     'May_wdrl_total_gallons',
                     'Jun_wdrl_total_gallons',
                     'Jul_wdrl_total_gallons',
                     'Aug_wdrl_total_gallons',
                     'Sep_wdrl_total_gallons',
                     'Oct_wdrl_total_gallons',
                     'Nov_wdrl_total_gallons',
                     'Dec_wdrl_total_gallons']
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
                           start_date='2011-01-01', end_date='2017-12-31',
                           period_stats={0: 'mean'},
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
    period_stats = {k: v.lower() for k, v in period_stats.items()}
    well_info, monthly_data = read_wdnr_monthly_water_use(wu_file, wu_points, model,
                                                          minimum_layer_thickness=minimum_layer_thickness)

    # slice the monthly values to the period of start_date, end_date
    # aggregate to mean values in m3/d
    # (this section will need some work for generalized transient run setup)
    inperiod = (monthly_data.datetime > start_date) & (monthly_data.datetime < end_date)
    monthly_data = monthly_data.loc[inperiod]

    # compute average daily flux using the sum and number of days for each site
    # (otherwise each month is weighted equally)
    # convert units from monthly gallons to daily gallons
    monthly_data['days'] = monthly_data.datetime.dt.daysinmonth

    # determine period for computing average pumping
    # make a dataframe for each stress period listed
    wel_data = []
    for per, stat in period_stats.items():
        if stat == 'mean':
            period_data = monthly_data.copy()
        # mean for given month (e.g. august mean)
        elif stat in months.keys() or stat in months.values():
            period_data = monthly_data.loc[monthly_data.month == months.get(stat, stat)].copy()
        else:
            raise ValueError('Unrecognized input for stat: {}'.format(stat))

        site_means = period_data.groupby('site_no').mean()
        site_sums = period_data.groupby('site_no').sum()
        site_means['gal_d'] = site_sums['gallons'] / site_sums['days']
        # conversion to model units is based on lenuni variable in DIS package
        site_means['flux'] = site_means.gal_d / conversions[model.dis.lenuni]
        site_means['per'] = per

        wel_data.append(well_info[['k', 'i', 'j']].join(site_means[['flux', 'per']]))

    wel_data = pd.concat(wel_data, axis=0)
    # water use fluxes should be negative
    if not wel_data.flux.max() <= 0:
        wel_data.loc[wel_data.flux.abs() != 0., 'flux'] *= -1
    wel_data['comments'] = ['site{:d}'.format(s) for s in wel_data.index]
    return wel_data


def resample_pumping_rates(wu_file, wu_points, model,
                           active_area=None,
                           minimum_layer_thickness=2,
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
    start_date : str (YYYY-MM-dd)
        Start date of time period to average.
    end_date : str (YYYY-MM-dd)
        End date of time period to average.
    period_stats : dict
        Dictionary of stats keyed by stress period. Stats include zero values, unless noted.
        keys : 0, 1, 2 ...
        values: str; indicate statistic to apply for each stress period
            'mean': mean pumping for period
            '<month>': average for a month of the year (e.g. 'august')
    active_area : str (shapefile path) or shapely.geometry.Polygon
        Polygon denoting active area of the model. If specified,
        wells are culled to this area instead of the model bounding box.
        (default None)
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
                                                          active_area=active_area,
                                                          minimum_layer_thickness=minimum_layer_thickness)
    print('resampling pumping rates in {} to model stress periods...'.format(wu_file))
    if dropna:
        print('wells with no data for a stress period will be dropped from that stress period')
    else:
        print('wells with no data for a stress period will be assigned {} pumping rates'.format(na_fill_value))
    t0 = time.time()
    # reindex the record at each site to the model stress periods
    dfs = []
    for site, sitedata in monthly_data.groupby('site_no'):
        sitedata.index = sitedata.datetime
        assert not sitedata.index.duplicated().any()

        if dropna:
            site_period_data = sitedata.reindex(model.perioddata.start_datetime).dropna(axis=1)
        else:
            site_period_data = sitedata.reindex(model.perioddata.start_datetime, fill_value=na_fill_value)
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
        site_period_data.index = model.perioddata.index

        # copy stress periods and lengths from master stress period table
        for col in ['perlen', 'per']:
            site_period_data[col] = model.perioddata[col]

        # convert units from monthly gallon totals to daily model length units
        site_period_data['gal_d'] = site_period_data['gallons'] / site_period_data['perlen']
        site_period_data['flux'] = site_period_data.gal_d / conversions[model.dis.lenuni]
        for col in ['i', 'j', 'k']:
            site_period_data[col] = well_info.loc[site, col]
        site_period_data.index = [site] * len(site_period_data)
        dfs.append(site_period_data[['k', 'i', 'j', 'flux', 'per']])
    wel_data = pd.concat(dfs)
    # water use fluxes should be negative
    if not wel_data.flux.max() <= 0:
        wel_data.loc[wel_data.flux.abs() != 0., 'flux'] *= -1
    wel_data['comments'] = ['site{:d}'.format(s) for s in wel_data.index]
    print("took {:.2f}s\n".format(time.time() - t0))
    return wel_data