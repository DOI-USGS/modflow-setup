import calendar
import numpy as np
import pandas as pd
import pytest
from ..wateruse import read_wdnr_monthly_water_use, resample_pumping_rates, get_mean_pumping_rates


@pytest.fixture
def wu_data(inset_with_dis):
    m = inset_with_dis
    well_info, monthly_data = read_wdnr_monthly_water_use(m.cfg['source_data']['water_use'],
                                                          m.cfg['source_data']['water_use_points'],
                                                          model=m,
                                                          minimum_layer_thickness=m.cfg['dis'][
                                                              'minimum_layer_thickness'])
    return well_info, monthly_data


def test_get_mean_pumping_rates(inset_with_dis, wu_data):
    m = inset_with_dis
    well_info, monthly_data = wu_data

    # WDNR specific formatting
    col_fmt = '{}_wdrl_gpm_amt'
    column_mappings = {'site_seq_no': 'site_no',
                       'wdrl_year': 'year',
                       'annual_wdrl_amt': 'annual_wdrl_total_gallons'
                       }
    start_date = '2011-01-01'
    end_date = '2017-12-31'
    years = range(int(start_date.split('-')[0]), int(end_date.split('-')[0])+1)

    df = get_mean_pumping_rates(well_info, monthly_data,
                                lenuni=m.dis.lenuni,
                                start_date=start_date, end_date=end_date)
    wu = pd.read_csv(m.cfg['source_data']['water_use'])
    wu.rename(columns=column_mappings, inplace=True)
    wu = wu.loc[wu.year.isin(years)].copy()
    wu['days'] = [365 if not calendar.isleap(y) else 366 for y in wu.year]

    monthlyQ_cols = [col_fmt.format(calendar.month_abbr[i]).lower()
                     for i in range(1, 13)]
    wu['annual_total_calc'] = wu[monthlyQ_cols].sum(axis=1)

    # verify that the monthly values match the annual totals
    assert np.allclose(wu['annual_wdrl_total_gallons'], wu['annual_total_calc'])

    sums = wu.dropna(subset=['annual_wdrl_total_gallons'], axis=0).groupby('site_no').sum()
    means = wu.dropna(subset=['annual_wdrl_total_gallons'], axis=0).groupby('site_no').mean()
    means['Q_m3d'] = sums['annual_wdrl_total_gallons'] / sums['days'] / conversions[2]

    sites = [int(s.strip('site')) for s in df.comments]
    means = means.loc[sites]
    compare = pd.DataFrame({'site_no': df.index,
                            'Q1': df['flux'],  # wel package flux computed by get_mean_pumping_rates
                            'Q2': -means['Q_m3d']})  # expected wel package flux
    compare['rpd'] = np.abs(np.abs(compare.Q2-compare.Q1)/compare.Q1)
    compare.dropna(subset=['rpd'], axis=0, inplace=True)

    # verify that fluxes computed by get_ss_pumping_rates are same as those above
    assert np.allclose(compare.Q1, compare.Q2)


def test_resample_pumping_rates(inset_with_transient_parent, wu_data):

    m = inset_with_transient_parent
    well_info, monthly_data = wu_data
    assert m.perioddata is not None

    # test with transient first stress period
    wu_file = m.cfg['source_data']['water_use']
    wu_points = m.cfg['source_data']['water_use_points']
    well_info, monthly_data = read_wdnr_monthly_water_use(wu_file, wu_points, m)

    wu_resampled = resample_pumping_rates(well_info, monthly_data, m.perioddata, m.dis.lenuni)

    for site in wu_resampled.index.unique():
        loc = (monthly_data.site_no == site) & \
              (monthly_data.year.isin(m.perioddata['start_datetime'].dt.year.unique()))
        site_data = monthly_data.loc[loc].sort_values(by=['year', 'month'])
        site_data['flux'] = site_data['gallons']/m.perioddata.perlen.values / conversions[m.dis.lenuni]
        assert np.allclose(-site_data.flux.values, wu_resampled.flux.values)


def test_resample_ss_first_period(inset_with_transient_parent, wu_data):
    m = inset_with_transient_parent
    well_info, monthly_data = wu_data

    steadystate_start_date = '2011-01-01'
    steadystate_end_date = '2017-12-31'
    nper = m.cfg['dis']['nper']
    perlen = list(m.cfg['dis']['perlen'])
    m.cfg['dis']['nper'] = nper + 1
    m.cfg['dis']['steady'] = [True] + [False] * nper
    m.cfg['dis']['perlen'] = [1] + perlen
    m.cfg['dis']['nstp'] = [1] + [5] * nper
    m.cfg['dis']['tsmult'] = [1] + [1.2] * nper

    m._set_perioddata()
    m.perioddata['parent_sp'] = [0] + list(range(nper))
    assert m.perioddata.steady[0]
    assert m.perioddata.perlen[0] == 1

    # test with transient first stress period
    m.setup_wel()
    df = m.wel.stress_period_data.get_dataframe()

    # get expected steady state rates for period 0
    df2 = get_mean_pumping_rates(well_info, monthly_data,
                                 lenuni=m.dis.lenuni,
                                 start_date=steadystate_start_date, end_date=steadystate_end_date,
                                 period_stats={0: 'mean'}
                                 )
    # reference fluxes by k, i, j locations
    wel_ss_fluxes = dict(zip(zip(df.k, df.i, df.j), df.flux0))
    expected_ss_fluxes = dict(zip(zip(df2.k, df2.i, df2.j), df2.flux))

    # put the site numbers back with the well package data
    sites = dict(zip(zip(well_info.k, well_info.i, well_info.j), well_info.index))
    df.index = [sites[(k, i, j)] for k, i, j in zip(df.k, df.i, df.j)]

    # verify that the period 0 fluxes are average values for
    # period between start and end dates
    for k, v in wel_ss_fluxes.items():
        if k in expected_ss_fluxes:
            assert np.allclose(v, expected_ss_fluxes[k])

    # verify that the subsequent transient fluxes are the monthly values for those sites
    transient_perioddata = m.perioddata.iloc[1:].copy()
    for site in df.index.unique():
        loc = (monthly_data.site_no == site) & \
              (monthly_data.year.isin(transient_perioddata['start_datetime'].dt.year.unique()))
        site_data = monthly_data.loc[loc].sort_values(by=['year', 'month'])
        site_data['flux'] = site_data['gallons'] / transient_perioddata.perlen.values / conversions[m.dis.lenuni]
        for i, q in enumerate(site_data.flux.values):
            col = 'flux{}'.format(i + 1)
            value = 0.0
            if col in df.columns:
                value = df.loc[site, col]
            assert np.allclose(-q, value)