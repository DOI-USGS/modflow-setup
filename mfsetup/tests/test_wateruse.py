import calendar

import numpy as np
import pandas as pd
import pytest

from mfsetup.wateruse import (
    get_mean_pumping_rates,
    read_wdnr_monthly_water_use,
    resample_pumping_rates,
)

conversions = {1: 7.48052, # gallons per cubic foot
               2: 264.172} # gallons per cubic meter


@pytest.fixture
def wu_data(pfl_nwt_with_dis_bas6):
    m = pfl_nwt_with_dis_bas6
    well_info, monthly_data = read_wdnr_monthly_water_use(m.cfg['wel']['source_data']['wdnr_dataset']['water_use'],
                                                          m.cfg['wel']['source_data']['wdnr_dataset']['water_use_points'],
                                                          model=m,
                                                          minimum_layer_thickness=m.cfg['dis'][
                                                              'minimum_layer_thickness'])
    return well_info, monthly_data


def test_get_mean_pumping_rates(pfl_nwt_with_dis_bas6):
    m = pfl_nwt_with_dis_bas6
    #well_info, monthly_data = wu_data

    # WDNR specific formatting
    col_fmt = '{}_wdrl_gpm_amt'
    column_mappings = {'site_seq_no': 'site_no',
                       'wdrl_year': 'year',
                       'annual_wdrl_amt': 'annual_wdrl_total_gallons'
                       }
    start_date = '2012-01-01'
    end_date = '2018-12-31'
    years = range(int(start_date.split('-')[0]), int(end_date.split('-')[0])+1)

    wu_file = m.cfg['wel']['source_data']['wdnr_dataset']['water_use']
    wu_points = m.cfg['wel']['source_data']['wdnr_dataset']['water_use_points']
    df = get_mean_pumping_rates(wu_file, wu_points, m,
                                period_stats={0: ['mean', '2012-01-01', '2018-12-31']},
                                start_date=start_date, end_date=end_date)
    wu = pd.read_csv(wu_file)
    wu.rename(columns=column_mappings, inplace=True)
    wu = wu.loc[wu.year.isin(years)].copy()
    wu['days'] = [365 if not calendar.isleap(y) else 366 for y in wu.year]

    monthlyQ_cols = [col_fmt.format(calendar.month_abbr[i]).lower()
                     for i in range(1, 13)]
    wu['annual_total_calc'] = wu[monthlyQ_cols].sum(axis=1)

    # verify that the monthly values match the annual totals
    assert np.allclose(wu['annual_wdrl_total_gallons'], wu['annual_total_calc'])

    sums = wu.dropna(subset=['annual_wdrl_total_gallons'], axis=0).groupby('site_no').sum(numeric_only=True)
    means = wu.dropna(subset=['annual_wdrl_total_gallons'], axis=0).groupby('site_no').mean(numeric_only=True)
    means['Q_m3d'] = sums['annual_wdrl_total_gallons'] / sums['days'] / conversions[2]

    sites = [int(s.strip('site')) for s in df.boundname]
    means = means.loc[sites]
    compare = pd.DataFrame({'site_no': df.index,
                            'Q1': df['q'],  # wel package flux computed by get_mean_pumping_rates
                            'Q2': -means['Q_m3d']})  # expected wel package flux
    compare['rpd'] = np.abs(np.abs(compare.Q2-compare.Q1)/compare.Q1)
    compare.dropna(subset=['rpd'], axis=0, inplace=True)

    # verify that fluxes computed by get_ss_pumping_rates are same as those above
    assert np.allclose(compare.Q1, compare.Q2, rtol=0.02)


def test_resample_pumping_rates(pleasant_nwt_with_dis_bas6):

    m = pleasant_nwt_with_dis_bas6
    assert m.perioddata is not None
    perioddata = m.perioddata.copy()
    perioddata.index = perioddata.start_datetime

    # test with transient first stress period
    active_area = m.modelgrid.bbox.buffer(10000)
    wu_file = m.cfg['wel']['source_data']['wdnr_dataset']['water_use']
    wu_points = m.cfg['wel']['source_data']['wdnr_dataset']['water_use_points']
    well_info, monthly_data = read_wdnr_monthly_water_use(wu_file, wu_points, m,
                                                          active_area=active_area
                                                          )
    wu_resampled = resample_pumping_rates(wu_file, wu_points, m,
                                          active_area=active_area
                                          )

    for site in wu_resampled.index.unique():
        loc = (monthly_data.site_no == site) & \
              (monthly_data.year.isin(perioddata.iloc[1:]['start_datetime'].dt.year.unique()))
        site_data = monthly_data.loc[loc].sort_values(by=['year', 'month'])
        ndays = perioddata.iloc[1:].loc[site_data.datetime, 'perlen']
        periods = perioddata.iloc[1:].loc[site_data.datetime, 'per'].values
        site_data['q'] = site_data['gallons'].values/ndays.values / conversions[m.dis.lenuni]
        wur_loc = [True if str(site) in r.boundname.lower() and r.per in periods
                   else False for i, r in wu_resampled.iterrows()]
        assert len(-site_data.q.values) == len(wu_resampled.loc[wur_loc, 'q'].values)
        assert np.allclose(-site_data.q.values, wu_resampled.loc[wur_loc, 'q'].values)


@pytest.mark.skip("still working on this test")
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

    m.perioddata
    m.perioddata['parent_sp'] = [0] + list(range(nper))
    assert m.perioddata.steady[0]
    assert m.perioddata.perlen[0] == 1

    # test with transient first stress period
    m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    df = m.wel.stress_period_data.get_dataframe()

    # get expected steady state rates for period 0
    df2 = get_mean_pumping_rates(well_info, monthly_data,
                                 lenuni=m.dis.lenuni,
                                 start_date=steadystate_start_date, end_date=steadystate_end_date,
                                 period_stats={0: 'mean'}
                                 )
    # reference fluxes by k, i, j locations
    wel_ss_fluxes = dict(zip(zip(df.k, df.i, df.j), df.q0))
    expected_ss_fluxes = dict(zip(zip(df2.k, df2.i, df2.j), df2.q))

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
        site_data['q'] = site_data['gallons'] / transient_perioddata.perlen.values / conversions[m.dis.lenuni]
        for i, q in enumerate(site_data.q.values):
            col = 'q{}'.format(i + 1)
            value = 0.0
            if col in df.columns:
                value = df.loc[site, col]
            assert np.allclose(-q, value)
