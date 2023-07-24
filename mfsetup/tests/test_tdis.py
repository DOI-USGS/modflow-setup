"""
Test setup of temporal discretization (perioddata table attribute).

See documentation for pandas.date_range method for generating time discretization,
and the reference within on frequency strings.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
"""
import os

import numpy as np
import pandas as pd
import pytest

from mfsetup.mf6model import MF6model
from mfsetup.sourcedata import TransientTabularSourceData
from mfsetup.tdis import (
    aggregate_dataframe_to_stress_period,
    get_parent_stress_periods,
    setup_perioddata_group,
)


@pytest.fixture(scope='function')
def pd0(shellmound_model):
    """Perioddata defined in shellmound.yml config file.
    3 groups:
    1) initial steady-state
    2) single transient period with start and end date
    3) frequency-defined periods with start and end date
    """
    m = shellmound_model #deepcopy(model)
    pd0 = m.perioddata.copy()
    return pd0


@pytest.fixture(scope='function')
def pd1(shellmound_model):
    """Perioddata defined with start_date_time, nper, perlen, nstp, and tsmult
    specified explicitly.
    """
    m = shellmound_model
    m.cfg['sto']['steady'] = {0: True,
                              1: False}

    # Explicit stress period setup
    nper = 11  # total number of MODFLOW periods (including initial steady-state)
    m.cfg['tdis']['perioddata'] = {}
    m.cfg['tdis']['options']['start_date_time'] = '2008-10-01'
    m.cfg['tdis']['perioddata']['perlen'] = [1] * nper
    m.cfg['tdis']['perioddata']['nstp'] = [5] * nper
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd1 = m.perioddata.copy()
    return pd1


@pytest.fixture(scope='function')
def pd2(shellmound_model):
    """Perioddata defined with start_date_time, frequency and nper.
    """
    m = shellmound_model
    m.cfg['tdis']['options']['end_date_time'] = None
    m.cfg['tdis']['perioddata']['perlen'] = None
    m.cfg['tdis']['dimensions']['nper'] = 11
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m.cfg['tdis']['perioddata']['nstp'] = 5
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd2 = m.perioddata.copy()
    return pd2


@pytest.fixture(scope='function')
def pd3(shellmound_model):
    """Perioddata defined with start_date_time, end_datetime, and nper
    """
    m = shellmound_model
    m.cfg['tdis']['perioddata']['end_date_time'] = '2008-10-11'
    m.cfg['tdis']['perioddata']['freq'] = None
    m._perioddata = None
    pd3 = m.perioddata.copy()
    return pd3


@pytest.fixture(scope='function')
def pd4(shellmound_model):
    """Perioddata defined with start_date_time, end_datetime, and freq.
    """
    m = shellmound_model
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m._perioddata = None
    pd4 = m.perioddata.copy()
    return pd4


@pytest.fixture(scope='function')
def pd5(shellmound_model):
    """Perioddata defined with end_datetime, freq and nper.
    """
    m = shellmound_model
    m.cfg['tdis']['options']['start_date_time'] = None
    m.cfg['tdis']['perioddata']['start_date_time'] = None
    m.cfg['tdis']['perioddata']['end_date_time'] = '2008-10-12'
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m.cfg['tdis']['perioddata']['nper'] = 11
    m.cfg['tdis']['perioddata']['perlen'] = None
    m.cfg['tdis']['perioddata']['nstp'] = 5
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd5 = m.perioddata.copy()
    return pd5


@pytest.fixture(scope='function')
def pd6(shellmound_model):
    """Perioddata defined with month-end frequency
    """
    m = shellmound_model
    # month end vs month start freq
    m.cfg['tdis']['perioddata']['freq'] = '6M'
    m.cfg['tdis']['options']['start_date_time'] = '2007-04-01'
    m.cfg['tdis']['perioddata']['end_date_time'] = '2015-10-01'
    m.cfg['tdis']['perioddata']['nstp'] = 15
    m._perioddata = None
    pd6 = m.perioddata.copy()
    return pd6


def test_pd0_freq_last_end_date_time(shellmound_model, pd0):
    """When perioddata are set-up based on start date, end date and freq,
    verify that the last end-date is consistent with what was specified.
    """
    m = shellmound_model
    assert pd0 is not None
    assert pd0['end_datetime'].iloc[-1] == \
           pd.Timestamp(m.cfg['tdis']['perioddata']['group 3']['end_date_time']) #+ pd.Timedelta(1, unit='d')


def test_pd1_explicit_perioddata_setup(pd1, shellmound_model):
    """Test perioddata setup with start_date_time, nper, perlen, nstp, and tsmult
    specified explicitly.
    """
    m = shellmound_model
    assert pd1['start_datetime'][0] == pd1['start_datetime'][1] == pd1['end_datetime'][0]
    assert pd1['end_datetime'][1] == pd.Timestamp(m.cfg['tdis']['options']['start_date_time']) + \
           pd.Timedelta(m.cfg['tdis']['perioddata']['perlen'][1], unit=m.time_units)
    assert pd1['nstp'][0] == 1
    assert pd1['tsmult'][0] == 1


def test_pd2_start_date_freq_nper(pd1, pd2):
    """Since perlen wasn't explicitly specified,
    pd2 will have the 11 periods at freq 'D' (like pd1)
    but with a steady-state first stress period of length 1
    in other words, perlen discretization with freq
    only applies to transient stress periods
    """
    assert pd2.iloc[:-1].equals(pd1)


def test_pd3_start_end_dates_nper(pd1, pd3):
    assert pd3.equals(pd1)


def test_pd4_start_end_dates_freq(pd1, pd4):
    assert pd4.equals(pd1)


def test_pd5_end_date_freq_nper(pd1, pd5):
    assert pd5.iloc[:-1].equals(pd1)


@pytest.mark.skip(reason='incomplete')
def test_pd6(pd0, pd6):
    pd0_g1_3 = pd.concat([pd0.iloc[:1], pd0.iloc[2:]])
    for c in pd0_g1_3[['perlen', 'start_datetime', 'end_datetime']]:
        assert np.array_equal(pd6[c].values, pd0_g1_3[c].values)


def test_pd_date_range():
    """Test that pandas date range is producing the results we expect.
    """
    dates = pd.date_range('2007-04-01', '2015-10-01', periods=None, freq='6MS')
    assert len(dates) == 18
    assert dates[-1] == pd.Timestamp('2015-10-01')


@pytest.mark.parametrize('copy_periods, nper', (('all', 'm.nper'),  # copy all stress periods
                                                ('all', 13),  # copy all stress periods up to 13
                                                ([0], 'm.nper'),  # repeat parent stress period 0
                                                ([2], 'm.nper'),  # repeat parent stress period 2
                                                ([1, 2], 'm.nper')  # include parent stress periods 1 and 2, repeating 2
                                                ))
def test_get_parent_stress_periods(copy_periods, nper, basic_model_instance, request):
    m = basic_model_instance
    if m.version != 'mf6':
        return
    if nper == 'm.nper':
        nper = m.nper
    test_name = request.node.name.split('[')[1].strip(']')
    if m.name == 'pfl' and copy_periods not in ('all', [0]):
        return
    expected = {'pfl_nwt-all-m.nper': [0, 0],  # one parent model stress period, 'all' input
                'pfl_nwt-all-13': [0] * nper,  # one parent model stress period, 'all' input
                'pfl_nwt-copy_periods2-m.nper': [0, 0],  # one parent model stress period, input=[0]
                'pleasant_nwt-all-m.nper': list(range(nper)),  # many parent model stress periods, input='all'
                'pleasant_nwt-all-13': list(range(nper)),  # many parent model stress periods, input='all'
                'pleasant_nwt-copy_periods2-m.nper': [0]*nper,  # many parent model stress periods, input=[0]
                'pleasant_nwt-copy_periods3-m.nper': [2] * nper,   # many parent model stress periods, input=[2]
                'pleasant_nwt-copy_periods4-m.nper': [1] + [2] * (nper-1),    # many parent model stress periods, input=[1, 2]
                'get_pleasant_mf6-all-m.nper': list(range(nper)),
                'get_pleasant_mf6-all-13': list(range(nper)),
                'get_pleasant_mf6-copy_periods2-m.nper': [0]*nper,
                'get_pleasant_mf6-copy_periods3-m.nper': [2] * nper,
                'get_pleasant_mf6-copy_periods4-m.nper': [1] + [2] * (nper-1),
                }

    # test getting list of parent stress periods corresponding to inset stress periods
    result = get_parent_stress_periods(m.parent, nper=nper,
                                       parent_stress_periods=copy_periods)
    assert result == expected[test_name]
    assert len(result) == nper
    assert not any(set(result).difference(set(range(m.parent.nper))))

    # test adding parent stress periods to perioddata table
    m.cfg['parent']['copy_stress_periods'] = copy_periods
    if m.version != 'mf6':
        m.cfg['dis']['nper'] = nper
        for var in ['perlen', 'nstp', 'tsmult', 'steady']:
            del m.cfg['dis'][var]
    m._set_parent()
    m._perioddata = None
    m.perioddata
    assert np.array_equal(m.perioddata['parent_sp'], np.array(expected[test_name]))


@pytest.mark.parametrize('dates', [('2007-04-01', '2007-03-31'),
                                   ('2008-04-01', '2008-09-30'),
                                   ('2008-10-01', '2009-03-31')])
@pytest.mark.parametrize('sourcefile', ['tables/sp69_pumping_from_meras21_m3.csv',
                                        'tables/iwum_m3_1M.csv',
                                        'tables/iwum_m3_6M.csv'])
def test_aggregate_dataframe_to_stress_period(shellmound_datapath, sourcefile, dates):
    """
    dates
    1) similar to initial steady-state period that doesn't represent a real time period
    2) period that spans one or more periods in source data
    3) period that doesn't span any periods in source data

    sourcefiles

    'tables/sp69_pumping_from_meras21_m3.csv'
        case where dest. period is completely within the start and end dates for the source data
        (source start date is before dest start date; source end date is after dest end date)

    'tables/iwum_m3_6M.csv'
        case where 1) start date coincides with start date in source data; end date spans
        one period in source data. 2) start and end date do not span any periods
        in source data (should return a result of length 0)

    'tables/iwum_m3_1M.csv'
        case where 1) start date coincides with start date in source data; end date spans
        multiple periods in source data. 2) start and end date do not span any periods
        in source data (should return a result of length 0)
    Returns
    -------

    """
    start, end = dates
    welldata = pd.read_csv(os.path.join(shellmound_datapath, sourcefile
                                        ))

    welldata['start_datetime'] = pd.to_datetime(welldata.start_datetime)
    welldata['end_datetime'] = pd.to_datetime(welldata.end_datetime)
    duplicate_well = welldata.groupby('node').get_group(welldata.node.values[0])
    welldata = pd.concat([welldata, duplicate_well], axis=0)
    start_datetime = pd.Timestamp(start)
    end_datetime = pd.Timestamp(end)  # pandas convention of including last day
    result = aggregate_dataframe_to_stress_period(welldata, id_column='node', data_column='flux_m3',
                                                  datetime_column='start_datetime', end_datetime_column='end_datetime',
                                                  start_datetime=start_datetime, end_datetime=end_datetime,
                                                  period_stat='mean', resolve_duplicates_with='sum')
    overlap = (welldata.start_datetime < end_datetime) & \
                               (welldata.end_datetime > start_datetime)
    #period_inside_welldata = (welldata.start_datetime < start_datetime) & \
    #                         (welldata.end_datetime > end_datetime)
    #overlap = welldata_overlaps_period #| period_inside_welldata

    # for each location (id), take the mean across source data time periods
    if end_datetime < start_datetime:
        assert result['flux_m3'].sum() == 0
    if not any(overlap):
        assert len(result) == 0
    if any(overlap):
        groupbedby = welldata.loc[overlap].copy().groupby(['start_datetime', 'node'])
        agg = groupbedby.sum(numeric_only=True).reset_index()
        agg = agg.groupby('node').mean().reset_index()
        expected_sum = agg['flux_m3'].sum()
        if duplicate_well.node.values[0] in agg.index:
            dw_overlaps = (duplicate_well.start_datetime < end_datetime) & \
                    (duplicate_well.end_datetime > start_datetime)
            expected_sum += duplicate_well.loc[dw_overlaps, 'flux_m3'].mean()
        assert np.allclose(result['flux_m3'].sum(), expected_sum)


@pytest.fixture()
def dest_model(shellmound_simulation, project_root_path):
    m = MF6model(simulation=shellmound_simulation)
    m._config_path = os.path.join(project_root_path, 'mfsetup/tests/data')
    return m


@pytest.fixture()
def obsdata():
    return pd.DataFrame({'datetime': ['2001-01-01', # per 0, site 2000
                                  '2002-01-01', # per 0, site 2001
                                  '2002-02-02', # per 0, site 2001
                                  '2014-10-02', # per 1, site 2002
                                  '2015-01-01', # per 1, site 2002
                                  '2015-01-02', # per 1, site 2002
                                  '2015-01-02', # per 1, site 2000
                                  '2017-01-01', #        site 2003
                                  '2018-01-01', #        site 2003
                                  '2019-01-01'], #       site 2000
                         'flow_m3d': [1, 1, 0, 4, 2, 3, 2, 1, 0, 10],
                         'comment': ['measured',
                                     'measured',
                                     'measured',
                                     'estimated',
                                     'estimated',
                                     'estimated',
                                     'estimated',
                                     'measured',
                                     'measured',
                                     'estimated'
                                     ],
                         'site_no': [2000,
                                     2001,
                                     2001,
                                     2002,
                                     2002,
                                     2002,
                                     2000,
                                     2003,
                                     2003,
                                     2000
                                     ],
                         'line_id': [1002000,
                                     1002001,
                                     1002001,
                                     1002002,
                                     1002002,
                                     1002002,
                                     1002000,
                                     1002003,
                                     1002003,
                                     1002000
                                     ]
                         })


@pytest.fixture()
def times():
    start_datetime = pd.to_datetime(['2001-01-01', '2014-10-01'])
    end_datetime = pd.to_datetime(['2014-09-30', '2015-09-30'])
    perlen = [(edt - sdt).days for edt, sdt in zip(start_datetime, end_datetime)]
    times = pd.DataFrame({'start_datetime': start_datetime,
                          'end_datetime': end_datetime,
                          'per': [0, 1],
                          'perlen': perlen,
                          'steady': False})
    return times


@pytest.mark.parametrize('category_col', (None, 'comment'))
def test_aggregate_values_with_categories(obsdata, times, category_col, dest_model, tmpdir):
    csvfile = os.path.join(tmpdir, 'obsdata.csv')
    obsdata['flow_ft3d'] = obsdata['flow_m3d']/(.3048**3)
    obsdata.to_csv(csvfile, index=False)
    dest_model._perioddata = times
    sd = TransientTabularSourceData(csvfile, data_columns=['flow_m3d', 'flow_ft3d'], datetime_column='datetime', id_column='line_id',
                                    x_col='x', y_col='y', end_datetime_column=None, period_stats={0: 'mean'},
                                    length_units='unknown', time_units='unknown', volume_units=None,
                                    column_mappings=None, category_column=category_col,
                                    resolve_duplicates_with='raise error',
                                    dest_model=dest_model)
    results = sd.get_data()
    # check that multiple data_columns get passed through
    # and treated the same
    assert np.allclose(results['flow_m3d']/results['flow_ft3d'],.3048**3)
    results.sort_values(by=['per', 'site_no'], inplace=True)
    assert np.array_equal(results.site_no.values,
                          np.array([2000, 2001, 2000, 2002]))
    assert np.array_equal(results['flow_m3d'].values,
                          np.array([1.000000,  # 2000 value in per 0
                                    0.500000,  # 2001 average for per 0
                                    2.000000,  # 2000 value in per 1
                                    3.000000,  # 2002 average for period 1
                                    ]))  #  2003 has no values within model timeframe
    if category_col is not None:
        assert np.array_equal(results['n_measured'].values, [1, 2, 0, 0])
        assert np.array_equal(results['n_estimated'].values, [0, 0, 1, 3])


def test_setup_perioddata_group():
    other_args = {
        'model_time_units': 'days',
        'nper': 1, 'nstp': 5,
        'oc_saverecord': {0: ['save head last', 'save budget last']},
        'steady': {0: False}, 'tsmult': 1.5}
    data = {'start_date_time': '2012-01-01',
            'end_date_time': '2018-12-31',
            'freq': '1MS',
            }
    data.update(other_args)
    results = setup_perioddata_group(**data)
    assert len(results) == 84
    assert results['end_datetime'][83] == pd.Timestamp('2018-12-31')
