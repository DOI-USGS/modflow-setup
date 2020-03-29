"""
Test setup of temporal discretization (perioddata table attribute).

See documentation for pandas.date_range method for generating time discretization,
and the reference within on frequency strings.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
"""
import copy
import numpy as np
import pandas as pd
import pytest


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
    m.cfg['tdis']['perioddata']['end_date_time'] = '2008-10-12'
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
    verify that the last end-date is the beginning of the next day (end of end-date)."""
    m = shellmound_model
    assert pd0 is not None
    assert pd0['end_datetime'].iloc[-1] == \
           pd.Timestamp(m.cfg['tdis']['perioddata']['group 3']['end_date_time']) + pd.Timedelta(1, unit='d')


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


@pytest.mark.skip("still need to fix this; workaround in the meantime is just to specify an extra period")
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


@pytest.mark.skip("still need to fix this")
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
