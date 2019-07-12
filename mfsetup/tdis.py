"""
Functions related to temporal discretization
"""
import calendar
import numpy as np
import pandas as pd
from .units import pandas_units


months = {v.lower(): k for k, v in enumerate(calendar.month_name) if k > 0}


def convert_freq_to_period_start(freq):
    """convert pandas frequency to period start"""
    if isinstance(freq, str):
        for prefix in ['M', 'Q', 'A', 'Y']:
            if prefix in freq.upper() and "S" not in freq.upper():
                freq = freq.replace(prefix, "{}S".format(prefix)).upper()
        return freq


def setup_perioddata(start_date, end_date=None,
                     nper=1, perlen=1, model_time_units=None, freq=None,
                     steady={0: True,
                             1: False},
                     nstp=10, tsmult=1.5,
                     oc={0: ['save head last',
                             'save budget last']},
                     ):
    """Sets up time discretization for a model; outputs a DataFrame with
    stress period dates/times and properties. Stress periods can be established
    with an established explicitly by specifying perlen as a list of period lengths in
    model units. Or, stress periods can be established using three of the
    start_date, end_date, nper, and freq arguments, similar to the
    pandas.date_range function.
    (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

    Parameters
    ----------
    start_date : str or datetime-like
        Left bound for generating stress period dates. See pandas documenation.
    end_date : str or datetime-like, optional
        Right bound for generating stress period dates. See pandas documenation.
    nper : int, optional
        Number of stress periods. Only used if perlen is None, or in combination with freq
        if an end_date isn't specified.
    perlen : sequence or None, optional
        A list of stress period lengths in model time units. Or specify as None and
        specify 3 of start_date, end_date, nper and/or freq.
    model_time_units : str, optional
        'days' or 'seconds'.
    freq : str or DateOffset, default None
        For setting up uniform stress periods between a start and end date, or of length nper.
        Same as argument to pandas.date_range. Frequency strings can have multiples,
        e.g. ‘6MS’ for a 6 month interval on the start of each month.
        See the pandas documentation for a list of frequency aliases. Note: Only "start"
        frequences (e.g. MS vs M for "month end") are supported.
    steady : dict
        Dictionary with zero-based stress periods as keys and boolean values. Similar to MODFLOW-6
        input, the information specified for a period will continue to apply until
        information for another perior is specified.
    nstp : int or sequence
        Number of timesteps in a stress period. Must be a integer if perlen=None.
    nstp : int or sequence
        Timestep multiplier for a stress period. Must be a integer if perlen=None.
    oc : dict
        Dictionary with zero-based stress periods as keys and output control options as values.
        Similar to MODFLOW-6 input, the information specified for a period will
        continue to apply until information for another perior is specified.

    Returns
    -------
    perrioddata : pandas.DataFrame
        DataFrame summarizing stress period information.
        Has columns:
        start_datetime : pandas datetimes; start date/time of each stress period
        (does not include steady-state periods)
        end_datetime : pandas datetimes; end date/time of each stress period
        (does not include steady-state periods)
        time : float; cumulative MODFLOW time (includes steady-state periods)
        per : int, zero-based stress period
        perlen : float; stress period length in model time units
        nstp : int; number of timesteps in the stress period
        tsmult : int; timestep multiplier for stress period
        steady : bool; True=steady-state, False=Transient
        oc : dict; MODFLOW-6 output control options

    """
    freq = convert_freq_to_period_start(freq)

    txt = "Specify perlen as a list of lengths in model units, or\nspecify 3" \
          "of start_date, end_date, nper and/or freq."
    # Explicitly specified stress period lengths
    if perlen is not None:
        datetimes = [pd.Timestamp(start_date)]
        for i, length in enumerate(perlen[1:]):
            datetimes.append(datetimes[i] + pd.Timedelta(length, unit=model_time_units))
        time = np.cumsum(perlen) # time in MODFLOW units

    # Set up datetimes based on 3 of start_date, end_date, nper and/or freq (scalar perlen)
    else:
        assert np.isscalar(nstp), "nstp: {}; nstp must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(nstp, txt)
        assert np.isscalar(tsmult), "tsmult: {}; tsmult must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(tsmult, txt)
        periods = None
        if end_date is None:
            # start_date, periods and freq
            # (i.e. nper periods of length perlen starting on stat_date)
            if freq is not None:
                periods = nper
            else:
                raise ValueError("Unrecognized input for perlen: {}.\n{}".format(perlen, txt))
        else:
            # end_date and freq and periods
            if start_date is None:
                periods = nper
            # start_date, end_date and frequency
            elif freq is not None:
                pass
            # start_date, end_date and (linearly spaced) periods
            # (i.e. nper periods of uniform length between start_date and end_date)
            if freq is None:
                periods = nper #-1 if steady[0] else nper
        datetimes = pd.date_range(start_date, end_date, periods=periods, freq=freq)
        if start_date is not None:
            #assert pd.Timestamp(start_date) == datetimes[0]
            pass
        else:
            start_date = datetimes[0]
        time = getattr((datetimes - pd.Timestamp(start_date)), model_time_units).values

        # if first period is steady-state, don't include in generated date_range
        if steady[0]:
            datetimes = [datetimes[0]] + datetimes[:-1].tolist()
            perlen = [1] + np.diff(time).tolist()
            time += 1
        else:
            datetimes = datetimes[:-1]
            time = time[1:]

    perioddata = pd.DataFrame({'datetime': datetimes,
                               'time': time,
                               'per': range(len(datetimes)),
                               'perlen': np.array(perlen).astype(float),
                               'nstp': nstp,
                               'tsmult': tsmult,
                               })

    # specify steady-state or transient for each period, filling empty
    # periods with previous state (same logic as MF6 input)
    issteady = [True]
    for i in range(len(perioddata)):
        issteady.append(steady.get(i, issteady[i]))
    perioddata['steady'] = issteady[1:]

    # set up output control, using previous value to fill empty periods
    # (same as MF6)
    oclist = [None]
    for i in range(len(perioddata)):
        oclist.append(oc.get(i, oclist[i]))
    perioddata['oc'] = oclist[1:]

    # create start and end datetime columns;
    # correct the datetime to only increment for transient stress periods
    start_datetime = [pd.Timestamp(start_date)]
    end_datetime = []
    for i, r in perioddata.iterrows():
        if r.steady:
            end_datetime.append(start_datetime[i])
        else:
            end_datetime.append(start_datetime[i] + pd.Timedelta(r.perlen, unit=model_time_units))
        start_datetime.append(end_datetime[i])

    perioddata['start_datetime'] = start_datetime[:-1]
    perioddata['end_datetime'] = end_datetime
    cols = ['start_datetime', 'end_datetime', 'time', 'per', 'perlen', 'nstp', 'tsmult', 'steady', 'oc']
    perioddata = perioddata.drop('datetime', axis=1)[cols]

    # correct nstp and tsmult to be 1 for steady-state periods
    perioddata.loc[perioddata.steady, 'nstp'] = 1
    perioddata.loc[perioddata.steady, 'tsmult'] = 1
    return perioddata
