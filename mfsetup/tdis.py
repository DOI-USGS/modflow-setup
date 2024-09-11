"""
Functions related to temporal discretization
"""
import calendar
import copy
import datetime as dt
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import mfsetup
from mfsetup.checks import is_valid_perioddata
from mfsetup.utils import get_input_arguments, print_item

months = {v.lower(): k for k, v in enumerate(calendar.month_name) if k > 0}


def convert_freq_to_period_start(freq):
    """convert pandas frequency to period start"""
    if isinstance(freq, str):
        for prefix in ['M', 'Q', 'A', 'Y']:
            if prefix in freq.upper() and "S" not in freq.upper():
                freq = freq.replace(prefix, "{}S".format(prefix)).upper()
        return freq


def get_parent_stress_periods(parent_model, nper=None,
                              parent_stress_periods='all'):

    parent_sp = copy.copy(parent_stress_periods)
    parent_model_nper = parent_model.modeltime.nper

    # use all stress periods from parent model
    if isinstance(parent_sp, str) and parent_sp.lower() == 'all':
        if nper is None:  # or nper < parent_model.nper:
            nper = parent_model_nper
            parent_sp = list(range(nper))
        elif nper > parent_model_nper:
            parent_sp = list(range(parent_model_nper))
            for i in range(nper - parent_model_nper):
                parent_sp.append(parent_sp[-1])
        else:
            parent_sp = list(range(nper))

    # use only specified stress periods from parent model
    elif isinstance(parent_sp, list):
        # limit parent stress periods to include
        # to those in parent model and nper specified for pfl_nwt
        if nper is None:
            nper = len(parent_sp)

        perlen = [parent_model.modeltime.perlen[0]]
        for i, p in enumerate(parent_sp):
            if i == nper:
                break
            if p == parent_model_nper:
                break
            if p > 0 and p >= parent_sp[-1] and len(parent_sp) < nper:
                parent_sp.append(p)
                perlen.append(parent_model.modeltime.perlen[p])
        if nper < len(parent_sp):
            nper = len(parent_sp)
        else:
            n_parent_per = len(parent_sp)
            for i in range(nper - n_parent_per):
                parent_sp.append(parent_sp[-1])

    # no parent stress periods specified,
    # default to just using first stress period
    # (repeating if necessary;
    # for example if creating transient inset model with steady bc from parent)
    else:
        if nper is None:
            nper = 1
        parent_sp = [0]
        for i in range(nper - 1):
            parent_sp.append(parent_sp[-1])
    assert len(parent_sp) == nper
    return parent_sp


def parse_perioddata_groups(perioddata_dict,
                            **kwargs):
    """Reorganize input in perioddata dict into
    a list of groups (dicts).
    """
    perioddata_groups = []
    defaults = {
        'start_date_time': '1970-01-01'
    }
    defaults.update(kwargs)
    group0 = defaults.copy()

    valid_txt = "if transient: perlen specified or 3 of start_date_time, " \
                "end_date_time, nper or freq;\n" \
                "if steady: nper or perlen specified. Default perlen " \
                "for steady-state periods is 1."
    for k, v in perioddata_dict.items():
        if 'group' in k.lower():
            data = defaults.copy()
            data.update(v)
            if is_valid_perioddata(data):
                data = get_input_arguments(data, setup_perioddata_group,
                                           errors='raise')
                perioddata_groups.append(data)
            else:
                print_item(k, data)
                prefix = "perioddata input for {} must have".format(k)
                raise Exception(prefix + valid_txt)
        elif 'perioddata' in k.lower():
            perioddata_groups += parse_perioddata_groups(perioddata_dict[k], **defaults)
        else:
            group0[k] = v
    if len(perioddata_groups) == 0:
        if not is_valid_perioddata(group0):
            print_item('perioddata:', group0)
            prefix = "perioddata input must have"
            raise Exception(prefix + valid_txt)
        data = get_input_arguments(group0, setup_perioddata_group)
        perioddata_groups = [data]
    for group in perioddata_groups:
        if 'steady' in group:
            if np.isscalar(group['steady']) or group['steady'] is None:
                group['steady'] = {0: group['steady']}
            elif not isinstance(group['steady'], dict):
                group['steady'] = {i: s for i, s in enumerate(group['steady'])}
    return perioddata_groups


def setup_perioddata_group(start_date_time, end_date_time=None,
                           nper=None, perlen=None, model_time_units='days', freq=None,
                           steady={0: True, 1: False},
                           nstp=10, tsmult=1.5,
                           oc_saverecord={0: ['save head last',
                                              'save budget last']},
                           ):
    """Sets up time discretization for a model; outputs a DataFrame with
    stress period dates/times and properties. Stress periods can be established
    by explicitly specifying perlen as a list of period lengths in
    model units. Or, stress periods can be generated via :func:`pandas.date_range`,
    using three of the start_date_time, end_date_time, nper, and freq arguments.

    Parameters
    ----------
    start_date_time : str or datetime-like
        Left bound for generating stress period dates. See :func:`pandas.date_range`.
    end_date_time : str or datetime-like, optional
        Right bound for generating stress period dates. See :func:`pandas.date_range`.
    nper : int, optional
        Number of stress periods. Only used if perlen is None, or in combination with freq
        if an end_date_time isn't specified.
    perlen : sequence or None, optional
        A list of stress period lengths in model time units. Or specify as None and
        specify 3 of start_date_time, end_date_time, nper and/or freq.
    model_time_units : str, optional
        'days' or 'seconds'.
        By default, 'days'.
    freq : str or DateOffset, default None
        For setting up uniform stress periods between a start and end date, or of length nper.
        Same as argument to pandas.date_range. Frequency strings can have multiples,
        e.g. ‘6MS’ for a 6 month interval on the start of each month.
        See the pandas documentation for a list of frequency aliases. Note: Only "start"
        frequences (e.g. MS vs M for "month end") are supported.
    steady : dict
        Dictionary with zero-based stress periods as keys and boolean values. Similar to MODFLOW-6
        input, the information specified for a period will continue to apply until
        information for another period is specified.
    nstp : int or sequence
        Number of timesteps in a stress period. Must be an integer if perlen=None.
    nstp : int or sequence
        Timestep multiplier for a stress period. Must be an integer if perlen=None.
    oc_saverecord : dict
        Dictionary with zero-based stress periods as keys and output control options as values.
        Similar to MODFLOW-6 input, the information specified for a period will
        continue to apply until information for another period is specified.

    Returns
    -------
    perioddata : pandas.DataFrame
        DataFrame summarizing stress period information. Data columns:

        ==================  ================  ==============================================
        **start_datetime**  pandas datetimes  start date/time of each stress period
        **end_datetime**    pandas datetimes  end date/time of each stress period
        **time**            float             cumulative MODFLOW time at end of period
        **per**             int               zero-based stress period
        **perlen**          float             stress period length in model time units
        **nstp**            int               number of timesteps in the stress period
        **tsmult**          int               timestep multiplier for stress period
        **steady**          bool              True=steady-state, False=Transient
        **oc**              dict              MODFLOW-6 output control options
        ==================  ================  ==============================================

    Notes
    -----
    *Initial steady-state period*

    If the first stress period is specified as steady-state (``steady[0] == True``),
    the period length (perlen) in MODFLOW time is automatically set to 1. If subsequent
    stress periods are specified, or if no end-date is specified, the end date for
    the initial steady-state stress period is set equal to the start date. In the latter case,
    the assumption is that the specified start date represents the start of the transient simulation,
    and the initial steady-state (which is time-invarient anyways) is intended to produce a valid
    starting condition. If only a single steady-state stress period is specified with an end date,
    then that end date is retained.

    *MODFLOW time vs real time*

    The ``time`` column of the output DataFrame represents time in the MODFLOW simulation,
    which cannot have zero-lengths for any period. Therefore, initial steady-state periods
    are automatically assigned lengths of one (as described above), and MODFLOW time is incremented
    accordingly. If the model has an initial steady-state period, this means that subsequent MODFLOW
    times will be 1 time unit greater than the acutal date-times.

    *End-dates*

    Specified ``end_date_time`` represents the right bound of the time discretization,
    or in other words, the time increment *after* the last time increment to be
    simulated. For example, ``end_date_time='2019-01-01'`` would mean that
    ``'2018-12-31'`` is the last date simulated by the model
    (which ends at ``2019-01-01 00:00:00``).



    """
    specified_start_datetime = None
    if start_date_time is not None:
        specified_start_datetime = pd.Timestamp(start_date_time)
    elif end_date_time is None:
        raise ValueError('If no start_datetime, must specify end_datetime')
    specified_end_datetime = None
    if end_date_time is not None:
        specified_end_datetime = pd.Timestamp(end_date_time)

    # if times are specified by start & end dates and freq,
    # period is determined by pd.date_range
    if all({specified_start_datetime, specified_end_datetime, freq}):
        nper = None
    freq = convert_freq_to_period_start(freq)
    oc = oc_saverecord
    if not isinstance(steady, dict):
        steady = {i: v for i, v in enumerate(steady)}

    # nstp and tsmult need to be lists
    if not np.isscalar(nstp):
        nstp = list(nstp)
    if not np.isscalar(tsmult):
        tsmult = list(tsmult)

    txt = "Specify perlen as a list of lengths in model units, or\nspecify 3 " \
          "of start_date_time, end_date_time, nper and/or freq."

    # Explicitly specified stress period lengths
    start_datetime = []  # datetimes at period starts
    end_datetime = []  # datetimes at period ends
    if perlen is not None:
        if np.isscalar(perlen):
            perlen = [perlen]
        start_datetime = [specified_start_datetime]
        if len(perlen) > 1:
            for i, length in enumerate(perlen):
                # initial steady-state period
                # set perlen to 0
                # and start/end dates to be equal
                if i == 0 and steady[0]:
                    next_start = start_datetime[i]
                    perlen[0] == 1
                else:
                    next_start = start_datetime[i] + \
                                 pd.Timedelta(length, unit=model_time_units)
                start_datetime.append(next_start)
            end_datetime = pd.to_datetime(start_datetime[1:])
            start_datetime = pd.to_datetime(start_datetime[:-1])
        # single specified stress period length
        else:
            end_datetime = [specified_start_datetime + pd.Timedelta(perlen[0],
                                                                    unit=model_time_units)]
        time = np.cumsum(perlen)  # time at end of period, in MODFLOW units

    # single steady-state period
    elif nper == 1 and steady[0]:
        perlen = [1]
        time = [1]
        start_datetime = pd.to_datetime([specified_start_datetime])
        if specified_end_datetime is not None:
            end_datetime = pd.to_datetime([specified_end_datetime])
        else:
            end_datetime = pd.to_datetime([specified_start_datetime])

    # Set up datetimes based on 3 of start_date_time, specified_end_datetime, nper and/or freq (scalar perlen)
    else:
        assert np.isscalar(nstp), "nstp: {}; nstp must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(nstp, txt)
        assert np.isscalar(tsmult), "tsmult: {}; tsmult must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(tsmult, txt)
        periods = None
        if specified_end_datetime is None:
            # start_date_time, periods and freq
            # (i.e. nper periods of length perlen starting on stat_date)
            if freq is not None:
                periods = nper
            else:
                raise ValueError("Unrecognized input for perlen: {}.\n{}".format(perlen, txt))
        else:
            # specified_end_datetime and freq and periods
            if specified_start_datetime is None:
                periods = nper + 1
            # start_date_time, specified_end_datetime and uniform periods
            # (i.e. nper periods of uniform length between start_date_time and specified_end_datetime)
            elif freq is None:
                periods = nper #-1 if steady[0] else nper
            # start_date_time, specified_end_datetime and frequency
            elif freq is not None:
                pass
        datetimes = pd.date_range(specified_start_datetime, specified_end_datetime,
                                  periods=periods, freq=freq)
        # if end_datetime, periods and freq were specified
        if specified_start_datetime is None:
            specified_start_datetime = datetimes[0]
            start_datetime = datetimes[:-1]
            end_datetime = datetimes[1:]
            time_edges = getattr((datetimes - start_datetime[0]),
                                 model_time_units).tolist()
            perlen = np.diff(time_edges)
            # time is elapsed time at the end of each period
            time = time_edges[1:]
        else:
            start_datetime = datetimes
            end_datetime = pd.to_datetime(datetimes[1:].tolist() +
                                          [specified_end_datetime])
            # Edge case of end date falling on the start date freq
            # (zero-length sp at end)
            if end_datetime[-1] == start_datetime[-1]:
                start_datetime = start_datetime[:-1]
                end_datetime = end_datetime[:-1]
            time_edges = getattr((end_datetime - start_datetime[0]),
                                 model_time_units).tolist()
            time_edges = [0] + time_edges
            perlen = np.diff(time_edges)
            # time is elapsed time at the end of each period
            time = time_edges[1:]
        #if len(datetimes) == 1:
        #    perlen = [(specified_end_datetime - specified_start_datetime).days]
        #    time = np.array(perlen)

        # if first period is steady-state,
        # insert it at the beginning of the generated range
        # (only do for pd.date_range -based discretization)
        if steady[0]:
            start_datetime = [start_datetime[0]] + start_datetime.tolist()
            end_datetime = [start_datetime[0]] + end_datetime.tolist()
            perlen = [1] + list(perlen)
            time = [1] + (np.array(time) + 1).tolist()
            if isinstance(nstp, list):
                nstp = [1] + nstp
            if isinstance(tsmult, list):
                tsmult = [1] + tsmult

    perioddata = pd.DataFrame({
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
        'time': time,
        'per': range(len(time)),
        'perlen': np.array(perlen).astype(float),
        'nstp': nstp,
        'tsmult': tsmult,
    })

    # specify steady-state or transient for each period, filling empty
    # periods with previous state (same logic as MF6 input)
    issteady = [steady[0]]
    for i in range(len(perioddata)):
        issteady.append(steady.get(i, issteady[i]))
    perioddata['steady'] = issteady[1:]
    perioddata['steady'] = perioddata['steady'].astype(bool)

    # set up output control, using previous value to fill empty periods
    # (same as MF6)
    oclist = [None]
    for i in range(len(perioddata)):
        oclist.append(oc.get(i, oclist[i]))
    perioddata['oc'] = oclist[1:]

    # correct nstp and tsmult to be 1 for steady-state periods
    perioddata.loc[perioddata.steady.values, 'nstp'] = 1
    perioddata.loc[perioddata.steady.values, 'tsmult'] = 1
    return perioddata


def concat_periodata_groups(perioddata_groups, time_units='days'):
    """Concatenate multiple perioddata DataFrames, but sort
    result on (absolute) datetimes and increment model time and stress period
    numbers accordingly."""

    # update any missing variables in the groups with global variables
    group_dfs = []
    for i, group in enumerate(perioddata_groups):
        group.update({'model_time_units': time_units,
                      })
        df = setup_perioddata_group(**group)
        group_dfs.append(df)

    df = pd.concat(group_dfs).sort_values(by=['end_datetime'])
    perlen = np.ones(len(df))
    perlen[~df.steady.values] = df.loc[~df.steady.values, 'perlen']
    df['time'] = np.cumsum(perlen)
    df['per'] = range(len(df))
    df.index = range(len(df))
    return df


def setup_perioddata(model,
                     tdis_perioddata_config,
                     default_start_datetime=None,
                     nper=None,
                     steady=None, time_units='days',
                     oc_saverecord=None, parent_model=None,
                     parent_stress_periods=None,
                     ):
    """Sets up the perioddata DataFrame that is used to reference model
    stress period start and end times to real date time.

    Parameters
    ----------
    model : _type_
        _description_
    tdis_perioddata_config : dict
        ``perioddata:``, ``tdis:`` (MODFLOW 6 models) or ``dis:`` (MODFLOW-2005 models)
        block from the Modflow-setup configuration file.
    default_start_datetime : str, optional
        Start date for model from the tdis: options: block in the configuration file,
        or ``model.modeltime.start_datetime`` Flopy attribute. Only used
        where start_datetime information is missing, for example if a group
        for an initial steady-state period in ``tdis_perioddata_config``
        doesn't have a start_datetime: entry. By default, None, in which case
        the default start_datetime of 1970-01-01 may be applied by
        py:func:`setup_perioddata_group`.
    nper : int, optional
        Number of stress periods. Only used if nper is specified in the
        tdis: dimensions: block of the configuration file and
        not in a perioddata group.
    steady : bool, sequence or dict
        Whether each period is steady-state or transient. Only used
        if steady is specified in the tdis: or sto: configuration file
        blocks (MODFLOW 6 models) or the dis: block (MODFLOW-2005 models),
        and not in perioddata groups.
    time_units : str, optional
        Model time units, by default 'days'.
    oc_saverecord : dict, optional
        Output control settings, keyed by stress period. Only
        used to record this information in the stress period data table.
    parent_model : flopy model instance, optional
        Parent model, if model is an inset.
    parent_stress_periods : list of ints, optional
        Parent model stress periods to apply to the inset model
        (read from the parent: copy_stress_periods: item in the
        configuration file).

    Returns
    -------
    perioddata : DataFrame
        Table of stress period information with columns:

        ============== =========================================
        start_datetime Start date of each model stress period
        end_datetime   End date of each model stress period
        time           MODFLOW elapsed time, in days [#f1]_
        per            Model stress period number
        perlen         Stress period length (days)
        nstp           Number of timesteps in stress period
        tsmult         Timestep multiplier
        steady         Steady-state or transient
        oc             Output control setting for MODFLOW
        parent_sp      Corresponding parent model stress period
        ============== =========================================

    Notes
    -----
    perioddata is also saved to stress_period_data.csv in the tables folder
    (usually `/tables`).

    .. rubric:: Footnotes

    .. [#f1] Modflow elapsed time includes the time lengths specified for
        any steady-state periods (at least 1 day). Therefore if the model
        has an initial steady-state period with a ``perlen`` of one day,
        the elapsed time at the model start date will already be 1 day.
    """
    # get start_date_time from parent if available and start_date_time wasn't specified
    # only apply to tdis_perioddata_config if it wasn't specified there
    if tdis_perioddata_config.get('start_datetime', '1970-01-01') == '1970-01-01' and \
            default_start_datetime != '1970-01-01':
        tdis_perioddata_config['start_date_time'] = default_start_datetime

    # option to define stress periods in table prior to model build
    if 'csvfile' in tdis_perioddata_config:
        csvfile = Path(model._config_path) / tdis_perioddata_config['csvfile']['filename']
        perioddata = pd.read_csv(csvfile)
        defaults = {
            'start_datetime_column': 'start_datetime',
            'end_datetime_column': 'end_datetime',
            'steady_column': 'steady',
            'nstp_column': 'nstp',
            'tsmult_column': 'tsmult'
        }

        csv_config = tdis_perioddata_config['csvfile']
        renames = {csv_config.get(k): v
                   for k, v in defaults.items() if k in csv_config}
        perioddata.rename(columns=renames, inplace=True)
        required_cols = defaults.values()
        for col in required_cols:
            if col not in perioddata.columns:
                raise KeyError(f"{col} column missing in supplied stress "
                               f"period table {csvfile}.")
        perioddata['start_datetime'] = pd.to_datetime(perioddata['start_datetime'])
        perioddata['end_datetime'] = pd.to_datetime(perioddata['end_datetime'])
        perioddata['per'] = np.arange(len(perioddata))
        perlen = getattr((perioddata['end_datetime'] -
                          perioddata['start_datetime']).dt,
                          model.time_units).tolist()
        # set initial steady-state stress period to at least length 1
        if perioddata['steady'][0] and perlen[0] < 1:
            perlen[0] = 1
        perioddata['perlen'] = perlen
        perioddata['time'] = np.cumsum(perlen)
        cols = ['start_datetime', 'end_datetime', 'time',
                'per', 'perlen', 'nstp', 'tsmult', 'steady']
        # option to supply Output Contorl INstructions as well
        if 'oc' in perioddata.columns:
            cols.append('oc')
        perioddata = perioddata[cols]
        # some validation
        assert np.all(perioddata['perlen'] > 0)
        assert np.all(np.diff(perioddata['time']) > 0)
    # define stress periods from perioddata group blocks in configuration file
    else:
        perioddata_groups = parse_perioddata_groups(tdis_perioddata_config,
                                                    nper=nper, steady=steady,
                                                    start_date_time=default_start_datetime)
        # set up the perioddata table from the groups
        perioddata = concat_periodata_groups(perioddata_groups, time_units)

    # assign parent model stress periods to each inset model stress period
    parent_sp = None
    if parent_model is not None:
        if parent_stress_periods is not None:
            # parent_sp has parent model stress period corresponding
            # to each inset model stress period (len=nper)
            # the same parent stress period can be specified for multiple inset model periods
            parent_sp = get_parent_stress_periods(parent_model, nper=len(perioddata),
                                                    parent_stress_periods=parent_stress_periods)
        elif model._is_lgr:
            parent_sp = perioddata['per'].values

        # add corresponding stress periods in parent model if there are any
        perioddata['parent_sp'] = parent_sp
    assert np.array_equal(perioddata['per'].values, np.arange(len(perioddata)))
    return perioddata


def aggregate_dataframe_to_stress_period(data, id_column, data_column, datetime_column='datetime',
                                         end_datetime_column=None, category_column=None,
                                         start_datetime=None, end_datetime=None, period_stat='mean',
                                         resolve_duplicates_with='raise error'):
    """Aggregate time-series data in a DataFrame to a single value representing
    a period defined by a start and end date.

    Parameters
    ----------
    data : DataFrame
        Must have an id_column, data_column, datetime_column, and optionally,
        an end_datetime_column.
    id_column : str
        Column in data with location identifier (e.g. node or well id).
    data_column : str or list
        Column(s) in data with values to aggregate.
    datetime_column : str
        Column in data with times for each value. For downsampling of multiple values in data
        to a longer period represented by start_datetime and end_datetime, this is all that is needed.
        Aggregated values will include values in datetime_column that are >= start_datetime and < end_datetime.
        In other words, datetime_column represents the start of each time interval in data.
        Values can be strings (e.g. YYYY-MM-DD) or pandas Timestamps. By default, None.
    end_datetime_column : str
        Column in data with end times for period represented by each value. This is only needed
        for upsampling, where the interval defined by start_datetime and end_datetime is smaller
        than the time intervals in data. The row(s) in data that have a datetime_column value < end_datetime,
        and an end_datetime_column value > start_datetime will be retained in aggregated.
        Values can be strings (e.g. YYYY-MM-DD) or pandas Timestamps. By default, None.
    start_datetime : str or pandas.Timestamp
        Start time of aggregation period. Only used if an aggregation start
        and end time are not given in period_stat. If None, and no start
        and end time are specified in period_stat, the first time in datetime_column is used.
        By default, None.
    end_datetime : str or pandas.Timestamp
        End time of aggregation period. Only used if an aggregation start
        and end time are not given in period_stat. If None, and no start
        and end time are specified in period_stat, the last time in datetime_column is used.
        By default, None.
    period_stat : str, list, or NoneType
        Method for aggregating data. By default, 'mean'.

        * Strings will be passed to DataFrame.groupby
          as the aggregation method. For example, ``'mean'`` would result in DataFrame.groupby().mean().
        * If period_stat is None, ``'mean'`` is used.
        * Lists of length 2 can be used to specify a statistic for a month (e.g. ``['mean', 'august']``),
          or for a time period that can be represented as a single string in pandas.
          For example, ``['mean', '2014']`` would average all values in the year 2014; ``['mean', '2014-01']``
          would average all values in January of 2014, etc. Basically, if the string
          can be used to slice a DataFrame or Series, it can be used here.
        * Lists of length 3 can be used to specify a statistic and a start and end date.
          For example, ``['mean', '2014-01-01', '2014-03-31']`` would average the values for
          the first three months of 2014.
    resolve_duplicates_with : {'sum', 'mean', 'first', 'raise error'}
        Method for reducing duplicates (of times, sites and measured or estimated category).
        By default, 'raise error' will result in a ValueError if duplicates are encountered.
        Otherwise any aggregate method in pandas can be used (e.g. DataFrame.groupby().<method>())

    Returns
    -------
    aggregated : DataFrame
        Aggregated values. Columns are the same as data, except the time column
        is named 'start_datetime'. In other words, aggregated periods are represented by
        their start dates (as opposed to midpoint dates or end dates).

    """
    data = data.copy()

    if data.index.name == datetime_column:
        data.sort_index(inplace=True)
    else:
        data.sort_values(by=datetime_column, inplace=True)

    if isinstance(period_stat, str):
        period_stat = [period_stat]
    elif period_stat is None:
        period_stat = ['mean']
    else:
        period_stat = period_stat.copy()
    if isinstance(data_column, str):
        data_columns = [data_column]
    else:
        data_columns = data_column

    if len(data_columns) > 1:
        pass

    start, end = None, None
    if isinstance(period_stat, list):
        stat = period_stat.pop(0)

        # stat for specified period
        if len(period_stat) == 2:
            start, end = period_stat
            period_data = data.loc[start:end]

        # stat specified by single item
        elif len(period_stat) == 1:
            period = period_stat.pop()
            # stat for a specified month
            if period in months.keys() or period in months.values():
                period_data = data.loc[data.index.dt.month == months.get(period, period)]

            # stat for a period specified by single string (e.g. '2014', '2014-01', etc.)
            else:
                period_data = data.loc[period]

        # no time period in source data specified for statistic; use start/end of current model period
        elif len(period_stat) == 0:
            assert datetime_column in data.columns, \
                "datetime_column needed for " \
                "resampling irregular data to model stress periods"
            if data[datetime_column].dtype == object:
                data[datetime_column] = pd.to_datetime(data[datetime_column])
            if end_datetime_column in data.columns and \
                    data[end_datetime_column].dtype == object:
                data[end_datetime_column] = pd.to_datetime(data[end_datetime_column])
            if start_datetime is None:
                start_datetime = data[datetime_column].iloc[0]
            if end_datetime is None:
                end_datetime = data[datetime_column].iloc[-1]
            # >= includes the start datetime
            # if there is no end_datetime column, select values that have start_datetimes within the period
            # this excludes values that start before the period but don't have an end date
            if end_datetime_column not in data.columns:
                data_overlaps_period = (data[datetime_column] < end_datetime) & \
                                       (data[datetime_column] >= start_datetime)
            # if some end_datetimes are missing, assume end_datetime is the period end
            # this assumes that missing end datetimes indicate pumping that continues to the end of the simulation
            elif data[end_datetime_column].isna().any():
                data.loc[data[end_datetime_column].isna(), 'end_datetime'] = end_datetime
                data_overlaps_period = (data[datetime_column] < end_datetime) & \
                                       (data[end_datetime_column] >= start_datetime)
            # otherwise, select values with start datetimes that are before the period end
            # and end datetimes that are after the period start
            # in other words, include all values that overlap in time with the period
            else:
                if data[end_datetime_column].dtype == object:
                    data[end_datetime_column] = pd.to_datetime(data[end_datetime_column])
                data_overlaps_period = (data[datetime_column] < end_datetime) & \
                                       (data[end_datetime_column] > start_datetime)
            period_data = data.loc[data_overlaps_period]

        else:
            raise Exception("")

    # create category column if there is none, to conform to logic below
    categories = False
    if category_column is None:
        category_column = 'category'
        period_data[category_column] = 'measured'
    elif category_column not in period_data.columns:
        raise KeyError('category_column: {} not in data'.format(category_column))
    else:
        categories = True

    # compute statistic on data
    # ensure that ids are unique in each time period
    # by summing multiple id instances by period
    # (only sum the data column)
    # check for duplicates with same time, id, and category (measured vs estimated)
    duplicated = pd.Series(list(zip(period_data[datetime_column],
                                    period_data[id_column],
                                    period_data[category_column]))).duplicated()
    aggregated = period_data.groupby(id_column).first()
    for data_column in data_columns:
        if any(duplicated):
            if resolve_duplicates_with == 'raise error':
                duplicate_info = period_data.loc[duplicated.values]
                msg = 'The following locations are duplicates which need to be resolved:\n'.format(duplicate_info.__str__())
                raise ValueError(msg)
            period_data.index.name = None
            by_period = period_data.groupby([id_column, datetime_column]).first().reset_index()
            agg_groupedby = getattr(period_data.groupby([id_column, datetime_column]),
                                    resolve_duplicates_with)(numeric_only=True)
            by_period[data_column] = agg_groupedby[data_column].values
            period_data = by_period
        agg_groupedby = getattr(period_data.groupby(id_column), stat)(numeric_only=True)
        aggregated[data_column] = agg_groupedby[data_column].values
    # if category column was argued, get counts of measured vs estimated
    # for each measurement location, for current stress period
    if categories:
        counts = period_data.groupby([id_column, category_column]).size().unstack(fill_value=0)
        for col in 'measured', 'estimated':
            if col not in counts.columns:
                counts[col] = 0
            aggregated['n_{}'.format(col)] = counts[col]
    aggregated.reset_index(inplace=True)

    # add datetime back in
    aggregated['start_datetime'] = start if start is not None else start_datetime
    # enforce consistent datetime dtypes
    # (otherwise pd.concat of multiple outputs from this function may fail)
    for col in 'start_datetime', 'end_datetime':
        if col in aggregated.columns:
            aggregated[col] = aggregated[col].astype('datetime64[ns]')

    # drop original datetime column, which doesn't reflect dates for period averages
    drop_cols = [datetime_column]
    if not categories:  # drop category column if it was created
        drop_cols.append(category_column)
    aggregated.drop(drop_cols, axis=1, inplace=True)
    return aggregated


def aggregate_xarray_to_stress_period(data, datetime_coords_name='time',
                                      start_datetime=None, end_datetime=None,
                                      period_stat='mean'):

    period_stat = copy.copy(period_stat)
    if isinstance(start_datetime, pd.Timestamp):
        start_datetime = start_datetime.strftime('%Y-%m-%d')
    if isinstance(end_datetime, pd.Timestamp):
        end_datetime = end_datetime.strftime('%Y-%m-%d')
    if isinstance(period_stat, str):
        period_stat = [period_stat]
    elif period_stat is None:
        period_stat = ['mean']

    if isinstance(period_stat, list):
        stat = period_stat.pop(0)

        # stat for specified period
        if len(period_stat) == 2:
            start, end = period_stat
            arr = data.loc[start:end].values

        # stat specified by single item
        elif len(period_stat) == 1:
            period = period_stat.pop()
            # stat for a specified month
            if period in months.keys() or period in months.values():
                arr = data.loc[data[datetime_coords_name].dt.month == months.get(period, period)].values

            # stat for a period specified by single string (e.g. '2014', '2014-01', etc.)
            else:
                arr = data.loc[period].values

        # no period specified; use start/end of current period
        elif len(period_stat) == 0:

            assert datetime_coords_name in data.coords, \
                "datetime_column needed for " \
                "resampling irregular data to model stress periods"
            # not sure if this is needed for xarray
            if data[datetime_coords_name].dtype == object:
                data[datetime_coords_name] = pd.to_datetime(data[datetime_coords_name])
            # default to aggregating whole dataset
            # if start_ and end_datetime not provided
            if start_datetime is None:
                start_datetime = data[datetime_coords_name].values[0]
            if end_datetime is None:
                end_datetime = data[datetime_coords_name].values[-1]
            # >= includes the start datetime
            # for now, in comparison to aggregate_dataframe_to_stress_period() fn
            # for tabular data (pandas)
            # assume that xarray data does not have an end_datetime column
            # (infer the end datetimes)
            arr = data.loc[start_datetime:end_datetime].values

        else:
            raise Exception("")

    # compute statistic on data
    aggregated = getattr(arr, stat)(axis=0)

    return aggregated


def add_date_comments_to_tdis(tdis_file, start_dates, end_dates=None):
    """Add stress period start and end dates to a tdis file as comments;
    add modflow-setup version info to tdis file header.
    """
    tempfile = tdis_file + '.temp'
    shutil.copy(tdis_file, tempfile)
    with open(tempfile) as src:
        with open(tdis_file, 'w') as dest:
            header = ''
            read_header = True
            for line in src:
                if read_header and len(line) > 0 and \
                        line.strip()[0] in {'#', '!', '//'}:
                    header += line
                elif 'begin options' in ' '.join(line.lower().split()):
                    if 'modflow-setup' not in header:
                        if 'flopy' in header.lower():
                            mfsetup_text = '# via '
                        else:
                            mfsetup_text = '# File created by '
                        mfsetup_text += 'modflow-setup version {}'.format(mfsetup.__version__)
                        mfsetup_text += ' at {:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now())
                        header += mfsetup_text + '\n'
                    dest.write(header)
                    read_header = False
                    dest.write(line)
                elif 'begin perioddata' in ' '.join(line.lower().split()):
                    dest.write(line)
                    dest.write(2*' ' + '# perlen nstp tsmult\n')

                    for i, line in enumerate(src):
                        if 'end perioddata' in ' '.join(line.lower().split()):
                            dest.write(line)
                            break
                        else:
                            line = 2*' ' + line.strip() + f'  # period {i+1}: {start_dates[i]:%Y-%m-%d}'
                            if end_dates is not None:
                                line += f' to {end_dates[i]:%Y-%m-%d}'
                            line += '\n'
                            dest.write(line)
                else:
                    dest.write(line)
    os.remove(tempfile)
