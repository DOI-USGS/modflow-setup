"""
Functions related to temporal discretization
"""
import calendar
import numpy as np
import pandas as pd
from .checks import is_valid_perioddata
from .utils import print_item

months = {v.lower(): k for k, v in enumerate(calendar.month_name) if k > 0}


def convert_freq_to_period_start(freq):
    """convert pandas frequency to period start"""
    if isinstance(freq, str):
        for prefix in ['M', 'Q', 'A', 'Y']:
            if prefix in freq.upper() and "S" not in freq.upper():
                freq = freq.replace(prefix, "{}S".format(prefix)).upper()
        return freq


def parse_perioddata_groups(perioddata_dict, defaults={}):
    """Reorganize input in perioddata dict into
    a list group (dicts).
    """
    perioddata = perioddata_dict.copy()
    perioddata_groups = []
    group0 = defaults.copy()

    valid_txt = "if transient: perlen specified or 3 of start_date_time, " \
                "end_date_time, nper or freq;\n" \
                "if steady: nper or perlen specified. Default perlen " \
                "for steady-state periods is 1."
    for k, v in perioddata.items():
        if 'group' in k.lower():
            data = defaults.copy()
            data.update(v)
            if is_valid_perioddata(data):
                perioddata_groups.append(data)
            else:
                print_item(k, data)
                prefix = "perioddata input for {} must have".format(k)
                raise Exception(prefix + valid_txt)
        else:
            group0[k] = v
    if len(perioddata_groups) == 0:
        if not is_valid_perioddata(group0):
            print_item('perioddata:', group0)
            prefix = "perioddata input must have"
            raise Exception(prefix + valid_txt)

        perioddata_groups = [group0]
    for group in perioddata_groups:
        if 'steady' in group:
            if np.isscalar(group['steady']):
                group['steady'] = {0: group['steady']}
            elif not isinstance(group['steady'], dict):
                group['steady'] = {i: s for i, s in enumerate(group['steady'])}
    return perioddata_groups


def setup_perioddata_group(start_date_time, end_date_time=None,
                           nper=1, perlen=None, model_time_units=None, freq=None,
                           steady={0: True,
                             1: False},
                           nstp=10, tsmult=1.5,
                           oc_saverecord={0: ['save head last',
                             'save budget last']},
                           ):
    """Sets up time discretization for a model; outputs a DataFrame with
    stress period dates/times and properties. Stress periods can be established
    with an established explicitly by specifying perlen as a list of period lengths in
    model units. Or, stress periods can be established using three of the
    start_date, end_date_time, nper, and freq arguments, similar to the
    pandas.date_range function.
    (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

    Parameters
    ----------
    start_date_time_time : str or datetime-like
        Left bound for generating stress period dates. See pandas documenation.
    end_date_time : str or datetime-like, optional
        Right bound for generating stress period dates. See pandas documenation.
    nper : int, optional
        Number of stress periods. Only used if perlen is None, or in combination with freq
        if an end_date_time isn't specified.
    perlen : sequence or None, optional
        A list of stress period lengths in model time units. Or specify as None and
        specify 3 of start_date_time, end_date_time, nper and/or freq.
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
        information for another period is specified.
    nstp : int or sequence
        Number of timesteps in a stress period. Must be a integer if perlen=None.
    nstp : int or sequence
        Timestep multiplier for a stress period. Must be a integer if perlen=None.
    oc_saverecord : dict
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
        time : float; cumulative MODFLOW time at end of period
            (includes steady-state periods)
        per : int, zero-based stress period
        perlen : float; stress period length in model time units
        nstp : int; number of timesteps in the stress period
        tsmult : int; timestep multiplier for stress period
        steady : bool; True=steady-state, False=Transient
        oc : dict; MODFLOW-6 output control options

    """
    freq = convert_freq_to_period_start(freq)
    oc = oc_saverecord

    txt = "Specify perlen as a list of lengths in model units, or\nspecify 3 " \
          "of start_date_time, end_date_time, nper and/or freq."
    # Explicitly specified stress period lengths
    if perlen is not None:
        if np.isscalar(perlen):
            perlen = [perlen]
        datetimes = [pd.Timestamp(start_date_time)]
        if len(perlen) > 1:
            for i, length in enumerate(perlen[1:]):
                datetimes.append(datetimes[i] + pd.Timedelta(length, unit=model_time_units))
        time = np.cumsum(perlen) # time in MODFLOW units
    elif nper == 1 and steady[0]:
        perlen = [1]
        time = [1]
        #datetimes = [pd.Timestamp(start_date_time)]

    # Set up datetimes based on 3 of start_date_time, end_date_time, nper and/or freq (scalar perlen)
    else:
        assert np.isscalar(nstp), "nstp: {}; nstp must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(nstp, txt)
        assert np.isscalar(tsmult), "tsmult: {}; tsmult must be a scalar if perlen " \
                                  "is not specified explicitly as a list.\n{}".format(tsmult, txt)
        periods = None
        if end_date_time is None:
            # start_date_time, periods and freq
            # (i.e. nper periods of length perlen starting on stat_date)
            if freq is not None:
                periods = nper
            else:
                raise ValueError("Unrecognized input for perlen: {}.\n{}".format(perlen, txt))
        else:
            # end_date_time and freq and periods
            if start_date_time is None:
                periods = nper + 1
            # start_date_time, end_date_time and (linearly spaced) periods
            # (i.e. nper periods of uniform length between start_date_time and end_date_time)
            elif freq is None:
                periods = nper #-1 if steady[0] else nper
            # start_date_time, end_date_time and frequency
            elif freq is not None:
                pass
        datetimes = pd.date_range(start_date_time, end_date_time, periods=periods, freq=freq)
        if start_date_time is None:
            start_date_time = datetimes[0]  # in case end_date_time, periods and freq were specified
        if len(datetimes) == 1:
            perlen = [(pd.Timestamp(end_date_time) - pd.Timestamp(start_date_time)).days]
            time = np.array(perlen)
        else:
            # time is at the end of each stress period
            time = getattr((datetimes - pd.Timestamp(start_date_time)), model_time_units).tolist()

            # get the last (end) time, if it wasn't included in datetimes
            if datetimes[0] == pd.Timestamp(start_date_time):
                if end_date_time is not None:
                    last_time = getattr((pd.Timestamp(end_date_time) -
                                         pd.Timestamp(start_date_time)),
                                        model_time_units)
                else:
                    end_datetimes = pd.date_range(start_date_time,
                                                  periods=len(datetimes) + 1,
                                                  freq=freq)
                    last_time = getattr((end_datetimes[-1] -
                                         pd.Timestamp(start_date_time)),
                                         model_time_units)
                if last_time != time[-1]:
                    time += [last_time]
            if time[0] != 0:
                time = [0] + time
            perlen = np.diff(time)
            time = np.array(time[1:])
            assert len(perlen) == len(time)# == len(datetimes)

        # if first period is steady-state,
        # insert it at the beginning of the generated range
        # this should only apply to cases where nper > 1
        if steady[0]:
            #datetimes = [datetimes[0]] + datetimes.tolist()  #  datetimes[:-1].tolist()
            perlen = [1] + list(perlen)
            time = [1] + (time + 1).tolist()
        else:
            pass
            #datetimes = datetimes[:-1]
            #perlen = np.diff(time).tolist()
            #time = time[1:]

    perioddata = pd.DataFrame({#'datetime': datetimes,
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

    # set up output control, using previous value to fill empty periods
    # (same as MF6)
    oclist = [None]
    for i in range(len(perioddata)):
        oclist.append(oc.get(i, oclist[i]))
    perioddata['oc'] = oclist[1:]

    # create start and end datetime columns;
    # correct the datetime to only increment for transient stress periods
    start_datetime = [pd.Timestamp(start_date_time)]
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
    #perioddata = perioddata.drop('datetime', axis=1)[cols]

    # correct nstp and tsmult to be 1 for steady-state periods
    perioddata.loc[perioddata.steady, 'nstp'] = 1
    perioddata.loc[perioddata.steady, 'tsmult'] = 1
    return perioddata


def setup_perioddata(cfg, time_units='days'):
    # get period data groups
    defaults = {'start_date_time': cfg['tdis']['options'].get('start_date_time'),
                'nper': cfg['tdis'].get('dimensions', {}).get('nper'),
                'steady': cfg['sto']['steady'],
                'oc_saverecord': cfg['oc'].get('saverecord', {0: ['save head last',
                                                                  'save budget last']})
                }
    perioddata_groups = parse_perioddata_groups(cfg['tdis']['perioddata'], defaults)

    # update any missing variables in the groups with global variables
    group_dfs = []
    for i, group in enumerate(perioddata_groups):
        group.update({'model_time_units': time_units,
                      })
        df = setup_perioddata_group(**group)
        group_dfs.append(df)

    # concatenate groups into single dataframe of perioddata
    perioddata = concat_periodata_groups(group_dfs)
    return perioddata


def concat_periodata_groups(groups):
    """Concatenate multiple perioddata DataFrames, but sort
    result on (absolute) datetimes and increment model time and stress period
    numbers accordingly."""
    df = pd.concat(groups).sort_values(by=['end_datetime'])
    perlen = np.ones(len(df))
    perlen[~df.steady.values] = df.loc[~df.steady.values, 'perlen']
    df['time'] = np.cumsum(perlen)
    df['per'] = range(len(df))
    df.index = range(len(df))
    return df


def aggregate_dataframe_to_stress_period(data, start_datetime, end_datetime,
                                         period_stat, id_column, data_column):
    """

    Parameters
    ----------
    data : pd.DataFrame
        Must have 'start_datetime' and 'end_datetime' columns with dates
        indicating the time bounds associated with each row (e.g. the time
        period for a specified model stress)
    start_datetime
    end_datetime
    period_stat
    id_column : column with location identifier (e.g. node or well id)
    data_column : column with data to aggregate (e.g. fluxes)

    Returns
    -------

    Notes
    -----
    pandas reindex could potentially be used for cases where
    the source time series needs to be resampled to the model stress periods
    e.g.
    def reindex_by_date(df):
        dates = [start_datetime, end_datetime]
        return df.reindex(dates, method='nearest')
    welldata.groupby('node').apply(reindex_by_date).reset_index(0, drop=True)
    """

    #if isinstance(start_datetime, pd.Timestamp):
    #    start_datetime = start_datetime.strftime('%Y-%m-%d')
    #if isinstance(end_datetime, pd.Timestamp):
    #    end_datetime = end_datetime.strftime('%Y-%m-%d')
    if isinstance(period_stat, str):
        period_stat = [period_stat]
    elif period_stat is None:
        period_stat = ['mean']
    if isinstance(data_column, str):
        data_columns = [data_column]
    else:
        data_columns = data_column

    if len(data_columns) > 1:
        pass

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
            assert 'start_datetime' in data.columns and 'end_datetime' in data.columns, \
                "start_datetime and end_datetime columns needed for " \
                "resampling irregular data to model stress periods"
            for col in ['start_datetime', 'end_datetime']:
                if data[col].dtype == np.object:
                    data[col] = pd.to_datetime(data[col])
            data_overlaps_period = (data.start_datetime < end_datetime) & \
                                   (data.end_datetime > start_datetime)
            period_data = data.loc[data_overlaps_period]

        else:
            raise Exception("")

    # compute statistic on data
    # ensure that ids are unique in each time period
    # by summing multiple id instances by period
    # (only sum the data column)
    if len(period_data) > 0:
        period_data.index.name = None
        by_period = period_data.groupby([id_column, 'start_datetime']).first().reset_index()
        by_period[data_column] = period_data.groupby([id_column, 'start_datetime']).sum()[data_column].values
        period_data = by_period
    aggregated = period_data.groupby(id_column).first().reset_index()
    aggregated[data_column] = getattr(period_data.groupby(id_column), stat)()[data_column].values
    aggregated['start_datetime'] = start_datetime # add datetime back in
    #aggregated.index.name = None
    #aggregated[id_column] = aggregated.index
    return aggregated


def aggregate_xarray_to_stress_period(data, start_datetime, end_datetime,
                                      period_stat, datetime_column):

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
                arr = data.loc[data[datetime_column].dt.month == months.get(period, period)].values

            # stat for a period specified by single string (e.g. '2014', '2014-01', etc.)
            else:
                arr = data.loc[period].values

        # no period specified; use start/end of current period
        elif len(period_stat) == 0:
            arr = data.loc[start_datetime:end_datetime].values

        else:
            raise Exception("")

    # compute statistic on data
    period_stat = getattr(arr, stat)(axis=0)

    return period_stat