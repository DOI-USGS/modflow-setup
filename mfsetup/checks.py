"""
Module with functions to check input data.
"""
import numpy as np
from .fileio import load_array


def is_valid_perioddata(data):
    """Check that a dictionary of period data has enough information
    (based on key names) to set up stress periods.
    Perlen must be explicitly input, or 3 of start_date_time, end_date_time,
    nper and/or freq must be specified. This is analogous to the input
    requirements for the pandas.date_range method for generating
    time discretization
    (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)
    """
    perlen = data.get('perlen') is not None
    steady = data.get('steady', False)
    if isinstance(steady, dict):
        steady = steady.get(0)
    if not steady:
        included = [k for k in ['nper', 'start_date_time', 'end_date_time', 'freq']
                    if data.get(k) is not None]
        has3 = len(included) >= 3
        return perlen or has3
    else:
        nper = data.get('nper') is not None
        return nper or perlen


def check_external_files_for_nans(files_list):
    has_nans = []
    for f in files_list:
        arr = load_array(f)
        if np.any(np.isnan(arr)):
            has_nans.append(f)
    return has_nans
