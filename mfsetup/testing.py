import numpy as np


def compare_float_arrays(a1, a2):
    txt = ""
    for name, where_nan in {'array 1': np.where(np.isnan(a1)),
                            'array 2': np.where(np.isnan(a2))}.items():
        nvalues = len(where_nan)
        if nvalues > 0:
            txt += "{} nans in {}: {}\n".format(nvalues, name, where_nan)
    max_abs_diff = np.nanmax(np.abs(a2 - a1))
    txt += 'Max absolute difference: {}\n'.format(max_abs_diff)
    max_rel_diff = np.nanmax(rpd(a1, a2))
    txt += 'Max relative difference: {}\n'.format(max_rel_diff)
    txt += 'RMSE: {}\n'.format(rms_error(a1, a2))
    return txt


def rms_error(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2))


def rpd(v1, v2):
    return np.abs(v1 - v2)/np.nanmean([v1, v2])


def dtypeisinteger(dtype):
    try:
        if dtype == int:
            return True
    except:
        pass
    try:
        if issubclass(dtype, np.integer):
            return True
    except:
        return False


def dtypeisfloat(dtype):
    try:
        if dtype == float:
            return True
    except:
        pass
    try:
        if issubclass(dtype, np.floating):
            return True
    except:
        return False