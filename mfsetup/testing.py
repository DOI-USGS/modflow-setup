import numpy as np


def rms_error(array1, array2):
    return np.sqrt(np.mean((array1 - array2) ** 2))


def rpd(v1, v2):
    return np.abs(v1 - v2)/np.mean([v1, v2])


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