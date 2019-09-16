import numpy as np
from ..testing import dtypeisfloat, dtypeisinteger


def test_dtypeisfloat():

    assert dtypeisfloat(float)
    assert dtypeisfloat(np.float)
    assert dtypeisfloat(np.float32)
    assert dtypeisfloat(np.float64)
    assert not dtypeisfloat(int)
    assert not dtypeisfloat(np.int64)


def test_dtypeisinteger():
    assert dtypeisinteger(int)
    assert dtypeisinteger(np.int)
    assert dtypeisinteger(np.int32)
    assert dtypeisinteger(np.int64)
    assert not dtypeisinteger(float)
    assert not dtypeisinteger(np.float64)

