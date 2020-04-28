import numpy as np
from ..testing import (dtypeisfloat, dtypeisinteger, point_is_on_nhg)


def test_rtree():
    from rtree import index


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


def test_point_is_on_nhg():
    x = np.arange(5) * 1000 + 177955.0
    y = np.arange(5) * 1000 + 939285.0
    assert point_is_on_nhg(x, y, offset='edge')

