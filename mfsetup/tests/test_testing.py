import numpy as np
import pytest

from mfsetup.testing import dtypeisfloat, dtypeisinteger, point_is_on_nhg


def test_rtree():
    pass


def test_dtypeisfloat():

    assert dtypeisfloat(float)
    assert dtypeisfloat(np.float32)
    assert dtypeisfloat(np.float64)
    assert not dtypeisfloat(int)
    assert not dtypeisfloat(np.int64)


def test_dtypeisinteger():
    assert dtypeisinteger(int)
    assert dtypeisinteger(np.int32)
    assert dtypeisinteger(np.int64)
    assert not dtypeisinteger(float)
    assert not dtypeisinteger(np.float64)


@pytest.mark.parametrize('x,y', ((177955.0, 939285.0),
                                 (np.arange(5) * 1000 + 177955.0,
                                  np.arange(5) * 1000 + 939285.0),
                                 (np.arange(20) * 250 + 177955.0,
                                  np.arange(20) * 250 + 939285.0)
                                 ))
def test_point_is_on_nhg(x, y):
    assert point_is_on_nhg(x, y, offset='edge')
