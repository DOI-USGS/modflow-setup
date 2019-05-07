import numpy as np
import pytest
from ..discretization import fix_model_layer_conflicts, verify_minimum_layer_thickness


@pytest.fixture(scope="module")
def top():
    def top(nlay, nrow, ncol):
        return np.ones((nrow, ncol), dtype=float) * nlay
    return top

@pytest.fixture(scope="module")
def botm():
    def botm(nlay, nrow, ncol):
        return np.ones((nlay, nrow, ncol)) * np.reshape(np.arange(nlay)[::-1], (nlay, 1, 1))
    return botm

@pytest.fixture(scope="module")
def idomain(botm):
    """Make an idomain with some inactive cells"""
    def idomain(botm):
        nlay, nrow, ncol = botm.shape
        nr = int(np.floor(nrow*.35))
        nc = int(np.floor(ncol*.35))
        idomain = np.zeros((nlay, nrow, ncol), dtype=int)
        idomain[:, nr:-nr, nc:-nc] = 1
        idomain[-1, :, :] = 1
        return idomain.astype(int)
    return idomain

def test_conflicts(top, botm, idomain):
    nlay, nrow, ncol = 13, 100, 100
    minimum_thickness = 1.0
    top = top(nlay, nrow, ncol)
    botm = botm(nlay, nrow, ncol)
    idomain = idomain(botm)

    isvalid = verify_minimum_layer_thickness(top, botm, idomain, minimum_thickness)
    assert isvalid
    botm[0, 0, 0] = -1
    isvalid = verify_minimum_layer_thickness(top, botm, idomain, minimum_thickness)
    assert isvalid
    i, j = int(nrow/2), int(ncol/2)
    botm[0, i, j] = -1
    isvalid = verify_minimum_layer_thickness(top, botm, idomain, minimum_thickness)
    assert not isvalid
    botm2 = fix_model_layer_conflicts(top, botm, idomain, minimum_thickness)
    isvalid = verify_minimum_layer_thickness(top, botm2, idomain, minimum_thickness)
    assert isvalid