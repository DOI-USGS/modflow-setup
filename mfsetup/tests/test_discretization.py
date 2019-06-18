import numpy as np
import pytest
from ..discretization import fix_model_layer_conflicts, verify_minimum_layer_thickness, fill_layers


pytest.fixture(scope="function")
def idomain(botm):
    nlay, nrow, ncol = botm.shape
    nr = int(np.floor(nrow*.35))
    nc = int(np.floor(ncol*.35))
    idomain = np.zeros((nlay, nrow, ncol), dtype=int)
    idomain[:, nr:-nr, nc:-nc] = 1
    idomain[-1, :, :] = 1
    return idomain.astype(int)


def test_conflicts():
    nlay, nrow, ncol = 13, 100, 100
    minimum_thickness = 1.0
    top = np.ones((nrow, ncol), dtype=float) * nlay
    botm = np.ones((nlay, nrow, ncol)) * np.reshape(np.arange(nlay)[::-1], (nlay, 1, 1))

    def idomain(botm):
        nlay, nrow, ncol = botm.shape
        nr = int(np.floor(nrow * .35))
        nc = int(np.floor(ncol * .35))
        idomain = np.zeros((nlay, nrow, ncol), dtype=int)
        idomain[:, nr:-nr, nc:-nc] = 1
        idomain[-1, :, :] = 1
        return idomain.astype(int)
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


def test_fill_layers():

    nlay, nrow, ncol = 10, 10, 10
    all_layers = np.zeros((nlay, nrow, ncol), dtype=float) * np.nan
    ni = 6
    nj= 3
    all_layers[0, 2:2+ni, 2:2+ni] = 10
    all_layers[2] = 8
    all_layers[5, 2:2+ni, 2:2+nj] = 5
    all_layers[9] = 1
    filled = fill_layers(all_layers)
    a = np.array([ni*ni, ni*ni, nrow*ncol,
                  ni*nj, ni*nj, ni*nj, ni*nj, ni*nj, ni*nj,
                  nrow*ncol])
    b = np.arange(1, 11, dtype=float)[::-1]
    assert np.array_equal(np.nansum(filled, axis=(1, 2)),
                          a*b)
    make_plot = False
    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(nlay):
            lw = 0.5
            if i in [0, 2, 5, 9]:
                lw = 2
            ax.plot(all_layers[i, 5, :], lw=lw)