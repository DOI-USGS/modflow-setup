import numpy as np
import pandas as pd
import pytest
from ..discretization import (fix_model_layer_conflicts, verify_minimum_layer_thickness,
                              fill_empty_layers, fill_cells_vertically, make_idomain, make_ibound,
                              get_layer_thicknesses, create_vertical_pass_through_cells,
                              deactivate_idomain_above, find_remove_isolated_cells)


@pytest.fixture(scope="function")
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


@pytest.fixture
def all_layers():
    """Sample layer elevation grid where some layers
    are completely pinched out (no botm elevation specified) and
    others partially pinched out (botms specified locally).

    Returns
    -------

    """
    nlay, nrow, ncol = 9, 10, 10
    all_layers = np.zeros((nlay + 1, nrow, ncol), dtype=float) * np.nan
    ni = 6
    nj = 3
    all_layers[0, 2:2 + ni, 2:2 + ni] = 10  # locally specified botms
    # layer 1 is completely pinched out
    all_layers[2] = 8  # global botm elevation of 8
    all_layers[5, 2:2 + ni, 2:2 + nj] = 5  # locally specified botms
    all_layers[9] = 1
    return all_layers


def test_fill_layers(all_layers):
    nlay, nrow, ncol = all_layers.shape
    ni = len(set(np.where(~np.isnan(all_layers[0]))[0]))
    nj = len(set(np.where(~np.isnan(all_layers[5]))[1]))
    filled = fill_empty_layers(all_layers)
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


def test_fill_na(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()
    botm[-2] = 2
    botm[-2, 2, 2] = np.nan

    top, botm = fill_cells_vertically(top, botm)
    filled = all_layers.copy()
    filled[0] = top
    filled[1:] = botm
    assert filled[:, 2, 2].tolist() == [10., 10.,  8.,  8.,  8.,  5.,  5.,  5.,  5, 1.]
    assert filled[:, 0, 0].tolist() == [8] * 8 + [2, 1]


def test_make_ibound(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()
    nodata = -9999
    botm[-1, 0, 0] = nodata
    botm[-2] = 2
    botm[-2, 2, 2] = np.nan
    botm[:, 3, 3] = np.arange(1, 10)[::-1] # column of cells all eq. to min thickness
    filled_top, filled_botm = fill_cells_vertically(top, botm)
    ibound = make_ibound(top, botm, nodata=nodata,
                           minimum_layer_thickness=1,
                           drop_thin_cells=True,
                           tol=1e-4)
    # test ibound based on nans
    assert np.array_equal(ibound[:, 2, 2].astype(bool), ~np.isnan(botm[:, 2, 2]))
    # test ibound based on nodata
    assert ibound[-1, 0, 0] == 0
    # test ibound based on layer thickness
    # unlike for idomain, individual cells < min thickness are not deactivated
    # (unless all cells at i, j location are < min thickness + tol)
    assert ibound[-1].sum() == 98
    assert ibound[-2].sum() == 98
    # test that nans in the model top result in the highest active botms being excluded
    # (these cells have valid botms, but no tops)
    assert ibound[:, 0, 0].sum() == 1
    # in all_layers, cells with valid tops are idomain=0
    # because all botms in layer 1 are nans
    assert ibound[0].sum() == 0

    # test edge case of values that match the layer thickness when tol=0
    ibound = make_ibound(top, botm, nodata=nodata,
                           minimum_layer_thickness=1,
                           drop_thin_cells=True,
                           tol=0)
    assert ibound[-1].sum() == 99


def test_make_idomain(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()
    nodata = -9999
    botm[-1, 0, 0] = nodata
    botm[-2] = 2
    botm[-2, 2, 2] = np.nan
    idomain = make_idomain(top, botm, nodata=nodata,
                           minimum_layer_thickness=1,
                           drop_thin_cells=True,
                           tol=1e-4)
    # test idomain based on nans
    assert np.array_equal(idomain[:, 2, 2].astype(bool), ~np.isnan(botm[:, 2, 2]))
    # test idomain based on nodata
    assert idomain[-1, 0, 0] == 0
    # test idomain based on layer thickness
    assert idomain[-1].sum() == 1
    assert idomain[-2].sum() == 99
    # test that nans in the model top result in the highest active botms being excluded
    # (these cells have valid botms, but no tops)
    assert idomain[:, 0, 0].sum() == 1
    # in all_layers, cells with valid tops are idomain=0
    # because all botms in layer 1 are nans
    assert idomain[0].sum() == 0

    # test edge case of values that match the layer thickness when tol=0
    idomain = make_idomain(top, botm, nodata=nodata,
                           minimum_layer_thickness=1,
                           drop_thin_cells=True,
                           tol=0)
    assert idomain[-1].sum() == 99


def test_get_layer_thicknesses(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()

    thicknesses = get_layer_thicknesses(top, botm)
    assert thicknesses[-1, 0, 0] == 7
    b = thicknesses[:, 0, 0]
    assert np.array_equal(b[~np.isnan(b)], np.array([7.]))
    expected = np.zeros(botm.shape[0]) * np.nan
    expected[1] = 2
    expected[4] = 3
    expected[-1] = 4
    assert np.allclose(thicknesses[:, 2, 2].copy(), expected, equal_nan=True)


def test_deactivate_idomain_above(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()
    idomain = make_idomain(top, botm,
                           minimum_layer_thickness=1,
                           drop_thin_cells=True,
                           tol=1e-4)
    packagedata = pd.DataFrame({'cellid': [(2, 2, 2),
                                    (8, 3, 3)]})
    idomain2 = deactivate_idomain_above(idomain, packagedata)
    assert idomain2[:, 2, 2].sum() == idomain[:, 2, 2].sum() -1
    assert idomain2[:, 3, 3].sum() == 1


def test_find_remove_isolated_cells():
    idomain = np.zeros((2, 1000, 1000), dtype=int)
    idomain[:, 200:-200, 200:-200] = 1
    idomain[:, 202:203, 202:203] = 0
    idomain[:, 300:330, 300:301] = 0
    idomain[:, 305:306, 305:306] = 0
    idomain[:, 10:13, 10:13] = 1
    idomain[:, 20:22, 20:22] = 1
    idomain[:, -20:-18, -20:-18] = 1

    result = find_remove_isolated_cells(idomain, minimum_cluster_size=10)
    assert result.shape == idomain.shape
    # no inactive cells should have been filled
    assert result.sum() == idomain[:, 200:-200, 200:-200].sum()
    result = find_remove_isolated_cells(idomain, minimum_cluster_size=9)
    # cluster of 9 active cells was not removed
    assert result.sum() == idomain[:, 200:-200, 200:-200].sum() + 9 * 2

    idomain = np.zeros((40, 40), dtype=int)
    idomain[5:-3, 2:7] = 1
    idomain[3:5, 7:8] = 1  # 2 pixels
    idomain[10, 8:11] = 1  # 3 pixels
    idomain[20, 7] = 1  # pixel that has only one connection
    result = find_remove_isolated_cells(idomain, minimum_cluster_size=10)
    assert result.sum() == idomain.sum() - 6


def test_create_vertical_pass_through_cells():
    idomain = np.zeros((5, 3, 3), dtype=int)
    idomain[0] = 1
    idomain[-1] = 1
    idomain[:, 0, :] = 0
    idomain[0, 0, -1] = -1  # put in a passthrough cell where there shouldn't be one
    idomain[1:-1, 1, 1] = -1
    passthru = create_vertical_pass_through_cells(idomain)
    assert np.issubdtype(passthru.dtype, np.integer)
    idomain[0, 0, -1] = 0
    assert np.array_equal(passthru[[0, -1]], idomain[[0, -1]])
    assert np.all(passthru[1:-1, 1:, :] == -1)
    assert not np.any(np.all(passthru <= 0, axis=0) &
                      (np.sum(passthru, axis=0) < 0))


