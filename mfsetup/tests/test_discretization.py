import numpy as np
import pandas as pd
import pytest

from mfsetup.discretization import (
    create_vertical_pass_through_cells,
    deactivate_idomain_above,
    fill_cells_vertically,
    fill_empty_layers,
    find_remove_isolated_cells,
    fix_model_layer_conflicts,
    get_layer_thicknesses,
    make_ibound,
    make_idomain,
    make_irch,
    populate_values,
    verify_minimum_layer_thickness,
    voxels_to_layers,
)


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


def test_conflicts_negative_layers():
    top = np.ones((2, 2), dtype=float) * 115
    botm = [130.0, 110.0, 90.0, 70.0, 50.0, 30.0, 10.0, -10.0, -30.0, -50.0, -70.0, -90.0, -110.0, -130.0,
            -6.611952336182595, -609.1836436464844]
    botm = np.transpose(np.array(botm) * np.ones((2, 2, len(botm))))
    result = fix_model_layer_conflicts(top, botm, minimum_thickness=1)


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
        from matplotlib import pyplot as plt
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

    botm = fill_cells_vertically(top, botm)
    filled = all_layers.copy()
    filled[0] = top
    filled[1:] = botm
    assert filled[:, 2, 2].tolist() == [10., 10.,  8.,  8.,  8.,  5.,  5.,  5.,  5, 1.]
    # only compare layers 1:
    # because fill_cells_vertically doesn't modify the model top
    # (assuming that in most cases, there will be value
    # for the model top elevation at each cell location)
    assert filled[1:, 0, 0].tolist() == [8.] * 7 + [2., 1.]


def test_make_ibound(all_layers):
    top = all_layers[0].copy()
    botm = all_layers[1:].copy()
    nodata = -9999
    botm[-1, 0, 0] = nodata
    botm[-2] = 2
    botm[-2, 2, 2] = np.nan
    botm[:, 3, 3] = np.arange(1, 10)[::-1] # column of cells all eq. to min thickness
    filled_botm = fill_cells_vertically(top, botm)
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

    # docstring examples
    top = np.reshape([[10]]* 4, (2, 2))
    botm = np.reshape([[np.nan,  8., np.nan, np.nan, np.nan,  2., np.nan]]*4, (2, 2, 7)).transpose(2, 0, 1)
    result = get_layer_thicknesses(top, botm)
    np.testing.assert_almost_equal(result[:, 0, 0], [np.nan,  2., np.nan, np.nan, np.nan,  6., np.nan])

    top = np.reshape([[10]] * 4, (2, 2))
    botm = np.reshape([[9, 8., 8, 6, 3, 2., -10]] * 4, (2, 2, 7)).transpose(2, 0, 1)
    result = get_layer_thicknesses(top, botm)
    assert np.allclose(result[:, 0, 0], [1.,  1., 0., 2., 3.,  1., 12.])

    all_layers2 = np.stack([top] + [b for b in botm])
    assert np.allclose(np.abs(np.diff(all_layers2, axis=0)), result)


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


def test_populate_values():
    v = populate_values({0: 1.0, 2: 2.0})
    assert v == {0: 1.0, 1: 1.5, 2: 2.0}
    v = populate_values({0: 1.0, 2: 2.0}, array_shape=(2, 2))
    assert [v.shape for k, v in v.items()] == [(2, 2)] * 3
    assert [v.sum() for k, v in v.items()] == [4, 6, 8]


@pytest.mark.parametrize('model_botm', (np.array([[[10, -5, 0], # 3D botm array
                                                  [10, 5, 0],
                                                  [9, 5, 0]],
                                                 [[9, -6, 0],
                                                  [10, 4, 0],
                                                  [7, 4, 0]]]),
                                        np.array([[10, -5, 0],  # 2D botm array
                                                  [10, 5, 0],
                                                  [9, 5, 0]])
                                        ))
@pytest.mark.parametrize('z_edges', ([20, 15, 10, 5, 0],  # 1D array of voxel edges (uniform elevations)
                                     # 2D array of voxel edges
                                     (np.ones((3, 3, 5)) * np.array([20, 15, 10, 5, 0])).transpose(2, 0, 1)
                                     ))
def test_voxels_to_layers(z_edges, model_botm):
    data = np.array([[[5, 4, 0],
                      [5, 4, 3],
                      [5, 4, 0]],
                     [[5, 4, 0],
                      [5, 4, 3],
                      [5, 4, 3]],
                     [[5, 4, 0],
                      [5, 4, 3],
                      [0, 0, 0]],
                     [[5, 4, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                     ])
    nlay, nrow, ncol = data.shape
    model_top = np.array([[25, 20, 19],
                          [20, 20, 20],
                          [20, 20, 16]])
    result = voxels_to_layers(data, z_edges, model_top=model_top,
                              model_botm=model_botm, no_data_value=0)
    if len(model_botm.shape) == 2:
        n_botm_lay = 1
    else:
        n_botm_lay = model_botm.shape[0]
    assert result.shape == (nlay+1+n_botm_lay, nrow, ncol)  # voxel top edge + botm layer
    assert result[-1, 0, 0] == 0  # botm was reset to 0 in this location
    # at this location, last two layers have no data, so same bottom as second layer
    # extend_botm=False by default, so another layer is added, extending from lowest voxel edge to botm
    assert np.allclose(result[:, 2, 0], [20., 15., 10., 10., 10., 9., 7.][:result.shape[0]])
    # at this location, model botm is at 10, but voxel with botm=5 has data,
    # so model botm is pushed downward to 5.
    assert np.allclose(result[:, 1, 0], [20., 15., 10., 5, 5, 5, 5][:result.shape[0]])

    assert result[0, 2, 2] == 16  # consistent with model top

    result = voxels_to_layers(data, z_edges, model_top=model_top,
                              model_botm=model_botm, extend_top=False, no_data_value=0)
    assert result.shape == (nlay+2+n_botm_lay, nrow, ncol)  # voxel top edge + top layer + botm layer

    result = voxels_to_layers(data, z_edges, model_top=model_top,
                              model_botm=model_botm, extend_botm=True, no_data_value=0)
    assert result[-1, 0, 1] == -5  # voxel edge was reset to -5 in this location
    assert np.allclose(result[:, 0, 2], [19., 19., 19., 19.,  0.])


def test_make_irch(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    irch = make_irch(m.idomain)

    m.setup_rch()
    m.rch.write()
    written_irch = np.loadtxt('external/irch.dat')
    idm_argmax = np.argmax(m.idomain, axis=0)
    assert np.allclose(written_irch, irch)
    assert np.allclose(written_irch, idm_argmax + 1)
