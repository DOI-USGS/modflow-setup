"""
Test lake package functionality
"""
import os

import numpy as np
import pandas as pd
import pytest

from mfsetup import MF6model
from mfsetup.fileio import load_array
from mfsetup.lakes import (
    PrismSourceData,
    get_flux_variable_from_config,
    get_horizontal_connections,
    setup_lake_connectiondata,
    setup_lake_info,
)
from mfsetup.tests.test_pleasant_mf6_inset import pleasant_mf6_cfg


@pytest.fixture
def source_data_from_prism_cases(project_root_path):
    cases = [{'climate':
                  {'filenames':
                       {600059060: os.path.join(project_root_path, 'examples/data/pleasant/source_data/PRISM_ppt_tmean_stable_4km_189501_201901_43.9850_-89.5522.csv')
                        },
                   'format': 'prism',
                   'period_stats':
                       {0: ['mean', '2012-01-01', '2012-12-31'],  # average daily rate for model period for initial steady state
                        1: 'mean'}
                   }
              }
    ]
    return cases


def get_prism_data(prism_datafile, model_start, model_end):
    """Read data from prism file; subset to model time period
    and convert units."""
    df = pd.read_csv(prism_datafile, header=None,
                     skiprows=11, names=['datetime', 'ppt_inches', 'tmean'])
    df.index = pd.to_datetime(df.datetime)

    # subset data to the model timeperiod
    df_model = df.loc[model_start:model_end].iloc[:-1]  # drop last day

    # compute mean precip in meters/day
    df_model['ppt_m'] = df_model['ppt_inches']*.3048/12
    df_model['ppt_md'] = df_model['ppt_m']/df_model.index.days_in_month
    df_model['tmean_c'] = (5/9) * (df_model['tmean'] - 32)
    return df_model


def test_parse_prism_source_data(source_data_from_prism_cases, pleasant_nwt_with_grid):
    m = pleasant_nwt_with_grid
    cases = source_data_from_prism_cases
    sd = PrismSourceData.from_config(cases[0]['climate'], dest_model=m)
    data = sd.get_data()
    assert np.array_equal(data.per, m.perioddata.per)
    prism = get_prism_data(sd.filenames[600059060],
                           m.perioddata['start_datetime'][0],
                           m.perioddata['end_datetime'].values[-1])
    assert np.allclose(data.loc[data.per == 0, 'temp'], prism['tmean_c'].mean())
    assert np.allclose(data.loc[data.per == 0, 'precipitation'], prism['ppt_md'].mean())
    assert np.allclose(data.loc[1:, 'temp'], prism['tmean_c'])
    assert np.allclose(data.loc[1:, 'precipitation'], prism['ppt_md'])


def test_setup_lake_info(get_pleasant_mf6_with_dis):

    m = get_pleasant_mf6_with_dis
    result = setup_lake_info(m)
    for id in result.lak_id:
        loc = m._lakarr_2d == id
        strt = result.loc[result.lak_id == id, 'strt'].values[0]
        assert np.allclose(strt, m.dis.top.array[loc].min())

    # test setting up lake info without any lake package
    del m.cfg['lak']
    result = setup_lake_info(m)
    assert result is None
    assert m.high_k_lake_recharge is None


def test_setup_lake_connectiondata(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    df = setup_lake_connectiondata(m, for_external_file=False)
    df['k'], df['i'], df['j'] = zip(*df['cellid'])
    vertical_connections = df.loc[df.claktype == 'vertical']
    lakezones = load_array(m.cfg['intermediate_data']['lakzones'][0])
    litleak = m.cfg['lak']['source_data']['littoral_leakance']
    profleak = m.cfg['lak']['source_data']['profundal_leakance']

    # 2D array showing all vertical connections in lake package
    lakarr2d_6 = np.zeros((m.nrow, m.ncol), dtype=bool)
    lakarr2d_6[vertical_connections.i.values, vertical_connections.j.values] = True

    # verify that number of vert. connection locations is consistent between lakarr and mf6 list input
    assert np.sum(m.lakarr.sum(axis=0) > 0) == np.sum(lakarr2d_6)

    # verify that there is only one vertical connection at each location
    ij_locations = set(zip(vertical_connections.i, vertical_connections.j))
    assert len(vertical_connections) == len(ij_locations)

    # verify that the connections are in the same place (horizontally)
    assert not np.any((m.lakarr.sum(axis=0) > 0) != lakarr2d_6)

    # check that the number of vert. connections in each layer is consistent
    lake_thickness = (m.lakarr > 0).sum(axis=0)
    for k in range(1, m.nlay+1):

        # lake connections in current layer
        i, j = np.where(lake_thickness == k)
        highest_active_layer = np.argmax(m.idomain[:, i, j], axis=0)
        connection_cellids = list(zip(highest_active_layer, i, j))
        kvc = vertical_connections.loc[vertical_connections.cellid.isin(connection_cellids)]
        # by definition, number of vert. connections in kvc is same as # cells with lake_thickness == k

        # verity that specified leakances are consistent with lake zones
        assert np.sum(kvc.bedleak == profleak) == np.sum(lakezones[lake_thickness == k] == 100)
        assert np.sum(kvc.bedleak == litleak) == np.sum(lakezones[lake_thickness == k] == 1)


@pytest.mark.parametrize('connection_info', (False, True))
def test_get_horizontal_connections(tmpdir, connection_info):
    nlay, nrow, ncol = 2, 20, 20
    lakarr = np.zeros((nlay, nrow, ncol))
    lakarr[0, 4, 7] = 1
    lakarr[0, 11:15, 9] = 1
    lakarr[0, 15, 10] = 1
    lakarr[0, 14, 8] = 1
    lakarr[0, 11:13, 6] = 1
    lakarr[0, 3:5, 8] = 1
    lakarr[0, 6:10, 5:-5] = 1
    lakarr[0, 5:11, 6:-6] = 1
    lakarr[1, 7:-7, 7:-7] = 1
    layer_elevations = np.zeros((nlay + 1, nrow, ncol))
    layer_elevations[0] = 2
    layer_elevations[1] = 1
    delr = np.ones(ncol)
    delc = np.ones(nrow)
    # get_horizontal_connections finds connections
    # from areas == 1 to areas == 0
    # the returned cells are always within the area == 1
    # lakarr has non-lake cells == 0
    # we want connections from non-lake cells to lake cells
    # (which are conceputalized as not being part of the gwflow solution)
    # so we need to invert lakarr so that non-lake cells == 1
    lakarr_inv = (lakarr == 0).astype(int)
    # cval is the
    connections = get_horizontal_connections(lakarr_inv, connection_info=connection_info,
                                             layer_elevations=layer_elevations,
                                             delr=delr, delc=delc)

    from scipy.ndimage import sobel

    try:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(2, 2)
        ax = ax.flat
        ax[0].imshow(lakarr[0])
        ax[0].set_title('lake extent')

        # sobel
        sobel_x = sobel(lakarr_inv[0], axis=1, mode='reflect') #, cval=1.)
        sobel_x[lakarr_inv[0] == 0] = 0
        sobel_y = sobel(lakarr_inv[0], axis=0, mode='reflect') #, cval=1.)
        sobel_y[lakarr_inv[0] == 0] = 0
        im = ax[1].imshow(sobel_x)
        ax[1].set_title('sobel filter applied in x direction')
        ax[2].imshow(sobel_y)
        ax[2].set_title('sobel filter applied in y direction')
        fig.colorbar(im, ax=ax[1])
        fig.colorbar(im, ax=ax[2])

        # get_horizontal_connections
        k, i, j = zip(*connections.cellid)
        hconn = np.zeros(lakarr.shape)
        hconn[0][lakarr[0] == 1] = 1
        hconn[k, i, j] = 2
        im = ax[3].imshow(hconn[0])
        ax[3].set_title('results from\nlakes.get_horizontal_connections()')
        plt.tight_layout()
        plt.savefig(os.path.join(tmpdir, 'horizontal_connections.pdf'))
    except:
        pass

    for k, lakarr2d in enumerate(lakarr_inv):
        sobel_x = sobel(lakarr2d, axis=1, mode='reflect') #, cval=1.)
        sobel_x[lakarr2d == 0] = 0
        sobel_y = sobel(lakarr2d, axis=0, mode='reflect') #, cval=1.)
        sobel_y[lakarr2d == 0] = 0

        ncon = np.sum(np.abs(sobel_x) > 1) + np.sum(np.abs(sobel_y) > 1)
        assert ncon == np.sum(connections['k'] == k)


@pytest.mark.parametrize('modflow_version', ('mf2005', 'mf6'))
@pytest.mark.parametrize('config, nlakes, nper, expected', (
    # single global value; gets repeated to all lakes and periods
    ({'precipitation': 0.01}, 1, 1, [0.01]),
    ({'precipitation': 0.01}, 2, 1, [0.01] * 2),
    ({'precipitation': 0.01}, 2, 2, [0.01] * 4),
    # steady value for each lake
    ({'precipitation': {0: 0.01, 1: 0.02}}, 2, 2, [0.01]*2 + [0.02]*2),
    ({'precipitation': [0.01, 0.02]}, 2, 2, [0.01, 0.02]*2),
    #pytest.param({'precipitation': [0.01, 0.02]}, 2, 2, None,
    #             marks=pytest.mark.xfail(reason='multiple lakes require dictionary input')),
    # time-varying values for one or more lakes
    ({'precipitation': [0.01, 0.011]}, 1, 2, [0.01, 0.011]),
    ({'precipitation': [0.01, 0.011]}, 1, 3, [0.01, 0.011, 0.011]),
    pytest.param({'precipitation': [0.01, 0.011, 0.012]}, 1, 2, None,
                 marks=pytest.mark.xfail(reason='more values than periods')),
    ({'precipitation': {0: [0.01, 0.011],
                        1: [0.02, 0.021]}}, 2, 2, [0.01, 0.011, 0.02, 0.021]),
    # last value gets repeated to remaining stress periods
    ({'precipitation': {0: [0.01, 0.011],
                        1: [0.02, 0.021]}}, 2, 3,
     [0.01, 0.011, 0.011, 0.02, 0.021, 0.021]),
    pytest.param({'precipitation': {0: [0.01, 0.011],
                        1: [0.02, 0.021]}}, 3, 3, None,
                 marks=pytest.mark.xfail(reason='no values for lake 3')),
    pytest.param({'precipitation': {0: [0.01, 0.011],
                        2: [0.02, 0.021]}}, 2, 3, None,
                 marks=pytest.mark.xfail(reason='no values for lake 1 or nonconsecutive lake numbers')),

))
def test_get_flux_variable_from_config(modflow_version, config, nlakes, nper, expected
                                       ):
    variable = 'precipitation'
    if modflow_version == 'mf6':
        config = {'perioddata': config}
    results = get_flux_variable_from_config(variable, config, nlakes, nper)
    # repeat values for each lake
    assert results == expected


def test_lake_idomain(pleasant_mf6_cfg):
    """For MODFLOW 6 models (at least), if no bathymetry raster is supplied,
    the lake should be simulated 'above' the grid, connecting vertically with
    the top model layer, and cells within the lake footprint should only
    be inactive if they are pinched out based on the original layer surfaces
    (cells containing lake shouldn't be inactive).
    """
    cfg = pleasant_mf6_cfg.copy()
    del cfg['lak']['source_data']['bathymetry_raster']
    m = MF6model(cfg=cfg)
    m.setup_dis()
    m.setup_tdis()
    m.setup_lak()


    k, i, j = zip(*m.lak.connectiondata.array['cellid'])
    # only a small number of cells
    # should have connections to layers below the top layer
    # note that this 5% criterian might change
    # if the test problem changes substantially
    assert np.sum(np.array(k) > 0)/np.sum(np.array(k) == 0) < 0.05
    # all connections should be to active cells
    assert all(m.dis.idomain.array[k, i, j] == 1)
