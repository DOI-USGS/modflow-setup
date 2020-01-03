"""
Test lake package functionality
"""
import numpy as np
import pandas as pd
import pytest
from mfsetup.fileio import load_array
from mfsetup.lakes import (PrismSourceData, setup_lake_info,
                           setup_lake_connectiondata,
                           get_horizontal_connections)


@pytest.fixture
def source_data_from_prism_cases():
    cases = [{'climate':
                  {'filenames':
                       {600059060: 'pleasant/source_data/PRISM_ppt_tmean_stable_4km_189501_201901_43.9850_-89.5522.csv'
                        },
                   'format': 'prism',
                   'period_stats':
                       {0: ['mean', '2012-01-01', '2018-12-31'],  # average daily rate for model period for initial steady state
                        1: 'mean'}
                   }
              }
    ]
    return cases


def get_prism_data(prism_datafile):
    """Read data from prism file; subset to model time period
    and convert units."""
    df = pd.read_csv(prism_datafile, header=None,
                     skiprows=11, names=['datetime', 'ppt_inches', 'tmean'])
    df.index = pd.to_datetime(df.datetime)

    # subset data to the model timeperiod
    model_start, model_end = '2012-01-01', '2018-12-31'
    df_model = df.loc[model_start:model_end]

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
    prism = get_prism_data(sd.filenames[600059060])
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
    assert m.lake_recharge is None


def test_setup_lake_connectiondata(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    df = setup_lake_connectiondata(m)
    df['k'] = [c[0] for c in df['cellid']]
    vertical_connections = df.loc[df.claktype == 'vertical']
    lakezones = load_array(m.cfg['intermediate_data']['lakzones'][0])
    litleak = m.cfg['lak']['source_data']['littoral_leakance']
    profleak = m.cfg['lak']['source_data']['profundal_leakance']
    for k, lakarr2d in enumerate(m.lakarr):
        kvc = vertical_connections.loc[vertical_connections.k == k]
        assert np.sum(lakarr2d > 0) == len(kvc)
        assert np.sum(kvc.bedleak == profleak) == np.sum(lakezones == 100)
        assert np.sum(kvc.bedleak == litleak) > 0
        assert np.sum(kvc.bedleak == litleak) <= np.sum(lakezones == 1)


    j=2


def test_get_horizontal_connections():
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
    connections = get_horizontal_connections(lakarr, layer_elevations, delr, delc)
    connections['k'] = [c[0] for c in connections['cellid']]

    from scipy.ndimage import sobel
    for k, lakarr2d in enumerate(lakarr):
        sobel_x = sobel(lakarr2d, axis=1, mode='constant', cval=0.)
        sobel_x[lakarr2d == 1] = 0
        sobel_y = sobel(lakarr2d, axis=0, mode='constant', cval=0.)
        sobel_y[lakarr2d == 1] = 0

        ncon = np.sum(np.abs(sobel_x) > 1) + np.sum(np.abs(sobel_y) > 1)
        assert ncon == np.sum(connections['k'] == k)
