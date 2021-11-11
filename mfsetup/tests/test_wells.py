# TODO: tests for functions in wells.py
import platform
import sys

import numpy as np
import pandas as pd
import pytest

from mfsetup.wells import (
    assign_layers_from_screen_top_botm,
    get_open_interval_thickness,
    get_package_stress_period_data,
    setup_wel_data,
)


@pytest.fixture(scope='function')
def all_layers(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    all_layers = np.zeros((m.nlay+1, m.nrow, m.ncol))
    all_layers[0] = m.dis.top.array
    all_layers[1:] = m.dis.botm.array
    return all_layers


def test_minimum_well_layer_thickness(shellmound_model_with_dis, all_layers):

    m = shellmound_model_with_dis
    minthickness = 2
    m.cfg['wel']['source_data']['csvfiles']['vertical_flux_distribution']\
        ['minimum_layer_thickness'] = minthickness
    df = setup_wel_data(m, for_external_files=False)
    assert np.all((-np.diff(all_layers, axis=0))[df.k, df.i, df.j] > minthickness)


def test_assign_layers_from_screen_top_botm(shellmound_model_with_dis, test_data_path):
    model = shellmound_model_with_dis
    model.setup_npf()

    # edit the dis and hk arrays at a chosen location
    all_layers = np.stack([model.dis.top.array] + [l for l in model.dis.botm.array])
    i, j = 10, 10
    thicknesses = np.ones(len(all_layers)) * 10
    thicknesses[0] = 0
    thicknesses[1:6] *= 100
    # reset the layer elevations
    all_layers[:, i, j] = all_layers[0, 10, 10] - np.add.accumulate(thicknesses)
    model.dis.botm = all_layers[1:]

    # reset hk, so that thickest layers at i, j
    # have low T
    hk = model.npf.k.array.copy()
    hk[:, i, j] = 1e-3
    hk[6:] = 100
    model.npf.k = hk

    # input well data
    welldata = pd.DataFrame({
        'i': [i],
        'j': [j],
        'q': [0.],
        'screen_top': [all_layers[0, i, j]],
        'screen_botm': [all_layers[-1, i, j]],
        'site_no': 'test_well'
    })
    results = assign_layers_from_screen_top_botm(welldata, model,
                                                 flux_col='q',
                                                 screen_top_col='screen_top',
                                                 screen_botm_col='screen_botm',
                                                 label_col='site_no',
                                                 across_layers=False,
                                                 distribute_by='transmissivity',
                                                 minimum_layer_thickness=10.)
    assert results.loc[0, 'k'] == 12
    results = assign_layers_from_screen_top_botm(welldata, model,
                                                 flux_col='q',
                                                 screen_top_col='screen_top',
                                                 screen_botm_col='screen_botm',
                                                 label_col='site_no',
                                                 across_layers=False,
                                                 distribute_by='thickness',
                                                 minimum_layer_thickness=10.)
    assert results.loc[0, 'k'] == 4

    # test effect of starting heads (sat. thickness) on well placement
    model.cfg['ic']['griddata']['strt'] = model.dis.botm.array[5]
    model.setup_ic()
    results = assign_layers_from_screen_top_botm(welldata, model,
                                                 flux_col='q',
                                                 screen_top_col='screen_top',
                                                 screen_botm_col='screen_botm',
                                                 label_col='site_no',
                                                 across_layers=False,
                                                 distribute_by='thickness',
                                                 minimum_layer_thickness=10.)
    assert results.loc[0, 'k'] == 12


def test_get_open_interval_thicknesses(shellmound_model_with_dis, all_layers):

    m = shellmound_model_with_dis
    i = [10] * m.ncol
    j = list(range(m.ncol))
    screen_top = all_layers[0, i, j]
    screen_botm = all_layers[-1:, i, j]
    b = get_open_interval_thickness(m, i=i, j=j,
                                    screen_top=screen_top,
                                    screen_botm=screen_botm)
    diffs = -np.diff(all_layers[:, i, j], axis=0)
    layers_from_diffs = np.argmax(diffs, axis=0)
    layers = np.argmax(b, axis=0)
    assert np.array_equal(layers, layers_from_diffs)
    assert np.allclose(b[layers, list(range(b.shape[1]))],
                       diffs[layers_from_diffs, list(range(diffs.shape[1]))])
    assert np.allclose(b[:, 0], -np.diff(all_layers[:, i[0], j[0]]))
    # TODO: test with partially penetrating wells


@pytest.mark.skipif((sys.version_info < (3,9)) & ('macos' in platform.platform().lower()),
                    reason='obscure decode issue with pfl nwt model top external file')
def test_get_package_stress_period_data(models_with_dis):
    m = models_with_dis
    m.cfg['wel']['external_files'] = False
    wel = m.setup_wel()
    result = get_package_stress_period_data(models_with_dis, package_name='wel')
    assert isinstance(result, pd.DataFrame)
    assert len({'k', 'i', 'j'}.intersection(result.columns)) == 3
    assert 'cellid' not in result.columns
    if models_with_dis.name == 'shellmound':
        assert np.array_equal(result.per.unique(),
                              np.arange(1, models_with_dis.nper))
    elif models_with_dis.name == 'pfl':
        assert np.array_equal(result.per.unique(),
                              np.arange(models_with_dis.nper))
