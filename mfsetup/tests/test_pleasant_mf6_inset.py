"""
Tests for Pleasant Lake inset case, MODFLOW-6 version
* creating MODFLOW-6 inset model from MODFLOW-NWT parent
* MODFLOW-6 Lake package
"""
import copy
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import pytest
import flopy
mf6 = flopy.mf6
from mfsetup import MF6model
from mfsetup.discretization import (get_layer_thicknesses, find_remove_isolated_cells)
from mfsetup.fileio import load_cfg, load_array
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pleasant_mf6_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/pleasant_mf6_test.yml'


@pytest.fixture(scope="function")
def pleasant_mf6_cfg(pleasant_mf6_test_cfg_path):
    cfg = load_cfg(pleasant_mf6_test_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def pleasant_simulation(pleasant_mf6_cfg):
    cfg = pleasant_mf6_cfg.copy()
    sim = mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="function")
def get_pleasant_mf6(pleasant_mf6_cfg, pleasant_simulation):
    print('creating Pleasant Lake MF6model instance from cfgfile...')
    cfg = pleasant_mf6_cfg.copy()
    cfg['model']['simulation'] = pleasant_simulation
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def get_pleasant_mf6_with_grid(get_pleasant_mf6):
    print('creating Pleasant Lake MFnwtModel instance with grid...')
    m = copy.deepcopy(get_pleasant_mf6)
    cfg = m.cfg.copy()
    cfg['setup_grid']['grid_file'] = m.cfg['setup_grid'].pop('output_files').pop('grid_file')
    sd = cfg['setup_grid'].pop('source_data').pop('features_shapefile')
    sd['features_shapefile'] = sd.pop('filename')
    cfg['setup_grid'].update(sd)
    kwargs = get_input_arguments(cfg['setup_grid'], m.setup_grid)
    m.setup_grid(**kwargs)
    return m


def test_dis_setup(get_pleasant_mf6_with_grid):

    m = get_pleasant_mf6_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    botm = m.dis.botm.array.copy()
    assert isinstance(dis, mf6.ModflowGwfdis)
    assert 'DIS' in m.get_package_list()
    # verify that units got conveted correctly
    assert m.dis.top.array.mean() < 100
    assert m.dis.length_units.array == 'meters'

    arrayfiles = m.cfg['intermediate_data']['top'] + \
                 m.cfg['intermediate_data']['botm'] + \
                 m.cfg['intermediate_data']['idomain']
    for f in arrayfiles:
        assert os.path.exists(f)
        fname = os.path.splitext(os.path.split(f)[1])[0]
        k = ''.join([s for s in fname if s.isdigit()])
        var = fname.strip(k)
        data = np.loadtxt(f)
        model_array = getattr(m.dis, var).array
        if len(k) > 0:
            k = int(k)
            model_array = model_array[k]
        assert np.array_equal(model_array, data)

   # test that written idomain array reflects supplied shapefile of active area
    active_area = rasterize(m.cfg['dis']['source_data']['idomain']['filename'],
                            m.modelgrid)
    isactive = active_area == 1
    written_idomain = load_array(m.cfg['dis']['griddata']['idomain'])
    assert np.all(written_idomain[:, ~isactive] <= 0)

    # test idomain from just layer elevations
    del m.cfg['dis']['griddata']['idomain']
    dis = m.setup_dis()
    top = dis.top.array.copy()
    top[top == m._nodata_value] = np.nan
    botm = dis.botm.array.copy()
    botm[botm == m._nodata_value] = np.nan
    thickness = get_layer_thicknesses(top, botm)
    invalid_botms = np.ones_like(botm)
    invalid_botms[np.isnan(botm)] = 0
    invalid_botms[thickness < 1.0001] = 0
    # these two arrays are not equal
    # because isolated cells haven't been removed from the second one
    # this verifies that _set_idomain is removing them
    assert not np.array_equal(m.idomain[:, isactive].sum(axis=1),
                          invalid_botms[:, isactive].sum(axis=1))
    invalid_botms = find_remove_isolated_cells(invalid_botms, minimum_cluster_size=10)
    active_cells = m.idomain[:, isactive].copy()
    active_cells[active_cells < 0] = 0  # need to do this because some idomain cells are -1
    assert np.array_equal(active_cells.sum(axis=1),
                          invalid_botms[:, isactive].sum(axis=1))

    # test recreating package from external arrays
    m.remove_package('dis')
    assert m.cfg['dis']['griddata']['top'] is not None
    assert m.cfg['dis']['griddata']['botm'] is not None
    dis = m.setup_dis()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array == 1],
                          botm[m.dis.idomain.array == 1])

    # test recreating just the top from the external array
    m.remove_package('dis')
    m.cfg['dis']['remake_top'] = False
    m.cfg['dis']['griddata']['botm'] = None
    dis = m.setup_dis()
    dis.write()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array == 1],
                          botm[m.dis.idomain.array == 1])
    arrayfiles = m.cfg['dis']['griddata']['top']
    for f in arrayfiles:
        assert os.path.exists(f['filename'])
    assert os.path.exists(os.path.join(m.model_ws, dis.filename))

    # dis package idomain should be consistent with model property
    updated_idomain = m.idomain
    assert np.array_equal(m.dis.idomain.array, updated_idomain)

    # check that units were converted (or not)
    assert np.allclose(dis.top.array.mean(), 40, atol=10)
    mcaq = m.cfg['dis']['source_data']['botm']['filenames'][3]
    assert 'mcaq' in mcaq
    with rasterio.open(mcaq) as src:
        mcaq_data = src.read(1)
        mcaq_data[mcaq_data == src.meta['nodata']] = np.nan
    assert np.allclose(m.dis.botm.array[3].mean() / .3048, np.nanmean(mcaq_data), atol=5)