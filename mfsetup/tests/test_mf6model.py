import sys
sys.path.append('..')
import time
from copy import deepcopy
import shutil
import os
import pytest
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
import flopy
mf6 = flopy.mf6
from ..discretization import get_layer_thicknesses
from ..fileio import load
from ..mf6model import MF6model
from ..utils import get_input_arguments


@pytest.fixture(scope="module")
def cfg(mf6_test_cfg_path):
    cfg = MF6model.load_cfg(mf6_test_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="module", autouse=True)
def reset_dirs(cfg):
    cfg = cfg.copy()
    folders = [cfg['intermediate_data']['output_folder'],
               cfg.get('external_path', cfg['model'].get('external_path')),
               cfg['gisdir']
               ]
    for folder in folders:
        #if not os.path.isdir(folder):
        #    os.makedirs(folder)
        #else:
        #    shutil.rmtree(folder)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


@pytest.fixture(scope="function")
def simulation(cfg):
    cfg = cfg.copy()
    sim = mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="function")
def model(cfg, simulation):
    cfg = cfg.copy()
    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = simulation
    kwargs = get_input_arguments(cfg['model'], MF6model)
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def model_with_grid(model):
    #model = deepcopy(model)
    model.setup_grid()
    return model


@pytest.fixture(scope="function")
def model_with_dis(model_with_grid):
    print('pytest fixture model_with_grid')
    m = model_with_grid  #deepcopy(inset_with_grid)
    m.setup_tdis()
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="module")
def model_setup(cfg_path):
    for folder in ['shellmound', 'tmp']:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    m = MF6model.setup_from_yaml(cfg_path)
    m.write()
    return m


def test_load_cfg(cfg, mf6_test_cfg_path):
    relative_model_ws = '../tmp/shellmound'
    ws = os.path.normpath(os.path.join(os.path.abspath(os.path.split(mf6_test_cfg_path)[0]),
                                                                       relative_model_ws))
    cfg = cfg
    assert cfg['simulation']['sim_ws'] == ws
    assert cfg['intermediate_data']['output_folder'] == os.path.join(ws, 'tmp')


def test_simulation(simulation):
    assert True


def test_model(model):
    assert True


def test_model_with_grid(model_with_grid):
    assert True


def test_external_file_path_setup(model):

    m = model #deepcopy(model)

    assert os.path.exists(os.path.join(m.cfg['simulation']['sim_ws'],
                                       m.external_path))
    top_filename = m.cfg['dis']['top_filename']
    botm_file_fmt = m.cfg['dis']['botm_filename_fmt']
    m.setup_external_filepaths('dis', 'top',
                                   top_filename,
                                   nfiles=1)
    m.setup_external_filepaths('dis', 'botm',
                                   botm_file_fmt,
                                   nfiles=m.nlay)
    assert m.cfg['intermediate_data']['top'] == \
           [os.path.normpath(os.path.join(m.tmpdir, os.path.split(top_filename)[-1]))]
    assert m.cfg['intermediate_data']['botm'] == \
           [os.path.normpath(os.path.join(m.tmpdir, botm_file_fmt).format(i))
                                  for i in range(m.nlay)]
    assert m.cfg['dis']['top'] == \
           [{'filename': os.path.normpath(os.path.join(m.model_ws,
                        m.external_path,
                        os.path.split(top_filename)[-1]))}]
    assert m.cfg['dis']['botm'] == \
           [{'filename': os.path.normpath(os.path.join(m.model_ws,
                         m.external_path,
                         botm_file_fmt.format(i)))} for i in range(m.nlay)]


def test_perrioddata(model):
    m = model #deepcopy(model)
    pd0 = m.perioddata.copy()
    assert pd0 is not None

    m.cfg['sto']['steady'] = {0: True,
                              1: False}
    # Explicit stress period setup
    m.cfg['tdis']['options']['start_date_time'] = '2008-10-01'
    m.cfg['tdis']['perioddata']['perlen'] = [1] * 11
    m.cfg['tdis']['perioddata']['nstp'] = [5] * 11
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd1 = m.perioddata.copy()
    assert pd1['start_datetime'][0] == pd1['start_datetime'][1] == pd1['end_datetime'][0]
    assert pd1['end_datetime'][1] == pd.Timestamp(m.cfg['tdis']['options']['start_date_time']) + \
           pd.Timedelta(m.cfg['tdis']['perioddata']['perlen'][1], unit=m.time_units)
    assert pd1['nstp'][0] == 1
    assert pd1['tsmult'][0] == 1

    # Start date, freq and nper
    m.cfg['tdis']['options']['end_date_time'] = None
    m.cfg['tdis']['perioddata']['perlen'] = None
    m.cfg['tdis']['dimensions']['nper'] = 11
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m.cfg['tdis']['perioddata']['nstp'] = 5
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd2 = m.perioddata.copy()
    assert pd2.equals(pd1)

    # Start date, end date, and nper
    m.cfg['tdis']['options']['end_date_time'] = '2008-10-11'
    m.cfg['tdis']['perioddata']['freq'] = None
    m._perioddata = None
    pd3 = m.perioddata.copy()
    assert pd3.equals(pd1)

    # Start date, end date, and freq
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m._perioddata = None
    pd4 = m.perioddata.copy()
    assert pd4.equals(pd1)

    # end date, freq and nper
    m.cfg['tdis']['options']['start_date_time'] = None
    m._perioddata = None
    pd5 = m.perioddata.copy()
    assert pd5.equals(pd1)

    # month end vs month start freq
    m.cfg['tdis']['perioddata']['freq'] = '6M'
    m.cfg['tdis']['options']['start_date_time'] = '2008-10-01'
    m.cfg['tdis']['options']['end_date_time'] = '2016-10-01'
    m.cfg['tdis']['perioddata']['nstp'] = 15
    m._perioddata = None
    pd6 = m.perioddata.copy()
    assert pd6.equals(pd0)


def test_set_lakarr(model_with_dis):
    m = model_with_dis
    if 'lak' in m.package_list:
        lakes_shapefile = m.cfg['lak'].get('source_data', {}).get('lakes_shapefile')
        assert lakes_shapefile is not None
        assert m._lakarr2d.sum() > 0
        assert m._isbc2d.sum() > 0  # requires
        assert m.isbc.sum() > 0  # requires DIS package
        assert m.lakarr.sum() > 0  # requires isbc to be set
        if m.version == 'mf6':
            externalfiles = m.cfg['external_files']['lakarr']
        else:
            externalfiles = m.cfg['intermediate_data']['lakarr']
        assert isinstance(externalfiles, dict)
        assert isinstance(externalfiles[0], list)
        for f in externalfiles[0]:
            assert os.path.exists(f)
    else:
        assert m._lakarr2d.sum() == 0
        assert m._isbc2d.sum() == 0
        assert m.isbc.sum() == 0  # requires DIS package
        assert m.lakarr.sum() == 0  # requires isbc to be set


def test_dis_setup(model_with_grid):

    m = model_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    assert isinstance(dis, mf6.ModflowGwfdis)
    assert 'DIS' in m.get_package_list()
    arrayfiles = m.cfg['intermediate_data']['top'] + \
                 m.cfg['intermediate_data']['botm'] + \
                 m.cfg['intermediate_data']['idomain']
    for f in arrayfiles:
        assert os.path.exists(f)

    # test idomain
    top = dis.top.array.copy()
    top[top == m._nodata_value] = np.nan
    botm = dis.botm.array.copy()
    botm[botm == m._nodata_value] = np.nan
    thickness = get_layer_thicknesses(top, botm)
    invalid_botms = np.ones_like(botm)
    invalid_botms[np.isnan(botm)] = 0
    invalid_botms[thickness < 1.0001] = 0
    assert np.array_equal(m.idomain.sum(axis=(1, 2)),
                          invalid_botms.sum(axis=(1, 2)))

    # test writing of top array from
    # intermediate array
    m.remove_package('dis')
    m.cfg['dis']['remake_top'] = False
    dis = m.setup_dis()
    dis.write()
    arrayfiles = m.cfg['dis']['top']
    for f in arrayfiles:
        assert os.path.exists(f['filename'])
    assert os.path.exists(os.path.join(m.model_ws, dis.filename))

    # check that units were converted (or not)
    assert np.allclose(dis.top.array.mean(), 126, atol=10)
    mcaq = m.cfg['dis']['source_data']['botm']['filenames'][3]
    assert 'mcaq' in mcaq
    with rasterio.open(mcaq) as src:
        mcaq_data = src.read(1)
        mcaq_data[mcaq_data == src.meta['nodata']] = np.nan
    assert np.allclose(m.dis.botm.array[3].mean() / .3048, np.nanmean(mcaq_data), atol=5)


def test_tdis_setup(model):

    m = model #deepcopy(model)
    tdis = m.setup_tdis()
    assert isinstance(tdis, mf6.ModflowTdis)
    period_df = pd.DataFrame(tdis.perioddata.array)
    period_df['perlen'] = period_df['perlen'].astype(float)
    assert period_df.equals(m.perioddata[['perlen', 'nstp', 'tsmult']])


def test_sto_setup(model_with_dis):

    m = model_with_dis  #deepcopy(model_with_grid)
    sto = m.setup_sto()
    assert isinstance(sto, mf6.ModflowGwfsto)
    assert np.allclose(sto.sy.array.mean(), m.cfg['sto']['griddata']['sy'])
    assert np.allclose(sto.ss.array.mean(), m.cfg['sto']['griddata']['ss'])


def test_yaml_setup(mf6_test_cfg_path):
    m = model_setup  #deepcopy(model_setup)
    try:
        success, buff = m.run_model(silent=False)
    except:
        pass
    #assert success, 'model run did not terminate successfully'

