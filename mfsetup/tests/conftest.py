import os
import shutil
import pytest
import flopy
fm = flopy.modflow
mf6 = flopy.mf6
from mfsetup import MF6model, MFnwtModel
from ..fileio import load_cfg
from ..utils import get_input_arguments


@pytest.fixture(scope="session")
def project_root_path():
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return os.path.normpath(os.path.join(filepath, '../../'))


@pytest.fixture(scope="session")
def demfile(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound/rasters/meras_100m_dem.tif'


@pytest.fixture(scope="session")
def pfl_nwt_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/pfl_nwt_test.yml'


@pytest.fixture(scope="function")
def pfl_nwt_cfg(pfl_nwt_test_cfg_path):
    cfg = load_cfg(pfl_nwt_test_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['model']['model_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def pfl_nwt(pfl_nwt_cfg):
    print('pytest fixture pfl_nwt')
    cfg = pfl_nwt_cfg.copy()
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="function")
def pfl_nwt_with_grid(pfl_nwt):
    print('pytest fixture pfl_nwt_with_grid')
    m = pfl_nwt  #deepcopy(pfl_nwt)
    cfg = pfl_nwt.cfg.copy()
    cfg['setup_grid']['grid_file'] = pfl_nwt.cfg['setup_grid'].pop('output_files').pop('grid_file')
    sd = cfg['setup_grid'].pop('source_data').pop('features_shapefile')
    sd['features_shapefile'] = sd.pop('filename')
    cfg['setup_grid'].update(sd)
    kwargs = get_input_arguments(cfg['setup_grid'], m.setup_grid)
    m.setup_grid(**kwargs)
    return pfl_nwt


@pytest.fixture(scope="function")
def pfl_nwt_with_dis(pfl_nwt_with_grid):
    print('pytest fixture pfl_nwt_with_dis')
    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    m.cfg['dis']['remake_arrays'] = True
    m.cfg['dis']['regrid_top_from_dem'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="function")
def pfl_nwt_with_dis_bas6(pfl_nwt_with_dis):
    print('pytest fixture pfl_nwt_with_dis_bas6')
    bas = pfl_nwt_with_dis.setup_bas6()
    return pfl_nwt_with_dis


@pytest.fixture(scope="session")
def shellmound_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound.yml'


@pytest.fixture(scope="function")
def shellmound_datapath(shellmound_cfg_path):
    return os.path.join(os.path.split(shellmound_cfg_path)[0], 'shellmound')


@pytest.fixture(scope="module")
def shellmound_cfg(shellmound_cfg_path):
    cfg = MF6model.load_cfg(shellmound_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def shellmound_simulation(shellmound_cfg):
    cfg = shellmound_cfg.copy()
    sim = mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="function")
def shellmound_model(shellmound_cfg, shellmound_simulation):
    cfg = shellmound_cfg.copy()
    cfg['model']['simulation'] = shellmound_simulation
    cfg = MF6model._parse_modflowgwf_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def shellmound_model_with_grid(shellmound_model):
    model = shellmound_model  #deepcopy(shellmound_model)
    model.setup_grid()
    return model


@pytest.fixture(scope="function")
def shellmound_model_with_dis(shellmound_model_with_grid):
    print('pytest fixture model_with_grid')
    m = shellmound_model_with_grid  #deepcopy(pfl_nwt_with_grid)
    m.setup_tdis()
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="session", autouse=True)
def tmpdir(project_root_path):
    folder = project_root_path + '/mfsetup/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['shellmound_model_with_dis',
                        'pfl_nwt_with_dis_bas6'])
def models_with_dis(request,
                    shellmound_model_with_dis,
                    pfl_nwt_with_dis_bas6):
    return {'shellmound_model_with_dis': shellmound_model_with_dis,
            'pfl_nwt_with_dis_bas6': pfl_nwt_with_dis_bas6}[request.param]