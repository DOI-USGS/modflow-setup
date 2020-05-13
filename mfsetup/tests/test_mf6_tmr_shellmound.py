import os
import flopy.mf6 as mf6
import pytest
from mfsetup import MF6model
from mfsetup.fileio import load_cfg
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def shellmound_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound_inset2.yml'


@pytest.fixture(scope="function")
def shellmound_datapath(shellmound_cfg_path):
    return os.path.join(os.path.split(shellmound_cfg_path)[0], 'shellmound')


@pytest.fixture(scope="module")
def shellmound_cfg(shellmound_cfg_path):
    cfg = load_cfg(shellmound_cfg_path, default_file='/mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def shellmound_simulation(shellmound_cfg):
    cfg = shellmound_cfg.copy()
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    return sim


@pytest.fixture(scope="function")
def shellmound_model(shellmound_cfg, shellmound_simulation):
    cfg = shellmound_cfg.copy()
    cfg['model']['simulation'] = shellmound_simulation
    cfg = MF6model._parse_model_kwargs(cfg)
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