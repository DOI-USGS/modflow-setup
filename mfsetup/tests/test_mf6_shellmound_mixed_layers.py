"""
Test setting up the shellmound model layering with a mix of raster surfaces and uniform elevation values
(for incorporating property info from a voxel grid, for example)
"""
import os

import flopy
import pytest

mf6 = flopy.mf6
from mfsetup.fileio import load_cfg
from mfsetup.mf6model import MF6model
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="module")
def shellmound_cfg(project_root_path):
    shellmound_cfg_path = project_root_path / 'mfsetup/tests/data/shellmound_flat_layers.yml'
    cfg = load_cfg(shellmound_cfg_path, default_file='mf6_defaults.yml')
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
