import os
import shutil
import pytest
from ..mfmodel import MF6model

@pytest.fixture(scope="module")
def cfg_path():
    return 'mfsetup/tests/data/shellmound.yml'

@pytest.fixture(scope="module")
def cfg(cfg_path):
    cfg = MF6model.load_cfg(cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg

@pytest.fixture(scope="module", autouse=True)
def tmpdir():
    folder = 'mfsetup/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder


@pytest.fixture(scope="module", autouse=True)
def reset_dirs(cfg):
    folders = [cfg['intermediate_data']['tmpdir'],
               cfg['external_path'],
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