import os
import shutil
import pytest
from ..mf6model import MF6model


@pytest.fixture(scope="session")
def mfnwt_inset_test_cfg_path():
    return 'mfsetup/tests/data/mfnwt_inset_test.yml'


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
