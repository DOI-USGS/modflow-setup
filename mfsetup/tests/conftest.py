import os
import shutil
import pytest
from ..mf6model import MF6model


@pytest.fixture(scope="session")
def demfile():
    return 'mfsetup/tests/data/shellmound/rasters/meras_100m_dem.tif'


@pytest.fixture(scope="session")
def mfnwt_inset_test_cfg_path():
    return 'mfsetup/tests/data/mfnwt_inset_test.yml'


@pytest.fixture(scope="module")
def mf6_test_cfg_path():
    return 'mfsetup/tests/data/shellmound.yml'


@pytest.fixture(scope="module", autouse=True)
def tmpdir():
    folder = 'mfsetup/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

