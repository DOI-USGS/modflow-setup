import os
import shutil
import pytest


@pytest.fixture(scope="session")
def project_root_path():
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return os.path.normpath(os.path.join(filepath, '../../'))

@pytest.fixture(scope="session")
def demfile(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound/rasters/meras_100m_dem.tif'


@pytest.fixture(scope="session")
def mfnwt_inset_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/mfnwt_inset_test.yml'


@pytest.fixture(scope="module")
def mf6_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound.yml'


@pytest.fixture(scope="module", autouse=True)
def tmpdir(project_root_path):
    folder = project_root_path + '/mfsetup/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

