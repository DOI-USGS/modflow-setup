import os

import pytest

from mfsetup import __version__
from mfsetup.model_version import get_versions
from mfsetup.tests.conftest import basic_model_instance


@pytest.mark.parametrize('start_version', (
    #'0' start_version=0 is the default, and
    # only produces result below if there are no tags
    '3.0.0',
    '3.0',
    '3'
                                           )
                         )
def test_get_versions(start_version):
    rest = get_versions(path='.',
                        start_version='')
    result = get_versions(path='.',
                          start_version=start_version)
    #
    assert result['version'] == start_version + rest['version']


def test_get_version_without_git(project_root_path):
    path = os.path.join(project_root_path, '..')
    result = get_versions(path=path,
                          start_version='0')
    assert result['version'] == '0+unknown'


def test_model_version(basic_model_instance):
    version = basic_model_instance.model_version
    # check that model_version attribute of for test cases
    # is reading the git version info the same as versioneer
    assert version['version'] == __version__


def test_write_model_version(basic_model_instance):
    m = basic_model_instance
    os.chdir(m._abs_model_ws)
    version = m.model_version
    m.setup_dis()
    m.setup_tdis()
    m.setup_oc()
    m.setup_solver()
    m.write_input()
