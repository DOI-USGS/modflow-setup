"""
Tests for utils.py module
"""
import pytest

from mfsetup.utils import flatten, update


@pytest.fixture(scope="function")
def multilevel_dict():
    d = {'a': 1,
         'b': {'c': 3},
         'c': {'b': 2,
               'c': 0,
               'd': {'d': 4,
                     'c': 3}}}
    return d


@pytest.fixture(scope="function")
def default_config():
    defaults = {'parent': {'junk': 0},
         'model': {'name': 'junk'}
         }
    return defaults


@pytest.fixture(scope="function")
def specified_config():
    updates = {'model': {'name': 'modelname',
                   'model_ws': 'path'
                   }
               }
    return updates


def test_flatten(multilevel_dict):
    d = flatten(multilevel_dict)
    assert d == dict(zip(['a', 'b', 'c', 'd'], range(1, 5)))


@pytest.mark.skip(reason="need to change update() so that only the included dictionary blocks are updated")
def test_update(default_config, specified_config):
    result = update(default_config, specified_config)

    # test that only keys in specified are updated
    assert 'parent' not in result
