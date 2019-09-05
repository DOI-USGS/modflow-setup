"""
Tests for utils.py module
"""
import pytest
from ..utils import flatten


@pytest.fixture(scope="function")
def multilevel_dict():
    d = {'a': 1,
         'b': {'c': 3},
         'c': {'b': 2,
               'c': 0,
               'd': {'d': 4,
                     'c': 3}}}
    return d


def test_flatten(multilevel_dict):
    d = flatten(multilevel_dict)
    assert d == dict(zip(['a', 'b', 'c', 'd'], range(1, 5)))

