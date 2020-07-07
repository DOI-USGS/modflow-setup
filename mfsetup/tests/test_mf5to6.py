import pytest

from mfsetup.mf5to6 import (
    get_package_name,
    get_variable_name,
    get_variable_package_name,
)


@pytest.mark.parametrize('version_var_expected', [('mfnwt', 'k', 'hk'),
                                                  ('mfnwt', 'k33', 'vka'),
                                                  ('mfnwt', 'botm', 'botm'),
                                                  ('mfnwt', 'idomain', 'ibound'),
                                                  ('mf6', 'hk', 'k'),
                                                  ('mf6', 'vka', 'k33'),
                                                  ('mf6', 'ibound', 'idomain'),
                                                  ('mf6', 'botm', 'botm'),
                                                  ])
def test_get_variable_name(version_var_expected):
    model_version, var, expected = version_var_expected
    result = get_variable_name(var, model_version)
    assert result == expected


@pytest.mark.parametrize('version_var_default_expected',
                         [('mfnwt', 'idomain', 'dis', 'bas6'),
                          ('mfnwt', 'strt', 'ic', 'bas6'),
                          ('mfnwt', 'botm', 'dis', 'dis'),
                          ('mfnwt', 'sy', 'npf', 'upw'),
                          ('mfnwt', 'ss', 'npf', 'upw'),
                          ('mfnwt', 'k', 'npf', 'upw'),
                          ('mfnwt', 'k33', 'npf', 'upw'),
                          ('mf6', 'ibound', 'bas6', 'dis'),
                          ('mf6', 'strt', 'bas6', 'ic'),
                          ('mf6', 'sy', 'upw', 'sto'),
                          ('mf6', 'ss', None, 'sto'),
                          ('mf6', 'hk', 'upw', 'npf'),
                          ('mf6', 'vka', None, 'npf'),
                          ('mf2005', 'botm', 'dis', 'dis'),
                          ('mf2005', 'sy', 'upw', 'lpf'),
                          ('mf2005', 'ss', 'npf', 'lpf'),
                          ('mf2005', 'k', None, 'lpf'),
                          ('mf2005', 'k33', 'npf', 'lpf'),
                          ])
def test_get_variable_package_name(version_var_default_expected):
    model_version, var, default, expected = version_var_default_expected
    result = get_variable_package_name(var, model_version, default)
    assert result == expected


@pytest.mark.parametrize('version_package_expected',
                         [('mfnwt', 'npf', {'upw'}),
                          ('mfnwt', 'sto', {'upw'}),
                          ('mfnwt', 'ic', {'bas6'}),
                          ('mfnwt', 'dis', {'dis', 'bas6'}),
                          ('mfnwt', 'tdis', {'dis'}),
                          ('mfnwt', 'oc', {'oc'}),
                          ('mf6', 'dis', {'dis', 'tdis'}),
                          ('mf6', 'upw', {'npf', 'sto'}),
                          ('mf6', 'lpf', {'npf', 'sto'}),
                          ('mf6', 'bas6', {'ic', 'dis'}),
                          ('mf6', 'oc', {'oc'}),
                          ('mf2005',  'npf', {'lpf'}),
                          ('mf2005', 'sto', {'lpf'}),
                          ('mf2005', 'ic', {'bas6'}),
                          ('mf2005', 'dis', {'dis', 'bas6'}),
                          ('mf2005', 'tdis', {'dis'}),
                          ])
def test_get_package_name(version_package_expected):
    model_version, package, expected = version_package_expected
    result = get_package_name(package, model_version)
    assert result == expected
