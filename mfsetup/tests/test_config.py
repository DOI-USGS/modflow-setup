"""
Test functions in config.py
"""
from mfsetup.config import validate_configuration


def test_validate_length_units(pfl_nwt_cfg):

    cfg = pfl_nwt_cfg

    cfg['dis']['length_units'] = 'feet'
    cfg['dis']['lenuni'] = 2  # meters (default)

    validate_configuration(cfg)

    assert cfg['dis']['lenuni'] == 1  # lenuni should have been changed to match length_units
