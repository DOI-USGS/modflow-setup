"""
Tests for different ways modflow-setup might be used
"""
import pytest

from mfsetup import MF6model, MFnwtModel


@pytest.mark.parametrize('cfg_file', ('shellmound.yml',))
def test_just_grid_setup_mf6(cfg_file, test_data_path):
    cfg_file = test_data_path / cfg_file
    m = MF6model(cfg=cfg_file)
    m.setup_grid()
    j=2


@pytest.mark.parametrize('cfg_file', ('pfl_nwt_test.yml',))
def test_just_grid_setup_mfnwt(cfg_file, test_data_path):
    cfg_file = test_data_path / cfg_file
    m = MFnwtModel(cfg=cfg_file)
    m.setup_grid()
    j=2
