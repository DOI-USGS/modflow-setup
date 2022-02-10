"""
Tests for different ways modflow-setup might be used
"""
import os
import shutil
from pathlib import Path

import pytest
from flopy import mf6

from mfsetup import MF6model, MFnwtModel
from mfsetup.utils import get_input_arguments


@pytest.mark.parametrize('cfg_file', ('shellmound.yml',))
def test_just_grid_setup_mf6(cfg_file, test_data_path, shellmound_cfg):

    wd = Path().cwd()
    # from cfg_file
    cfg_file = test_data_path / cfg_file
    m = MF6model(cfg=cfg_file)
    m.setup_grid()

    # reset working directory
    os.chdir(wd)
    # model workspace should be set correctly
    # expected absolute path to model_ws
    assert m.model_ws == Path('.')
    expected_model_ws = cfg_file.parent / m.cfg['simulation']['sim_ws']
    assert expected_model_ws.is_absolute()
    assert Path(m._abs_model_ws).samefile(expected_model_ws)

    # grid bbox file
    # (should be written if grid was made)
    assert (expected_model_ws / 'postproc/shps/shellmound_bbox.shp').exists()
    # reset the model_ws
    shutil.rmtree(expected_model_ws)

    # from cfg dict
    # check that model_ws still gets set correctly without a cfg file
    cfg = shellmound_cfg.copy()
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)

    # reset working directory
    os.chdir(wd)
    assert m.model_ws == Path('.')
    assert Path(m._abs_model_ws).samefile(expected_model_ws)
    # Note: the only reason this works is because
    # the correct absolute path to the cfg file and/or
    # the sim_ws is already set in the cfg dictionary


@pytest.mark.parametrize('cfg_file', ('pfl_nwt_test.yml',))
def test_just_grid_setup_mfnwt(cfg_file, test_data_path):
    cfg_file = test_data_path / cfg_file
    m = MFnwtModel(cfg=cfg_file)
    m.setup_grid()
    j=2
