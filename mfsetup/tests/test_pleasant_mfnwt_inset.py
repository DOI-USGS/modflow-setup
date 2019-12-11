"""
Tests for Pleasant Lake inset case
* MODFLOW-NWT
* SFR + Lake package
* Lake precip and evap specified with PRISM data; evap computed using evaporation.hamon_evaporation
* transient parent model with initial steady-state
"""
from copy import copy, deepcopy
import os
import pytest
import pandas as pd
import flopy
fm = flopy.modflow
from mfsetup import MFnwtModel
from ..fileio import exe_exists, load_cfg
from ..utils import get_input_arguments


def test_perioddata(pleasant_nwt):
    m = pleasant_nwt
    m._set_perioddata()
    assert m.perioddata['start_datetime'][0] == pd.Timestamp(m.cfg['dis']['start_date_time'])


def test_setup_lak(pleasant_nwt_with_dis_bas6):
    m = pleasant_nwt_with_dis_bas6
    m.setup_lak()
    j=2