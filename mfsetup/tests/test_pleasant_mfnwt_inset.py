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
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from mfsetup import MFnwtModel
from .test_lakes import get_prism_data


def test_perioddata(pleasant_nwt):
    m = pleasant_nwt
    m._set_perioddata()
    assert m.perioddata['start_datetime'][0] == pd.Timestamp(m.cfg['dis']['start_date_time'])


def test_setup_lak(pleasant_nwt_with_dis_bas6):
    m = pleasant_nwt_with_dis_bas6
    lak = m.setup_lak()
    lak.write_file()
    lak = fm.ModflowLak.load(lak.fn_path, m)
    datafile = '../../data/pleasant/source_data/PRISM_ppt_tmean_stable_4km_189501_201901_43.9850_-89.5522.csv'
    prism = get_prism_data(datafile)
    precip = [lak.flux_data[per][0][0] for per in range(1, m.nper)]
    assert np.allclose(lak.flux_data[0][0][0], prism['ppt_md'].mean())
    assert np.allclose(precip, prism['ppt_md'])


def test_model_setup(full_pleasant_nwt):
    m = full_pleasant_nwt
    assert isinstance(m, MFnwtModel)


def test_model_setup_and_run(full_pleasant_nwt_with_model_run):
    m = full_pleasant_nwt_with_model_run
