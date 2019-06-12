import sys
sys.path.append('..')
import time
from copy import deepcopy
import shutil
import os
import pytest
import numpy as np
import pandas as pd
from shapely.geometry import box
import flopy
mf6 = flopy.mf6
from ..fileio import load
from ..mf6model import MF6model


@pytest.fixture(scope="module")
def simulation(cfg):
    sim = mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="module")
def model(cfg, simulation):
    cfg = cfg.copy()
    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = simulation
    m = MF6model(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="module")
def model_with_grid(model):
    #model = deepcopy(model)
    model.setup_grid()
    return model


@pytest.fixture(scope="module")
def model_setup(cfg_path):
    for folder in ['shellmound', 'tmp']:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    m = MF6model.setup_from_yaml(cfg_path)
    m.write()
    return m


def test_simulation(simulation):
    assert True


def test_model(model):
    assert True


def test_model_with_grid(model_with_grid):
    assert True

def test_external_file_path_setup(model):

    m = model #deepcopy(model)
    top_filename = m.cfg['dis']['top_filename']
    botm_file_fmt = m.cfg['dis']['botm_filename_fmt']
    m.setup_external_filepaths('dis', 'top',
                                   top_filename,
                                   nfiles=1)
    m.setup_external_filepaths('dis', 'botm',
                                   botm_file_fmt,
                                   nfiles=m.nlay)
    assert m.cfg['intermediate_data']['top'] == \
           os.path.join(m.tmpdir, os.path.split(top_filename)[-1])
    assert m.cfg['intermediate_data']['botm'] == \
           [os.path.join(m.tmpdir, botm_file_fmt).format(i)
                                  for i in range(m.nlay)]
    assert m.cfg['dis']['top'] == \
           os.path.join(m.model_ws,
                        m.external_path,
                        os.path.split(top_filename)[-1])
    assert m.cfg['dis']['botm'] == \
           [os.path.join(m.model_ws,
                         m.external_path,
                         botm_file_fmt.format(i)) for i in range(m.nlay)]

def test_perrioddata(model):
    m = model #deepcopy(model)
    pd0 = m.perioddata.copy()
    assert pd0 is not None

    m.cfg['sto']['steady'] = {0: True,
                              1: False}
    # Explicit stress period setup
    m.cfg['tdis']['options']['start_date_time'] = '2008-10-01'
    m.cfg['tdis']['perioddata']['perlen'] = [1] * 11
    m.cfg['tdis']['perioddata']['nstp'] = [5] * 11
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd1 = m.perioddata.copy()
    assert pd1['start_datetime'][0] == pd1['start_datetime'][1] == pd1['end_datetime'][0]
    assert pd1['end_datetime'][1] == pd.Timestamp(m.cfg['tdis']['options']['start_date_time']) + \
           pd.Timedelta(m.cfg['tdis']['perioddata']['perlen'][1], unit=m.time_units)
    assert pd1['nstp'][0] == 1
    assert pd1['tsmult'][0] == 1

    # Start date, freq and nper
    m.cfg['tdis']['options']['end_date_time'] = None
    m.cfg['tdis']['perioddata']['perlen'] = None
    m.cfg['tdis']['dimensions']['nper'] = 11
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m.cfg['tdis']['perioddata']['nstp'] = 5
    m.cfg['tdis']['perioddata']['tsmult'] = 1.5
    m._perioddata = None
    pd2 = m.perioddata.copy()
    assert pd2.equals(pd1)

    # Start date, end date, and nper
    m.cfg['tdis']['options']['end_date_time'] = '2008-10-11'
    m.cfg['tdis']['perioddata']['freq'] = None
    m._perioddata = None
    pd3 = m.perioddata.copy()
    assert pd3.equals(pd1)

    # Start date, end date, and freq
    m.cfg['tdis']['perioddata']['freq'] = 'D'
    m._perioddata = None
    pd4 = m.perioddata.copy()
    assert pd4.equals(pd1)

    # end date, freq and nper
    m.cfg['tdis']['options']['start_date_time'] = None
    m._perioddata = None
    pd5 = m.perioddata.copy()
    assert pd5.equals(pd1)

    # month end vs month start freq
    m.cfg['tdis']['perioddata']['freq'] = '6M'
    m.cfg['tdis']['options']['start_date_time'] = '2008-10-01'
    m.cfg['tdis']['options']['end_date_time'] = '2016-10-01'
    m.cfg['tdis']['perioddata']['nstp'] = 15
    m._perioddata = None
    pd6 = m.perioddata.copy()
    assert pd6.equals(pd0)

def test_dis_setup(model_with_grid):

    m = model_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_arrays'] = True
    m.cfg['dis']['regrid_top_from_dem'] = True
    dis = m.setup_dis()
    arrayfiles = [m.cfg['intermediate_data']['top']] + \
                 m.cfg['intermediate_data']['botm']
    for f in arrayfiles:
        assert os.path.exists(f)

    # test writing of MODFLOW arrays from
    # intermediate arrays
    m.cfg['dis']['remake_arrays'] = False
    m.cfg['dis']['regrid_top_from_dem'] = False
    dis = m.setup_dis()
    dis.write()
    arrayfiles = [m.cfg['dis']['top']] + \
                 m.cfg['dis']['botm']
    for f in arrayfiles:
        assert os.path.exists(f)
    assert os.path.exists(os.path.join(m.model_ws, dis.filename))

    # test shapefile export
    dis.export('{}/dis.shp'.format(m.cfg['gisdir']))
    # need to add assertion for shapefile bounds being in right place
    assert True


def test_tdis_setup(model):

    m = model #deepcopy(model)
    tdis = m.setup_tdis()
    assert isinstance(tdis, mf6.ModflowTdis)
    period_df = pd.DataFrame(tdis.perioddata.array)
    period_df['perlen'] = period_df['perlen'].astype(float)
    assert period_df.equals(m.perioddata[['perlen', 'nstp', 'tsmult']])


def test_sto_setup(model_with_grid):

    m = deepcopy(model_with_grid)
    sto = m.setup_sto()
    m.cfg['dis']['nper'] = 4
    m.cfg['dis']['perlen'] = [1, 1, 1, 1]
    m.cfg['dis']['nstp'] = [1, 1, 1, 1]
    m.cfg['dis']['tsmult'] = [1, 1, 1, 1]
    m.cfg['dis']['steady'] = [1, 0, 0, 1]
    # check settings
    #assert m.cfg['dis']['steady'] == [True, False, False, True]
    #assert dis.steady.array.tolist() == [True, False, False, True]


def test_yaml_setup(model_setup):
    m = deepcopy(model_setup)
    try:
        success, buff = m.run_model(silent=False)
    except:
        pass
    #assert success, 'model run did not terminate successfully'

