import sys
sys.path.append('..')
import time
import shutil
import os
import pytest
import numpy as np
import pandas as pd
from shapely.geometry import box
import flopy
from ..fileio import load
from ..mfmodel import MF6model


@pytest.fixture(scope="module")
def simulation(cfg):
    sim = flopy.mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="module")
def model(cfg, simulation):
    cfg['model']['simulation'] = simulation
    m = MF6model(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="module")
def model_with_grid(model):
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


def test_external_file_path_setup(model):

    top_filename = model.cfg['dis']['top_filename']
    botm_file_fmt = model.cfg['dis']['botm_filename_fmt']
    model.setup_external_filepaths('dis', 'top',
                                   top_filename,
                                   nfiles=1)
    model.setup_external_filepaths('dis', 'botm',
                                   botm_file_fmt,
                                   nfiles=model.nlay)
    assert model.cfg['intermediate_data']['top'] == \
           os.path.join(model.tmpdir, os.path.split(top_filename)[-1])
    assert model.cfg['intermediate_data']['botm'] == \
           [os.path.join(model.tmpdir, botm_file_fmt).format(i)
                                  for i in range(model.nlay)]
    assert model.cfg['dis']['top'] == \
           os.path.join(model.model_ws,
                        model.external_path,
                        os.path.split(top_filename)[-1])
    assert model.cfg['dis']['botm'] == \
           [os.path.join(model.model_ws,
                         model.external_path,
                         botm_file_fmt.format(i)) for i in range(model.nlay)]


def test_dis_setup(model_with_grid):

    m = model_with_grid
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


def test_sto_setup(model_with_grid):

    m = model_with_grid
    m.cfg['dis']['nper'] = 4
    m.cfg['dis']['perlen'] = [1, 1, 1, 1]
    m.cfg['dis']['nstp'] = [1, 1, 1, 1]
    m.cfg['dis']['tsmult'] = [1, 1, 1, 1]
    m.cfg['dis']['steady'] = [1, 0, 0, 1]
    # check settings
    #assert m.cfg['dis']['steady'] == [True, False, False, True]
    #assert dis.steady.array.tolist() == [True, False, False, True]

def test_yaml_setup(model_setup):

    m = model_setup
    try:
        success, buff = m.run_model(silent=False)
    except:
        pass
    #assert success, 'model run did not terminate successfully'

