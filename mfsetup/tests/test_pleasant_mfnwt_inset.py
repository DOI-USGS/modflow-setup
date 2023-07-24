"""
Tests for Pleasant Lake inset case
* MODFLOW-NWT
* SFR + Lake package
* Lake precip and evap specified with PRISM data; evap computed using evaporation.hamon_evaporation
* transient parent model with initial steady-state; copy unspecified data from parent
"""
import os

import flopy
import numpy as np
import pandas as pd

fm = flopy.modflow
import pytest

from mfsetup import MFnwtModel
from mfsetup.discretization import find_remove_isolated_cells
from mfsetup.fileio import load_array
from mfsetup.tests.test_lakes import get_prism_data


def test_perioddata(get_pleasant_nwt):
    m = get_pleasant_nwt
    m.perioddata
    assert m.perioddata['start_datetime'][0] == pd.Timestamp(m.cfg['dis']['start_date_time'])


def test_ibound(pleasant_nwt_with_dis):
    m = pleasant_nwt_with_dis
    # use pleasant lake extent as ibound
    is_pleasant_lake = m.lakarr[0]
    # clear out lake info, just for this test function
    m.cfg['model']['packages'].remove('lak')
    del m.cfg['lak']['source_data']
    # specify path relative to cfg file
    m.cfg['bas6']['source_data']['ibound'] = {'filename': '../../../examples/data/pleasant/source_data/shps/all_lakes.shp'}
    m._reset_bc_arrays()
    bas6 = m.setup_bas6()
    bas6.write_file()
    assert np.array_equal(m.ibound, m.bas6.ibound.array)
    # find_remove_isolated_cells is run on ibound array but not in Lake setup
    assert np.array_equal(m.ibound[0], find_remove_isolated_cells(is_pleasant_lake))
    ibound = load_array(m.cfg['bas6']['ibound'])
    assert np.array_equal(m.ibound, ibound)


@pytest.mark.skip('issue with flopy loading modflow-nwt lake package')
def test_setup_lak(pleasant_nwt_with_dis_bas6):
    m = pleasant_nwt_with_dis_bas6
    lak = m.setup_lak()
    lak.write_file()
    assert os.path.exists(lak.fn_path)
    lak = fm.ModflowLak.load(lak.fn_path, m)
    datafile = '../../../examples/data/pleasant/source_data/PRISM_ppt_tmean_stable_4km_189501_201901_43.9850_-89.5522.csv'
    prism = get_prism_data(datafile)
    precip = [lak.flux_data[per][0][0] for per in range(1, m.nper)]
    assert np.allclose(lak.flux_data[0][0][0], prism['ppt_md'].mean())
    assert np.allclose(precip, prism['ppt_md'])


def test_ghb_setup(get_pleasant_nwt_with_dis_bas6):
    m = get_pleasant_nwt_with_dis_bas6
    ghb = m.setup_ghb(**m.cfg['ghb'], **m.cfg['ghb']['mfsetup_options'])
    ghb.write_file()
    assert os.path.exists(ghb.fn_path)
    assert isinstance(ghb, fm.ModflowGhb)
    assert ghb.stress_period_data is not None

    # check for inactive cells
    spd0 = ghb.stress_period_data[0]
    k, i, j = spd0['k'], spd0['i'], spd0['j']
    inactive_cells = m.ibound[k, i, j] < 1
    assert not np.any(inactive_cells)

    # check that heads are above layer botms
    assert np.all(spd0['bhead'] > m.dis.botm.array[k, i, j])
    assert np.all(spd0['cond'] == m.cfg['ghb']['source_data']['cond'])


def test_wel_setup(get_pleasant_nwt_with_dis_bas6):

    m = get_pleasant_nwt_with_dis_bas6
    m.setup_upw()

    # test without tmr
    m.cfg['model']['perimeter_boundary_type'] = 'specified head'
    wel = m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    wel.write_file()
    assert os.path.exists(m.cfg['wel']['output_files']['lookup_file'])
    df = pd.read_csv(m.cfg['wel']['output_files']['lookup_file'])
    bfluxes0 = df.loc[(df.boundname == 'boundary_flux') & (df.per == 0)]
    assert len(bfluxes0) == 0
    # verify that water use fluxes are negative
    assert wel.stress_period_data[0]['flux'].max() <= 0.
    # verify that water use fluxes are in sp after 0
    # assuming that no wells shut off
    nwells0 = len(wel.stress_period_data[0][wel.stress_period_data[0]['flux'] != 0])
    n_added_wels = len(m.cfg['wel']['wells'])
    for k, spd in wel.stress_period_data.data.items():
        if k == 0:
            continue
        assert len(spd) >= nwells0 + n_added_wels


def test_oc_setup(get_pleasant_nwt_with_dis_bas6):
    m = get_pleasant_nwt_with_dis_bas6  # deepcopy(model)
    oc = m.setup_oc()
    # oc stress period data should be filled
    assert len(oc.stress_period_data) == m.nper

def test_model_setup(full_pleasant_nwt):
    m = full_pleasant_nwt
    assert isinstance(m, MFnwtModel)

    # test load_only
    package_list = [s.lower() for s in m.parent.get_package_list()]
    load_only = [s.lower() for s in m.cfg['parent']['load_only']]
    assert package_list == load_only

    # verify that gage package was written
    # verify that observation data were added and written
    obs = pd.read_csv(m.cfg['sfr']['source_data']['observations']['filename'])
    assert len(m.sfrdata.observations) == len(obs)
    expected = obs[m.cfg['sfr']['source_data']['observations']['obsname_column']].astype(str).tolist()
    assert m.sfrdata.observations['obsname'].tolist() == expected
    sfr_obs_filename = os.path.join(m.model_ws, m.sfrdata.observations_file)
    assert 'GAGE' in m.get_package_list()
    assert os.path.exists(sfr_obs_filename)
    with open(sfr_obs_filename) as src:
        gagedata = src.read()
    assert gagedata == '3 \n-1 -250 1 \n1 22 251 0 \n2 2 252 0 \n'


def test_model_setup_and_run(pleasant_nwt_model_run):
    m = pleasant_nwt_model_run
