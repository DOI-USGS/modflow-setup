"""
Tests for the tmr.py module

Notes
-----
Some relevant tests are also in the following modules
test_mf6_tmr_shellmound.py
"""
from pathlib import Path
from subprocess import PIPE, Popen

import flopy
import numpy as np
import pandas as pd
import pytest
from flopy.utils import Mf6ListBudget, MfListBudget
from flopy.utils import binaryfile as bf

from mfsetup.discretization import get_layer
from mfsetup.fileio import exe_exists
from mfsetup.grid import MFsetupGrid, get_ij
from mfsetup.tmr import Tmr, get_qx_qy_qz
from mfsetup.zbud import write_zonebudget6_input

from .test_mf6_tmr_shellmound import (
    shellmound_tmr_cfg,
    shellmound_tmr_cfg_path,
    shellmound_tmr_model,
    shellmound_tmr_model_with_dis,
    shellmound_tmr_model_with_grid,
    shellmound_tmr_model_with_refined_dis,
    shellmound_tmr_simulation,
    shellmound_tmr_small_rectangular_inset,
)
from .test_pleasant_mf6_inset import get_pleasant_mf6_with_dis


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['get_pleasant_mf6_with_dis',
                        'pleasant_nwt_with_dis_bas6'])
def pleasant_model(request,
                   get_pleasant_mf6_with_dis,
                   pleasant_nwt_with_dis_bas6):
    return {'get_pleasant_mf6_with_dis': get_pleasant_mf6_with_dis,
            'pleasant_nwt_with_dis_bas6': pleasant_nwt_with_dis_bas6}[request.param]


@pytest.mark.parametrize('specific_discharge',(False, True))
def test_get_qx_qy_qz(tmpdir, parent_model_mf6, parent_model_nwt, specific_discharge):
    """Compare get_qx_qy_qz results between mf6 and nwt
    """
    mf6_ws = Path(tmpdir) / 'perimeter_bc_demo/parent'

    # get results for MF6
    m6 = parent_model_mf6
    qx6, qy6, qz6 = get_qx_qy_qz(mf6_ws / 'tmr_parent.cbc', binary_grid_file=mf6_ws / 'tmr_parent.dis.grb',
                                version='mf6',
                                specific_discharge=specific_discharge,
                                modelgrid=m6.modelgrid,
                                headfile=mf6_ws / 'tmr_parent.hds')


    # get results for MFNWT
    mfnwt_ws = Path(tmpdir) / 'perimeter_bc_demo/parent_nwt'

    mnwt = parent_model_nwt
    qxnwt, qynwt, qznwt = get_qx_qy_qz(mfnwt_ws / 'tmr_parent_nwt.cbc',
                                       version='mfnwt',
                                       specific_discharge=specific_discharge,
                                       modelgrid=mnwt.modelgrid,
                                       headfile=mfnwt_ws / 'tmr_parent_nwt.hds')
    #import matplotlib.pyplot as plt
    #fig, axes = plt.subplots(1, 2)
    #axes = axes.flat
    #pmv = flopy.plot.PlotMapView(model=parent_model_mf6, ax=axes[0], layer=0)
    #pmv.plot_bc('CHD_0')
    #pmv.plot_grid()
    #pmv2 = flopy.plot.PlotMapView(model=parent_model_nwt, ax=axes[1], layer=0)
    #pmv2.plot_bc('CHD')
    #pmv2.plot_grid()

    assert np.allclose(qx6,qxnwt,atol=1e-2)
    assert np.allclose(qy6,qynwt,atol=1e-2)
    assert np.allclose(qz6,qznwt,atol=1e-2)


def test_tmr_new(pleasant_model):
    m = pleasant_model
    parent_headfile = Path(m.cfg['chd']['perimeter_boundary']['parent_head_file'])
    parent_cellbudgetfile = parent_headfile.with_suffix('.cbc')

    tmr = Tmr(m.parent, m,
                 parent_head_file=parent_headfile)

    results = tmr.get_inset_boundary_values(for_external_files=False)
    assert np.all(results.columns ==
                  ['k', 'i', 'j', 'per', 'head'])
    # indices should be zero-based
    assert results['k'].min() == 0
    # non NaN heads
    assert not results['head'].isna().any()
    # no heads below cell bottoms
    cell_botms = m.dis.botm.array[results['k'], results['i'], results['j']]
    assert not np.any(results['head'] < cell_botms)
    # no duplicate heads
    results['cellid'] = list(zip(results.per, results.k, results.i, results.j))
    assert not results.cellid.duplicated().any()

    # test external files case
    # and with connections defined by layer
    tmr.define_connections_by = 'by_layer'
    tmr._inset_boundary_cells = None   # reset properties
    tmr._interp_weights_heads = None
    results = tmr.get_inset_boundary_values(for_external_files=True)
    # '#k' required for header row
    assert np.all(results.columns ==
                  ['#k', 'i', 'j', 'per', 'head'])
    # indices should be one-based (written directly to external files)
    assert results['#k'].min() == 1


def test_get_boundary_cells_shapefile(shellmound_tmr_model_with_dis, test_data_path, tmpdir):
    m = shellmound_tmr_model_with_dis

    from mfexport import export
    export(m, m.modelgrid, 'dis', 'idomain', pdfs=False, output_path=tmpdir)
    boundary_shapefile = test_data_path / 'shellmound/tmr_parent/gis/irregular_boundary.shp'
    tmr = Tmr(m.parent, m,
                 inset_parent_period_mapping=m.parent_stress_periods,
                 boundary_type='head')
    df = tmr.get_inset_boundary_cells(shapefile=boundary_shapefile)
    assert np.all(df.columns == ['k', 'i', 'j', 'cellface', 'top', 'botm', 'idomain',
                                 'cellid', 'geometry'])
    # option to write a shapefile of boundary cells for visual inspection
    write_shapefile = False
    if write_shapefile:
        out_shp = Path(tmpdir, 'shps/bcells.shp')
        df.drop('cellid', axis=1).to_file(out_shp)


@pytest.fixture
def test_model_properties():
    properties = {
        'h': 3000,
        'w': 3000,
        'inset_xoff': 1000,
        'inset_yoff': 1000,
        'ncells_side': 60,
        'nlay': 1,
        'top': 20, #30,
        'botm': 0,
        'west_bhead_value': 29,
        'lake_level': 28,
        'lake_width': 200
    }
    return properties


@pytest.fixture
def parent_model_mf6(tmpdir, mf6_exe, test_model_properties):
    """Make a simpmle parent model for TMR perimeter boundary tests,
    with inflow from west that curves to outflow to the north.
    """
    # set up simulation
    name = 'tmr_parent'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/parent'
    model_ws.mkdir(exist_ok=True, parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(model_ws))
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=1,
                                perioddata=[(1.0, 1, 1.0)])
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="MODERATE",
                    outer_dvclose=1e-4,
                    inner_dvclose=1e-4)
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name, newtonoptions='newton')

    h = test_model_properties['h']
    w = test_model_properties['w']
    ncells_side = test_model_properties['ncells_side']
    dx = w/ncells_side
    dis = flopy.mf6.ModflowGwfdis(model, nlay=test_model_properties['nlay'],
                                  nrow=ncells_side, ncol=ncells_side,
                                delr=dx, delc=dx,
                                top=test_model_properties['top'], botm=test_model_properties['botm']
                                )
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1.0, k33=1.0, save_flows=True)
    # set up CHD boundaries
    # for eastward flow through the west boundary
    # curving to northward flow through the north boundary
    chd_start_pos = int(ncells_side / 2)
    nchd_side = ncells_side - chd_start_pos
    w_heads = list(np.ones((nchd_side)) * test_model_properties['west_bhead_value'])
    w_heads_i = list(range(chd_start_pos, ncells_side))
    w_heads_j = [0] * len(w_heads)
    n_heads = list(np.array(w_heads) - 2.)
    n_heads_i = [0] * len(n_heads)
    n_heads_j = w_heads_i
    # make a CHD "lake" in the middle (to anchor heads within the inset model domain)
    # (the parent also needs to have the lake if we want to compare solutions)
    lake_width = test_model_properties['lake_width']
    lake_lower_left = ncells_side / 2 - (lake_width / 2 / dx)
    lake_upper_right = ncells_side / 2 + (lake_width / 2 / dx)
    lake_heads_i, lake_heads_j = np.meshgrid(np.arange(lake_lower_left,lake_upper_right),
                                             np.arange(lake_lower_left,lake_upper_right))
    lake_heads_i = list(lake_heads_i.ravel().astype(int))
    lake_heads_j = list(lake_heads_j.ravel().astype(int))
    # make the stress period data
    spd = pd.DataFrame({'k': 0,
                        'i': w_heads_i + n_heads_i + lake_heads_i,
                        'j': w_heads_j + n_heads_j + lake_heads_j,
                        # set the lake level at 28 (between level of west and north boundaries)
                        'head': w_heads + n_heads + ([test_model_properties['lake_level']] * len(lake_heads_j))
                        })
    spd['cellid'] = list(zip(spd['k'], spd['i'], spd['j']))
    spd_rec = spd[['cellid', 'head']].to_records(index=False)

    start = test_model_properties['top'] * np.ones_like(dis.botm.array)
    start[0, w_heads_i, w_heads_j] = w_heads
    start[0, n_heads_i, n_heads_j] = n_heads
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    oc = flopy.mf6.ModflowGwfoc(model,
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                head_filerecord=f"{name}.hds",
                                budget_filerecord=f"{name}.cbc",
                                )
    chd = flopy.mf6.ModflowGwfchd(model, maxbound=len(spd_rec),
                                stress_period_data=spd_rec,
                                save_flows=True)
    # rcha = flopy.mf6.ModflowGwfrcha(model, recharge=5.e-06)

    sim.write_simulation()
    # run the model
    sim.exe_name = mf6_exe
    success = False
    if exe_exists(mf6_exe):
        success, buff = sim.run_simulation()
        if not success:
            list_file = model.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return model


@pytest.fixture
def inset_model_mf6(tmpdir, mf6_exe, test_model_properties):
    """Make a simple inset model to go in parent model
    """
    name = 'tmr_inset'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/inset'
    model_ws.mkdir(exist_ok=True, parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name='mf6',
                    sim_ws=str(model_ws))
    setattr(sim.simulation_data, 'use_pandas', False)
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=1,
                                perioddata=[(1.0, 1, 1.0)])
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="MODERATE",
                    outer_dvclose=1e-4,
                    outer_maximum=150,
                    inner_dvclose=1e-6,
                    inner_maximum=100)
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name, newtonoptions='newton')
    h = int(test_model_properties['h']/3)
    w = int(test_model_properties['w']/3)
    ncells_side = int(test_model_properties['ncells_side']/3)
    dx = w/ncells_side
    dis = flopy.mf6.ModflowGwfdis(model, nlay=test_model_properties['nlay'],
                                  nrow=ncells_side, ncol=ncells_side,
                                delr=dx, delc=dx,
                                top=test_model_properties['top'],
                                botm=test_model_properties['botm']
                                )
    start = test_model_properties['top'] * np.ones_like(dis.botm.array)
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1.0, k33=1.0, save_flows=True)
    oc = flopy.mf6.ModflowGwfoc(model,
                            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                            head_filerecord=f"{name}.hds",
                            budget_filerecord=f"{name}.cbc",
                            )
    # rcha = flopy.mf6.ModflowGwfrcha(model, recharge=5.e-06)

    # inset model needs a lake at the exact same position
    lake_width = test_model_properties['lake_width']
    lake_lower_left = ncells_side / 2 - (lake_width / 2 / dx)
    lake_upper_right = ncells_side / 2 + (lake_width / 2 / dx)
    i, j = np.meshgrid(np.arange(lake_lower_left,lake_upper_right),
                       np.arange(lake_lower_left,lake_upper_right))
    lake_chd = pd.DataFrame({'k': 0,
                            'i': i.ravel().astype(int),
                            'j': j.ravel().astype(int),
                            'head': test_model_properties['lake_level']
                            })
    lake_chd['cellid'] = list(zip(lake_chd['k'], lake_chd['i'], lake_chd['j']))
    one_cell_chd_rec = lake_chd[['cellid', 'head']].to_records(index=False)
    chd = flopy.mf6.ModflowGwfchd(model, maxbound=len(one_cell_chd_rec),
                                stress_period_data=one_cell_chd_rec,
                                save_flows=True)
    model._modelgrid = MFsetupGrid(delc=model.dis.delc.array, delr=model.dis.delr.array,
                                  top=model.dis.top.array, botm=model.dis.botm.array,
                                  xoff=test_model_properties['inset_xoff'],
                                  yoff=test_model_properties['inset_yoff'])
    model._mg_resync = False
    assert hasattr(model, 'modelgrid'), "something went wrong setting the modelgrid attribute"
    return model


@pytest.fixture
def parent_model_nwt(tmpdir, mfnwt_exe, test_model_properties):
    """Make a simpmle MFNWT parent model for TMR perimeter boundary tests,
    with inflow from west that curves to outflow to the north.

    """
    # set up simulation
    name = 'tmr_parent_nwt'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/parent_nwt'
    model_ws.mkdir(exist_ok=True, parents=True)

    mf = flopy.modflow.Modflow(name, model_ws=str(model_ws), version='mfnwt')
    nwt = flopy.modflow.ModflowNwt(mf, headtol=1e-4)

    h = test_model_properties['h']
    w = test_model_properties['w']
    ncells_side = test_model_properties['ncells_side']
    dx = w/ncells_side
    dis = flopy.modflow.ModflowDis(mf, nlay=test_model_properties['nlay'],
                                   nrow=ncells_side, ncol=ncells_side,
                                delr=dx, delc=dx,
                                top=test_model_properties['top'],
                                botm=test_model_properties['botm']
                                )
    upw = flopy.modflow.ModflowUpw(mf,laytyp=1, hk=1., vka=1.0, ipakcb=53)
    # set up CHD boundaries
    # for eastward flow through the west boundary
    # curving to northward flow through the north boundary
    chd_start_pos = int(ncells_side / 2)
    nchd_side = ncells_side - chd_start_pos
    w_heads = list(np.ones((nchd_side)) * test_model_properties['west_bhead_value'])
    w_heads_i = list(range(chd_start_pos, ncells_side))
    w_heads_j = [0] * len(w_heads)
    n_heads = list(np.array(w_heads) - 2.)
    n_heads_i = [0] * len(n_heads)
    n_heads_j = w_heads_i
    # make a CHD "lake" in the middle (to anchor heads within the inset model domain)
    # (the parent also needs to have the lake if we want to compare solutions)
    lake_width = test_model_properties['lake_width']
    lake_lower_left = ncells_side / 2 - (lake_width / 2 / dx)
    lake_upper_right = ncells_side / 2 + (lake_width / 2 / dx)
    lake_heads_i, lake_heads_j = np.meshgrid(np.arange(lake_lower_left,lake_upper_right),
                                             np.arange(lake_lower_left,lake_upper_right))
    lake_heads_i = list(lake_heads_i.ravel().astype(int))
    lake_heads_j = list(lake_heads_j.ravel().astype(int))
    # make the stress period data
    spd = pd.DataFrame({'k': 0,
                        'i': w_heads_i + n_heads_i + lake_heads_i,
                        'j': w_heads_j + n_heads_j + lake_heads_j,
                        # set the lake level at 28 (between level of west and north boundaries)
                        'shead': w_heads + n_heads +\
                            ([test_model_properties['lake_level']] * len(lake_heads_j)),
                        'ehead': w_heads + n_heads +\
                            ([test_model_properties['lake_level']] * len(lake_heads_j))
                        })
    spd_rec = flopy.modflow.ModflowChd.get_empty(len(spd))
    for col in ['k', 'i', 'j', 'shead', 'ehead']:
        spd_rec[col] = spd[col].values

    ibnd = np.ones([test_model_properties['nlay'], ncells_side, ncells_side])
    ibnd[0, w_heads_i, w_heads_j] = -1
    ibnd[0, n_heads_i, n_heads_j] = -1
    start = test_model_properties['top'] * np.ones_like(dis.botm.array)
    start[0, w_heads_i, w_heads_j] = w_heads
    start[0, n_heads_i, n_heads_j] = n_heads
    bas = flopy.modflow.ModflowBas(mf, ibound=ibnd, strt=start)
    oc = flopy.modflow.ModflowOc(mf,
                                stress_period_data={(0, 0): ['save head','save budget']})
    chd = flopy.modflow.ModflowChd(mf,stress_period_data=spd_rec,
                                   #save_flows=True
                                   )
    #rch = flopy.modflow.ModflowRch(mf, rech=5.e-06)

    mf.write_input()

    mf.exe_name = mfnwt_exe
    success = False
    if exe_exists(mfnwt_exe):
        success, buff = mf.run_model()
        if not success:
            list_file = mf.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return mf


@pytest.fixture
def inset_model_nwt(tmpdir, test_model_properties):
    """Make a simple inset model to go in MFNWT parent model

    TODO: make this match the MF6 inset model
    """
    name = 'tmr_inset_nwt'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/inset_nwt'
    model_ws.mkdir(exist_ok=True, parents=True)

    mf = flopy.modflow.Modflow(name, model_ws=str(model_ws), version='mfnwt')
    nwt = flopy.modflow.ModflowNwt(mf)
    h = int(test_model_properties['h']/3)
    w = int(test_model_properties['w']/3)
    ncells_side = int(test_model_properties['ncells_side']/3)
    dx = w/ncells_side
    dis = flopy.modflow.ModflowDis(mf, nlay=test_model_properties['nlay'],
                                   nrow=ncells_side, ncol=ncells_side,
                                delr=dx, delc=dx,
                                top=test_model_properties['top'],
                                botm=test_model_properties['botm']
                                )
    start = test_model_properties['top'] * np.ones_like(dis.botm.array)
    bas = flopy.modflow.ModflowBas(mf, strt=start)
    upw = flopy.modflow.ModflowUpw(mf, laytyp=1, hk=1., vka=1.0)
    oc = flopy.modflow.ModflowOc(mf,
                                stress_period_data={(0, 0): ['save head','save budget']})
    #rch = flopy.modflow.ModflowRch(mf, recharge=5.e-06)

    # inset model needs a lake at the exact same position
    lake_width = test_model_properties['lake_width']
    lake_lower_left = ncells_side / 2 - (lake_width / 2 / dx)
    lake_upper_right = ncells_side / 2 + (lake_width / 2 / dx)
    i, j = np.meshgrid(np.arange(lake_lower_left,lake_upper_right),
                       np.arange(lake_lower_left,lake_upper_right))
    lake_chd = pd.DataFrame({'k': 0,
                            'i': i.ravel().astype(int),
                            'j': j.ravel().astype(int),
                            'shead': test_model_properties['lake_level'],
                            'ehead': test_model_properties['lake_level']
                            })
    spd_rec = flopy.modflow.ModflowChd.get_empty(len(lake_chd))
    for col in ['k', 'i', 'j', 'shead', 'ehead']:
        spd_rec[col] = lake_chd[col].values
    chd = flopy.modflow.ModflowChd(mf, stress_period_data=spd_rec)
    mf._modelgrid = MFsetupGrid(delc=mf.dis.delc.array, delr=mf.dis.delr.array,
                                top=mf.dis.top.array, botm=mf.dis.botm.array,
                                xoff=test_model_properties['inset_xoff'],
                                yoff=test_model_properties['inset_yoff'])
    mf._mg_resync = False
    assert hasattr(mf, 'modelgrid'), "something went wrong setting the modelgrid attribute"
    return mf


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['parent_model_mf6',
                        'parent_model_nwt'])
def parent_model(request,
                   parent_model_mf6,
                   parent_model_nwt):
    """MODFLOW-NWT and MODFLOW-6 versions of the test case parent model."""
    return {'parent_model_mf6': parent_model_mf6,
            'parent_model_nwt': parent_model_nwt}[request.param]


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['inset_model_mf6',
                        'inset_model_nwt'])
def inset_model(request,
                 inset_model_mf6,
                 inset_model_nwt):
    """MODFLOW-NWT and MODFLOW-6 versions of the test case inset model."""
    return {'inset_model_mf6': inset_model_mf6,
            'inset_model_nwt': inset_model_nwt}[request.param]


def test_get_boundary_heads(parent_model, inset_model,
                            project_root_path,
                            mf6_exe, mfnwt_exe, zbud6_exe):
    """Test getting perimeter boundary head values from a parent model,
    for a TMR inset model that is a regular Flopy model with a Modflow-setup
    grid (MFsetupGrid).

    TODO: refactor test_get_boundary_heads to test different
    parent/inset versions like test_get_boundary_fluxes

    Parameters
    ----------
    parent_model : flopy model instance from pytest fixture
    inset_model : flopy model instance from pytest fixture
    project_root_path : absolute path to modflow setup root folder
    mf6_exe : Modflow 6 executable from pytest fixture
    zbud6_exe : Zonebudget 6 executable from pytest fixture
    """
    project_root_path = Path(project_root_path)

    #m = get_pleasant_mf6_with_dis
    #parent_ws = project_root_path / 'examples/data/pleasant/'
    #parent_model = parent_model_mf6
    if inset_model.version != parent_model.version:
        return
    m = inset_model #_mf6
    parent_ws = Path(parent_model.model_ws)
    #boundary_shapefile = parent_ws / 'gis/irregular_boundary.shp'
    parent_budget_file = parent_ws / f'{parent_model.name}.cbc'
    parent_head_file = parent_ws / f'{parent_model.name}.hds'
    parent_binary_grid_file = parent_ws / f'{parent_model.name}.dis.grb'
    tmr = Tmr(parent_model, m, parent_head_file=parent_head_file,
                 boundary_type='head',
                 )
    perimeter_df = tmr.get_inset_boundary_values()

    # set up the CHD package
    perimeter_df['cellid'] = list(perimeter_df[['k', 'i', 'j']].to_records(index=False))
    period_groups = perimeter_df.groupby('per')
    if m.version == 'mf6':
        spd = {}
        maxbound = 0
        for per, data in period_groups:
            spd[per] = data[['cellid', 'head']].to_records(index=False)
            if len(data) > maxbound:
                maxbound = len(data)
        chd = flopy.mf6.ModflowGwfchd(m, maxbound=maxbound,
                                    stress_period_data=spd,
                                    save_flows=True, filename=f'{m.name}-perimeter.chd')
        # not sure why this needs to be done again to retain modelgrid attribute
        m._mg_resync = False

        # write the inset model input files
        m.simulation.write_simulation()

        # run the inset model
        m.simulation.exe_name = mf6_exe
        success = False
        if exe_exists(mf6_exe):
            success, buff = m.simulation.run_simulation()
            if not success:
                list_file = m.name_file.list.array
                with open(list_file) as src:
                    list_output = src.read()
    else:
        spd = {}
        for per, data in period_groups:
            data['shead'] = data['head']
            data['ehead'] = data['head']
            # since we can only have 1 CHD package in MODFLOW-NWT
            # add existing lake CHD input to the perimeter CHD recarray
            lake_data = pd.DataFrame(m.chd.stress_period_data.data[0])
            data = pd.concat([data, lake_data])
            spd_rec = flopy.modflow.ModflowChd.get_empty(len(data))
            for col in ['k', 'i', 'j', 'shead', 'ehead']:
                spd_rec[col] = data[col].values
            spd[per] = spd_rec
        m.remove_package('CHD')
        chd = flopy.modflow.ModflowChd(m, stress_period_data=spd)

        # not sure why this needs to be done again to retain modelgrid attribute
        m._mg_resync = False

        m.write_input()
        m.exe_name = mfnwt_exe
        success = False
        if exe_exists(mfnwt_exe):
            success, buff = m.run_model()
            if not success:
                list_file = m.name_file.list.array
                with open(list_file) as src:
                    list_output = src.read()

    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

    # Zone Budget comparison only implemented for MF6
    if m.version == 'mf6':
        # Set up zone budget on the parent (for the inset model footprint)
        inset_footprint_within_parent = tmr.inset_zone_within_parent
        inset_lower_left = np.argmax(np.argmax(tmr.inset_zone_within_parent, axis=0))
        # shrink the inset footprint by 1
        # so that zone budget evaluates the interior faces
        # of the boundary cells
        inset_footprint_within_parent[:, inset_lower_left] = 0
        inset_footprint_within_parent[inset_lower_left, :] = 0
        inset_footprint_within_parent[:, -(inset_lower_left+1)] = 0
        inset_footprint_within_parent[-(inset_lower_left+1), :] = 0
        output_budget_name = Path(m.model_ws).absolute() / f"{m.name}-parent"
        write_zonebudget6_input(inset_footprint_within_parent, budgetfile=parent_budget_file,
                                binary_grid_file=parent_binary_grid_file,
                                outname=output_budget_name)
        # run zonebudget
        process = Popen([str(zbud6_exe), f'{m.name}-parent.zbud.nam'], cwd=Path(m.model_ws).absolute(),
                    stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        assert process.returncode == 0
        # read the zone budget output
        zb_results = pd.read_csv(output_budget_name.with_suffix('.zbud.csv'))
        zb_results['net_flux'] = zb_results['FROM ZONE 0'] - zb_results['TO ZONE 0']

        # run zonebudget on the inset model
        inset_nrow, inset_ncol = m.dis.top.array.shape
        inset_zone_budget_array = np.ones((inset_nrow, inset_ncol))
        inset_budget_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.cbc')
        inset_binary_grid_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.dis.grb')
        output_budget_name = Path(m.model_ws).absolute() / f"{m.name}-inset"
        write_zonebudget6_input(inset_zone_budget_array, budgetfile=inset_budget_file,
                                binary_grid_file=inset_binary_grid_file,
                                outname=output_budget_name)
        process = Popen([str(zbud6_exe), f'{m.name}-inset.zbud.nam'], cwd=Path(m.model_ws).absolute(),
                    stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        assert process.returncode == 0
        # read the zone budget output
        zb_results_inset = pd.read_csv(output_budget_name.with_suffix('.zbud.csv'))
        zb_results_inset['net_flux'] = zb_results_inset['FROM ZONE 0'] - zb_results_inset['TO ZONE 0']
        perimeter_chd_package = 'CHD_1'
        zb_results_inset['chd_net'] = zb_results_inset[f'{perimeter_chd_package}-CHD-IN'] -\
            zb_results_inset[f'{perimeter_chd_package}-CHD-OUT']

        # get the inset model boundary fluxes
        inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.lst')
        mfl = Mf6ListBudget(inset_list_file)
        df_flux, df_vol = mfl.get_dataframes()
        perimeter_chd_package_list = 'CHD2'
        df_flux['CHD_net'] = df_flux[f'{perimeter_chd_package_list}_IN'] -\
            df_flux[f'{perimeter_chd_package_list}_OUT']
        df_flux.reset_index(inplace=True)

        # compare the fluxes
        # check that total in/out fluxes match
        # Note: the listing file evaluates flux across the interior face of the boundary cells
        # so the zone budget results need to be for an inset footprint that is one cell smaller on each side
        assert np.allclose(df_flux[f'{perimeter_chd_package_list}_IN'], zb_results['FROM ZONE 0'], rtol=0.001)
        assert np.allclose(df_flux[f'{perimeter_chd_package_list}_OUT'], zb_results['TO ZONE 0'], rtol=0.001)

    else:
        # get the inset model boundary fluxes
        inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.list')
        mfl = MfListBudget(inset_list_file)
        df_flux, df_vol = mfl.get_dataframes()
        #perimeter_chd_package_list = 'CHD2'
        df_flux['CHD_net'] = df_flux[f'CONSTANT_HEAD_IN'] - df_flux[f'CONSTANT_HEAD_OUT']
        df_flux.reset_index(inplace=True)

        # for MODFLOW-NWT, do a simpler absolute comparison
        # (based on values from MF6 model)
        # compare to total CHD flux (perimeter + lake)
        assert np.allclose(df_flux['CONSTANT_HEAD_IN'], 16.88, rtol=0.01)
        assert np.allclose(df_flux['CONSTANT_HEAD_OUT'], 16.88, rtol=0.01)

    # check that the heads were applied correctly
    inset_heads_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.hds')
    hdsobj = bf.HeadFile(inset_heads_file)
    allhds = hdsobj.get_alldata()
    k, i, j, per = perimeter_df[['k', 'i', 'j', 'per']].T.values
    perimeter_df['inset_head'] = allhds[per, k, i, j]
    assert np.allclose(perimeter_df['inset_head'].values,
                       perimeter_df['head'].values)


def test_get_boundary_fluxes(parent_model, inset_model,
                            project_root_path,
                            mf6_exe, mfnwt_exe, zbud6_exe):
    """Test getting perimeter boundary head values from a parent model,
    for a TMR inset model that is a regular Flopy model with a Modflow-setup
    grid (MFsetupGrid).

    Parameters
    ----------
    parent_model : flopy model instance from pytest fixture
    inset_model : flopy model instance from pytest fixture
    project_root_path : absolute path to modflow setup root folder
    mf6_exe : Modflow 6 executable from pytest fixture
    zbud6_exe : Zonebudget 6 executable from pytest fixture
    """
    project_root_path = Path(project_root_path)

    #m = get_pleasant_mf6_with_dis
    #parent_ws = project_root_path / 'examples/data/pleasant/'
    #parent_model = parent_model_mf6
    #if inset_model.version != parent_model.version:
    #    return
    m = inset_model #_mf6
    parent_ws = Path(parent_model.model_ws)
    #boundary_shapefile = parent_ws / 'gis/irregular_boundary.shp'
    parent_budget_file = parent_ws / f'{parent_model.name}.cbc'
    parent_head_file = parent_ws / f'{parent_model.name}.hds'
    if parent_model.version == 'mf6':
        parent_binary_grid_file = parent_ws / f'{parent_model.name}.dis.grb'
    else:
        parent_binary_grid_file = None
    tmr = Tmr(parent_model, m, parent_cell_budget_file=parent_budget_file,
                 parent_binary_grid_file=parent_binary_grid_file,
                 parent_head_file=parent_head_file,
                 boundary_type='flux',
                 )
    perimeter_df = tmr.get_inset_boundary_values()

    # set up the WEL package
    perimeter_df['cellid'] = list(perimeter_df[['k', 'i', 'j']].to_records(index=False))
    period_groups = perimeter_df.groupby('per')
    if m.version == 'mf6':
        spd = {}
        maxbound = 0
        for per, data in period_groups:
            spd[per] = data[['cellid', 'q']].to_records(index=False)
            if len(data) > maxbound:
                maxbound = len(data)
        wel = flopy.mf6.ModflowGwfwel(m, maxbound=maxbound,
                                    stress_period_data=spd,
                                    save_flows=True, filename=f'{m.name}-perimeter.wel')
        # not sure why this needs to be done again to retain modelgrid attribute
        m._mg_resync = False

        # write the inset model input files
        m.simulation.write_simulation()

        # run the inset model
        m.simulation.exe_name = mf6_exe
        success = False
        if exe_exists(mf6_exe):
            success, buff = m.simulation.run_simulation()
            if not success:
                list_file = m.name_file.list.array
                with open(list_file) as src:
                    list_output = src.read()
    else:
        spd = {}
        for per, data in period_groups:
            spd_rec = flopy.modflow.ModflowWel.get_empty(len(data))
            for col in ['k', 'i', 'j']:
                spd_rec[col] = data[col].values
            spd_rec['flux'] = data['q'].values
            spd[per] = spd_rec
        wel = flopy.modflow.ModflowWel(m, stress_period_data=spd)

        # not sure why this needs to be done again to retain modelgrid attribute
        m._mg_resync = False

        m.write_input()
        m.exe_name = mfnwt_exe
        success = False
        if exe_exists(mfnwt_exe):
            success, buff = m.run_model()
            if not success:
                list_file = m.name_file.list.array
                with open(list_file) as src:
                    list_output = src.read()

    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

    # Zone Budget comparison only implemented for MF6
    if parent_model.version == 'mf6':
        # Set up zone budget on the parent (for the inset model footprint)
        inset_footprint_within_parent = tmr.inset_zone_within_parent
        inset_lower_left = np.argmax(np.argmax(tmr.inset_zone_within_parent, axis=0))
        # unlike for heads, leave the zone budget footprint in the parent model
        # the same as the inset model area
        # since we want to compare to the boundary fluxes (WEL package) directly
        # (in constrast to CHD package fluxes,
        #  which are evaluated at the interior faces of the boundary cells)
        output_budget_name = Path(m.model_ws).absolute() / f"{m.name}-parent"
        write_zonebudget6_input(inset_footprint_within_parent, budgetfile=parent_budget_file,
                                binary_grid_file=parent_binary_grid_file,
                                outname=output_budget_name)
        # run zonebudget
        process = Popen([str(zbud6_exe), f'{m.name}-parent.zbud.nam'], cwd=Path(m.model_ws).absolute(),
                    stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        assert process.returncode == 0
        # read the zone budget output
        zb_results = pd.read_csv(output_budget_name.with_suffix('.zbud.csv'))
        zb_results['net_flux'] = zb_results['FROM ZONE 0'] - zb_results['TO ZONE 0']

        # run zonebudget on the inset model
        if m.version == 'mf6':
            inset_nrow, inset_ncol = m.dis.top.array.shape
            inset_zone_budget_array = np.ones((inset_nrow, inset_ncol))
            inset_budget_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.cbc')
            inset_binary_grid_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.dis.grb')
            output_budget_name = Path(m.model_ws).absolute() / f"{m.name}-inset"
            write_zonebudget6_input(inset_zone_budget_array, budgetfile=inset_budget_file,
                                    binary_grid_file=inset_binary_grid_file,
                                    outname=output_budget_name)
            process = Popen([str(zbud6_exe), f'{m.name}-inset.zbud.nam'], cwd=Path(m.model_ws).absolute(),
                        stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            assert process.returncode == 0
            # read the zone budget output
            zb_results_inset = pd.read_csv(output_budget_name.with_suffix('.zbud.csv'))
            zb_results_inset['net_flux'] = zb_results_inset['FROM ZONE 0'] - zb_results_inset['TO ZONE 0']
            #perimeter_chd_package = 'CHD_1'
            zb_results_inset['bound_net'] = zb_results_inset['WEL-IN'] -\
                zb_results_inset['WEL-OUT']
            # compare inset/parent zone budget results
            assert np.allclose(zb_results_inset['WEL-IN'], zb_results['FROM ZONE 0'], rtol=0.05)
            assert np.allclose(zb_results_inset['WEL-OUT'], zb_results['TO ZONE 0'], rtol=0.05)

        if m.version == 'mf6':
            # get the inset model boundary fluxes
            inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.lst')
            mfl = Mf6ListBudget(inset_list_file)
            wel_in_col, wel_out_col = 'WEL_IN', 'WEL_OUT'
        else:
            # get the inset model boundary fluxes
            inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.list')
            mfl = MfListBudget(inset_list_file)
            wel_in_col, wel_out_col = 'WELLS_IN', 'WELLS_OUT'

        df_flux, df_vol = mfl.get_dataframes()
        df_flux['bound_net'] = df_flux[wel_in_col] - df_flux[wel_out_col]
        df_flux.reset_index(inplace=True)

        # compare the fluxes
        # check that total in/out fluxes match
        # Note: the listing file evaluates flux across the interior face of the boundary cells
        # so the zone budget results need to be for an inset footprint that is one cell smaller on each side
        assert np.allclose(df_flux[wel_in_col], zb_results['FROM ZONE 0'], rtol=0.02)
        assert np.allclose(df_flux[wel_out_col], zb_results['TO ZONE 0'], rtol=0.02)

    else:
        if m.version == 'mf6':
            # get the inset model boundary fluxes
            inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.lst')
            mfl = Mf6ListBudget(inset_list_file)
            wel_in_col, wel_out_col = 'WEL_IN', 'WEL_OUT'
        else:
            # get the inset model boundary fluxes
            inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.list')
            mfl = MfListBudget(inset_list_file)
            wel_in_col, wel_out_col = 'WELLS_IN', 'WELLS_OUT'

        df_flux, df_vol = mfl.get_dataframes()
        df_flux['bound_net'] = df_flux[wel_in_col] - df_flux[wel_out_col]
        df_flux.reset_index(inplace=True)

        # for MODFLOW-NWT, do a simpler absolute comparison
        # (based on values from MF6 model)
        assert np.allclose(df_flux[wel_in_col], 14.2125, rtol=0.05)
        assert np.allclose(df_flux[wel_out_col], 14.2125, rtol=0.05)

    # compare heads between the parent and inset models
    parent_hds = bf.HeadFile(parent_ws / f"{parent_model.name}.hds")
    parent_heads = parent_hds.get_data(kstpkper=(0, 0))
    inset_hds = bf.HeadFile(Path(m.model_ws).absolute() / f"{m.name}.hds")
    inset_heads = inset_hds.get_data(kstpkper=(0, 0))
    head_diff = inset_heads[0] - parent_heads[0, 20:40, 20:40]
    rms_error = np.sqrt(np.mean(head_diff.ravel()**2))
    assert np.allclose(inset_heads[0].ravel(),
                       parent_heads[0, 20:40, 20:40].ravel(), atol=0.01)

    # make some figures
    make_figures = True
    if make_figures:
        from matplotlib import pyplot as plt


        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        pmv = flopy.plot.PlotMapView(model=parent_model, ax=ax)
        arr = pmv.plot_array(parent_heads)
        contours = pmv.contour_array(parent_heads, colors="white", levels=np.linspace(27, 29, 21))
        ax.clabel(contours, fmt="%2.2f")
        plt.colorbar(arr, shrink=0.5, ax=ax)
        ax.set_title("Simulated Heads")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        pmv = flopy.plot.PlotMapView(model=m, ax=ax)
        arr = pmv.plot_array(inset_heads, vmin=parent_heads.min(), vmax=parent_heads.max())
        contours = pmv.contour_array(inset_heads, colors="red", levels=np.linspace(27, 29, 21))
        ax.clabel(contours, fmt="%2.2f")
        plt.colorbar(arr, shrink=0.5, ax=ax)
        ax.set_title("Simulated Heads")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.savefig(project_root_path / 'mfsetup/tests/tmp/perimeter_bc_demo/head_comp.pdf')

        fig, ax = plt.subplots()
        pmv = flopy.plot.PlotMapView(model=m, ax=ax)
        head_diff = inset_heads[0] - parent_heads[0, 20:40, 20:40]
        arr = pmv.plot_array(head_diff)
        contours = pmv.contour_array(head_diff, colors="w",
                                     levels=np.linspace(head_diff.min(), head_diff.max(), 10))
        ax.clabel(contours, fmt="%2.4f")
        plt.colorbar(arr, shrink=0.5, ax=ax)
        ax.set_title("Difference in head (pos. values indicate inset > parent)")
        plt.savefig(project_root_path / 'mfsetup/tests/tmp/perimeter_bc_demo/head_diff.pdf')

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        axes = axes.flat
        pmv = flopy.plot.PlotMapView(model=parent_model, ax=axes[0], layer=0)
        #pmv.plot_bc('CHD_0')
        pmv.plot_bc('CHD')
        pmv.plot_grid()
        pmv2 = flopy.plot.PlotMapView(model=m, ax=axes[1], layer=0)
        pmv2.plot_bc('CHD')
        pmv2.plot_bc('WEL')
        pmv2.plot_grid()
        plt.savefig(project_root_path / 'mfsetup/tests/tmp/perimeter_bc_demo/grid_comp.pdf')


def test_parent_xyzcellfacecenters(parent_model_mf6, inset_model_mf6):
    parent_ws = Path(parent_model_mf6.model_ws)
    #boundary_shapefile = parent_ws / 'gis/irregular_boundary.shp'
    parent_budget_file = parent_ws / f'{parent_model_mf6.name}.cbc'
    parent_head_file = parent_ws / f'{parent_model_mf6.name}.hds'
    parent_binary_grid_file = parent_ws / f'{parent_model_mf6.name}.dis.grb'
    tmr = Tmr(parent_model_mf6, inset_model_mf6, parent_cell_budget_file=parent_budget_file,
                 parent_binary_grid_file=parent_binary_grid_file,
                 parent_head_file=parent_head_file,
                 boundary_type='flux',
                 )
    result = tmr.parent_xyzcellfacecenters
    assert not {'right', 'bottom'}.difference(result.keys())
    for item in result.values():
        for i in range(len(item)):
            assert item[i].shape == tmr.parent_xyzcellcenters[i].shape
    # in theory, the actual values returned by parent_xyzcellfacecenters
    # are tested in test_grid.py::test_get_cellface_midpoint
