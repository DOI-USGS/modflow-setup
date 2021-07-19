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
from flopy.utils import Mf6ListBudget
from flopy.utils import binaryfile as bf

from mfsetup.discretization import get_layer
from mfsetup.fileio import exe_exists
from mfsetup.grid import MFsetupGrid, get_ij
from mfsetup.tmr import Tmr, TmrNew, get_qx_qy_qz
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


@pytest.fixture(scope='function')
def tmr(pleasant_model):
    m = pleasant_model
    tmr = Tmr(m.parent, m,
              parent_head_file=m.cfg['chd']['perimeter_boundary']['parent_head_file'],
              inset_parent_layer_mapping=m.parent_layers,
              inset_parent_period_mapping=m.parent_stress_periods)
    return tmr


@pytest.fixture(scope='function')
def parent_heads(tmr):
    headfile = tmr.hpth
    hdsobj = bf.HeadFile(headfile)
    yield hdsobj  # provide the fixture value
    # code below yield statement is executed after test function finishes
    print("closing the heads file")
    hdsobj.close()


def test_get_inset_boundary_heads(tmr, parent_heads):
    """Verify that inset model specified head boundary accurately
    reflects parent model head solution, including when cells
    are dry or missing (e.g. pinched out cells in MF6).
    """
    bheads_df = tmr.get_inset_boundary_heads(for_external_files=False)
    groups = bheads_df.groupby('per')
    all_kstpkper = parent_heads.get_kstpkper()
    kstpkper_list = [all_kstpkper[0], all_kstpkper[-1]]
    for kstp, kper in kstpkper_list:
        hds = parent_heads.get_data(kstpkper=(kstp, kper))
        df = groups.get_group(kper)
        df['cellid'] = list(zip(df.k, df.i, df.j))
        # check for duplicate locations (esp. corners)
        # in mf2005, duplicate chd heads will be summed
        assert not df.cellid.duplicated().any()

        # x, y, z locations of inset model boundary cells
        ix = tmr.inset.modelgrid.xcellcenters[df.i, df.j]
        iy = tmr.inset.modelgrid.ycellcenters[df.i, df.j]
        iz = tmr.inset.modelgrid.zcellcenters[df.k, df.i, df.j]

        # parent model grid cells associated with inset boundary cells
        i, j = get_ij(tmr.parent.modelgrid, ix, iy)
        k = get_layer(tmr.parent.dis.botm.array, i, j, iz)

        # error between parent heads and inset heads
        # todo: interpolate parent head solution to inset points for comparison


@pytest.mark.parametrize('specific_discharge',(False, True))
@pytest.mark.parametrize('version', ('mf6', 'mfnwt'))
def test_get_qx_qy_qz(test_data_path, version, specific_discharge):
    if version == 'mf6':
        cell_budget_file = test_data_path / 'shellmound/tmr_parent/shellmound.cbc'
        binary_grid_file = test_data_path / 'shellmound/tmr_parent/shellmound.dis.grb'
        model_top = None
        model_bottom_array = None
    else:
        cell_budget_file = test_data_path / 'plainfieldlakes/pfl.cbc'
        model_top = np.loadtxt(test_data_path / 'plainfieldlakes/external/top.dat')
        botms = []
        for i in range(4):
            arr = np.loadtxt(test_data_path / f'plainfieldlakes/external/botm{i}.dat')
            botms.append(arr)
        model_bottom_array = np.array(botms)
    qx, qy, qz = get_qx_qy_qz(cell_budget_file, binary_grid_file=binary_grid_file,
                              version=version,
                              model_top=model_top, model_bottom_array=model_bottom_array,
                              specific_discharge=specific_discharge)
    j=2


def test_tmr_new(pleasant_model):
    m = pleasant_model
    parent_headfile = Path(m.cfg['chd']['perimeter_boundary']['parent_head_file'])
    parent_cellbudgetfile = parent_headfile.with_suffix('.cbc')

    tmr = TmrNew(m.parent, m,
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
    tmr = TmrNew(m.parent, m,
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
def parent_model_mf6(tmpdir, mf6_exe):
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
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE")
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name)

    ncells_side = 30
    dis = flopy.mf6.ModflowGwfdis(model, nlay=3, nrow=ncells_side, ncol=ncells_side,
                                delr=100, delc=100,
                                top=30., botm=[20.,10.,0.]
                                )
    start = 30. * np.ones_like(dis.botm.array)
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1., save_flows=True)
    # set up CHD boundaries
    # for eastward flow through the west boundary
    # curving to northward flow through the north boundary
    chd_start_pos = int(ncells_side / 2)
    nchd_side = ncells_side - chd_start_pos
    w_heads = list(np.ones((nchd_side)) * 29.)
    w_heads_i = list(range(chd_start_pos, ncells_side))
    w_heads_j = [0] * len(w_heads)
    n_heads = list(np.array(w_heads) - 2.)
    n_heads_i = [0] * len(n_heads)
    n_heads_j = w_heads_i
    perim_chd = pd.DataFrame({'k': 0,
                            'i': w_heads_i + n_heads_i,
                            'j': w_heads_j + n_heads_j,
                            'head': w_heads + n_heads
                            })
    perim_chd['cellid'] = list(zip(perim_chd['k'], perim_chd['i'], perim_chd['j']))
    perim_chd_rec = perim_chd[['cellid', 'head']].to_records(index=False)

    chd = flopy.mf6.ModflowGwfchd(model, maxbound=len(perim_chd_rec),
                                stress_period_data=perim_chd_rec,
                                save_flows=True)
    oc = flopy.mf6.ModflowGwfoc(model,
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                head_filerecord=f"{name}.hds",
                                budget_filerecord=f"{name}.cbc",
                                )
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
def inset_model_mf6(tmpdir, mf6_exe):
    """Make a simple inset model to go in parent model
    """
    name = 'tmr_inset'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/inset'
    model_ws.mkdir(exist_ok=True, parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name='mf6',
                    sim_ws=str(model_ws))
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=1,
                                perioddata=[(1.0, 1, 1.0)])
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE")
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name)

    ncells_side = 100
    dis = flopy.mf6.ModflowGwfdis(model, nlay=3, nrow=ncells_side, ncol=ncells_side,
                                delr=10, delc=10,
                                top=30., botm=[20.,10.,0.]
                                )
    start = 30. * np.ones_like(dis.botm.array)
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1., save_flows=True)

    oc = flopy.mf6.ModflowGwfoc(model,
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                head_filerecord=f"{name}.hds",
                                budget_filerecord=f"{name}.cbc",
                                )
    model._modelgrid = MFsetupGrid(delc=model.dis.delc.array, delr=model.dis.delr.array,
                                  top=model.dis.top.array, botm=model.dis.botm.array,
                                  xoff=1000, yoff=1000)
    model._mg_resync = False
    assert hasattr(model, 'modelgrid'), "something went wrong setting the modelgrid attribute"
    return model


@pytest.fixture
def parent_model_nwt(tmpdir, mf6_exe):
    """Make a simpmle parent model for TMR perimeter boundary tests,
    with inflow from west that curves to outflow to the north.

    TODO : convert this to NWT model
    """
    # set up simulation
    name = 'tmr_parent'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/parent'
    model_ws.mkdir(exist_ok=True, parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(model_ws))
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=1,
                                perioddata=[(1.0, 1, 1.0)])
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE")
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name)

    ncells_side = 30
    dis = flopy.mf6.ModflowGwfdis(model, nlay=3, nrow=ncells_side, ncol=ncells_side,
                                delr=100, delc=100,
                                top=30., botm=[20.,10.,0.]
                                )
    start = 30. * np.ones_like(dis.botm.array)
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1., save_flows=True)
    # set up CHD boundaries
    # for eastward flow through the west boundary
    # curving to northward flow through the north boundary
    chd_start_pos = int(ncells_side / 2)
    nchd_side = ncells_side - chd_start_pos
    w_heads = list(np.ones((nchd_side)) * 29.)
    w_heads_i = list(range(chd_start_pos, ncells_side))
    w_heads_j = [0] * len(w_heads)
    n_heads = list(np.array(w_heads) - 2.)
    n_heads_i = [0] * len(n_heads)
    n_heads_j = w_heads_i
    perim_chd = pd.DataFrame({'k': 0,
                            'i': w_heads_i + n_heads_i,
                            'j': w_heads_j + n_heads_j,
                            'head': w_heads + n_heads
                            })
    perim_chd['cellid'] = list(zip(perim_chd['k'], perim_chd['i'], perim_chd['j']))
    perim_chd_rec = perim_chd[['cellid', 'head']].to_records(index=False)

    chd = flopy.mf6.ModflowGwfchd(model, maxbound=len(perim_chd_rec),
                                stress_period_data=perim_chd_rec,
                                save_flows=True)
    oc = flopy.mf6.ModflowGwfoc(model,
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                head_filerecord=f"{name}.hds",
                                budget_filerecord=f"{name}.cbc",
                                )
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
def inset_model_nwt(tmpdir, mf6_exe):
    """Make a simple inset model to go in parent model

    TODO: convert this to MODFLOW NWT model
    """
    name = 'tmr_inset'
    model_ws = Path(tmpdir) / 'perimeter_bc_demo/inset'
    model_ws.mkdir(exist_ok=True, parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name='mf6',
                    sim_ws=str(model_ws))
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=1,
                                perioddata=[(1.0, 1, 1.0)])
    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE")
    # create model instance
    model = flopy.mf6.ModflowGwf(sim, modelname=name)

    ncells_side = 100
    dis = flopy.mf6.ModflowGwfdis(model, nlay=3, nrow=ncells_side, ncol=ncells_side,
                                delr=10, delc=10,
                                top=30., botm=[20.,10.,0.]
                                )
    start = 30. * np.ones_like(dis.botm.array)
    ic = flopy.mf6.ModflowGwfic(model, pname="ic", strt=start)
    npf = flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1., save_flows=True)

    oc = flopy.mf6.ModflowGwfoc(model,
                                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                head_filerecord=f"{name}.hds",
                                budget_filerecord=f"{name}.cbc",
                                )
    model._modelgrid = MFsetupGrid(delc=model.dis.delc.array, delr=model.dis.delr.array,
                                  top=model.dis.top.array, botm=model.dis.botm.array,
                                  xoff=1000, yoff=1000)
    model._mg_resync = False
    assert hasattr(model, 'modelgrid'), "something went wrong setting the modelgrid attribute"
    return model


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
def parent_model(request,
                 inset_model_mf6,
                 inset_model_nwt):
    """MODFLOW-NWT and MODFLOW-6 versions of the test case inset model."""
    return {'inset_model_mf6': inset_model_mf6,
            'inset_model_nwt': inset_model_nwt}[request.param]


def test_get_boundary_heads(parent_model, inset_model,
                            project_root_path,
                            mf6_exe, zbud6_exe):
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
    m = inset_model
    parent_ws = Path(parent_model.model_ws)
    #boundary_shapefile = parent_ws / 'gis/irregular_boundary.shp'
    parent_budget_file = parent_ws / f'{parent_model.name}.cbc'
    parent_head_file = parent_ws / f'{parent_model.name}.hds'
    parent_binary_grid_file = parent_ws / f'{parent_model.name}.dis.grb'
    tmr = TmrNew(parent_model, m, parent_head_file=parent_head_file,
                 boundary_type='head',
                 )
    perimeter_df = tmr.get_inset_boundary_values()

    # set up the CHD package
    perimeter_df['cellid'] = list(perimeter_df[['k', 'i', 'j']].to_records(index=False))
    period_groups = perimeter_df.groupby('per')
    spd = {}
    maxbound = 0
    for per, data in period_groups:
        spd[per] = data[['cellid', 'head']].to_records(index=False)
        if len(data) > maxbound:
            maxbound = len(data)
    chd = flopy.mf6.ModflowGwfchd(inset_model, maxbound=maxbound,
                                  stress_period_data=spd,
                                  save_flows=True)
    # not sure why this needs to be done again to retain modelgrid attribute
    inset_model._mg_resync = False

    # write the inset model input files
    m.simulation.write_simulation()

    # Set up zone budget on the parent (for the inset model footprint)
    inset_footprint_within_parent = tmr.inset_zone_within_parent
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

    # run the inset model
    m.simulation.exe_name = mf6_exe
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

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
    zb_results_inset['chd_net'] = zb_results_inset['CHD-IN'] - zb_results_inset['CHD-OUT']

    # get the inset model boundary fluxes
    inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.lst')
    mfl = Mf6ListBudget(inset_list_file)
    df_flux, df_vol = mfl.get_dataframes()
    df_flux['CHD_net'] = df_flux['CHD_IN'] - df_flux['CHD_OUT']
    df_flux.reset_index(inplace=True)

    # compare the fluxes
    # check that total in/out fluxes are within 5%
    assert np.allclose(df_flux['CHD_IN'], zb_results['FROM ZONE 0'], rtol=0.05)
    assert np.allclose(df_flux['CHD_OUT'], zb_results['TO ZONE 0'], rtol=0.05)

    # check that the heads were applied correctly
    inset_heads_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.hds')
    hdsobj = bf.HeadFile(inset_heads_file)
    allhds = hdsobj.get_alldata()
    k, i, j, per = perimeter_df[['k', 'i', 'j', 'per']].T.values
    perimeter_df['inset_head'] = allhds[per, k, i, j]
    assert np.allclose(perimeter_df['inset_head'].values,
                       perimeter_df['head'].values)


def test_get_boundary_fluxes(parent_model_mf6, inset_model_mf6,
                            project_root_path,
                            mf6_exe, zbud6_exe):
    """Test getting perimeter boundary flux values from a parent model,
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
    # TODO : change parent_model_mf6 and inset_model_mf6 fixtures to
    # parent_model and inset_model to run MODFLOW-NWT version as well
    project_root_path = Path(project_root_path)

    #m = get_pleasant_mf6_with_dis
    #parent_ws = project_root_path / 'examples/data/pleasant/'
    m = inset_model
    parent_ws = Path(parent_model.model_ws)
    #boundary_shapefile = parent_ws / 'gis/irregular_boundary.shp'
    parent_budget_file = parent_ws / f'{parent_model.name}.cbc'
    parent_head_file = parent_ws / f'{parent_model.name}.hds'
    parent_binary_grid_file = parent_ws / f'{parent_model.name}.dis.grb'
    tmr = TmrNew(parent_model, m, parent_cell_budget_file=parent_budget_file,
                 parent_binary_grid_file=parent_binary_grid_file,
                 boundary_type='flux',
                 )
    perimeter_df = tmr.get_inset_boundary_values()

    # set up the WEL package
    perimeter_df['cellid'] = list(perimeter_df[['k', 'i', 'j']].to_records(index=False))
    period_groups = perimeter_df.groupby('per')
    spd = {}
    maxbound = 0
    for per, data in period_groups:
        spd[per] = data[['cellid', 'q']].to_records(index=False)
        if len(data) > maxbound:
            maxbound = len(data)
    wel = flopy.mf6.ModflowGwfwel(inset_model, maxbound=maxbound,
                                  stress_period_data=spd,
                                  save_flows=True)
    # not sure why this needs to be done again to retain modelgrid attribute
    inset_model._mg_resync = False

    # write the inset model input files
    m.simulation.write_simulation()

    # Set up zone budget on the parent (for the inset model footprint)
    inset_footprint_within_parent = tmr.inset_zone_within_parent
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

    # run the inset model
    m.simulation.exe_name = mf6_exe
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

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
    zb_results_inset['wel_net'] = zb_results_inset['WEL-IN'] - zb_results_inset['WEL-OUT']

    # get the inset model boundary fluxes
    inset_list_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.lst')
    mfl = Mf6ListBudget(inset_list_file)
    df_flux, df_vol = mfl.get_dataframes()
    df_flux['wel_net'] = df_flux['WEL_IN'] - df_flux['WEL_OUT']
    df_flux.reset_index(inplace=True)

    # compare the fluxes
    # check that total in/out fluxes are within 5%
    assert np.allclose(df_flux['WEL_IN'], zb_results['FROM ZONE 0'], rtol=0.05)
    assert np.allclose(df_flux['WEL_OUT'], zb_results['TO ZONE 0'], rtol=0.05)

    # compare the heads
    # create a second Tmr object that gets the heads from the parent
    tmr_head = TmrNew(parent_model, m, parent_head_file=parent_head_file,
                      boundary_type='head',
                      )
    perimeter_heads_df = tmr_head.get_inset_boundary_values()

    inset_heads_file = (Path(m.model_ws).absolute() / m.name).with_suffix('.hds')
    hdsobj = bf.HeadFile(inset_heads_file)
    allhds = hdsobj.get_alldata()
    k, i, j, per = perimeter_heads_df[['k', 'i', 'j', 'per']].T.values
    perimeter_heads_df['inset_head'] = allhds[per, k, i, j]

    # TODO: ? apply a suitable tolerance here
    assert np.allclose(perimeter_heads_df['inset_head'].values,
                       perimeter_heads_df['head'].values, atol=0.01)
