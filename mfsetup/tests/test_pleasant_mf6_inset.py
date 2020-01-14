"""
Tests for Pleasant Lake inset case, MODFLOW-6 version
* creating MODFLOW-6 inset model from MODFLOW-NWT parent
* MODFLOW-6 Lake package
"""
import copy
import os
import glob
import numpy as np
import pandas as pd
import rasterio
import pytest
import flopy
mf6 = flopy.mf6
from mfsetup import MF6model
from mfsetup.checks import check_external_files_for_nans
from mfsetup.fileio import load_cfg, read_mf6_block, exe_exists
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pleasant_mf6_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/pleasant_mf6_test.yml'


@pytest.fixture(scope="function")
def pleasant_mf6_cfg(pleasant_mf6_test_cfg_path):
    cfg = load_cfg(pleasant_mf6_test_cfg_path,
                   default_file='/mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def pleasant_simulation(pleasant_mf6_cfg):
    cfg = pleasant_mf6_cfg.copy()
    sim = mf6.MFSimulation(**cfg['simulation'])
    return sim


@pytest.fixture(scope="function")
def get_pleasant_mf6(pleasant_mf6_cfg, pleasant_simulation):
    print('creating Pleasant Lake MF6model instance from cfgfile...')
    cfg = pleasant_mf6_cfg.copy()
    cfg['model']['simulation'] = pleasant_simulation
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def get_pleasant_mf6_with_grid(get_pleasant_mf6):
    print('creating Pleasant Lake MFnwtModel instance with grid...')
    m = copy.deepcopy(get_pleasant_mf6)
    m.setup_grid()
    return m


@pytest.fixture(scope="function")
def get_pleasant_mf6_with_dis(get_pleasant_mf6_with_grid):
    print('creating Pleasant Lake MFnwtModel instance with grid...')
    m = copy.deepcopy(get_pleasant_mf6_with_grid)
    m.setup_tdis()
    m.setup_dis()
    return m


@pytest.fixture(scope="function")
def pleasant_mf6_setup_from_yaml(pleasant_mf6_test_cfg_path):
    m = MF6model.setup_from_yaml(pleasant_mf6_test_cfg_path)
    m.write_input()
    #if hasattr(m, 'sfr'):
    #    sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
    #    m.sfrdata.write_package(sfr_package_filename,
    #                                version='mf6',
    #                                options=['save_flows',
    #                                         'BUDGET FILEOUT shellmound.sfr.cbc',
    #                                         'STAGE FILEOUT shellmound.sfr.stage.bin',
    #                                         # 'OBS6 FILEIN {}'.format(sfr_obs_filename)
    #                                         # location of obs6 file relative to sfr package file (same folder)
    #                                         ]
    #                                )
    return m


@pytest.fixture(scope="function")
def pleasant_mf6_model_run(pleasant_mf6_setup_from_yaml, mf6_exe):
    m = copy.deepcopy(pleasant_mf6_setup_from_yaml)
    m.simulation.exe_name = mf6_exe
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


def test_model(get_pleasant_mf6_with_grid):
    m = get_pleasant_mf6_with_grid
    assert m.version == 'mf6'
    assert 'UPW' in m.parent.get_package_list()


def test_perioddata(get_pleasant_mf6):
    m = get_pleasant_mf6
    m._set_perioddata()
    assert m.perioddata['start_datetime'][0] == pd.Timestamp(m.cfg['tdis']['options']['start_date_time'])


def test_tdis_setup(get_pleasant_mf6):

    m = get_pleasant_mf6 #deepcopy(model)
    tdis = m.setup_tdis()
    tdis.write()
    assert os.path.exists(os.path.join(m.model_ws, tdis.filename))
    assert isinstance(tdis, mf6.ModflowTdis)
    period_df = pd.DataFrame(tdis.perioddata.array)
    period_df['perlen'] = period_df['perlen'].astype(np.float64)
    period_df['nstp'] = period_df['nstp'].astype(np.int64)
    pd.testing.assert_frame_equal(period_df[['perlen', 'nstp', 'tsmult']],
                                  m.perioddata[['perlen', 'nstp', 'tsmult']])


def test_dis_setup(get_pleasant_mf6_with_grid):

    m = get_pleasant_mf6_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    botm = m.dis.botm.array.copy()
    assert isinstance(dis, mf6.ModflowGwfdis)
    assert 'DIS' in m.get_package_list()
    assert m.dis.length_units.array == 'meters'

    arrayfiles = m.cfg['intermediate_data']['top'] + \
                 m.cfg['intermediate_data']['botm'] + \
                 m.cfg['intermediate_data']['idomain']
    for f in arrayfiles:
        assert os.path.exists(f)
        fname = os.path.splitext(os.path.split(f)[1])[0]
        k = ''.join([s for s in fname if s.isdigit()])
        var = fname.strip(k)
        data = np.loadtxt(f)
        model_array = getattr(m.dis, var).array
        if len(k) > 0:
            k = int(k)
            model_array = model_array[k]
        assert np.array_equal(model_array, data)


def test_idomain(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    assert issubclass(m.idomain.dtype.type, np.integer)
    assert m.idomain.sum() == m.dis.idomain.array.sum()


def test_ic_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    ic = m.setup_ic()
    ic.write()
    assert os.path.exists(os.path.join(m.model_ws, ic.filename))
    assert isinstance(ic, mf6.ModflowGwfic)
    assert ic.strt.array.shape == m.dis.botm.array.shape


def test_sto_setup(get_pleasant_mf6_with_dis):

    m = get_pleasant_mf6_with_dis  #deepcopy(model_with_grid)
    sto = m.setup_sto()
    sto.write()
    assert os.path.exists(os.path.join(m.model_ws, sto.filename))
    assert isinstance(sto, mf6.ModflowGwfsto)
    for var in ['sy', 'ss']:
        model_array = getattr(sto, var).array
        for k, item in enumerate(m.cfg['sto']['griddata'][var]):
            f = item['filename']
            assert os.path.exists(f)
            data = np.loadtxt(f)
            assert np.array_equal(model_array[k], data)


def test_npf_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    npf = m.setup_npf()
    npf.write()
    assert isinstance(npf, mf6.ModflowGwfnpf)
    assert os.path.exists(os.path.join(m.model_ws, npf.filename))


def test_obs_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    obs = m.setup_obs()
    obs.write()
    obsfile = os.path.join(m.model_ws, obs.filename)
    assert os.path.exists(obsfile)
    assert isinstance(obs, mf6.ModflowUtlobs)
    with open(obsfile) as obsdata:
        for line in obsdata:
            if 'fileout' in line.lower():
                _, _, _, fname = line.strip().split()
                assert fname == m.cfg['obs']['filename_fmt'].format(m.name)
                break


def test_oc_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    oc = m.setup_oc()
    oc.write()
    ocfile = os.path.join(m.model_ws, oc.filename)
    assert os.path.exists(ocfile)
    assert isinstance(oc, mf6.ModflowGwfoc)
    options = read_mf6_block(ocfile, 'options')
    options = {k: ' '.join(v).lower() for k, v in options.items()}
    perioddata = read_mf6_block(ocfile, 'period')
    assert 'fileout' in options['budget'] and '.cbc' in options['budget']
    assert 'fileout' in options['head'] and '.hds' in options['head']
    assert 'save head last' in perioddata[1]
    assert 'save budget last' in perioddata[1]


def test_rch_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    rch = m.setup_rch()
    rch.write()
    assert os.path.exists(os.path.join(m.model_ws, rch.filename))
    assert isinstance(rch, mf6.ModflowGwfrcha)
    assert rch.recharge is not None


def test_wel_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    wel = m.setup_wel()
    wel.write()
    assert os.path.exists(os.path.join(m.model_ws, wel.filename))
    assert isinstance(wel, mf6.ModflowGwfwel)
    assert wel.stress_period_data is not None


@pytest.mark.skip('not implemented yet')
def test_ghb_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    ghb = m.setup_ghb()
    ghb.write()
    assert os.path.exists(os.path.join(m.model_ws, ghb.filename))
    assert isinstance(ghb, mf6.ModflowGwfghb)
    assert ghb.stress_period_data is not None

    # check for inactive cells
    spd0 = ghb.stress_period_data.array[0]
    k, i, j = zip(*spd0['cellid'])
    inactive_cells = m.idomain[k, i, j] < 1
    assert not np.any(inactive_cells)

    # check that heads are above layer botms
    assert np.all(spd0['head'] > m.dis.botm.array[k, i, j])


def test_sfr_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    m.setup_sfr()
    m.sfr.write()
    assert os.path.exists(os.path.join(m.model_ws, m.sfr.filename))
    assert isinstance(m.sfr, mf6.ModflowGwfsfr)
    output_path = m.cfg['sfr']['output_path']
    shapefiles = ['{}/{}_sfr_cells.shp'.format(output_path, m.name),
                  '{}/{}_sfr_outlets.shp'.format(output_path, m.name),
                  #'{}/{}_sfr_inlets.shp'.format(output_path, m.name),
                  '{}/{}_sfr_lines.shp'.format(output_path, m.name),
                  '{}/{}_sfr_routing.shp'.format(output_path, m.name)
    ]
    for f in shapefiles:
        assert os.path.exists(f)
    assert m.sfrdata.model == m


def test_perimeter_boundary_setup(get_pleasant_mf6_with_dis):

    m = get_pleasant_mf6_with_dis  #deepcopy(pfl_nwt_with_dis)
    chd = m.setup_perimeter_boundary()
    chd.write()
    assert os.path.exists(os.path.join(m.model_ws, chd.filename))
    assert len(chd.stress_period_data.array) == len(set(m.cfg['parent']['copy_stress_periods']))
    assert len(m.get_boundary_cells()[0]) == (m.nrow*2 + m.ncol*2 - 4) * m.nlay  # total number of boundary cells
    # number of boundary heads;
    # can be less than number of active boundary cells if the (parent) water table is not always in (inset) layer 1
    assert len(chd.stress_period_data.array[0]) <= np.sum(m.idomain[m.get_boundary_cells()] == 1)

    # check for inactive cells
    spd0 = chd.stress_period_data.array[0]
    k, i, j = zip(*spd0['cellid'])
    inactive_cells = m.idomain[k, i, j] < 1
    assert not np.any(inactive_cells)

    # check that heads are above layer botms
    assert np.all(spd0['head'] > m.dis.botm.array[k, i, j])


def test_model_setup(pleasant_mf6_setup_from_yaml):
    m = pleasant_mf6_setup_from_yaml
    assert isinstance(m, MF6model)
    assert 'tdis' in m.simulation.package_key_dict
    assert 'ims' in m.simulation.package_key_dict
    assert m.get_package_list() == ['DIS', 'IC', 'NPF', 'STO', 'RCHA', 'OC', 'SFR', 'WEL_0', 'OBS_0', 'CHD_0']
    external_path = os.path.join(m.model_ws, 'external')
    external_files = glob.glob(external_path + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = '\n'.join(has_nans)
    if len(has_nans) > 0:
        assert False, has_nans


def test_model_setup_and_run(pleasant_mf6_model_run):
    m = pleasant_mf6_model_run

