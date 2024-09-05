"""
Tests for Pleasant Lake inset case, MODFLOW-6 version
* creating MODFLOW-6 inset model from MODFLOW-NWT parent
* MODFLOW-6 Lake package
"""
import copy
import glob
import os
import shutil
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pytest
from gisutils import get_values_at_points

mf6 = flopy.mf6
fm = flopy.modflow
from mfsetup import MF6model
from mfsetup.checks import check_external_files_for_nans
from mfsetup.fileio import exe_exists, load_cfg, read_lak_ggo, read_mf6_block
from mfsetup.testing import compare_inset_parent_values
from mfsetup.tests.plot import make_lake_xsections
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pleasant_mf6_test_cfg_path(project_root_path):
    return project_root_path / 'mfsetup/tests/data/pleasant_mf6_test.yml'


@pytest.fixture(scope="function")
def pleasant_mf6_cfg(pleasant_mf6_test_cfg_path):
    cfg = load_cfg(pleasant_mf6_test_cfg_path,
                   default_file='mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def pleasant_simulation(pleasant_mf6_cfg):
    cfg = pleasant_mf6_cfg
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    return sim


@pytest.fixture(scope="function")
def get_pleasant_mf6(pleasant_mf6_cfg, pleasant_simulation):
    print('creating Pleasant Lake MF6model instance from cfgfile...')
    cfg = copy.deepcopy(pleasant_mf6_cfg)
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
    print('creating Pleasant Lake MFnwtModel instance with DIS package...')
    m = copy.deepcopy(get_pleasant_mf6_with_grid)
    m.setup_tdis()
    m.setup_dis()
    return m


@pytest.fixture(scope="function")
def get_pleasant_mf6_with_sfr(get_pleasant_mf6_with_dis):
    m = copy.deepcopy(get_pleasant_mf6_with_dis)
    m.setup_sfr()
    return m


@pytest.fixture(scope="function")
def get_pleasant_mf6_with_lak(get_pleasant_mf6_with_dis):
    print('creating Pleasant Lake MFnwtModel instance with LAKE package...')
    m = copy.deepcopy(get_pleasant_mf6_with_dis)
    lak = m.setup_lak()
    lak.write()
    return m


@pytest.fixture(scope="function")
def pleasant_mf6_setup_from_yaml(pleasant_mf6_test_cfg_path):
    m = MF6model.setup_from_yaml(pleasant_mf6_test_cfg_path)
    m.write_input()
    #if hasattr(m, 'sfr'):
    #    sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
    #    m.sfrdata.write_package(sfr_package_filename,
    #                            version='mf6',
    #                            idomain=m.idomain,
    #                            options=['save_flows',
    #                                     'BUDGET FILEOUT shellmound.sfr.cbc',
    #                                     'STAGE FILEOUT shellmound.sfr.stage.bin',
    #                                     # 'OBS6 FILEIN {}'.format(sfr_obs_filename)
    #                                     # location of obs6 file relative to sfr package file (same folder)
    #                                     ]
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

    # test skip load
    package_list = [s.lower() for s in m.parent.get_package_list()]
    for package in m.cfg['parent']['skip_load']:
        assert package.lower() not in package_list


def test_perioddata(get_pleasant_mf6, pleasant_nwt):
    nwt = pleasant_nwt
    nwt.perioddata
    m = get_pleasant_mf6
    m.perioddata
    assert m.perioddata['start_datetime'][0] == pd.Timestamp(m.cfg['tdis']['options']['start_date_time'])
    pd.testing.assert_frame_equal(m.perioddata, nwt.perioddata, check_dtype=False)


def test_tdis_setup(get_pleasant_mf6):

    m = get_pleasant_mf6 #deepcopy(model)
    tdis = m.setup_tdis()
    tdis.write()
    assert os.path.exists(os.path.join(m.model_ws, tdis.filename))
    assert isinstance(tdis, mf6.ModflowTdis)
    period_df = pd.DataFrame(tdis.perioddata.array)
    period_df['perlen'] = period_df['perlen'].astype(float)
    period_df['nstp'] = period_df['nstp'].astype(m.perioddata['nstp'].dtype)
    pd.testing.assert_frame_equal(period_df[['perlen', 'nstp', 'tsmult']],
                                  m.perioddata[['perlen', 'nstp', 'tsmult']])

    # check that period start/end dates were added to tdis file
    m.write_input()
    results = read_mf6_block(m.simulation.tdis.filename, 'perioddata')
    for i, line in enumerate(results['perioddata'][1:]):
        start_date, end_date = line.split(':')[1].strip().split('to')
        expected_start_date = pd.Timestamp(m.perioddata.start_datetime[i])
        expected_end_date = pd.Timestamp(m.perioddata.end_datetime[i])
        assert pd.Timestamp(start_date) == expected_start_date
        assert pd.Timestamp(end_date) == expected_end_date


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
        var, *k = fname.split('_')
        data = np.loadtxt(f)
        model_array = getattr(m.dis, var).array
        if len(k) > 0:
            k = int(k[0])
            model_array = model_array[k]
        assert np.array_equal(model_array, data)


def test_dis_setup_inactive_thickness(get_pleasant_mf6_with_grid):

    m = get_pleasant_mf6_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    assert np.all(dis.thickness[m.idomain != 1] == 0.)
    assert np.all(dis.thickness[m.idomain == 1] >= m.cfg['dis']['minimum_layer_thickness'])


def test_idomain(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    assert issubclass(m.idomain.dtype.type, np.integer)
    assert m.idomain.sum() == m.dis.idomain.array.sum()


@pytest.mark.parametrize('from_binary', (False, True))
def test_ic_setup(get_pleasant_mf6_with_dis, from_binary):
    """Test starting heads setup from parent model strt array or parent model head solution
    (MODFLOW binary output)."""
    m = copy.deepcopy(get_pleasant_mf6_with_dis)
    if not from_binary:
        m.cfg['ic']['source_data']['strt'] = None
    ic = m.setup_ic()
    ic.write()
    assert os.path.exists(os.path.join(m.model_ws, ic.filename))
    assert isinstance(ic, mf6.ModflowGwfic)
    assert ic.strt.array.shape == m.dis.botm.array.shape

    assert m.ic.strt.array[m.dis.idomain.array > 0].min() > 250
    assert m.ic.strt.array[m.dis.idomain.array > 0].max() < 350


@pytest.mark.parametrize('simulate_high_k_lakes', (False, True))
def test_sto_setup(get_pleasant_mf6_with_dis, simulate_high_k_lakes):
    m = get_pleasant_mf6_with_dis  #deepcopy(model_with_grid)
    m.cfg['high_k_lakes']['simulate_high_k_lakes'] = simulate_high_k_lakes
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
    period_data = read_mf6_block(sto.filename, 'period')
    assert period_data[1] == ['steady-state']
    assert period_data[2] == ['transient']

    # compare values to parent model
    inset_parent_layer_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
    for var in ['ss', 'sy']:
        parent_array = m.parent.upw.__dict__[var].array
        inset_array = sto.__dict__[var].array
        # with addition of high-k lakes block,
        # ss has a different default value than parent
        if simulate_high_k_lakes and var == 'ss':
            continue
        compare_inset_parent_values(inset_array, parent_array,
                                    m.modelgrid, m.parent.modelgrid,
                                    inset_parent_layer_mapping,
                                    nodata=1.0,
                                    rtol=0.05
                                    )

    if not simulate_high_k_lakes:
        assert not np.any(m._isbc2d == 2)
        assert sto.sy.array.max() < m.cfg['high_k_lakes']['sy']
        assert sto.ss.array.min() > m.cfg['high_k_lakes']['ss']
    else:
        assert np.any(m._isbc2d == 2)
        assert sto.sy.array.max() == m.cfg['high_k_lakes']['sy']
        assert sto.ss.array.min() == m.cfg['high_k_lakes']['ss']


@pytest.mark.parametrize('simulate_high_k_lakes', (False, True))
def test_npf_setup(get_pleasant_mf6_with_dis, simulate_high_k_lakes):
    m = get_pleasant_mf6_with_dis
    m.cfg['high_k_lakes']['simulate_high_k_lakes'] = simulate_high_k_lakes
    npf = m.setup_npf()
    npf.write()
    assert isinstance(npf, mf6.ModflowGwfnpf)
    assert os.path.exists(os.path.join(m.model_ws, npf.filename))

    # compare values to parent model
    # mapping of variables and layers between parent and inset
    variables = {'hk': 'k',
                 'vka': 'k33',
                 }
    inset_parent_layer_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
    for parent_var, inset_var in variables.items():
        parent_array = m.parent.upw.__dict__[parent_var].array
        inset_array = npf.__dict__[inset_var].array
        compare_inset_parent_values(inset_array, parent_array,
                                    m.modelgrid, m.parent.modelgrid,
                                    inset_parent_layer_mapping,
                                    nodata=float(m.cfg['parent']['hiKlakes_value']),
                                    rtol=0.1
                                    )
    if not simulate_high_k_lakes:
        assert not np.any(m._isbc2d == 2)
        assert npf.k.array.max() < m.cfg['high_k_lakes']['high_k_value']
        # for now, k33 not adjusted in setting high-k lakes
    else:
        assert np.any(m._isbc2d == 2)
        assert npf.k.array.max() == m.cfg['high_k_lakes']['high_k_value']
        # for now, k33 not adjusted in setting high-k lakes


def test_obs_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    obs = m.setup_obs(**m.cfg['obs'])
    obs.write()
    obsfile = os.path.join(m.model_ws, obs.filename)
    assert os.path.exists(obsfile)
    assert isinstance(obs, mf6.ModflowUtlobs)
    with open(obsfile) as obsdata:
        for line in obsdata:
            if 'fileout' in line.lower():
                _, _, _, fname = line.strip().split()
                assert fname == m.cfg['obs']['mfsetup_options']['filename_fmt'].format(m.name)
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


@pytest.mark.parametrize('simulate_high_k_lakes', (False, True))
def test_rch_setup(get_pleasant_mf6_with_dis, simulate_high_k_lakes):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    m.cfg['high_k_lakes']['simulate_high_k_lakes'] = simulate_high_k_lakes
    rch = m.setup_rch()
    rch.write()
    assert os.path.exists(os.path.join(m.model_ws, rch.filename))
    assert isinstance(rch, mf6.ModflowGwfrcha)
    assert rch.recharge is not None

    if not simulate_high_k_lakes:
        assert not np.any(m._isbc2d == 2)
        assert np.all(rch.recharge.array.min(axis=(1, 2, 3)) >= 0)
    else:
        assert np.any(m._isbc2d == 2)
        assert np.any(rch.recharge.array.min(axis=(1, 2, 3)) < 0)


def test_wel_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    m.cfg['wel']['mfsetup_options']['external_files'] = False
    wel = m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    wel.write()
    assert os.path.exists(os.path.join(m.model_ws, wel.filename))
    assert isinstance(wel, mf6.ModflowGwfwel)
    assert wel.stress_period_data is not None

    # verify that periodata blocks were written
    output = read_mf6_block(wel.filename, 'period')
    for per, ra in wel.stress_period_data.data.items():
        assert len(output[per + 1]) == len(ra)


def test_lak_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    m.cfg['lak']['external_files'] = False
    m.cfg['lak']['horizontal_connections'] = True
    lak = m.setup_lak()
    lak.write()
    assert isinstance(lak, mf6.ModflowGwflak)
    package_filename = os.path.join(m.model_ws, lak.filename)
    assert os.path.exists(package_filename)
    for f in lak.tables.array['tab6_filename']:
        assert os.path.exists(f)
    options = read_mf6_block(package_filename, 'options')
    for var in ['boundnames', 'save_flows', 'obs6', 'surfdep',
                'time_conversion', 'length_conversion']:
        assert var in options
    assert float(options['time_conversion'][0]) == 86400. == lak.time_conversion.array
    assert float(options['length_conversion'][0]) == 1. == lak.length_conversion.array
    assert lak.nlakes.array == len(lak.tables.array)
    assert lak.packagedata.array['nlakeconn'][0] == len(lak.connectiondata.array)
    # verify that there are no connections to inactive cells
    k, i, j = zip(*lak.connectiondata.array['cellid'])
    inactive = m.dis.idomain.array[k, i, j] < 1
    assert not np.any(inactive)
    assert len(lak.perioddata.array) == m.nper
    lake_fluxes = m.lake_fluxes.copy()
    lake_fluxes['rainfall'] = lake_fluxes['precipitation']
    for per in range(m.nper):
        for var in ['rainfall', 'evaporation']:
            loc = m.lak.perioddata.array[0]['laksetting'] == var
            value = m.lak.perioddata.array[per]['laksetting_data'][loc][0]
            assert np.allclose(value, lake_fluxes.loc[per, var])

    # check the auxilliary table
    connections_lookup_file = m.cfg['lak']['output_files']['connections_lookup_file'].format(m.name)
    connections_lookup_file = os.path.join(m._tables_path, os.path.split(connections_lookup_file)[1])
    info = pd.read_csv(connections_lookup_file)
    assert not info.zone.isnull().any()
    assert not info.loc[info.claktype == 'horizontal', 'cellface'].isnull().any()

    # check the lake discretization
    import rasterio
    i, j = 35, 40  # point in the middle of the lake
    x = m.modelgrid.xcellcenters[i, j]
    y = m.modelgrid.ycellcenters[i, j]
    datum = m.dis.top.array[i, j]
    bathy_raster = m.cfg['lak']['source_data']['bathymetry_raster']['filename']
    with rasterio.open(bathy_raster) as src:
        bathy = np.squeeze(list(src.sample(zip([x], [y]))))
        bathy[(bathy == src.nodata) | (bathy == 0)] = np.nan

    # all inactive cells should have zero thickness
    edges = [m.dis.top.array[i, j]] + m.dis.botm.array[:, i, j].tolist()
    assert np.all((np.diff(edges) != 0) ==\
        m.dis.idomain.array[:, i, j].astype(bool))
    # original version of this test had the first two layers as inactive
    # not sure why
    #assert np.allclose(m.dis.botm.array[:2, i, j], m.dis.top[i, j])
    #assert np.allclose(m.dis.idomain.array[:2, i, j], 0)


def test_external_tables(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis
    lak = m.setup_lak()
    lak.write()
    assert os.path.exists(m.cfg['external_files']['lak_connectiondata'][0])
    blocks = read_mf6_block(lak.filename, 'connectiondata')
    assert blocks['connectiondata'][0].strip().split()[1].strip('\'') == \
           m.cfg['external_files']['lak_connectiondata'][0]

    wel = m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    wel.write()
    for f in m.cfg['external_files']['wel_stress_period_data'].values():
        assert os.path.exists(f)
    blocks = read_mf6_block(wel.filename, 'period')
    for period, block in blocks.items():
        assert block[0].strip().split()[1].strip('\'') in m.cfg['external_files']['wel_stress_period_data'].values()

    chd = m.setup_chd(**m.cfg['chd'], **m.cfg['chd']['mfsetup_options'])
    chd.write()
    for f in m.cfg['external_files']['chd_stress_period_data'].values():
        assert os.path.exists(f)
    blocks = read_mf6_block(chd.filename, 'period')
    for period, block in blocks.items():
        assert block[0].strip().split()[1].strip('\'') in m.cfg['external_files']['chd_stress_period_data'].values()


def test_lak_obs_setup(get_pleasant_mf6_with_dis):
    m = get_pleasant_mf6_with_dis  # deepcopy(model)
    lak = m.setup_lak()
    m.write()
    # todo: add lake obs tests
    obsinput = read_mf6_block('{}.lak.obs'.format(m.name), 'continuous')
    boundnames = set()
    for k, v in obsinput.items():
        for line in v:
            line = line.strip().split()
            variable = line[0]
            # these variables require a boundname to be specified
            # to get the value for the whole lake
            # otherwise, specific connection is monitored
            # (if ID2 is entered, otherwise the output is 0.)
            # verify that boundname is not an integer (mf6 requirement)
            if variable in ['lak', 'wetted-area', 'conductance']:
                assert not line[-1].isdigit()
                boundnames.add(line[-1])
    # check that the boundnames exist
    packagedata = read_mf6_block(lak.filename, 'packagedata')
    for line in packagedata['packagedata']:
        line = line.strip().split()
        assert line[-1] in boundnames


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


def test_sfr_setup(get_pleasant_mf6_with_sfr):
    m = get_pleasant_mf6_with_sfr
    m.write_input()
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

    i, j = m.sfrdata.reach_data['i'], m.sfrdata.reach_data['j']
    sfr_x = m.modelgrid.xcellcenters[i, j]
    sfr_y = m.modelgrid.ycellcenters[i, j]
    dem_values = get_values_at_points(
        m.cfg['sfr']['source_data']['dem']['filename'], sfr_x, sfr_y,
        points_crs=m.modelgrid.crs
        )
    # verify that DEM was sampled
    # cheesy test for DEM-based values
    # with the DEM sampling turned off,
    # the range of differences should be much more extreme
    assert np.std(m.sfrdata.reach_data['strtop'] - dem_values) < 1.
    assert np.allclose(m.sfrdata.reach_data['strtop'], dem_values, atol=3.5)


def test_write_sfr(get_pleasant_mf6_with_sfr):
    m = get_pleasant_mf6_with_sfr
    m.write_input()
    sfr_package_file = m.sfrdata.modflow_sfr2.fn_path
    options = read_mf6_block(sfr_package_file, 'options')
    assert 'save_flows' in options
    assert options['budget'] == ['fileout', 'pleasant_mf6.sfr.out.bin']
    assert options['stage'] == ['fileout', 'pleasant_mf6.sfr.stage.bin']
    assert options['obs6'] == ['filein', 'pleasant_mf6.sfr.obs']
    assert options['unit_conversion'] == ['86400.0']
    assert options['auxiliary'] == ['line_id']


def test_sfr_obs(get_pleasant_mf6_with_sfr):
    m = get_pleasant_mf6_with_sfr
    m.write_input()
    # verify that observation data were added and written
    sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
    obs = pd.read_csv(m.cfg['sfr']['source_data']['observations']['filename'])
    assert len(m.sfrdata.observations) == len(obs)
    expected = obs[m.cfg['sfr']['source_data']['observations']['obsname_column']].astype(str).tolist()
    assert m.sfrdata.observations['obsname'].tolist() == expected
    sfr_obs_filename = os.path.normpath(os.path.join(m.model_ws, m.sfrdata.observations_file))
    assert os.path.exists(sfr_obs_filename)
    obs_input = read_mf6_block(sfr_obs_filename, 'continuous')
    assert obs_input[sfr_obs_filename + '.output.csv'] == \
           ['# obsname obstype rno',
            '1000000 downstream-flow 22',
            '2000000 downstream-flow 31']


def test_perimeter_boundary_setup(get_pleasant_mf6_with_dis):

    m = get_pleasant_mf6_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.cfg['chd']['mfsetup_options']['external_files'] = False
    chd = m.setup_chd(**m.cfg['chd'], **m.cfg['chd']['mfsetup_options'])
    chd.write()
    assert os.path.exists(os.path.join(m.model_ws, chd.filename))
    assert len(chd.stress_period_data.array) == len(set(m.cfg['parent']['copy_stress_periods']))
    assert len(m.get_boundary_cells()[0]) == (m.nrow*2 + m.ncol*2 - 4) * m.nlay  # total number of boundary cells
    # number of boundary heads;
    # can be less than number of active boundary cells if the (parent) water table is not always in (inset) layer 1
    #assert len(chd.stress_period_data.array[0]) <= np.sum(m.idomain[m.get_boundary_cells()] == 1)

    # check for inactive cells
    spd0 = chd.stress_period_data.array[0]
    k, i, j = zip(*spd0['cellid'])
    inactive_cells = m.idomain[k, i, j] < 1
    assert not np.any(inactive_cells)

    # check that heads are above layer botms
    assert np.all(spd0['head'] > m.dis.botm.array[k, i, j])


def test_model_setup(pleasant_mf6_setup_from_yaml, tmpdir):
    m = pleasant_mf6_setup_from_yaml
    assert isinstance(m, MF6model)
    assert 'tdis' in m.simulation.package_key_dict
    assert 'ims' in m.simulation.package_key_dict
    assert set(m.get_package_list()) == {'DIS', 'IC', 'NPF', 'STO', 'RCHA_0', 'OC', 'SFR_0', 'LAK_0',
                                         'WEL_0',
                                         'OBS_1',  # lak obs todo: specify names of mf6 packages with multiple instances
                                         'CHD_0',
                                         'OBS_0',  # chd obs
                                         'OBS_2',  # head obs
                                         'OBS_3'  # head obs
                                         }
    external_path = os.path.join(m.model_ws, 'external')
    external_files = glob.glob(external_path + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = '\n'.join(has_nans)
    if len(has_nans) > 0:
        assert False, has_nans

    make_xsections = False
    if make_xsections:
        outpdf = Path('postproc/lake_xsections.pdf')
        make_lake_xsections(m, i_range=(30, 51), j_range=(30, 41),
                            bathymetry_raster=m.cfg['lak']['source_data']['bathymetry_raster']['filename'],
                            datum=298.73, outpdf=outpdf)


@pytest.mark.parametrize('remake_top', (True,
                                        pytest.param(False,
                                                     marks=pytest.mark.xfail(reason="model can't be built "))
                                        ))
def test_setup_from_yaml_issue(project_root_path, remake_top):
    cfg = MF6model.load_cfg(Path(project_root_path) / 'examples/pleasant_lgr_parent.yml')
    lgr_test_path = Path(project_root_path) / 'examples/pleasant_lgr'
    shutil.rmtree(lgr_test_path, ignore_errors=True)
    keep_keys = {'simulation', 'model',  'parent', 'setup_grid', 'dis', 'tdis',
                 'intermediate_data', 'postprocessing', 'filename'}
    new_cfg = {k: v for k, v in cfg.items() if k in keep_keys}
    new_cfg['model']['packages'] = ['dis']
    new_cfg['dis']['remake_top'] = remake_top
    del new_cfg['setup_grid']['lgr']
    MF6model.setup_from_cfg(new_cfg)


def test_check_external_files():
    external_files = glob.glob('mfsetup/tests/tmp/pleasant_mf6/external' + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = '\n'.join(has_nans)
    if len(has_nans) > 0:
        assert False, has_nans


#@pytest.mark.skip("still working on comparing mfnwt and mf6 versions of pleasant test case")
def test_mf6_results(tmpdir, project_root_path, pleasant_mf6_model_run, pleasant_nwt_model_run):
    #pleasant_mf6_model_run = None
    #pleasant_nwt_model_run = None
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    make_plot = False
    if pleasant_mf6_model_run is None:
        sim = mf6.MFSimulation.load('mfsim', sim_ws='{}/pleasant_mf6'.format(tmpdir))
        pleasant_mf6_model_run = sim.get_model('pleasant_mf6')
    if pleasant_nwt_model_run is None:
        pleasant_nwt_model_run = fm.Modflow.load('pleasant.nam',
                                                 model_ws='{}/pleasant_nwt'.format(tmpdir))

    # mass bal. results
    mfl = flopy.utils.MfListBudget('{}/pleasant_nwt/pleasant.list'.format(tmpdir))
    df_flux, df_vol = mfl.get_dataframes(start_datetime='2012-01-01')
    mfl6 = flopy.utils.Mf6ListBudget('{}/pleasant_mf6/pleasant_mf6.list'.format(tmpdir))
    df_flux6, df_vol6 = mfl6.get_dataframes(start_datetime='2012-01-01')
    mf6_terms = {'STORAGE_IN': ['STO-SS_IN', 'STO-SY_IN'],
                 'CONSTANT_HEAD_IN': 'CHD_IN',
                 'WELLS_IN': 'WEL_IN',
                 'RECHARGE_IN': 'RCH_IN',
                 'STREAM_LEAKAGE_IN': 'SFR_IN',
                 'LAKE__SEEPAGE_IN': 'LAK_IN',
                 'TOTAL_IN': 'TOTAL_IN'
                 }

    # compare the terms
    if make_plot:
        os.chdir(project_root_path)
        pdf_outfile = '../modflow-setup-dirty/pleasant_mfnwt_mf6_compare.pdf'
        with PdfPages(pdf_outfile) as pdf:
            for k, v in mf6_terms.items():
                term = k
                out_term = term.replace('IN', 'OUT')
                mf6_term = v
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax = df_flux[term].plot(c='C0')
                ax = (-df_flux[out_term]).plot(ax=ax, c='C0')
                if isinstance(mf6_term, list):
                    mf6_series = df_flux6[mf6_term].sum(axis=1)
                    mf6_out_term = [s.replace('IN', 'OUT') for s in mf6_term]
                    mf6_out_series = df_flux6[mf6_out_term].sum(axis=1)
                else:
                    mf6_out_term = mf6_term.replace('IN', 'OUT')
                    mf6_series = df_flux6[mf6_term]
                    mf6_out_series = df_flux6[mf6_out_term]
                mf6_series.plot(ax=ax, c='C1')
                (-mf6_out_series).plot(ax=ax, c='C1')
                h, l = ax.get_legend_handles_labels()
                ax.legend(h[::2], ['mfnwt', 'mf6'])
                ax.set_title(term.split('_')[0])
                pdf.savefig()
                plt.close()

    # head results
    HeadFile = flopy.utils.binaryfile.HeadFile
    mf6_hds_obj = HeadFile('{}/pleasant_mf6/pleasant_mf6.hds'.format(tmpdir))
    mfnwt_hds_obj = HeadFile('{}/pleasant_nwt/pleasant.hds'.format(tmpdir))
    assert np.allclose(mf6_hds_obj.get_times(), mfnwt_hds_obj.get_times(), rtol=1e-4)
    all_kstpkper = mf6_hds_obj.get_kstpkper()

    # compare heads along the boundary
    k, i, j = pleasant_nwt_model_run.get_boundary_cells(exclude_inactive=True)
    mf6_bhead_avg = []
    mfnwt_bhead_avg = []
    for kstp, kper in all_kstpkper:
        mf6_hds = mf6_hds_obj.get_data(kstpkper=(kstp, kper))
        mfnwt_hds = mfnwt_hds_obj.get_data(kstpkper=(kstp, kper))
        mf6_bhead_avg.append(mf6_hds[k, i, j].mean())
        mfnwt_bhead_avg.append(mfnwt_hds[k, i, j].mean())

        #last = [all_kstpkper-1]
        #mf6_hds = mf6_hds_obj.get_data(kstpkper=last)
        #mfnwt_hds = mfnwt_hds_obj.get_data(kstpkper=last)
        #from flopy.utils.postprocessing import get_water_table
        #mf6_wt = get_water_table(mf6_hds, nodata=1e30)
        #mfnwt_wt = get_water_table(mfnwt_hds, nodata=-9999)
        #loc = pleasant_mf6_model_run.dis.idomain.array == 1
        #rms = np.sqrt(np.mean((mf6_wt - mfnwt_wt) ** 2))

    if make_plot:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plt.plot(mf6_bhead_avg, label='mf6')
        plt.plot(mfnwt_bhead_avg, label='mfnwt')

        ax = df_flux[term].plot(c='C0')
        ax = (-df_flux[out_term]).plot(ax=ax, c='C0')
        if isinstance(mf6_term, list):
            mf6_series = df_flux6[mf6_term].sum(axis=1)
            mf6_out_term = [s.replace('IN', 'OUT') for s in mf6_term]
            mf6_out_series = df_flux6[mf6_out_term].sum(axis=1)
        else:
            mf6_out_term = mf6_term.replace('IN', 'OUT')
            mf6_series = df_flux6[mf6_term]
            mf6_out_series = df_flux6[mf6_out_term]
        mf6_series.plot(ax=ax, c='C1')
        (-mf6_out_series).plot(ax=ax, c='C1')
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::2], ['mfnwt', 'mf6'])
        ax.set_title(term.split('_')[0])
        pdf.savefig()
        plt.close()
        j=2

        # lake budget results
        #mf6_cb_obj = HeadFile('pleasant_mf6.cbc')
        #mfnwt_cb_obj = HeadFile('../pleasant_nwt/pleasant.cbc')

    # lake stage results
    # this also tests that the gage package is writing to the correct unit
    df_mf6 = pd.read_csv('{}/pleasant_mf6/lake1.obs.csv'.format(tmpdir))
    # this next line isn't working on Travis for some reason
    df_mfnwt = read_lak_ggo('{}/pleasant_nwt/lak1_600059060.ggo'.format(tmpdir),
                            model=pleasant_nwt_model_run)
    #if make_plot:
    #    plt.plot(df_mf6.time, df_mf6.STAGE, label='mf6')
    #    plt.plot(df_mfnwt.time, df_mfnwt.stageh, label='mfnwt')
    #    plt.legend()
    lake_stage_rms = np.sqrt(np.mean((df_mfnwt['stageh'].values - df_mf6['STAGE'].values) ** 2))
    # TODO: not sure why there is a 0.15 offset in stage between the two models
    # doesn't seem to be due to the ghb package
    # or the high-k lakes, unless they aren't implemented in the modflow-nwt version
    np.allclose(df_mfnwt['stageh'].values + 0.15, df_mf6['STAGE'].values, atol=0.01)
    #pdf.close()
