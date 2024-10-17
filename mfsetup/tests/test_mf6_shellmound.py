import sys
from pathlib import Path

sys.path.append('..')
import glob
import os
import platform
import shutil
from copy import deepcopy

import flopy
import numpy as np
import pandas as pd
import pytest
import rasterio

import xarray as xr

mf6 = flopy.mf6
from mfsetup import testing
from mfsetup.checks import check_external_files_for_nans
from mfsetup.discretization import (
    cellids_to_kij,
    find_remove_isolated_cells,
    get_layer_thicknesses,
)
from mfsetup.fileio import exe_exists, load, load_array, load_cfg, read_mf6_block
from mfsetup.grid import rasterize
from mfsetup.mf6model import MF6model
from mfsetup.units import convert_length_units
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="module", autouse=True)
def reset_dirs(shellmound_cfg):
    cfg = deepcopy(shellmound_cfg)
    folders = [cfg.get('intermediate_data', {}).get('output_folder', 'original-arrays/'),
               cfg['model'].get('external_path'),
               cfg['gisdir']
               ]
    for folder in folders:
        #if not os.path.isdir(folder):
        #    os.makedirs(folder)
        #else:
        #    shutil.rmtree(folder)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


@pytest.fixture(scope="function")
def model_with_sfr(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    sfr = m.setup_sfr()
    return m


@pytest.fixture(scope="module")
def model_setup(shellmound_cfg_path):
    for folder in ['shellmound', 'tmp']:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    m = MF6model.setup_from_yaml(shellmound_cfg_path)
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


def test_init(shellmound_cfg):
    cfg = deepcopy(shellmound_cfg)

    sim = mf6.MFSimulation()
    assert isinstance(sim, mf6.MFSimulation)

    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation, warn=False)
    sim = mf6.MFSimulation(**kwargs)
    assert isinstance(sim, mf6.MFSimulation)

    cfg['model']['packages'] = []
    cfg['model']['simulation'] = sim
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    # test initialization with no packages
    m = MF6model(cfg=cfg, **kwargs)
    assert isinstance(m, MF6model)

    # test initialization with no arguments
    m = MF6model(simulation=sim)
    assert isinstance(m, MF6model)


def test_parse_modflowgwf_kwargs(shellmound_cfg):
    cfg = deepcopy(shellmound_cfg)
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    sim_path = os.path.normpath(m.simulation.simulation_data.mfpath._sim_path).lower()
    assert sim_path == cfg['simulation']['sim_ws'].lower()
    m.write()

    # verify that options were written correctly to namefile
    # newton solver, but without underrelaxation
    nampath = os.path.join(m.model_ws, m.model_nam_file)
    options = read_mf6_block(nampath, 'options')
    assert os.path.normpath(options['list'][0]).lower() == \
           os.path.normpath(cfg['model']['list']).lower()
    for option in ['print_input', 'print_flows', 'save_flows']:
        if cfg['model'][option]:
            assert option in options
    assert len(options['newton']) == 0

    # newton solver, with underrelaxation
    cfg['model']['options']['newton_under_relaxation'] = True
    cfg = MF6model._parse_model_kwargs(cfg)
    assert cfg['model']['options']['newtonoptions'] == ['under_relaxation']
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.write()
    options = read_mf6_block(nampath, 'options')
    assert options['newton'] == ['under_relaxation']


def test_repr(shellmound_model, shellmound_model_with_grid):
    txt = shellmound_model.__repr__()
    assert isinstance(txt, str)
    # cheesy test that flopy repr isn't returned
    assert 'CRS:' in txt and 'Bounds:' in txt
    txt = shellmound_model_with_grid.__repr__()
    assert isinstance(txt, str)


def test_load_cfg(shellmound_cfg, shellmound_cfg_path):
    relative_model_ws = '../tmp/shellmound'
    ws = os.path.normpath(os.path.join(os.path.abspath(os.path.split(shellmound_cfg_path)[0]),
                                                                       relative_model_ws))
    cfg = shellmound_cfg
    assert cfg['simulation']['sim_ws'] == ws


def test_simulation(shellmound_simulation):
    sim = shellmound_simulation
    # verify that "continue" option was successfully translated
    # to flopy sim constructor arg "continue_"
    assert sim.name_file.continue_.array


def test_model(shellmound_model):
    model = shellmound_model
    assert model.exe_name == 'mf6'
    assert model.simulation.exe_name == 'mf6'

    # verify that cwd has been set to model_ws
    assert os.path.normpath(os.path.abspath(model.model_ws)) == \
           os.path.normpath(os.getcwd())
    # and that external files paths are correct relative to model_ws
    assert os.path.normpath(os.path.abspath(model.tmpdir)) == \
           os.path.normpath(os.path.join(os.path.abspath(model.model_ws), model.tmpdir))
    assert os.path.normpath(os.path.abspath(model.external_path)) == \
           os.path.normpath(os.path.join(os.path.abspath(model.model_ws), model.external_path))


def test_namefile(shellmound_model_with_dis):
    model = shellmound_model_with_dis
    model.write_input()

    # check that listing file was written correctly
    expected_listfile_name = model.cfg['model']['list_filename_fmt'].format(model.name)
    with open(model.namefile) as src:
        for line in src:
            if 'LIST' in line:
                assert line.strip().split()[-1] == expected_listfile_name


def test_ims_setup(shellmound_model):
    model = shellmound_model
    ims = model.setup_ims()

    assert ims.csv_outer_output_filerecord is not None

    assert ims.csv_outer_output_filerecord.array[0][0] == \
        'solver_outer_out.csv'


def test_snap_to_NHG(shellmound_cfg, shellmound_simulation):
    cfg = deepcopy(shellmound_cfg)
    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = shellmound_simulation
    cfg['setup_grid']['snap_to_NHG'] = True

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()

    # national grid parameters
    xul, yul = -2553045.0, 3907285.0  # upper left corner
    ngrows = 4000
    ngcols = 4980
    natCellsize = 1000

    # locations of left and top cell edges
    ngx = np.arange(ngcols) * natCellsize + xul
    ngy = np.arange(ngrows) * -natCellsize + yul

    x0, x1, y0, y1 = m.modelgrid.extent
    assert np.min(np.abs(ngx - x0)) == 0
    assert np.min(np.abs(ngy - y0)) == 0
    assert np.min(np.abs(ngx - x1)) == 0
    assert np.min(np.abs(ngy - y1)) == 0


def test_model_with_grid(shellmound_model_with_grid):
    m = shellmound_model_with_grid
    assert m.modelgrid is not None
    assert isinstance(m.cfg['grid'], dict)


@pytest.mark.parametrize('relative_external_paths', [True,
                                                     False])
def test_package_external_file_path_setup(shellmound_model_with_grid,
                                          relative_external_paths):
    m = shellmound_model_with_grid
    top_filename = m.cfg['dis']['top_filename_fmt']
    botm_file_fmt = m.cfg['dis']['botm_filename_fmt']
    m.relative_external_paths = relative_external_paths
    dis = m.setup_dis()
    dis.write()
    assert os.path.exists(dis.filename)
    with open(dis.filename) as src:
        for line in src:
            if 'open/close' in line.lower():
                path = line.strip().split()[1].strip().strip('\'')
                if not relative_external_paths:
                    assert os.path.isabs(path)
                else:
                    assert not os.path.isabs(path)

    assert m.cfg['intermediate_data']['top'] == \
           [os.path.normpath(os.path.join(m.tmpdir, os.path.split(top_filename)[-1]))]
    assert m.cfg['intermediate_data']['botm'] == \
           [os.path.normpath(os.path.join(m.tmpdir, botm_file_fmt).format(i))
                                  for i in range(m.nlay)]
    if not relative_external_paths:
        assert m.cfg['dis']['griddata']['top'] == \
               [{'filename': os.path.normpath(os.path.join(m.model_ws,
                            m.external_path,
                            os.path.split(top_filename)[-1]))}]
        assert m.cfg['dis']['griddata']['botm'] == \
               [{'filename': os.path.normpath(os.path.join(m.model_ws,
                             m.external_path,
                             botm_file_fmt.format(i)))} for i in range(m.nlay)]


def test_set_lakarr(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    if 'lake' in m.package_list:
        # lak not implemented yet for mf6
        #assert 'lak' in m.package_list
        lakes_shapefile = m.cfg['lak'].get('source_data', {}).get('lakes_shapefile')
        assert lakes_shapefile is not None
        assert m._lakarr2d.sum() > 0
        assert m._isbc2d.sum() > 0  # requires
        assert m.isbc.sum() > 0  # requires DIS package
        assert m.lakarr.sum() > 0  # requires isbc to be set
        if m.version == 'mf6':
            externalfiles = m.cfg['external_files']['lakarr']
        else:
            externalfiles = m.cfg['intermediate_data']['lakarr']
        assert isinstance(externalfiles, dict)
        assert isinstance(externalfiles[0], list)
        for f in externalfiles[0]:
            assert os.path.exists(f)
        m.cfg['model']['packages'].remove('lak')
        m._lakarr_2d = None
        m._isbc_2d = None
    else:
        assert m._lakarr2d.sum() == 0
        assert m._lakarr2d.sum() == 0
        assert m._isbc2d.sum() == 0
        assert m.isbc.sum() == 0  # requires DIS package
        assert m.lakarr.sum() == 0  # requires isbc to be set


def test_dis_setup(shellmound_model_with_grid):

    m = shellmound_model_with_grid #deepcopy(model_with_grid)
    # test intermediate array creation
    m.cfg['dis']['remake_top'] = True
    m.cfg['dis']['source_data']['top']['resample_method'] = 'nearest'
    m.cfg['dis']['source_data']['botm']['resample_method'] = 'nearest'
    dis = m.setup_dis()
    botm = m.dis.botm.array.copy()
    assert isinstance(dis, mf6.ModflowGwfdis)
    assert 'DIS' in m.get_package_list()
    # verify that units got conveted correctly
    assert m.dis.top.array.mean() < 100
    assert m.dis.length_units.array == 'meters'

    # verify that modelgrid was reset after building DIS
    mg = m.modelgrid
    assert (mg.nlay, mg.nrow, mg.ncol) == m.dis.botm.array.shape
    assert np.array_equal(mg.top, m.dis.top.array)
    assert np.array_equal(mg.botm, m.dis.botm.array)

    arrayfiles = m.cfg['intermediate_data']['top'] + \
                 m.cfg['intermediate_data']['botm'] + \
                 m.cfg['intermediate_data']['idomain']
    for f in arrayfiles:
        assert os.path.exists(f)
        fname = os.path.splitext(os.path.split(f)[1])[0]
        k = ''.join([s for s in fname if s.isdigit()])
        var = fname.split('_')[0]
        data = np.loadtxt(f)
        model_array = getattr(m.dis, var).array
        if len(k) > 0:
            k = int(k)
            model_array = model_array[k]
        assert np.array_equal(model_array, data)

   # test that written idomain array reflects supplied shapefile of active area
    active_area = rasterize(m.cfg['dis']['source_data']['idomain']['filename'],
                            m.modelgrid)
    isactive = active_area >= 1
    written_idomain = load_array(m.cfg['dis']['griddata']['idomain'])
    assert np.all(written_idomain[:, ~isactive] <= 0)

    # test idomain from just layer elevations
    del m.cfg['dis']['griddata']['idomain']
    dis = m.setup_dis()
    top = dis.top.array.copy()
    top[top == m._nodata_value] = np.nan
    botm = dis.botm.array.copy()
    botm[botm == m._nodata_value] = np.nan
    thickness = get_layer_thicknesses(top, botm)
    invalid_botms = np.ones_like(botm)
    invalid_botms[np.isnan(botm)] = 0
    invalid_botms[thickness < 1.0001] = 0

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(axes.flat):
        if i < 12:
            ax.imshow(m.idomain[i])
    # these two arrays are not equal
    # because isolated cells haven't been removed from the second one
    # this verifies that _set_idomain is removing them
    # note: 8/4/2024: after 012b6c6, this statement is True,
    # presumably because the isolated cells are getting culled as part of setup_dis()
    # TODO: might be better to test find_remove_isolated_cells directly
    # assert not np.array_equal(m.idomain[:, isactive].sum(axis=1),
    #                       invalid_botms[:, isactive].sum(axis=1))
    invalid_botms = find_remove_isolated_cells(invalid_botms, minimum_cluster_size=20)
    active_cells = m.idomain[:, isactive].copy()
    active_cells[active_cells < 0] = 0  # need to do this because some idomain cells are -1
    assert np.array_equal(active_cells.sum(axis=1),
                          invalid_botms[:, isactive].sum(axis=1))

    # test recreating package from external arrays
    m.remove_package('dis')
    assert m.cfg['dis']['griddata']['top'] is not None
    assert m.cfg['dis']['griddata']['botm'] is not None
    dis = m.setup_dis()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array >= 1],
                          botm[m.dis.idomain.array >= 1])

    # test recreating just the top from the external array
    m.remove_package('dis')
    m.cfg['dis']['remake_top'] = False
    m.cfg['dis']['griddata']['botm'] = None
    dis = m.setup_dis()
    dis.write()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array >= 1],
                          botm[m.dis.idomain.array >= 1])
    arrayfiles = m.cfg['dis']['griddata']['top']
    for f in arrayfiles:
        assert os.path.exists(f['filename'])
    assert os.path.exists(os.path.join(m.model_ws, dis.filename))

    # dis package idomain should be consistent with model property
    updated_idomain = m.idomain
    assert np.array_equal(m.dis.idomain.array, updated_idomain)

    # check that units were converted (or not)
    assert np.allclose(dis.top.array[dis.idomain.array[0] >= 1].mean(), 40, atol=10)
    mcaq = m.cfg['dis']['source_data']['botm']['filenames'][3]
    assert 'mcaq' in mcaq
    with rasterio.open(mcaq) as src:
        mcaq_data = src.read(1)
        mcaq_data[mcaq_data == src.meta['nodata']] = np.nan
    assert np.allclose(m.dis.botm.array[3][dis.idomain.array[3] == 1].mean() / .3048, np.nanmean(mcaq_data), atol=5)

    # check that original arrays were created in expected folder
    assert Path(m.cfg['intermediate_data']['output_folder']).is_dir()


def test_idomain(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    assert issubclass(m.idomain.dtype.type, np.integer)
    assert m.idomain.sum() == m.dis.idomain.array.sum()


def test_ic_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    ic = m.setup_ic()
    ic.write()
    assert os.path.exists(os.path.join(m.model_ws, ic.filename))
    assert isinstance(ic, mf6.ModflowGwfic)
    assert np.allclose(ic.strt.array.mean(axis=0), m.dis.top.array)


def test_tdis_setup(shellmound_model):

    m = shellmound_model #deepcopy(model)
    tdis = m.setup_tdis()
    tdis.write()
    assert os.path.exists(os.path.join(m.model_ws, tdis.filename))
    assert isinstance(tdis, mf6.ModflowTdis)
    period_df = pd.DataFrame(tdis.perioddata.array)
    period_df['perlen'] = period_df['perlen'].astype(float)
    period_df['nstp'] = period_df['nstp'].astype(m.perioddata['nstp'].dtype)
    pd.testing.assert_frame_equal(period_df[['perlen', 'nstp', 'tsmult']],
                                  m.perioddata[['perlen', 'nstp', 'tsmult']])


def test_sto_setup(shellmound_model_with_dis):

    m = shellmound_model_with_dis  #deepcopy(model_with_grid)
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


def test_npf_setup(shellmound_model_with_dis):
    m = deepcopy(shellmound_model_with_dis)

    # check time unit conversion
    m.cfg['npf']['source_data']['k']['time_units'] = 'seconds'

    npf = m.setup_npf()
    npf.write()
    assert os.path.exists(os.path.join(m.model_ws, npf.filename))

    # check that units were converted
    k3tif = m.cfg['npf']['source_data']['k']['filenames'][3]
    assert k3tif.endswith('k3.tif')
    with rasterio.open(k3tif) as src:
        data = src.read(1)
        data[data == src.meta['nodata']] = np.nan
    assert np.allclose(npf.k.array[3].mean() / (0.3048 * 86400), np.nanmean(data), atol=5)

    # TODO: add tests that Ks got distributed properly considering input and pinched layers


@pytest.mark.parametrize('config', [{'source_data':
                                         {'filenames': ['../../data/shellmound/tables/preprocessed_head_obs_info.csv'],
                                          'column_mappings':
                                              {'obsname': ['obsprefix']}
                                          },
                                     },
                                    {'source_data':
                                         {'filename': '../../data/shellmound/tables/head_obs_well_info2.csv',
                                          'column_mappings':
                                              {'obsname': ['obsprefix'],
                                               'x': 'x_5070',
                                               'y': 'y_5070'}
                                          },
                                     },
                                    ])
def test_obs_setup(shellmound_model_with_dis, config, project_root_path):
    m = shellmound_model_with_dis  # deepcopy(model)
    default_mf6_cfg = load(Path(project_root_path) / 'mfsetup/mf6_defaults.yml')
    kwargs = default_mf6_cfg['obs'].copy()
    kwargs.update(config)
    obs = m.setup_obs(**kwargs)
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


@pytest.mark.parametrize('input', [
    # flopy-style input
    {'saverecord': {0: {'head': 'last', 'budget': 'last'}}},
    # MODFLOW 6-style input
    {'period_options': {0: ['save head last', 'save budget last']}},
    # blank period to skip subsequent periods
    {'period_options': {0: ['save head last', 'save budget last'],
                        1: []}}
                                        ])
def test_oc_setup(shellmound_model_with_dis, input):
    cfg = {'head_fileout_fmt': '{}.hds',
           'budget_fileout_fmt': '{}.cbc'}
    cfg.update(input)
    m = shellmound_model_with_dis  # deepcopy(model)
    m.cfg['oc'] = cfg
    oc = m.setup_oc()
    oc.write()
    ocfile = os.path.join(m.model_ws, oc.filename)
    assert os.path.exists(ocfile)
    assert isinstance(oc, mf6.ModflowGwfoc)
    options = read_mf6_block(ocfile, 'options')
    options = {k: ' '.join(v).lower() for k, v in options.items()}
    perioddata = read_mf6_block(ocfile, 'period')
    # convert back to zero-based
    perioddata = {k-1:v for k, v in perioddata.items()}
    assert 'fileout' in options['budget'] and '.cbc' in options['budget']
    assert 'fileout' in options['head'] and '.hds' in options['head']
    if 'saverecord' in input:
        assert 'save head last' in perioddata[0]
        assert 'save budget last' in perioddata[0]
    else:
        assert perioddata == input['period_options']


def test_rch_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis  # deepcopy(model)
    rch = m.setup_rch()
    rch.write()
    # check for irch file
    irchfile = os.path.join(m.model_ws, m.cfg['rch']['irch'][0]['filename'])
    assert os.path.exists(irchfile)
    irch = load_array(os.path.join(m.model_ws, m.cfg['rch']['irch'][0]['filename']))
    assert irch.shape[0] == m.nrow
    assert irch.shape[1] == m.ncol

    assert os.path.exists(os.path.join(m.model_ws, rch.filename))
    assert isinstance(rch, mf6.ModflowGwfrcha)
    assert rch.recharge is not None
    # get the same data from the source file
    ds = xr.open_dataset(m.cfg['rch']['source_data']['recharge']['filename'])
    x = m.modelgrid.xcellcenters[0, :]
    y = m.modelgrid.ycellcenters[:, 0] # not sure if y should be flipped or not

    unit_conversion = convert_length_units('inches', 'meters')

    def get_period_values(start, end):
        #period_data = ds['net_infiltration'].loc[start:end].mean(axis=0)
        #dsi = period_data.interp(x=x, y=y, method='linear',
        #                         kwargs={'fill_value': np.nan,
        #                                 'bounds_error': True})
        dsi = ds.interp(x=x, y=y, method='linear',
                                 kwargs={'fill_value': np.nan,
                                         'bounds_error': True})
        period_data = dsi['net_infiltration'].loc[start:end].mean(axis=0)
        data = period_data.values * unit_conversion
        #data = dsi.values * unit_conversion
        return np.reshape(data, (m.nrow, m.ncol))

    # test steady-state avg. across all data
    values = get_period_values('2012-01-01', '2017-12-31')

    #assert np.allclose(values, m.rch.recharge.array[0, 0])
    # test period 1 avg. for those times
    start = m.cfg['rch']['source_data']['recharge']['period_stats'][1][1]
    end = m.cfg['rch']['source_data']['recharge']['period_stats'][1][2]
    values1 = get_period_values(start, end)
    assert testing.rpd(values1.mean(), m.rch.recharge.array[1, 0].mean()) < 0.01

    # check that nodata are written as 0.
    tmp = rch.recharge.array[:2].copy()
    tmp[0, 0, 0, 0] = np.nan
    tmp = {i: arr[0] for i, arr in enumerate(tmp)}
    m._setup_array('rch', 'recharge', datatype='transient2d',
                   data=tmp, write_fmt='%.6e',
                   write_nodata=0.)
    rech0 = load_array(m.cfg['rch']['recharge'][0])
    assert rech0[0, 0] == 0.
    assert rech0.min() >= 0.
    assert np.allclose(m.rch.recharge.array[0, 0].ravel(), rech0.ravel())


def test_direct_rch_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis
    del m.cfg['rch']['source_data']
    m.cfg['rch']['recharge'] = 1
    m.setup_rch()
    assert np.all(m.rch.recharge.array == 1)
    m.remove_package('rch')
    m.cfg['rch']['recharge'] = {0: 0, 1:1}
    m.setup_rch()
    assert m.rch.recharge.array[0].sum() == 0
    assert np.all(m.rch.recharge.array[1:] == 1)


@pytest.mark.parametrize('pckg_abbrv,head_variables,boundnames',
                         (('chd', ['head'], False),
                          ('drn', ['elev'], False),
                          ('ghb', ['bhead'], False),
                          ('riv', ['stage'], {'lake henry'}),
                          )
                         )
def test_basic_stress_package_setup(shellmound_model_with_dis, pckg_abbrv,
                                    head_variables, boundnames):
    flopy_package = getattr(mf6, f'ModflowGwf{pckg_abbrv}')
    m = shellmound_model_with_dis  # deepcopy(model)
    setup_method = getattr(m, f'setup_{pckg_abbrv}')
    pckg = setup_method(**m.cfg[pckg_abbrv], **m.cfg[pckg_abbrv]['mfsetup_options'])
    pckg.write()
    assert os.path.exists(os.path.join(m.model_ws, pckg.filename))
    assert isinstance(pckg, flopy_package)
    assert hasattr(m, pckg_abbrv)
    assert pckg.stress_period_data is not None
    df = pd.DataFrame(pckg.stress_period_data.data[0])
    assert len(df) > 0
    k, i, j = zip(*df['cellid'])
    cell_bottoms = m.dis.botm.array[k, i, j]
    for var in head_variables:
        assert np.all(df[var] > cell_bottoms)
    if pckg_abbrv == 'riv':
        assert np.all(df['stage'] > df['rbot'])
    if pckg_abbrv not in {'chd', 'ghb'}:
        assert np.all(df['cond'] == m.cfg[pckg_abbrv]['source_data']['cond'])
    # checks for basic transient csv input case
    if pckg_abbrv == 'chd':
        include_ids = m.cfg['chd']['source_data']['shapefile']['include_ids']
        in_df = pd.read_csv(m.cfg['chd']['source_data']['csvfile']['filename'])
        in_df = in_df.loc[in_df['comid'].isin(include_ids)]
        spd_heads = np.array(
            [ra[0][1] for per, ra in m.chd.stress_period_data.data.items()])
        # every stress period should have input
        assert len(m.chd.stress_period_data.data.keys()) == m.nper
        # heads in CHD package should match the CSV input,
        # after conversion to meters
        assert np.allclose(in_df['head'].values, spd_heads[1:]/.3048)
    if pckg_abbrv == 'drn':
        exclude_ids = list(map(str, m.cfg[pckg_abbrv]['source_data']['shapefile']['exclude_ids']))
        comids = [s.replace('feature-','') for s in np.unique(m.drn.stress_period_data.data[0]['boundname'])]
        assert not set(exclude_ids).intersection(comids)
        assert comids == ['18046236', '18047154']
    # more advanced transient input case with csv
    # and conductance values specified via a raster
    if pckg_abbrv == 'ghb':
        in_df = pd.read_csv(m.cfg['ghb']['source_data']['csvfile']['filename'])
        spd_heads = np.array(
            [ra[0][1] for per, ra in m.ghb.stress_period_data.data.items()])
        # every stress period should have input
        assert len(m.ghb.stress_period_data.data.keys()) == m.nper
        # heads in CHD package should match the CSV input,
        # after conversion to meters
        assert np.allclose(in_df['head'].values[-1], spd_heads[1:]/.3048)

        in_raster = m.cfg['ghb']['source_data']['cond']['filename']
        with rasterio.open(in_raster) as src:
            data = src.read(1)
            raster_cond = np.unique(data[data > 0])[0]
        spd_conds = np.array(
            [ra[0][2] for per, ra in m.ghb.stress_period_data.data.items()])
        assert np.allclose(spd_conds, raster_cond)
    if boundnames:
        assert 'boundname_column' in m.cfg[pckg_abbrv]['source_data']['shapefile']
        assert not set(df['boundname']).symmetric_difference(boundnames)


def test_wel_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis  # deepcopy(model)
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

    # check the stress_period_data against source data
    sums = [ra['q'].sum() if ra is not None else 0
            for ra in wel.stress_period_data.array]
    cellids = set()
    cellids2d = set()
    for per, ra in wel.stress_period_data.data.items():
        cellids.update(set(ra['cellid']))
        cellids2d.update(set([c[1:] for c in ra['cellid']]))

    # sum the rates from the source files
    min_thickness = m.cfg['wel']['source_data']['csvfiles']['vertical_flux_distribution']['minimum_layer_thickness']
    dfs = []
    for f in m.cfg['wel']['source_data']['csvfiles']['filenames']:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)

    # cull wells to within model area
    l, b, r, t = m.modelgrid.bounds
    outside = (df.x.values > r) | (df.x.values < l) | (df.y.values < b) | (df.y.values > t)
    df['outside'] = outside
    df = df.loc[~outside]
    df['start_datetime'] = pd.to_datetime(df.start_datetime)
    df['end_datetime'] = pd.to_datetime(df.end_datetime)
    from mfsetup.grid import get_ij
    i, j = get_ij(m.modelgrid, df.x.values, df.y.values)
    df['i'] = i
    df['j'] = j
    thicknesses = get_layer_thicknesses(m.dis.top.array, m.dis.botm.array, m.idomain)
    b = thicknesses[:, i, j]
    b[np.isnan(b)] = 0
    df['k'] = np.argmax(b, axis=0)
    df['laythick'] = b[df['k'].values, range(b.shape[1])]
    df['idomain'] = m.idomain[df['k'], i, j]
    valid_ij = (df['idomain'] == 1) & (df['laythick'] > min_thickness)  # nwell array of valid i, j locations (with at least one valid layer)
    culled = df.loc[~valid_ij].copy()  # wells in invalid i, j locations
    df = df.loc[valid_ij].copy()  # remaining wells
    cellids_2d_2 = set(list(zip(df['i'], df['j'])))
    df.index = df.start_datetime
    sums2 = []
    for i, r in m.perioddata.iterrows():
        end_datetime = r.end_datetime - pd.Timedelta(1, unit='d')
        welldata_overlaps_period = (df.start_datetime < end_datetime) & \
                                   (df.end_datetime > r.start_datetime)
        q = df.loc[welldata_overlaps_period, 'flux_m3'].sum()
        sums2.append(q)
    sums = np.array(sums)
    sums2 = np.array(sums2)
    # if this doesn't match
    # may be due to wells with invalid open intervals getting removed
    assert np.allclose(sums, sums2, rtol=0.01)

#@pytest.mark.skipif((os.environ.get('GITHUB_ACTIONS') == 'true') &\
#    ('macos' in platform.platform().lower()),
#                    reason='This test fails on macos CI for an unknown reason; passes locally on macos')
def test_sfr_setup(model_with_sfr, project_root_path):
    m = model_with_sfr
    m.sfr.write()
    assert os.path.exists(os.path.join(m.model_ws, m.sfr.filename))
    assert isinstance(m.sfr, mf6.ModflowGwfsfr)
    output_path = m._shapefiles_path
    shapefiles = ['{}/{}_sfr_cells.shp'.format(output_path, m.name),
                  '{}/{}_sfr_outlets.shp'.format(output_path, m.name),
                  #'{}/{}_sfr_inlets.shp'.format(output_path, m.name),
                  '{}/{}_sfr_lines.shp'.format(output_path, m.name),
                  '{}/{}_sfr_routing.shp'.format(output_path, m.name)
    ]
    for f in shapefiles:
        assert os.path.exists(f)
    assert m.sfrdata.model == m

    # verify that only unconnected sfr reaches are in
    # places where all layers are inactive
    k, i, j = m.sfrdata.reach_data.k, m.sfrdata.reach_data.i, m.sfrdata.reach_data.j
    reach_idomain = m.idomain[k, i, j]
    inactive_reaches = m.sfrdata.reach_data.loc[reach_idomain != 1]
    ki, ii, ji = inactive_reaches.k, inactive_reaches.i, inactive_reaches.j
    # 26, 13
    # verify that reaches were consolidated to one per cell
    assert len(m.sfrdata.reach_data.node.unique()) == len(m.sfrdata.reach_data)

    # check that add_outlets works
    expected_outlets = {'17957815', '17956199'}
    for outlet_id in expected_outlets:
        # handle either int-based or str-based line_ids
        assert int(outlet_id) in m.sfrdata.reach_data.line_id.astype(int).tolist()
        assert m.sfrdata.reach_data.loc[m.sfrdata.reach_data.line_id == outlet_id,
                                        'outseg'].sum() == 0

    # check that adding runoff works
    runoff_period_data = m.sfrdata.period_data.dropna(subset=['runoff'], axis=0)
    # only compare periods 2-7 (2007-04-01 to 2010-04-01)
    runoff_period_data = runoff_period_data.loc[2:].copy()
    runoff_period_data['line_id_in_model'] = runoff_period_data['line_id_in_model'].astype('int64')
    # sum runoff by line id for each period
    runoff_period_comid_sums = runoff_period_data.groupby(['per', 'line_id_in_model']).sum(numeric_only=True)
    # then take the mean for each line id across periods
    mean_period_data_runoff_by_comid = runoff_period_comid_sums.groupby('line_id_in_model').mean()
    # read in the input values
    df = pd.read_csv('../../data/shellmound/tables/swb_runoff_by_nhdplus_comid_m3d.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.loc['2007-04-01':]
    mean_input_runoff_by_comid = df.groupby('comid').mean()

    # todo: compare mean annual runoff between input and model period_data
    # ONLY the input that is in the model
    # something like
    # runoff_period_data.groupby(runoff_period_data.start_datetime.dt.year)['runoff'].sum().mean()
    # df.groupby(df.time.dt.year)['runoff_m3d'].sum().mean()
    # can't easily do this currently because just have comid in the input csv (no x, y info)
    mean_annual_roff = runoff_period_data.groupby(runoff_period_data.start_datetime.dt.year)\
        ['runoff'].sum().mean()
    # with just supplied flowlines, mean_annual_roff is ~386,829
    # with full routing info (including culled flowlines; in flowline_routing.csv),
    # mean_annual_roff is ~839,144
    # ~776,577 without the Tallahatchie
    assert mean_annual_roff > 7.7e5

    # compare
    mean_input_runoff_by_comid['in_model'] = mean_period_data_runoff_by_comid['runoff']
    mean_input_runoff_by_comid.dropna(inplace=True)
    mean_input_runoff_by_comid['diff'] = mean_input_runoff_by_comid['in_model'] - \
                                         mean_input_runoff_by_comid['runoff_m3d']
    # compute the absolute relative diff between input and sfr package
    mean_input_runoff_by_comid['abs_pct'] = np.abs(mean_input_runoff_by_comid['diff']/ \
                                                   mean_input_runoff_by_comid['runoff_m3d'])
    # for lines where the model has less runoff, the difference should mostly be small
    # (due to mismatch in averaging 6-month model stress periods vs. monthly input data)
    mean_input_runoff_by_comid.loc[mean_input_runoff_by_comid['diff'] < 0, 'abs_pct'].mean() < 0.05
    # the input data include all catchments (for all NHDPlus lines)
    # the model only includes NHDPlus lines with > 20k arbolate sum
    # SFRmaker routes all runoff from missing upstream catchments to the first downstream catchment
    # that is in the model. So any catchment in the model that is not a headwater in NHDPlus
    # will have runoff greater than the input data.

    # check minimum slope
    assert np.allclose(np.round(m.sfr.packagedata.array['rgrd'].min(), 6), \
                       np.round(m.cfg['sfr']['sfrmaker_options']['minimum_slope'], 6))
    # check that no observations were placed in cells with SFR
    #default_mf6_cfg = load(Path(project_root_path) / 'mfsetup/mf6_defaults.yml')
    #kwargs = default_mf6_cfg['obs'].copy()
    #kwargs.update(config)
    obs = m.setup_obs(**m.cfg['obs'])
    sfr_cells = set(m.sfr.packagedata.array['cellid'])
    obs_cells = set(m.obs[1].continuous.data['shellmound.head.obs']['id'])
    assert not any(obs_cells.intersection(sfr_cells))


def test_sfr_inflows_from_csv(model_with_sfr):
    m = model_with_sfr

    # compare input values resampled to 6 months to sfr period data
    inflow_input = pd.read_csv(m.cfg['sfr']['source_data']['inflows']['filename'])
    inflow_input['start_datetime'] = pd.to_datetime(inflow_input['datetime'])
    inflow_input.index = inflow_input['start_datetime']
    #sfr_pd = m.sfrdata.period_data.dropna(axis=1)
    sfr_pd = m.sfrdata.period_data.dropna(subset=['inflow'], axis=0).reset_index()
    sfr_pd.index = sfr_pd.start_datetime

    line_id = inflow_input['line_id'].unique()[0]
    left = inflow_input.loc[inflow_input.line_id == line_id].loc['2007-04-01':, 'flow_m3d'].resample('6MS').mean()
    lookup = dict(zip(sfr_pd.specified_line_id, sfr_pd.rno))
    rno = lookup.get(str(line_id), lookup.get(line_id))
    right = sfr_pd.loc[sfr_pd.rno == rno].loc['2007-04-01':, 'inflow']
    left = left.loc[:right.index[-1]]
    pd.testing.assert_series_equal(left, right, check_names=False, check_freq=False)


@pytest.mark.xfail(reason='flopy remove_package() issue')
def test_idomain_above_sfr(model_with_sfr):
    m = model_with_sfr
    sfr = m.sfr
    # get the kij locations of sfr reaches
    k, i, j = zip(*sfr.reach_data[['k', 'i', 'j']])

    # verify that streambed tops are above layer bottoms
    assert np.all(sfr.packagedata.array['rtp'] > np.all(m.dis.botm.array[k, i, j]))

    # test that idomain above sfr cells is being set to 0
    # by setting all botms above streambed tops
    new_botm = m.dis.botm.array.copy()
    new_top = m.dis.top.array.copy()
    new_botm[:-1, i, j] = 9999
    new_botm[-1, i, j] = 9990
    new_top[i, j] = 9999
    np.savetxt(m.cfg['dis']['griddata']['top'][0]['filename'], new_top)
    m.dis.botm = new_botm
    m.dis.top = new_top
    # reset external files for model top
    # (that are used to cache an original version of the model top
    # prior to any adjustment to lake bottoms)
    from pathlib import Path
    original_top_file = Path(m.tmpdir,
                             f"{m.name}_{m.cfg['dis']['top_filename_fmt']}.original")
    original_top_file.unlink()
    # if original_top_file is not found or invalid,
    # the routine in sourcedata.setup_array for setting up the botm array
    # attempts to write original_top_file from
    # m.cfg['intermediate_data']['top']
    # successive calls to sourcedata.setup_array
    # in the context of setting up the bottom array
    # then reference this "original" top,
    # so if adjustments to lake bathymetry are made,
    # they are only made relative to the "original" top,
    # and not a revised top (which would keep pushing the bottoms downward)
    np.savetxt(m.cfg['intermediate_data']['top'][0], new_top)

    m.remove_package(sfr)
    m._reset_bc_arrays()
    assert not np.any(m._isbc2d == 4)
    sfr = m.setup_sfr()

    # test loading a 3d array from a filelist
    idomain = load_array(m.cfg['dis']['griddata']['idomain'])
    assert np.array_equal(m.idomain, idomain)
    # dis package idomain of model instance attached to sfrdata
    # forms basis for identifying unconnected cells
    assert np.array_equal(m.idomain, m.sfrdata.model.idomain)
    assert np.array_equal(m.idomain, m.sfrdata.model.dis.idomain.array)

    # verify that dis package file still references external file
    m.dis.write()
    fname = os.path.join(m.model_ws, m.dis.filename)
    assert os.path.getsize(fname) < 3e3

    # idomain should be zero everywhere there's a sfr reach
    # except for in the botm layer
    # (verifies that model botm was reset to accomdate SFR reaches)
    assert np.array_equal(m.sfr.reach_data.i, i)
    assert np.array_equal(m.sfr.reach_data.j, j)
    k, i, j = cellids_to_kij(sfr.packagedata.array['cellid'])
    assert idomain[:-1, i, j].sum() == 0
    active = np.array([True if c != 'none' else False for c in sfr.packagedata.array['cellid']])
    assert idomain[-1, i, j].sum() == active.sum()
    # assert np.all(m.dis.botm.array[:-1, i, j] > 9980)
    assert np.all(m.dis.botm.array[-1, i, j] < 100)


@pytest.fixture(scope="module")
def model_setup_and_run(model_setup, mf6_exe):
    m = model_setup  #deepcopy(model_setup)
    m.simulation.exe_name = mf6_exe

    dis_idomain = m.dis.idomain.array.copy()
    for i, d in enumerate(m.cfg['dis']['griddata']['idomain']):
        arr = load_array(d['filename'])
        assert np.array_equal(m.idomain[i], arr)
        assert np.array_equal(dis_idomain[i], arr)
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


def test_model_setup_no_nans(model_setup):
    m = model_setup
    external_path = os.path.join(m.model_ws, 'external')
    external_files = glob.glob(external_path + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = '\n'.join(has_nans)
    if len(has_nans) > 0:
        assert False, has_nans

    # verify that "continue" option was successfully translated
    # to flopy sim constructor arg "continue_"
    assert m.simulation.name_file.continue_.array

    # verify that basic stress packages were built
    # and that there is only one RIV package
    package_list = m.get_package_list()
    for pckg in 'CHD_0', 'DRN_0', 'GHB_0', 'RIV_0':
        assert pckg in package_list
    assert 'RIV_1' not in package_list
    m.write_input()
    df = pd.read_csv(m.model_ws / 'external/riv_000.dat', delim_whitespace=True)
    # single RIV external file should have Lake Henry (from riv: block)
    # and Tallahatchie/Yazoo (via SFRmaker to_riv:)
    # does flopy force boundnames to lower case as of v3.3.7?
    assert set(df.boundname) == {'yazoo river', 'lake henry', 'tallahatchie river'}


def test_model_setup_and_run(model_setup_and_run):
    m = model_setup_and_run

    # check that original arrays folder was deleted
    # (on finish of setup_from_yaml workflow)
    # this also tests whether m.model_ws is a pathlib object
    # can't use m.tmpdir property here
    # because the property remakes the folder if it's missing
    assert not (m.model_ws / m.cfg['intermediate_data']['output_folder']).exists()


@pytest.mark.parametrize('load_only', (None, ['dis']))
def test_load(model_setup, shellmound_cfg_path, load_only):
    m = model_setup  #deepcopy(pfl_nwt_setup_from_yaml)
    m2 = MF6model.load_from_config(shellmound_cfg_path,
                                   load_only=load_only)
    if load_only is None:
        assert m == m2
    else:
        assert m2.get_package_list() == ['DIS']


def test_packagelist(shellmound_cfg_path):

    cfg = load_cfg(shellmound_cfg_path, default_file='mf6_defaults.yml')

    packages = cfg['model']['packages']
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    cfg['model']['simulation'] = sim

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    assert m.package_list == [p for p in m._package_setup_order if p in packages]


def test_delete_tmpdir_on_start(shellmound_cfg_path, tmpdir):
    tmpdir = Path(tmpdir) / 'shellmound/original-arrays'
    # Make the temporary 'original files' folder
    # like it already exists
    # add a stale file
    tmpdir.mkdir()
    stale_file = tmpdir / 'junk.txt'
    with open(stale_file, 'w') as dest:
        dest.write('junk!')
    m = MF6model(cfg=shellmound_cfg_path)
    # the temporary 'original files' folder
    # should have been deleted and remade on init of MF6model
    assert not stale_file.exists()
