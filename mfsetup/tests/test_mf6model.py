import sys
sys.path.append('..')
import time
from copy import deepcopy
import shutil
import os
import glob
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from shapely.geometry import box
import flopy
mf6 = flopy.mf6
from ..checks import check_external_files_for_nans
from ..discretization import get_layer_thicknesses, find_remove_isolated_cells
from ..fileio import load_array, exe_exists, read_mf6_block, load_cfg
from ..grid import rasterize
from ..mf6model import MF6model
from .. import testing
from ..units import convert_length_units
from ..utils import get_input_arguments


@pytest.fixture(scope="module", autouse=True)
def reset_dirs(shellmound_cfg):
    cfg = shellmound_cfg.copy()
    folders = [cfg['intermediate_data']['output_folder'],
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
    if hasattr(m, 'sfr'):
        sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
        m.sfrdata.write_package(sfr_package_filename,
                                    version='mf6',
                                    options=['save_flows',
                                             'BUDGET FILEOUT shellmound.sfr.cbc',
                                             'STAGE FILEOUT shellmound.sfr.stage.bin',
                                             # 'OBS6 FILEIN {}'.format(sfr_obs_filename)
                                             # location of obs6 file relative to sfr package file (same folder)
                                             ]
                                    )
    return m


def test_init(shellmound_cfg):
    cfg = shellmound_cfg.copy()
    sim = mf6.MFSimulation(**cfg['simulation'])
    assert isinstance(sim, mf6.MFSimulation)

    sim = mf6.MFSimulation()
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
    cfg = shellmound_cfg.copy()
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
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
    assert cfg['intermediate_data']['output_folder'] == os.path.join(ws, 'tmp')


def test_simulation(shellmound_simulation):
    assert True


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


def test_snap_to_NHG(shellmound_cfg, shellmound_simulation):
    cfg = shellmound_cfg.copy()
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


def test_perrioddata(shellmound_model):
    m = shellmound_model #deepcopy(model)
    pd0 = m.perioddata.copy()
    assert pd0 is not None
    assert pd0['end_datetime'].iloc[-1] == \
           pd.Timestamp(m.cfg['tdis']['perioddata']['group 3']['end_date_time'])


    m.cfg['sto']['steady'] = {0: True,
                              1: False}

    # save the group input for later tests
    pdinput_with_groups = {k: v for k, v in m.cfg['tdis']['perioddata'].items()
                           if 'group' in k}

    # Explicit stress period setup
    m.cfg['tdis']['perioddata'] = {k: v for k, v in m.cfg['tdis']['perioddata'].items()
                                   if 'group' not in k}
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
    # since perlen wasn't explicitly specified,
    # this dataframe will have the 11 periods at freq 'D' (like pd1)
    # but with a steady-state first stress period of length 1
    # in other words, perlen discretization with freq
    # only applies to transient stress periods
    assert pd2.iloc[:-1].equals(pd1)

    # Start date, end date, and nper
    m.cfg['tdis']['perioddata']['end_date_time'] = '2008-10-11'
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
    m.cfg['tdis']['perioddata']['end_date_time'] = '2008-10-12'
    m._perioddata = None
    pd5 = m.perioddata.copy()
    assert pd5.iloc[:-1].equals(pd1)

    # month end vs month start freq
    m.cfg['tdis']['perioddata']['freq'] = '6M'
    m.cfg['tdis']['options']['start_date_time'] = '2007-04-01'
    m.cfg['tdis']['perioddata']['end_date_time'] = '2015-10-01'
    m.cfg['tdis']['perioddata']['nstp'] = 15
    m._perioddata = None
    pd6 = m.perioddata.copy()
    pd0_g1_3 = pd.concat([pd0.iloc[:1], pd0.iloc[2:]])
    for c in pd0_g1_3[['perlen', 'start_datetime', 'end_datetime']]:
        np.array_equal(pd6[c].values, pd0_g1_3[c].values)


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
    dis = m.setup_dis()
    botm = m.dis.botm.array.copy()
    assert isinstance(dis, mf6.ModflowGwfdis)
    assert 'DIS' in m.get_package_list()
    # verify that units got conveted correctly
    assert m.dis.top.array.mean() < 100
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

   # test that written idomain array reflects supplied shapefile of active area
    active_area = rasterize(m.cfg['dis']['source_data']['idomain']['filename'],
                            m.modelgrid)
    isactive = active_area == 1
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
    # these two arrays are not equal
    # because isolated cells haven't been removed from the second one
    # this verifies that _set_idomain is removing them
    assert not np.array_equal(m.idomain[:, isactive].sum(axis=1),
                          invalid_botms[:, isactive].sum(axis=1))
    invalid_botms = find_remove_isolated_cells(invalid_botms, minimum_cluster_size=10)
    active_cells = m.idomain[:, isactive].copy()
    active_cells[active_cells < 0] = 0  # need to do this because some idomain cells are -1
    assert np.array_equal(active_cells.sum(axis=1),
                          invalid_botms[:, isactive].sum(axis=1))

    # test recreating package from external arrays
    m.remove_package('dis')
    assert m.cfg['dis']['griddata']['top'] is not None
    assert m.cfg['dis']['griddata']['botm'] is not None
    dis = m.setup_dis()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array == 1],
                          botm[m.dis.idomain.array == 1])

    # test recreating just the top from the external array
    m.remove_package('dis')
    m.cfg['dis']['remake_top'] = False
    m.cfg['dis']['griddata']['botm'] = None
    dis = m.setup_dis()
    dis.write()
    assert np.array_equal(m.dis.botm.array[m.dis.idomain.array == 1],
                          botm[m.dis.idomain.array == 1])
    arrayfiles = m.cfg['dis']['griddata']['top']
    for f in arrayfiles:
        assert os.path.exists(f['filename'])
    assert os.path.exists(os.path.join(m.model_ws, dis.filename))

    # dis package idomain should be consistent with model property
    updated_idomain = m.idomain
    assert np.array_equal(m.dis.idomain.array, updated_idomain)

    # check that units were converted (or not)
    assert np.allclose(dis.top.array.mean(), 40, atol=10)
    mcaq = m.cfg['dis']['source_data']['botm']['filenames'][3]
    assert 'mcaq' in mcaq
    with rasterio.open(mcaq) as src:
        mcaq_data = src.read(1)
        mcaq_data[mcaq_data == src.meta['nodata']] = np.nan
    assert np.allclose(m.dis.botm.array[3].mean() / .3048, np.nanmean(mcaq_data), atol=5)


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
    period_df['perlen'] = period_df['perlen'].astype(np.float64)
    period_df['nstp'] = period_df['nstp'].astype(np.int64)
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
    m = shellmound_model_with_dis
    npf = m.setup_npf()
    npf.write()
    assert os.path.exists(os.path.join(m.model_ws, npf.filename))

    # check that units were converted
    k3tif = m.cfg['npf']['source_data']['k']['filenames'][3]
    assert k3tif.endswith('k3.tif')
    with rasterio.open(k3tif) as src:
        data = src.read(1)
        data[data == src.meta['nodata']] = np.nan
    assert np.allclose(npf.k.array[3].mean() / .3048, np.nanmean(data), atol=5)

    # TODO: add tests that Ks got distributed properly considering input and pinched layers


@pytest.mark.parametrize('config', [{'source_data':
                                         {'filenames': ['../../data/shellmound/tables/head_obs_well_info.csv'],
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
def test_obs_setup(shellmound_model_with_dis, config):
    m = shellmound_model_with_dis  # deepcopy(model)
    defaults = {'default_columns':
                    {'x_location_col': 'x', # x coordinates in wtm
                     'y_location_col': 'y' # y coordinates in wtm
                     },
                'filename_fmt': '{}.head.obs',  # only head obs supported at this point
                'options':
                    {'digits': 10,
                     'print_input': True}
                }
    defaults.update(config)
    m.cfg['obs'] = defaults
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


@pytest.mark.parametrize('options', [{'saverecord': {0: {'head': 'last',
                                                         'budget': 'last'}}},
                                     {'period_options': {0: ['save head last',
                                                             'save budget last']}}
                                        ])
def test_oc_setup(shellmound_model_with_dis, options):
    cfg = {'head_fileout_fmt': '{}.hds',
           'budget_fileout_fmt': '{}.cbc'}
    cfg.update(options)
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
    assert 'fileout' in options['budget'] and '.cbc' in options['budget']
    assert 'fileout' in options['head'] and '.hds' in options['head']
    assert 'save head last' in perioddata[1]
    assert 'save budget last' in perioddata[1]


def test_rch_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis  # deepcopy(model)
    rch = m.setup_rch()
    rch.write()
    assert os.path.exists(os.path.join(m.model_ws, rch.filename))
    assert isinstance(rch, mf6.ModflowGwfrcha)
    assert rch.recharge is not None

    # get the same data from the source file
    ds = xr.open_dataset(m.cfg['rch']['source_data']['recharge']['filename'])
    x = xr.DataArray(m.modelgrid.xcellcenters.ravel(), dims='z')
    y = xr.DataArray(m.modelgrid.ycellcenters.ravel(), dims='z')

    unit_conversion = convert_length_units('inches', 'meters')

    def get_period_values(start, end):
        period_data = ds['net_infiltration'].loc[start:end].mean(axis=0)
        dsi = period_data.interp(x=x, y=y, method='linear',
                                 kwargs={'fill_value': np.nan,
                                         'bounds_error': True})
        data = dsi.values * unit_conversion
        return np.reshape(data, (m.nrow, m.ncol))

    # test steady-state avg. across all data
    values = get_period_values('2012-01-01', '2017-12-31')

    #assert np.allclose(values, m.rch.recharge.array[0, 0])
    # test period 1 avg. for those times
    values1 = get_period_values(m.perioddata['start_datetime'].values[1],
                                m.perioddata['end_datetime'].values[1])
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


def test_wel_setup(shellmound_model_with_dis):
    m = shellmound_model_with_dis  # deepcopy(model)
    wel = m.setup_wel()
    wel.write()
    assert os.path.exists(os.path.join(m.model_ws, wel.filename))
    assert isinstance(wel, mf6.ModflowGwfwel)
    assert wel.stress_period_data is not None
    sums = [ra['q'].sum() if ra is not None else 0
            for ra in wel.stress_period_data.array]

    # sum the rates from the source files
    dfs = []
    for f in m.cfg['wel']['source_data']['csvfiles']['filenames']:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    df['start_datetime'] = pd.to_datetime(df.start_datetime)
    df['end_datetime'] = pd.to_datetime(df.end_datetime)
    from ..grid import get_ij
    i, j = get_ij(m.modelgrid, df.x.values, df.y.values)
    idm = m.idomain[:, i, j]
    invalid = np.zeros(len(df), dtype=bool) #(df.screen_top == -9999.) & (df.screen_botm == -9999.)
    invalid = invalid | (idm.sum(axis=0) <= 0)
    df = df.loc[~invalid].copy()
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
    assert np.allclose(sums, sums2, rtol=0.001)


def test_sfr_setup(model_with_sfr):
    m = model_with_sfr
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


def test_idomain_above_sfr(model_with_sfr):
    m = model_with_sfr
    sfr = m.sfr
    # get the kij locations of sfr reaches
    cellids = sfr.packagedata.array['cellid'].tolist()
    deact_lays = [list(range(cellid[0])) for cellid in cellids]
    k, i, j = list(zip(*cellids))

    # verify that streambed tops are above layer bottoms
    assert np.all(sfr.packagedata.array['rtp'] > np.all(m.dis.botm.array[k, i, j]))

    # test that idomain above sfr cells is being set to 0
    # by setting all botms above streambed tops
    new_botm = m.dis.botm.array.copy()
    new_top = m.dis.top.array.copy()
    new_botm[:, i, j] = 9999
    new_top[i, j] = 9999
    np.savetxt(m.cfg['dis']['griddata']['top'][0]['filename'], new_top)
    m.dis.botm = new_botm
    #m.dis.top = new_top
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
    assert os.path.getsize(fname) < 2e3

    # idomain should be zero everywhere there's a sfr reach
    # except for in the botm layer
    # (verifies that model botm was reset to accomdate SFR reaches)
    assert np.array_equal(m.sfr.reach_data.i, i)
    assert np.array_equal(m.sfr.reach_data.j, j)
    assert idomain[:-1, i, j].sum() == 0
    assert idomain[-1, i, j].sum() == len(sfr.packagedata.array)
    assert np.all(m.dis.botm.array[:-1, i, j] > 9980)
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


def test_model_setup_and_run(model_setup_and_run):
    m = model_setup_and_run


def test_load(model_setup, shellmound_cfg_path):
    m = model_setup  #deepcopy(pfl_nwt_setup_from_yaml)
    m2 = MF6model.load(shellmound_cfg_path)
    assert m == m2


def test_packagelist(shellmound_cfg_path):

    cfg = load_cfg(shellmound_cfg_path, default_file='/mf6_defaults.yml')

    packages = cfg['model']['packages']
    sim = flopy.mf6.MFSimulation(**cfg['simulation'])
    cfg['model']['simulation'] = sim

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    assert m.package_list == [p for p in m._package_setup_order if p in packages]
