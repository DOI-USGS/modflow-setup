import sys

sys.path.append('..')
import filecmp
import glob
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pytest
import rasterio

fm = flopy.modflow
mf6 = flopy.mf6
from mfsetup import MFnwtModel
from mfsetup.checks import check_external_files_for_nans
from mfsetup.fileio import exe_exists, load, load_cfg, remove_file_header
from mfsetup.grid import MFsetupGrid, get_ij
from mfsetup.units import convert_length_units, convert_time_units
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pfl_nwt_setup_from_yaml(pfl_nwt_test_cfg_path):
    m = MFnwtModel.setup_from_yaml(pfl_nwt_test_cfg_path)
    m.write_input()
    return m


def write_namefile(model):
    """Write the namefile,
    making a backup copy if one already exists."""
    m = model
    namfile = os.path.join(m.model_ws, m.namefile)
    if os.path.exists(namfile):
        shutil.copy(namfile, namfile + '.bak')
    m.write_name_file()
    return namfile


def test_load_cfg(pfl_nwt_cfg):
    cfg = pfl_nwt_cfg
    assert True


def test_init(pfl_nwt_cfg):
    cfg = pfl_nwt_cfg.copy()
    cfg['model']['packages'] = []
    # test initialization with no packages
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    assert isinstance(m, MFnwtModel)

    # test initialization with no arguments
    m = MFnwtModel()
    assert isinstance(m, MFnwtModel)


def test_repr(pfl_nwt, pfl_nwt_with_grid):
    txt = pfl_nwt.__repr__()
    assert isinstance(txt, str)
    # cheesy test that flopy repr isn't returned
    assert 'CRS:' in txt and 'Bounds:' in txt
    txt = pfl_nwt_with_grid.__repr__()
    assert isinstance(txt, str)


def test_pfl_nwt(pfl_nwt):
    assert isinstance(pfl_nwt, MFnwtModel)


def test_pfl_nwt_with_grid(pfl_nwt_with_grid):
    assert pfl_nwt_with_grid.modelgrid is not None


def test_namefile(pfl_nwt_with_dis):
    model = pfl_nwt_with_dis
    model.write_input()

    # check that listing file was written correctly
    expected_listfile_name = model.cfg['model']['list_filename_fmt'].format(model.name)
    with open(model.namefile) as src:
        for line in src:
            if 'LIST' in line:
                assert line.strip().split()[-1] == expected_listfile_name


def test_perioddata(pfl_nwt):
    model = pfl_nwt
    assert np.array_equal(model.perioddata.steady, [True, False])


def test_set_parent_model(pfl_nwt):
    m = pfl_nwt
    assert isinstance(m.parent, fm.Modflow)
    assert isinstance(m.parent.perioddata, pd.DataFrame)
    assert isinstance(m.parent.modelgrid, MFsetupGrid)
    assert m.parent.modelgrid.nrow == m.parent.nrow
    assert m.parent.modelgrid.ncol == m.parent.ncol
    assert m.parent.modelgrid.nlay == m.parent.nlay


def test_set_parent_perioddata(pfl_nwt):
    perioddata = pfl_nwt.perioddata
    parent_perioddata = pfl_nwt.parent.perioddata
    cols = set(perioddata).difference({'parent_sp'})
    for c in cols:
        assert c in parent_perioddata.columns


def test_set_perioddata_tr_parent(inset_with_transient_parent):
    if inset_with_transient_parent is not None:
        m = deepcopy(inset_with_transient_parent)
        perioddata = m.perioddata
        assert pd.Timestamp(perioddata['start_datetime'].values[0]) == \
               pd.Timestamp(m.cfg['model']['start_date_time'])
        assert perioddata['time'].values[-1] == np.sum(m.cfg['dis']['perlen'])


def test_load_grid(pfl_nwt, pfl_nwt_with_grid):

    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    m2 = pfl_nwt  #deepcopy(pfl_nwt)
    m2.load_grid(m.cfg['setup_grid']['output_files']['grid_file'].format(m.name))
    assert m.cfg['grid'] == m2.cfg['grid']


def test_set_lakarr(pfl_nwt_with_dis):
    m = pfl_nwt_with_dis
    assert 'lak' in m.package_list
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
    assert m._lakarr2d.sum() == 0


def test_dis_setup(pfl_nwt_with_grid):

    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)

    # test intermediate array creation
    m.cfg['dis']['source_data']['top']['elevation_units'] = 'meters'
    m.cfg['dis']['lenuni'] = 2  # meters
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    assert 'DIS' in m.get_package_list()
    arrayfiles = m.cfg['intermediate_data']['top'] +\
                 m.cfg['intermediate_data']['botm']
    for f in arrayfiles:
        assert os.path.exists(f)

    # verify that modelgrid was reset after building DIS
    mg = m.modelgrid
    assert (mg.nlay, mg.nrow, mg.ncol) == m.dis.botm.array.shape
    assert np.array_equal(mg.top, m.dis.top.array)
    assert np.array_equal(mg.botm, m.dis.botm.array)

    # test using previously made external files as input
    if m.version == 'mf6':
        assert m.cfg['dis']['top'] == m.cfg['external_files']['top']
        assert m.cfg['dis']['botm'] == m.cfg['external_files']['botm']
    else:
        assert m.cfg['dis']['top'] == m.cfg['intermediate_data']['top']
        assert m.cfg['dis']['botm'] == m.cfg['intermediate_data']['botm']
    m.cfg['dis']['remake_top'] = False
    m.cfg['dis']['nper'] = 4
    m.cfg['dis']['perlen'] = [1, 1, 1, 1]
    m.cfg['dis']['nstp'] = [1, 1, 1, 1]
    m.cfg['dis']['tsmult'] = [1, 1, 1, 1]
    m.cfg['dis']['steady'] = [1, 0, 0, 1]
    m._perioddata = None
    dis = m.setup_dis()
    dis.write_file()
    arrayfiles = m.cfg['external_files']['top'] + \
                 m.cfg['external_files']['botm']
    for f in arrayfiles:
        assert os.path.exists(f)
    assert os.path.exists(dis.fn_path)

    # check settings
    assert m.cfg['dis']['steady'] == [True, False, False, True]
    assert dis.steady.array.tolist() == [True, False, False, True]

    # test unit conversion
    top_m = dis.top.array.copy()
    botm_m = dis.botm.array.copy()
    m.cfg['dis']['top'] = None  # arrays don't get remade if this has data
    m.cfg['dis']['botm'] = None
    for var in 'top', 'botm':
        if var in m.cfg['setup_grid']:
            del m.cfg['setup_grid'][var]
    m.cfg['dis']['remake_top'] = True
    m.cfg['dis']['lenuni'] = 1 # feet
    m.cfg['dis']['minimum_layer_thickness'] = 1/.3048 # feet
    m.cfg['setup_grid']['dxy'] = 20  # in CRS units
    m.remove_package('DIS')
    m.setup_grid()
    #m._reset_bc_arrays()
    assert m.cfg['parent']['length_units'] == 'meters'
    assert m.cfg['parent']['time_units'] == 'days'
    assert m.length_units == 'feet'
    original_top_file = Path(m.external_path,
                             f"{m.name}_{m.cfg['dis']['top_filename_fmt']}.original")
    original_top_file.unlink(missing_ok=True)
    dis = m.setup_dis()
    assert np.allclose(dis.top.array.mean() * convert_length_units(1, 2), top_m.mean())
    assert np.allclose(dis.botm.array.mean() * convert_length_units(1, 2), botm_m.mean())

    # check that original arrays were created in expected folder
    assert Path(m.cfg['intermediate_data']['output_folder']).is_dir()


@pytest.mark.parametrize('bas6_config,default_parent_source_data', (
    ('config file',True),  # test whatever is in the configuration file
    # with default_parent_source_data
    # starting heads are automatically resampled from parent model
    # unless a strt array or binary head file is provided
    #
    # test case of default configuration (no bas: block on configuration file)
    # starting heads are set to the model top
    ('defaults', False),
    # test case where just bas: (None configuration) is argued in configration file
    # (this overrides the defaults)
    # starting heads are set to the model top
    (None, False),
    # starting heads from a raster
    ({'source_data': {
        'strt': {
            'filename': 'plainfieldlakes/source_data/dem10m.tif'
        }
        }}, False),
    # need a strt variable
    pytest.param({'source_data': {
        'filename': 'plainfieldlakes/source_data/dem10m.tif'
        }}, False, marks=pytest.mark.xfail(reason='some bug')),
    # with a layer specified (gets repeated)
    ({'source_data': {
        'strt': {
            'filenames': {
                0: 'plainfieldlakes/source_data/dem10m.tif'
            }
        }
    }}, False),
    # starting heads from MODFLOW binary head output
    ({'source_data': {
        'strt': {
            'from_parent': {
                'binaryfile': 'plainfieldlakes/pfl.hds',
                'stress_period': 0
            }
        }
        }}, False)
)
                         )
def test_bas_setup(pfl_nwt_cfg, pfl_nwt_with_dis, bas6_config,
                   default_parent_source_data, project_root_path):
    """Test setup of the BAS6 package, especially the starting heads."""
    cfg = pfl_nwt_cfg.copy()
    project_root_path = Path(project_root_path)
    # load defaults here
    default_cfg = project_root_path / 'mfsetup/mfnwt_defaults.yml'
    defaults = load(default_cfg)
    if bas6_config == 'config file':
        pass
    elif bas6_config == 'defaults':
        cfg['bas6'] = defaults['bas6']
    elif bas6_config is None:
        cfg['bas6'] = defaultdict(dict)
    else:
        cfg['bas6'] = bas6_config
    cfg['parent']['default_source_data'] = default_parent_source_data

    m = MFnwtModel(cfg=cfg, **cfg['model'])
    m.setup_grid()
    m.setup_dis()

    bas = m.setup_bas6()

    assert bas.strt.array.shape == m.modelgrid.shape
    assert not np.isnan(bas.strt.array).any(axis=(0, 1, 2))

    # In the absence of a parent model with default_source_data
    # or input to strt
    # check that strt was set from model top
    if bas6_config in ('defaults', None):
        assert np.allclose(bas.strt.array,
                           np.array([m.dis.top.array] * m.modelgrid.nlay))
    # assumes that all rasters being tested
    # are the same as the dem used to make the model top
    elif isinstance(bas6_config, dict):
        if 'filename' in bas6_config.get('source_data', dict()) or\
            'filenames' in bas6_config.get('source_data', dict()):
            assert np.allclose(bas.strt.array,
                            np.array([m.dis.top.array] * m.modelgrid.nlay))
    # TODO: placeholder for more rigorous test that starting heads
    # are consistent with parent model head solution
    else:
        pass

    if bas6_config == 'defaults':
        # test intermediate array creation
        arrayfiles = m.cfg['intermediate_data']['strt'] + \
                    m.cfg['intermediate_data']['ibound']
                    #m.cfg['intermediate_data']['lakarr']
        for f in arrayfiles:
            assert os.path.exists(f)

        # test using previously made external files as input
        if m.version == 'mf6':
            assert m.cfg['bas6']['strt'] == m.cfg['external_files']['strt']
            assert m.cfg['bas6']['ibound'] == m.cfg['external_files']['ibound']
        else:
            assert m.cfg['bas6']['strt'] == m.cfg['intermediate_data']['strt']
            assert m.cfg['bas6']['ibound'] == m.cfg['intermediate_data']['ibound']
        bas = m.setup_bas6()
        bas.write_file()
        arrayfiles = m.cfg['bas6']['strt'] + \
                    m.cfg['bas6']['ibound']
        for f in arrayfiles:
            assert os.path.exists(f)
        assert os.path.exists(bas.fn_path)


@pytest.mark.parametrize('simulate_high_k_lakes', (False, True))
def test_rch_setup(pfl_nwt_with_dis, project_root_path, simulate_high_k_lakes):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.cfg['high_k_lakes']['simulate_high_k_lakes'] = simulate_high_k_lakes
    # test intermediate array creation from rech specified as scalars
    m.cfg['rch']['rech'] = [0.001, 0.002]
    #m.cfg['rch']['rech_length_units'] = 'meters'
    #m.cfg['rch']['rech_time_units'] = 'days'
    rch = m.setup_rch()
    arrayfiles = m.cfg['intermediate_data']['rech']
    assert len(arrayfiles) == len(m.cfg['rch']['rech'])
    for f in arrayfiles:
        assert os.path.exists(f)

    # test intermediate array creation from source_data
    # (rasters of different shapes)
    inf_array = 'mfsetup/tests/data/plainfieldlakes/source_data/' \
                'net_infiltration__2012-01-01_to_2017-12-31__1066_by_1145__SUM__INCHES_PER_YEAR.tif'
    inf_array = os.path.join(project_root_path, inf_array)
    with rasterio.open(inf_array) as src:
        inf_values = src.read(1)

    m.cfg['rch']['source_data']['rech']['filename'] = inf_array
    m.cfg['rch']['rech'] = None
    m.cfg['rch']['source_data']['rech']['length_units'] = 'inches'
    m.cfg['rch']['source_data']['rech']['time_units'] = 'years'
    rch = m.setup_rch()

    # spatial mean recharge in model should approx. match the GeoTiff (which covers a larger area)
    avg_in_yr = rch.rech.array[0, 0, :, :].mean() * convert_length_units('meters', 'inches') / \
        convert_time_units('days', 'years')
    assert np.allclose(avg_in_yr, inf_values.mean() * m.cfg['rch']['source_data']['rech']['mult'], rtol=0.25)
    arrayfiles = m.cfg['intermediate_data']['rech']
    for f in arrayfiles:
        assert os.path.exists(f)

    # check that high-K lake recharge was assigned correctly
    if simulate_high_k_lakes:
        highklake_recharge = m.rch.rech.array[:, 0, m.isbc[0] == 2].mean(axis=1)
        print(highklake_recharge)
        print(m.high_k_lake_recharge)
        assert np.allclose(highklake_recharge, m.high_k_lake_recharge)
    else:
        assert not np.any(m._isbc2d == 2)

    # test writing of MODFLOW arrays
    rch.write_file()
    assert m.cfg['rch']['rech'] is not None
    for f in m.cfg['rch']['rech']:
        assert os.path.exists(f)
    assert os.path.exists(rch.fn_path)

    # test intermediate array creation from rech specified as arrays
    # (of same shape; use MODFLOW arrays written above)
    rch = m.setup_rch()
    arrayfiles = m.cfg['intermediate_data']['rech']
    for f in arrayfiles:
        assert os.path.exists(f)


@pytest.mark.parametrize('simulate_high_k_lakes,case', [(False, 0),
                                                        (True, 0),
                                                        (False, 1)])
def test_upw_setup(pfl_nwt_with_dis, case, simulate_high_k_lakes):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.cfg['high_k_lakes']['simulate_high_k_lakes'] = simulate_high_k_lakes
    if case == 0:
        # test intermediate array creation
        m.cfg['upw']['remake_arrays'] = True
        upw = m.setup_upw()
        arrayfiles = m.cfg['intermediate_data']['hk'] + \
                     m.cfg['intermediate_data']['vka']
        for f in arrayfiles:
            assert os.path.exists(f)

        # check that lakes were set up properly
        if not simulate_high_k_lakes:
            assert not np.any(m._isbc2d == 2)
            assert upw.hk.array.max() < m.cfg['high_k_lakes']['high_k_value']
            assert upw.sy.array.min() < m.cfg['high_k_lakes']['sy']
            assert upw.ss.array.min() > m.cfg['high_k_lakes']['ss']
        else:
            assert np.any(m._isbc2d == 2)
            assert upw.hk.array.max() == m.cfg['high_k_lakes']['high_k_value']
            assert upw.sy.array.max() == m.cfg['high_k_lakes']['sy']
            assert np.allclose(upw.ss.array.min(), m.cfg['high_k_lakes']['ss'])

        # compare values to parent model
        for var in ['hk', 'vka']:
            ix, iy = m.modelgrid.xcellcenters.ravel(), m.modelgrid.ycellcenters.ravel()
            pi, pj = get_ij(m.parent.modelgrid, ix, iy)
            parent_layer = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
            for k, pk in parent_layer.items():
                parent_vals = m.parent.upw.__dict__[var].array[pk, pi, pj]
                inset_vals = upw.__dict__[var].array
                valid_parent = parent_vals != m.cfg['high_k_lakes'].get('high_k_value', -9999)
                valid_inset = inset_vals[k].ravel() != m.cfg['high_k_lakes'].get('high_k_value', -9999)
                parent_vals = parent_vals[valid_parent & valid_inset]
                inset_vals = inset_vals[k].ravel()[valid_parent & valid_inset]
                assert np.allclose(parent_vals, inset_vals, rtol=0.01)

    elif case == 1:
        # test changing vka to anisotropy
        m.cfg['upw']['layvka'] = [1, 1, 1, 1, 1]
        m.cfg['upw']['vka'] = [10, 10, 10, 10, 10]
        upw = m.setup_upw()
        assert np.array_equal(m.upw.layvka.array, np.array([1, 1, 1, 1, 1]))
        assert np.allclose(m.upw.vka.array.max(axis=(1, 2)),
                           np.array([10, 10, 10, 10, 10]))


def test_wel_setup(pfl_nwt_with_dis_bas6):
    m = pfl_nwt_with_dis_bas6  #deepcopy(pfl_nwt_with_dis)deepcopy(pfl_nwt_with_dis)
    m.setup_upw()
    # test without tmr
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


def test_wel_setup_drop_ids(pfl_nwt_with_dis_bas6):
    m = pfl_nwt_with_dis_bas6  # deepcopy(pfl_nwt_with_dis)deepcopy(pfl_nwt_with_dis)
    m.setup_upw()
    m.cfg['wel']['source_data']['wdnr_dataset']['drop_ids'] = [4026]
    m.cfg['wel']['mfsetup_options']['external_files'] = False
    wel = m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    df = pd.read_csv(m.cfg['wel']['output_files']['lookup_file'])
    assert 'site4026' not in df.boundname.tolist()
    assert len(df) == len(wel.stress_period_data[1])


def test_wel_setup_csv_by_per(pfl_nwt_with_dis_bas6):

    m = pfl_nwt_with_dis_bas6  # deepcopy(pfl_nwt_with_dis)deepcopy(pfl_nwt_with_dis)
    m.setup_upw()
    # test adding a wel from a csv file
    m.cfg['wel']['source_data']['csvfile'] = {
        'filename':'plainfieldlakes/source_data/added_wells.csv',
        'data_column': 'flux',
        'id_column': 'name',
        'datetime_column': 'datetime'
    }
    wel = m.setup_wel(**m.cfg['wel'], **m.cfg['wel']['mfsetup_options'])
    assert -2000 in wel.stress_period_data[1]['flux']


def test_mnw_setup(pfl_nwt_with_dis):

     m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
     mnw = m.setup_mnw2()
     mnw.write_file()
     assert True


def test_littoral_zone_buffer_width(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)

    # test huge buffer (no profundal zone)
    m.cfg['lak']['source_data']['littoral_zone_buffer_width'] = 200
    lak = m.setup_lak()
    assert not np.any(np.any(lak.bdlknc.array == m.cfg['lak']['source_data']['profundal_leakance']))
    m.remove_package('lak')
    # verify that there are less litoral cells if no buffer is specified
    # (there are always some, because of default 1.5 cell width buffer
    # around outside edge of lake, for fluxes through horizontal cell faces)
    m.cfg['lak']['source_data']['littoral_zone_buffer_width'] = 20
    lak = m.setup_lak()
    n_littoral_20 = np.sum(lak.bdlknc.array == m.cfg['lak']['source_data']['littoral_leakance'])
    m.remove_package('lak')
    m.cfg['lak']['source_data']['littoral_zone_buffer_width'] = 0
    lak = m.setup_lak()
    n_littoral_0 = np.sum(lak.bdlknc.array == m.cfg['lak']['source_data']['littoral_leakance'])
    assert n_littoral_0 < n_littoral_20


def test_lak_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)

    # fill in stage area volume file
    #df = pd.read_csv(m.cfg['lak']['stage_area_volume'])
    #cols = [s.lower() for s in df.columns]
    #if 'hydroid' not in cols:
    #    df['hydroid'] = m.cfg['lak']['include_lakes'][0]
    #    df.to_csv(m.cfg['lak']['stage_area_volume'], index=False)

    lak = m.setup_lak()
    lak.write_file()
    assert lak.bdlknc.array.sum() > 0
    assert not np.any(np.isnan(lak.bdlknc.array))
    assert np.any(lak.bdlknc.array == m.cfg['lak']['source_data']['littoral_leakance'])
    assert np.any(lak.bdlknc.array == m.cfg['lak']['source_data']['profundal_leakance'])
    lookup_file = Path(m._tables_path, Path(m.cfg['lak']['output_files']['lookup_file']).name)
    assert lookup_file.exists()
    assert lak.lakarr.array.sum() > 0
    tabfiles = m.cfg['lak']['tab_files']
    for f in tabfiles:
        assert os.path.exists(os.path.join(m.model_ws, f))
    namfile = write_namefile(m)
    with open(namfile) as src:
        txt = src.read()
    # kludge to deal with ugliness of lake package external file handling
    tab_files_argument = [os.path.relpath(f) for f in tabfiles]
    for f in tab_files_argument:
        assert f in txt

    # test setup of lak package with steady-state stress period > 1
    m.cfg['dis']['nper'] = 4
    m.cfg['dis']['perlen'] = [1, 1, 1, 1]
    m.cfg['dis']['nstp'] = [1, 1, 1, 1]
    m.cfg['dis']['tsmult'] = [1, 1, 1, 1]
    m.cfg['dis']['steady'] = [1, 0, 0, 1]
    m._perioddata = None
    dis = m.setup_dis()
    lak = m.setup_lak()
    lak.write_file()
    # verify that min/max stage were written to dataset 9 in last sp
    with open(lak.fn_path) as src:
        for line in src:
            if "Stress period 4" in line:
                ds9_entries = next(src).split('#')[0].strip().split()
    assert len(ds9_entries) == 6

    # check that order in lake lookup file is same as specified in include_ids
    lookup = pd.read_csv(lookup_file)
    include_ids = m.cfg['lak']['source_data']['lakes_shapefile']['include_ids']
    assert lookup.feat_id.tolist() == include_ids

    # check that tabfiles are in correct order
    with open(namfile) as src:
        units = []
        hydroids = []
        for line in src:
            if 'stage_area_volume' in line:
                _, unit, fname = line.strip().split()
                hydroid = int(os.path.split(fname)[1].split('_')[0])
                hydroids.append(hydroid)
                units.append(int(unit))
        inds = np.argsort(units)
        hydroids = np.array(hydroids)[inds]
    assert hydroids.tolist() == include_ids
    # restore the namefile to what was there previously
    if os.path.exists(namfile+'.bak'):
        shutil.copy(namfile+'.bak', namfile)


def test_nwt_setup(pfl_nwt, project_root_path):
    m = pfl_nwt  #deepcopy(pfl_nwt)
    m.cfg['nwt']['use_existing_file'] = project_root_path / 'mfsetup/tests/data/RGN_rjh_3_23_18.NWT'
    nwt = m.setup_nwt()
    nwt.write_file()
    m.cfg['nwt']['use_existing_file'] = None
    nwt = m.setup_nwt()
    nwt.write_file()


@pytest.mark.parametrize('input,expected', [
    # MODFLOW 6-style input
    ({'period_options': {0: ['save head last', 'save budget last'],
                         1: []}},
     {'stress_period_data': {(0, 0): ['save head', 'save budget'],
                            (1, 0): []}}),
    # MODFLOW 2005-style input
    ({'stress_period_data': {(0, 0): ['save head', 'save budget'],
                            (1, 0): []}},
     {'stress_period_data': {(0, 0): ['save head', 'save budget'],
                            (1, 0): []}})
])
def test_oc_setup(pfl_nwt, input, expected):
    m = pfl_nwt
    m.cfg['oc'].update(input)
    oc = m.setup_oc()
    assert oc.stress_period_data == expected['stress_period_data']

    # TODO: add datetime comments to OC file


def test_hyd_setup(pfl_nwt_with_dis_bas6):

    m = pfl_nwt_with_dis_bas6  #deepcopy(pfl_nwt_with_dis)
    hyd = m.setup_hyd(**m.cfg['hyd'])
    hyd.write_file()
    # verify that each head observation is in each layer
    df = pd.DataFrame(hyd.obsdata)
    heads = df.loc[df.arr == b'HD']
    nobs = len(set(heads.hydlbl))
    assert sorted(heads.klay.tolist()) == sorted(list(range(m.nlay)) * nobs)
    #m.cfg['hyd'][]


def test_lake_gag_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.cfg['gag']['lak_outtype'] = 1
    lak = m.setup_lak()
    gag = m.setup_gag()
    gag.write_file()
    for f in m.cfg['gag']['ggo_files']:
        assert f in m.output_fnames

    # check that lake numbers and units are negative
    # and that outtype is specified
    with open(gag.fn_path) as src:
        ngage = int(next(src).strip())
        for i, line in enumerate(src):
            if i == lak.nlakes:
                break
            lake_no, unit, outtype = line.strip().split()
            assert int(lake_no) < 0
            assert int(unit) < 0  # for reading outtype
            assert int(outtype) >= 0

    # check that ggos are in correct order
    include_ids = m.cfg['lak']['source_data']['lakes_shapefile']['include_ids']
    namfile = write_namefile(m)
    with open(namfile) as src:
        units = []
        hydroids = []
        for line in src:
            if '.ggo' in line:
                _, unit, fname = line.strip().split()
                hydroid = int(os.path.splitext(os.path.split(fname)[1].split('_')[1])[0])
                hydroids.append(hydroid)
                units.append(int(unit))
        inds = np.argsort(units)
        hydroids = np.array(hydroids)[inds]
    assert hydroids.tolist() == include_ids
    # restore the namefile to what was there previously
    if os.path.exists(namfile + '.bak'):
        shutil.copy(namfile + '.bak', namfile)


def test_perimeter_boundary_setup(pfl_nwt_with_dis_bas6):

    m = pfl_nwt_with_dis_bas6  #deepcopy(pfl_nwt_with_dis)
    chd = m.setup_chd(**m.cfg['chd'], **m.cfg['chd']['mfsetup_options'])
    chd.write_file()
    assert os.path.exists(chd.fn_path)
    assert len(chd.stress_period_data.data.keys()) == len(set(m.cfg['parent']['copy_stress_periods']))
    # number of boundary heads;
    # can be less than number of active boundary cells if the (parent) water table is not always in (inset) layer 1
    assert len(chd.stress_period_data[0]) <= np.sum(m.ibound[m.get_boundary_cells()] == 1)

    # check for inactive cells
    spd0 = chd.stress_period_data[0]
    k, i, j = spd0['k'], spd0['i'], spd0['j']
    inactive_cells = m.ibound[k, i, j] < 1
    assert not np.any(inactive_cells)

    # check that heads are above layer botms
    assert np.all(spd0['shead'] > m.dis.botm.array[k, i, j])
    assert np.all(spd0['ehead'] > m.dis.botm.array[k, i, j])


def test_sfr_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.setup_bas6()
    m.setup_sfr()
    assert m.sfr is None


def test_model_setup(pfl_nwt_setup_from_yaml):
    m = pfl_nwt_setup_from_yaml
    assert 'CHD' in m.get_package_list()


def test_model_setup_no_nans(pfl_nwt_setup_from_yaml):
    m = pfl_nwt_setup_from_yaml
    external_path = os.path.join(m.model_ws, 'external')
    external_files = glob.glob(external_path + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = '\n'.join(has_nans)
    if len(has_nans) > 0:
        assert False, has_nans


def test_model_setup_nans(pfl_nwt_setup_from_yaml):
    m = pfl_nwt_setup_from_yaml
    external_path = os.path.join(m.model_ws, 'external')
    bad_file = os.path.normpath('external/CHD_9999.dat')
    with open('external/CHD_0000.dat') as src:
        with open(bad_file, 'w') as dest:
            for i, line in enumerate(src):
                if i in [10, 11]:
                    values = line.strip().split()
                    values[-1] = 'NaN'
                    dest.write(' '.join(values) + '\n')
                dest.write(line)
    external_files = glob.glob(external_path + '/*')
    has_nans = check_external_files_for_nans(external_files)
    has_nans = [os.path.normpath(f) for f in has_nans]
    assert bad_file in has_nans
    os.remove(bad_file)


#@pytest.mark.skip("still working on wel")
def test_model_setup_and_run(model_setup_and_run):
    m = model_setup_and_run  #deepcopy(model_setup_and_run)

    # check that original arrays folder was deleted
    # (on finish of setup_from_yaml workflow)
    # this also tests whether m.model_ws is a pathlib object
    # can't use m.tmpdir property here
    # because the property remakes the folder if it's missing
    assert not (m.model_ws / m.cfg['intermediate_data']['output_folder']).exists()


def test_load(pfl_nwt_setup_from_yaml, pfl_nwt_test_cfg_path):
    m = pfl_nwt_setup_from_yaml  #deepcopy(pfl_nwt_setup_from_yaml)
    m2 = MFnwtModel.load(pfl_nwt_test_cfg_path)

    m3 = MFnwtModel.load(pfl_nwt_test_cfg_path, forgive=True)
    assert m == m2
    assert m2 == m3


def test_remake_a_package(pfl_nwt_setup_from_yaml, pfl_nwt_test_cfg_path):

    m = pfl_nwt_setup_from_yaml
    shutil.copy(m.lak.fn_path, 'lakefile1.lak')
    m2 = MFnwtModel.load(pfl_nwt_test_cfg_path, load_only=['dis'])
    lak = m2.setup_lak()
    lakefile2 = lak.fn_path
    lak.write_file()
    # scrub the headers of both files for the comparison
    remove_file_header('lakefile1.lak')
    remove_file_header(lakefile2)
    assert filecmp.cmp('lakefile1.lak', lakefile2)


@pytest.fixture(scope="function")
def inset_with_transient_parent(pfl_nwt_with_grid):
    # TODO: port LPR test case over from CSLS
    return None


@pytest.fixture(scope="session")
def model_setup_and_run(pfl_nwt_setup_from_yaml, mfnwt_exe):
    m = pfl_nwt_setup_from_yaml
    m.exe_name = mfnwt_exe
    success = False
    if exe_exists(mfnwt_exe):
        success, buff = m.run_model(silent=False)
        if not success:
            list_file = m.lst.fn_path
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


def test_packagelist(pfl_nwt_test_cfg_path):
    cfg = load_cfg(pfl_nwt_test_cfg_path, default_file='mfnwt_defaults.yml')

    assert len(cfg['model']['packages']) > 0
    kwargs = get_input_arguments(cfg['model'], MFnwtModel)
    m = MFnwtModel(cfg=cfg, **kwargs)
    assert m.package_list == [p for p in m._package_setup_order
                              if p in cfg['model']['packages']]
