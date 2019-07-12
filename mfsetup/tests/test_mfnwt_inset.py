import sys
sys.path.append('..')
import time
from copy import copy, deepcopy
import shutil
import os
import pytest
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from mfsetup import MFnwtModel
from mfsetup.units import convert_length_units
from mfsetup.utils import get_input_arguments



@pytest.fixture(scope="session")
def cfg(mfnwt_inset_test_cfg_path):
    cfg = MFnwtModel.load_cfg(mfnwt_inset_test_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['model']['model_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def inset(cfg):
    print('pytest fixture inset')
    cfg = cfg.copy()
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="session")
def inset_setup_from_yaml(mfnwt_inset_test_cfg_path):
    m = MFnwtModel.setup_from_yaml(mfnwt_inset_test_cfg_path)
    m.write_input()
    return m


@pytest.fixture(scope="function")
def inset_with_grid(inset):
    print('pytest fixture inset_with_grid')
    m = inset  #deepcopy(inset)
    cfg = inset.cfg.copy()
    cfg['setup_grid']['grid_file'] = inset.cfg['setup_grid'].pop('output_files').pop('grid_file')
    sd = cfg['setup_grid'].pop('source_data').pop('features_shapefile')
    sd['features_shapefile'] = sd.pop('filename')
    cfg['setup_grid'].update(sd)
    kwargs = get_input_arguments(cfg['setup_grid'], m.setup_grid)
    m.setup_grid(**kwargs)
    return inset


@pytest.fixture(scope="function")
def inset_with_dis(inset_with_grid):
    print('pytest fixture inset_with_dis')
    m = inset_with_grid  #deepcopy(inset_with_grid)
    m.cfg['dis']['remake_arrays'] = True
    m.cfg['dis']['regrid_top_from_dem'] = True
    dis = m.setup_dis()
    return m


def test_load_cfg(cfg):
    cfg = cfg
    assert True


def test_inset(inset):
    assert isinstance(inset, MFnwtModel)


def test_inset_with_grid(inset_with_grid):
    assert inset_with_grid.modelgrid is not None


def test_set_perioddata(inset_with_transient_parent):
    if inset_with_transient_parent is not None:
        m = deepcopy(inset_with_transient_parent)
        perioddata = m.perioddata
        assert pd.Timestamp(perioddata['start_datetime'].values[0]) == \
               pd.Timestamp(m.cfg['model']['start_date_time'])
        assert perioddata['time'].values[-1] == np.sum(m.cfg['dis']['perlen'])


def test_load_grid(inset, inset_with_grid):

    m = inset_with_grid  #deepcopy(inset_with_grid)
    m2 = inset  #deepcopy(inset)
    m2.load_grid(m.cfg['setup_grid']['grid_file'])
    assert m.cfg['grid'] == m2.cfg['grid']


def test_regrid_linear(inset_with_grid):

    from mfsetup.interpolate import regrid
    m = inset_with_grid  #deepcopy(inset_with_grid)
    arr = m.parent.dis.top.array

    # test basic regrid with no masking
    rg1 = m.regrid_from_parent(arr, method='linear')
    rg2 = regrid(arr, m.parent.modelgrid, m.modelgrid,
                 mask1=m.parent_mask,
                 method='linear')
    rg3 = regrid(arr, m.parent.modelgrid, m.modelgrid,
                 method='linear')
    assert np.allclose(rg1, rg2)
    # check that the results from regridding using a window
    # are close to regridding from whole parent grid
    # results won't match exactly, presumably because the
    # simplexes created from the parent grid are unlikely to be the same.
    assert np.allclose(rg1.mean(), rg3.mean(), atol=0.01, rtol=1e-4)


def test_regrid_linear_with_mask(inset_with_grid):

    from mfsetup.interpolate import regrid
    m = inset_with_grid  #deepcopy(inset_with_grid)
    arr = m.parent.dis.top.array

    # pick out some inset cells
    # find locations in parent to make mask
    imask_inset = np.arange(50)
    jmask_inset = np.arange(50)
    xmask_inset = m.modelgrid.xcellcenters[imask_inset, jmask_inset]
    ymask_inset = m.modelgrid.ycellcenters[imask_inset, jmask_inset]
    i = []
    j = []
    for x, y in zip(xmask_inset, ymask_inset):
        ii, jj = m.parent.modelgrid.intersect(x, y)
        i.append(ii)
        j.append(jj)
    #i = np.array(i)
    #j = np.array(j)
    #i, j = m.parent.modelgrid.get_ij(xmask_inset, ymask_inset)
    mask = np.ones(arr.shape)
    mask[i, j] = 0
    mask = mask.astype(bool)

    # test basic regrid with no masking
    rg1 = m.regrid_from_parent(arr, mask=mask, method='linear')
    rg2 = regrid(arr, m.parent.modelgrid, m.modelgrid, mask1=mask,
                 method='linear')
    assert np.allclose(rg1, rg2)


def test_regrid_nearest(inset_with_grid):

    from mfsetup.interpolate import regrid
    m = inset_with_grid  #deepcopy(inset_with_grid)
    arr = m.parent.dis.top.array

    # test basic regrid with no masking
    rg1 = m.regrid_from_parent(arr, method='nearest')
    rg2 = regrid(arr, m.parent.modelgrid, m.modelgrid, method='nearest')
    assert np.allclose(rg1, rg2)


def test_set_lakarr(inset_with_dis):
    m = inset_with_dis
    if 'lak' in m.package_list:
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
    else:
        assert m._lakarr2d.sum() == 0


def test_dis_setup(inset_with_grid):

    m = inset_with_grid  #deepcopy(inset_with_grid)

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
    m.cfg['dis']['remake_top'] = True
    m.cfg['dis']['lenuni'] = 1 # feet
    assert m.cfg['parent']['length_units'] == 'meters'
    assert m.cfg['parent']['time_units'] == 'days'
    assert m.length_units == 'feet'
    dis = m.setup_dis()
    assert np.allclose(dis.top.array.mean() * convert_length_units(1, 2), top_m.mean())
    assert np.allclose(dis.botm.array.mean() * convert_length_units(1, 2), botm_m.mean())


def test_bas_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)

    # test intermediate array creation
    bas = m.setup_bas6()
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


def test_rch_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)

    # test intermediate array creation from rech specified as scalars
    m.cfg['rch']['rech'] = [0.001, 0.002]
    m.cfg['rch']['rech_length_units'] = 'meters'
    m.cfg['rch']['rech_time_units'] = 'days'
    rch = m.setup_rch()
    arrayfiles = m.cfg['intermediate_data']['rech']
    assert len(arrayfiles) == len(m.cfg['rch']['rech'])
    for f in arrayfiles:
        assert os.path.exists(f)

    # test intermediate array creation from source_data
    # (rasters of different shapes)
    inf_array = 'mfsetup/tests/data/plainfieldlakes/source_data/' \
                'net_infiltration__2012-01-01_to_2017-12-31__1066_by_1145__SUM__INCHES_PER_YEAR.tif'

    m.cfg['rch']['source_data']['rech']['filename'] = inf_array
    m.cfg['rch']['rech'] = None
    m.cfg['rch']['source_data']['rech']['length_units'] = 'inches'
    m.cfg['rch']['source_data']['rech']['time_units'] = 'years'
    rch = m.setup_rch()
    arrayfiles = m.cfg['intermediate_data']['rech']
    assert len(arrayfiles) == 1
    for f in arrayfiles:
        assert os.path.exists(f)

    # check that high-K lake recharge was assigned correctly
    highklake_recharge = m.rch.rech.array[0, 0][m.isbc[0] == 2]
    assert np.diff(highklake_recharge).sum() == 0
    for per in range(len(highklake_recharge)):
        val = (m.cfg['lak']['precip'][per] - m.cfg['lak']['evap'][per])  # this won't pass if these aren't in model units
        assert np.allclose(highklake_recharge[per], val)

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


@pytest.mark.parametrize('case', [0, 1])
def test_upw_setup(inset_with_dis, case):

    m = inset_with_dis  #deepcopy(inset_with_dis)

    if case == 0:
        # test intermediate array creation
        m.cfg['upw']['remake_arrays'] = True
        upw = m.setup_upw()
        arrayfiles = m.cfg['intermediate_data']['hk'] + \
                     m.cfg['intermediate_data']['vka']
        for f in arrayfiles:
            assert os.path.exists(f)
        # check that lakes were set up properly
        hiKlakes_value = {}
        hiKlakes_value['hk'] = float(m.cfg['parent']['hiKlakes_value'])
        hiKlakes_value['sy'] = 1.0
        hiKlakes_value['ss'] = 1.0
        for var in ['hk', 'sy', 'ss']:
            arr = upw.__dict__[var].array
            for k, kvar in enumerate(arr):
                if not np.any(m.isbc[k] == 2):
                    assert kvar.max(axis=(0, 1)) < hiKlakes_value[var]
                else:
                    assert np.diff(kvar[m.isbc[k] == 2]).sum() == 0
                    assert kvar[m.isbc[k] == 2][0] == hiKlakes_value[var]
    elif case == 1:
        # test changing vka to anisotropy
        m.cfg['upw']['layvka'] = [1, 1, 1, 1, 1]
        m.cfg['upw']['vka'] = [10, 10, 10, 10, 10]
        upw = m.setup_upw()
        assert np.array_equal(m.upw.layvka.array, np.array([1, 1, 1, 1, 1]))
        assert np.allclose(m.upw.vka.array.max(axis=(1, 2)),
                           np.array([10, 10, 10, 10, 10]))


@pytest.mark.skip("still need to fix TMR")
def test_wel_tmr(inset_with_dis):
    m = inset_with_dis  #deepcopy(inset_with_dis)
    m.setup_upw()

    # test with tmr
    m.cfg['model']['perimeter_boundary_type'] = 'specified flux'
    wel = m.setup_wel()
    wel.write_file()
    assert os.path.exists(m.cfg['wel']['lookup_file'])
    df = pd.read_csv(m.cfg['wel']['lookup_file'])
    bfluxes0 = df.loc[(df.comments == 'boundary_flux') & (df.per == 0)]
    assert len(bfluxes0) == (m.nrow*2 + m.ncol*2) * m.nlay


@pytest.mark.skip("still working wel")
def test_wel_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)deepcopy(inset_with_dis)
    m.setup_upw()

    # test without tmr
    m.cfg['model']['perimeter_boundary_type'] = 'specified head'
    wel = m.setup_wel()
    wel.write_file()
    assert os.path.exists(m.cfg['wel']['lookup_file'])
    df = pd.read_csv(m.cfg['wel']['lookup_file'])
    bfluxes0 = df.loc[(df.comments == 'boundary_flux') & (df.per == 0)]
    assert len(bfluxes0) == 0
    # verify that water use fluxes are negative
    assert wel.stress_period_data[0]['flux'].max() <= 0.
    # verify that water use fluxes are in sp after 0
    # assuming that no wells shut off
    nwells0 = len(wel.stress_period_data[0][wel.stress_period_data[0]['flux'] != 0])
    n_added_wels = len(m.cfg['wel']['added_wells'])
    for k, spd in wel.stress_period_data.data.items():
        if k == 0:
            continue
        assert len(spd) >= nwells0 + n_added_wels

    # test adding a wel from a csv file
    m.cfg['wel']['added_wells'] = 'data/added_wells.csv'
    wel = m.setup_wel()
    assert -2000 in wel.stress_period_data[1]['flux']


@pytest.mark.skip("still working wel")
def test_wel_wu_resampling(inset_with_transient_parent):

    m = inset_with_transient_parent  #deepcopy(inset_with_transient_parent)
    m.cfg['upw']['hk'] = 1
    m.cfg['upw']['vka'] = 1
    m.setup_upw()

    # test without tmr
    m.cfg['model']['perimeter_boundary_type'] = 'specified head'
    m.cfg['wel']['period_stats'] = 'resample'
    wel = m.setup_wel()


def test_mnw_setup(inset_with_dis):

     m = inset_with_dis  #deepcopy(inset_with_dis)
     mnw = m.setup_mnw2()
     mnw.write_file()
     assert True


def test_lak_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)

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
    assert np.any(lak.bdlknc.array == m.cfg['lak']['littoral_leakance'])
    assert np.any(lak.bdlknc.array == m.cfg['lak']['profundal_leakance'])
    assert os.path.exists(m.cfg['lak']['output_files']['lookup_file'])
    assert lak.lakarr.array.sum() > 0
    tabfiles = m.cfg['lak']['tab_files']
    for f in tabfiles:
        assert os.path.exists(os.path.join(m.model_ws, f))
    namfile = os.path.join(m.model_ws, m.namefile)
    if os.path.exists(namfile):
        shutil.copy(namfile, namfile+'.bak')
    m.write_name_file()
    with open(namfile) as src:
        txt = src.read()
    # kludge to deal with ugliness of lake package external file handling
    tab_files_argument = [f.replace(m.model_ws, '').strip('/') for f in tabfiles]
    for f in tab_files_argument:
        assert f in txt
    if os.path.exists(namfile+'.bak'):
        shutil.copy(namfile+'.bak', namfile)

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


def test_nwt_setup(inset):

    m = inset  #deepcopy(inset)
    m.cfg['nwt']['use_existing_file'] = os.path.abspath('mfsetup/tests/data/RGN_rjh_3_23_18.NWT')
    nwt = m.setup_nwt()
    nwt.write_file()
    m.cfg['nwt']['use_existing_file'] = None
    nwt = m.setup_nwt()
    nwt.write_file()


def test_oc_setup(inset):
    m = inset
    oc = m.setup_oc()
    for (kper, kstp), words in oc.stress_period_data.items():
        assert kstp == m.perioddata.loc[kper, 'nstp'] - 1
        assert words == m.perioddata.loc[kper, 'oc']

    # TODO: add datetime comments to OC file


def test_hyd_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)
    hyd = m.setup_hyd()
    hyd.write_file()
    # verify that each head observation is in each layer
    df = pd.DataFrame(hyd.obsdata)
    heads = df.loc[df.arr == b'HD']
    nobs = len(set(heads.hydlbl))
    assert sorted(heads.klay.tolist()) == sorted(list(range(m.nlay)) * nobs)
    #m.cfg['hyd'][]


def test_lake_gag_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)
    lak = m.setup_lak()
    gag = m.setup_gag()
    gag.write_file()
    for f in m.cfg['gag']['ggo_files']:
        assert f in m.output_fnames


def test_chd_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)
    chd = m.setup_chd()
    chd.write_file()
    assert os.path.exists(chd.fn_path)
    assert len(chd.stress_period_data.data.keys()) == len(set(m.cfg['model']['parent_stress_periods']))
    assert len(chd.stress_period_data[0]) == (m.nrow*2 + m.ncol*2 - 4) * m.nlay


def test_sfr_setup(inset_with_dis):

    m = inset_with_dis  #deepcopy(inset_with_dis)
    m.setup_bas6()
    m.setup_sfr()
    assert m.sfr is None


@pytest.mark.skip("still working on wel")
def test_yaml_setup(inset_setup_with_model_run):
    m = inset_setup_with_model_run  #deepcopy(inset_setup_with_model_run)


@pytest.mark.skip("still working on wel")
def test_load(inset_setup_from_yaml, mfnwt_inset_test_cfg_path):
    m = inset_setup_from_yaml  #deepcopy(inset_setup_from_yaml)
    m2 = MFnwtModel.load(mfnwt_inset_test_cfg_path)
    assert m == m2


@pytest.mark.skip("still working on wel")
def test_remake_a_package(inset_setup_from_yaml, mfnwt_inset_test_cfg_path):

    m = inset_setup_from_yaml  #deepcopy(inset_setup)
    m2 = MFnwtModel.load(mfnwt_inset_test_cfg_path, load_only=['dis'])
    lak = m2.setup_lak()
    lak.write_file()


@pytest.fixture(scope="function")
def inset_with_transient_parent(inset_with_grid):
    # TODO: port LPR test case over from CSLS
    return None


@pytest.fixture(scope="session")
def inset_setup_with_model_run(inset_setup_from_yaml):
    m = inset_setup_from_yaml
    try:
        success, buff = m.run_model(silent=False)
    except:
        pass
    assert success, 'model run did not terminate successfully'
    return m
