import sys
sys.path.append('..')
import time
from copy import copy, deepcopy
import shutil
import os
import glob
import pytest
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from mfsetup import MFnwtModel
from ..checks import check_external_files_for_nans
from ..fileio import exe_exists, load_cfg
from ..units import convert_length_units
from ..utils import get_input_arguments


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


def test_perioddata(pfl_nwt):
    model = pfl_nwt
    assert np.array_equal(model.perioddata.steady, [True, False])


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
    m2.load_grid(m.cfg['setup_grid']['grid_file'])
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


def test_bas_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)

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


def test_rch_setup(pfl_nwt_with_dis, project_root_path):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)

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
    inf_array = os.path.join(project_root_path, inf_array)

    m.cfg['rch']['source_data']['rech']['filename'] = inf_array
    m.cfg['rch']['rech'] = None
    m.cfg['rch']['source_data']['rech']['length_units'] = 'inches'
    m.cfg['rch']['source_data']['rech']['time_units'] = 'years'
    rch = m.setup_rch()
    arrayfiles = m.cfg['intermediate_data']['rech']
    for f in arrayfiles:
        assert os.path.exists(f)

    # check that high-K lake recharge was assigned correctly
    highklake_recharge = m.rch.rech.array[:, 0, m.isbc[0] == 2].mean(axis=1)
    print(highklake_recharge)
    print(m.lake_recharge)
    assert np.allclose(highklake_recharge, m.lake_recharge)

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
def test_upw_setup(pfl_nwt_with_dis, case):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)

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
def test_wel_tmr(pfl_nwt_with_dis):
    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.setup_upw()

    # test with tmr
    m.cfg['model']['perimeter_boundary_type'] = 'specified flux'
    wel = m.setup_wel()
    wel.write_file()
    assert os.path.exists(m.cfg['wel']['lookup_file'])
    df = pd.read_csv(m.cfg['wel']['lookup_file'])
    bfluxes0 = df.loc[(df.comments == 'boundary_flux') & (df.per == 0)]
    assert len(bfluxes0) == (m.nrow*2 + m.ncol*2) * m.nlay


def test_wel_setup(pfl_nwt_with_dis_bas6):

    m = pfl_nwt_with_dis_bas6  #deepcopy(pfl_nwt_with_dis)deepcopy(pfl_nwt_with_dis)
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
    n_added_wels = len(m.cfg['wel']['wells'])
    for k, spd in wel.stress_period_data.data.items():
        if k == 0:
            continue
        assert len(spd) >= nwells0 + n_added_wels

    # test adding a wel from a csv file
    m.cfg['wel']['csvfile'] = 'data/added_wells.csv'
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


def test_mnw_setup(pfl_nwt_with_dis):

     m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
     mnw = m.setup_mnw2()
     mnw.write_file()
     assert True


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
    assert np.any(lak.bdlknc.array == m.cfg['lak']['littoral_leakance'])
    assert np.any(lak.bdlknc.array == m.cfg['lak']['profundal_leakance'])
    assert os.path.exists(m.cfg['lak']['output_files']['lookup_file'])
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
    lookup = pd.read_csv(m.cfg['lak']['output_files']['lookup_file'])
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
    m.cfg['nwt']['use_existing_file'] = project_root_path + '/mfsetup/tests/data/RGN_rjh_3_23_18.NWT'
    nwt = m.setup_nwt()
    nwt.write_file()
    m.cfg['nwt']['use_existing_file'] = None
    nwt = m.setup_nwt()
    nwt.write_file()


def test_oc_setup(pfl_nwt):
    m = pfl_nwt
    oc = m.setup_oc()
    for (kper, kstp), words in oc.stress_period_data.items():
        assert kstp == m.perioddata.loc[kper, 'nstp'] - 1
        assert words == m.perioddata.loc[kper, 'oc']

    # TODO: add datetime comments to OC file


def test_hyd_setup(pfl_nwt_with_dis_bas6):

    m = pfl_nwt_with_dis_bas6  #deepcopy(pfl_nwt_with_dis)
    hyd = m.setup_hyd()
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


def test_chd_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    chd = m.setup_chd()
    chd.write_file()
    assert os.path.exists(chd.fn_path)
    assert len(chd.stress_period_data.data.keys()) == len(set(m.cfg['parent']['copy_stress_periods']))
    assert len(chd.stress_period_data[0]) == (m.nrow*2 + m.ncol*2 - 4) * m.nlay


def test_sfr_setup(pfl_nwt_with_dis):

    m = pfl_nwt_with_dis  #deepcopy(pfl_nwt_with_dis)
    m.setup_bas6()
    m.setup_sfr()
    assert m.sfr is None


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


#@pytest.mark.skip("still working on wel")
def test_model_setup_and_run(model_setup_and_run):
    m = model_setup_and_run  #deepcopy(model_setup_and_run)


@pytest.mark.skip("needs some work")
def test_load(pfl_nwt_setup_from_yaml, pfl_nwt_test_cfg_path):
    m = pfl_nwt_setup_from_yaml  #deepcopy(pfl_nwt_setup_from_yaml)
    m2 = MFnwtModel.load(pfl_nwt_test_cfg_path)
    assert m == m2


@pytest.mark.skip("still working on wel")
def test_remake_a_package(pfl_nwt_setup_from_yaml, pfl_nwt_test_cfg_path):

    m = pfl_nwt_setup_from_yaml  #deepcopy(inset_setup)
    m2 = MFnwtModel.load(pfl_nwt_test_cfg_path, load_only=['dis'])
    lak = m2.setup_lak()
    lak.write_file()


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
    cfg = load_cfg(pfl_nwt_test_cfg_path)

    assert len(cfg['model']['packages']) > 0
    kwargs = get_input_arguments(cfg['model'], MFnwtModel)
    m = MFnwtModel(cfg=cfg, **kwargs)
    assert m.package_list == [p for p in m._package_setup_order
                              if p in cfg['model']['packages']]