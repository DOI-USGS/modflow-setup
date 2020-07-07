import os
import platform

import numpy as np
import pytest

from mfsetup.fileio import (
    dump_yml,
    exe_exists,
    load,
    load_array,
    load_cfg,
    load_modelgrid,
    load_yml,
    which,
)


@pytest.fixture
def module_tmpdir(tmpdir):
    module_tmpdir = os.path.join(tmpdir, os.path.splitext(os.path.split(__file__)[1])[0])
    if not os.path.isdir(module_tmpdir):
        os.makedirs(module_tmpdir)
    return module_tmpdir


@pytest.fixture
def external_files_path(module_tmpdir):
    external_files_path = os.path.join(module_tmpdir, 'external')
    if not os.path.isdir(external_files_path):
        os.makedirs(external_files_path)
    return external_files_path


@pytest.fixture
def data():
    return {'a': np.int64(5),
            'b': np.float64(30.48),
            'c': 5,
            'e': 5.,
            'f': [1, 2, 3],
            'g': [1.,'a', 3.],
            'h': {'a': 1.,
                  'b': 'a',
                  'c': 3.
                  }
            }


def test_dump_yml(data, tmpdir):
    outfile = os.path.join(tmpdir, 'junk.yml')
    dump_yml(outfile, data)
    data2 = load_yml(outfile)
    assert data == data2


def test_load_array(tmpdir):
    nodata = -9999
    size = (100, 100)
    a = np.random.randn(*size)
    a_nodata = a.copy()
    a[0:2, 0:2] = np.nan
    a_nodata[0:2, 0:2] = nodata
    f = '{}/junk.txt'.format(tmpdir)
    np.savetxt(f, a_nodata)
    b = load_array(f, nodata=nodata)
    np.testing.assert_allclose(a, b)


def test_load_grid():
    gridfile = '/Users/aleaf/Documents/CSLS/source/test/data/Transient_MODFLOW-NWT/LPR_parent_grid.yml'
    if os.path.exists(gridfile):
        modelgrid = load_modelgrid(gridfile)
        assert True
    else:
        pass


def test_load_cfg(pfl_nwt_test_cfg_path):
    cfg_pathed = load_cfg(pfl_nwt_test_cfg_path, default_file='/mfnwt_defaults.yml')
    cfg = load(pfl_nwt_test_cfg_path)
    config_file_location = os.path.split(os.path.abspath(pfl_nwt_test_cfg_path))[0]
    assert cfg_pathed['nwt']['use_existing_file'] is None

    p1 = os.path.normpath(cfg_pathed['model']['model_ws'])
    p2 = os.path.normpath(os.path.join(config_file_location, cfg['model']['model_ws']))
    assert p1 == p2

    p1 = os.path.normpath(cfg_pathed['rch']['source_data']['rech']['filenames'][0])
    p2 = os.path.normpath(os.path.join(config_file_location,
                                       cfg['rch']['source_data']['rech']['filenames'][0]))
    assert p1 == p2

    p1 = os.path.normpath(cfg_pathed['hyd']['source_data']['filenames'][0])
    p2 = os.path.normpath(os.path.join(config_file_location, cfg['hyd']['source_data']['filenames'][0]))
    assert p1 == p2


def test_whether_flopy_remembers_external_path(module_tmpdir, external_files_path):
    #m = fm.Modflow('junk', model_ws=module_tmpdir, external_path=external_files_path)
    #nlay, nrow, ncol = 1, 2, 2
    #arr = np.ones((nrow, ncol))
    #layfile = os.path.join(external_files_path, 'junk.dat')
    #np.savetxt(layfile, arr, fmt='%.2f')
    #dis = fm.ModflowDis(m, nrow=nrow, ncol=ncol, top=layfile, botm=[layfile])
    #m.write_input(check=False)

    #m2 = fm.Modflow.load('junk.nam', model_ws=module_tmpdir, external_path=external_files_path)
    #assert True
    pass


def test_which():
    badexe = which('junk')
    assert badexe is None


def test_exe_exists(modflow_executable):
    assert not exe_exists('junk')
    if "linux" in platform.platform().lower():
        assert exe_exists(modflow_executable)
    elif "darwin" in platform.platform().lower():
        assert exe_exists(modflow_executable)
        print('{} exists'.format(modflow_executable))
    else:
        assert exe_exists(modflow_executable)
        print('{} exists'.format(modflow_executable))
