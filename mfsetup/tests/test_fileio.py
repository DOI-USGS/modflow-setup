import os
import numpy as np
import pytest
from ..fileio import load, load_array, dump_yml, load_yml, load_modelgrid, load_cfg


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
    size = (100, 100)
    a = np.random.randn(*size)
    f = '{}/junk.txt'.format(tmpdir)
    np.savetxt(f, a)
    b = load_array(f)
    np.testing.assert_allclose(a, b)


def test_load_grid():
    gridfile = '/Users/aleaf/Documents/CSLS/source/test/data/Transient_MODFLOW-NWT/LPR_parent_grid.yml'
    if os.path.exists(gridfile):
        modelgrid = load_modelgrid(gridfile)
        assert True
    else:
        pass


def test_load_cfg(mfnwt_inset_test_cfg_path):
    cfg_pathed = load_cfg(mfnwt_inset_test_cfg_path)
    cfg = load(mfnwt_inset_test_cfg_path)
    config_file_location = os.path.split(os.path.abspath(mfnwt_inset_test_cfg_path))[0]
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

