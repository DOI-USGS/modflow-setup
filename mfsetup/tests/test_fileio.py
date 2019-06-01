import os
import numpy as np
import pytest
from ..fileio import load_array, dump_yml, load_yml, load_modelgrid



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
