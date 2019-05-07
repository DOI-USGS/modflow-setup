import numpy as np
from ..fileio import load_array


def test_load_array(tmpdir):
    size = (100, 100)
    a = np.random.randn(*size)
    f = '{}/junk.txt'.format(tmpdir)
    np.savetxt(f, a)
    b = load_array(f)
    np.testing.assert_allclose(a, b)