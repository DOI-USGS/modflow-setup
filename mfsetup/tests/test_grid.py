import copy
import os
import numpy as np
import fiona
import pytest
from gisutils import shp2df
from ..grid import MFsetupGrid

# TODO: add tests for grid.py

@pytest.fixture(scope='module')
def modelgrid():
    return MFsetupGrid(xoff=100., yoff=200., angrot=20.,
                       proj4='epsg:3070',
                       delr=np.ones(10), delc=np.ones(2))


def test_grid_eq(modelgrid):
    grid2 = copy.deepcopy(modelgrid)
    assert modelgrid == grid2


def test_grid_init():
    ncol = 394
    nrow = 414
    kwargs = {
    "delc": np.ones(nrow) * 5280.0 * .3048,
    "delr": np.ones(ncol) * 5280.0 * .3048,
    "epsg": 5070,
    "proj4": "+init=epsg:5070",
    "angrot": 0.0,
    "xul": 178389.0,
    "yul": 1604780.4160000002,
    "lenuni": 1
    }
    grid = MFsetupGrid(**kwargs)
    assert np.allclose(grid.yoffset, grid.yul - grid.nrow * 5280 * .3048)


def test_grid_write_shapefile(modelgrid, tmpdir):
    filename = os.path.join(tmpdir, 'grid.shp')
    modelgrid.write_shapefile(filename)
    with fiona.open(filename) as src:
        assert src.crs['init'] == 'epsg:3070'
        assert np.allclose(src.bounds, modelgrid.bounds)
    df = shp2df(filename)
    i, j = np.indices((modelgrid.nrow, modelgrid.ncol))
    assert np.array_equal(np.arange(len(df), dtype=int), df.node.values)
    assert np.array_equal(i.ravel(), df.i.values)
    assert np.array_equal(j.ravel(), df.j.values)


