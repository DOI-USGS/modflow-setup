import copy
import os

import fiona
import numpy as np
import pyproj
import pytest
from gisutils import shp2df

from mfsetup.fileio import dump, load_modelgrid
from mfsetup.grid import (
    MFsetupGrid,
    get_nearest_point_on_grid,
    get_point_on_national_hydrogeologic_grid,
)
from mfsetup.testing import point_is_on_nhg

# TODO: add tests for grid.py

@pytest.fixture(scope='module')
def modelgrid():
    return MFsetupGrid(xoff=100., yoff=200., angrot=20.,
                       proj_str='epsg:3070',
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
    "proj_str": "+init=epsg:5070",
    "angrot": 0.0,
    "xul": 178389.0,
    "yul": 1604780.4160000002,
    "lenuni": 1
    }
    grid = MFsetupGrid(**kwargs)
    assert np.allclose(grid.yoffset, grid.yul - grid.nrow * 5280 * .3048)
    assert isinstance(grid.crs, pyproj.crs.CRS)
    assert grid.crs.srs == 'EPSG:5070'


def test_grid_write_shapefile(modelgrid, tmpdir):
    # test writing the grid cells
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

    # test writing the bounding box
    bbox_filename = os.path.join(tmpdir, 'grid_bbox.shp')
    modelgrid.write_shapefile(bbox_filename)
    with fiona.open(filename) as src:
        assert src.crs['init'] == 'epsg:3070'
        assert np.allclose(src.bounds, modelgrid.bounds)


@pytest.mark.parametrize('xul,yul,height,width,dx,dy,rotation,input,expected', ((0, 0, 3, 2, 1, -1, 45,
                                                                                 (0, -0.7),
                                                                                 (0, -np.sqrt(2)/2)),
                                                                                (0, 0, 2, 2, 5, -10, 0,
                                                                                 (4.9, -9),
                                                                                 (2.5, -5)),
                                                                                (0, 0, 3, 2, 5, -10, 0,
                                                                                 (5.1, -21),
                                                                                 (7.5, -25)))
                         )
def test_get_nearest_point_on_grid(xul, yul, height, width, dx, dy, rotation, input, expected):
    x, y = input
    result = get_nearest_point_on_grid(x, y, transform=None,
                                       xul=xul, yul=yul,
                                       dx=dx, dy=dy, rotation=rotation,
                                       offset='center')
    assert np.allclose(result, expected)


@pytest.mark.parametrize('offset', ('center', 'edge'))
def test_get_point_on_national_hydrogeologic_grid(offset):
    x, y = 178389, 938512
    x_nhg, y_nhg = get_point_on_national_hydrogeologic_grid(x, y, offset=offset)
    assert point_is_on_nhg(x_nhg, y_nhg, offset=offset)


def test_load_modelgrid(tmpdir):
    cfg = {'xoff': 100, 'yoff': 100, 'angrot': 20.,
           'proj_str': 'epsg:3070',
           'delr': np.ones(10).tolist(), 'delc': np.ones(2).tolist()}
    grid1 = MFsetupGrid(**cfg)
    grid_file = os.path.join(tmpdir, 'test_grid.json')
    dump(grid_file, cfg)

    grid2 = load_modelgrid(grid_file)
    assert grid1 == grid2
