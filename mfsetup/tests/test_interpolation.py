import os

import numpy as np
import pytest
import rasterio
from scipy.interpolate import griddata, interpn

import xarray as xr
from mfsetup.grid import MFsetupGrid
from mfsetup.interpolate import (
    Interpolator,
    get_source_dest_model_xys,
    interp_weights,
    interp_weights_kdtree,
    interpolate,
)
from mfsetup.testing import compare_float_arrays


@pytest.fixture
def dem_DataArray(demfile,):
    with rasterio.open(demfile) as src:
        meta = src.meta
        data = src.read(1)

    dx, rot, xul, rot, dy, yul = src.transform[:6]
    x = np.arange(src.width) * dx + xul
    y = np.arange(src.height) * dy + yul
    data[data == meta['nodata']] = np.nan
    data = xr.DataArray(data, coords={'x': x,
                                      'y': y},
                        dims=('y', 'x'))
    return data


@pytest.fixture
def modelgrid(dem_DataArray):
    da = dem_DataArray
    nrow, ncol = da.shape
    res = np.diff(da.x.values)[0]
    kwargs = {}
    kwargs['xoff'] = da.x.values[20]
    kwargs['yoff'] = da.y.values[-20]
    kwargs['delr'] = np.ones(ncol-40) * res
    kwargs['delc'] = np.ones(nrow-40) * res
    return MFsetupGrid(**kwargs)


@pytest.mark.skip(reason="incomplete")
def test_interp(dem_DataArray, modelgrid):

    da = dem_DataArray
    mg = modelgrid
    x = mg.xcellcenters.ravel()
    y = mg.ycellcenters.ravel()

    # interpolate to the model grid points from xarray dataset points
    xDA = xr.DataArray(x, dims='z')  # model grid points as xarray dataset
    yDA = xr.DataArray(y, dims='z')
    dsi = da.interp(x=xDA, y=yDA, method='linear')
    dsi = np.reshape(dsi.values, (mg.nrow, mg.ncol))

    # interpolate to the model grid points using griddata
    X, Y = np.meshgrid(da.x.values, da.y.values)
    X = X.ravel()
    Y = Y.ravel()
    xyz = np.array([X, Y]).transpose() # source data points are xarray datapoints
    uvw = np.array([x, y]).transpose()
    gdi = griddata(xyz, da.values.ravel(), uvw, method='linear')
    gdi = np.reshape(gdi, (mg.nrow, mg.ncol))

    # interpolate to the model grid points using interpn
    points = (da.x.values, da.y.values[::-1])
    itni = interpn(points, np.flipud(da.values.transpose()), uvw, method='linear')
    itni = np.reshape(itni, (mg.nrow, mg.ncol))
    assert True


# even though test runs locally on Windows 10, and on Travis
@pytest.mark.xfail(os.environ.get('CI', 'False').lower() == 'true',
                   reason="")
def test_interp_weights(pfl_nwt_with_grid):
    m = pfl_nwt_with_grid
    parent_xy, inset_xy = get_source_dest_model_xys(m.parent,
                                                    m)
    inds, weights = interp_weights(parent_xy, inset_xy)
    assert np.all(weights >= 0), print(np.where(weights < 0))


def test_interp_weights_3d_rescaling():
    """The d=3 branch of interp_weights rescales the coordinates
    before building the Delaunay triangulation, mirroring
    scipy.interpolate.griddata(rescale=True). This keeps the
    triangulation well-conditioned for point clouds with extreme
    aspect ratios, e.g. a model grid whose vertical (z) extent is
    small relative to its horizontal (x, y) extent.

    Test verifies that 3D weights reproduce griddata with rescale=True
    """
    rng = np.random.default_rng(7)

    # source point cloud with a severe aspect ratio:
    # x, y span tens of thousands of units while z spans ~1,
    # with jitter so the cloud is irregular like real model layers.
    nx, ny, nz = 12, 12, 6
    x = np.linspace(0, 80000, nx)
    y = np.linspace(0, 80000, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(float)
    xyz[:, 0] += rng.uniform(-3000, 3000, len(xyz))
    xyz[:, 1] += rng.uniform(-3000, 3000, len(xyz))
    xyz[:, 2] += rng.uniform(-0.05, 0.05, len(xyz))

    # a non-linear field, so the choice of simplex (and therefore the
    # rescaling) actually affects the interpolated result
    def field(pts):
        return (np.sin(pts[:, 0] / 20000.0) * np.cos(pts[:, 1] / 20000.0)
                + 5.0 * pts[:, 2])

    source_values = field(xyz)

    # destination points well inside the convex hull
    n_dest = 300
    uvw = np.column_stack([
        rng.uniform(x[2], x[-3], n_dest),
        rng.uniform(y[2], y[-3], n_dest),
        rng.uniform(z[1], z[-2], n_dest),
    ])

    vtx, weights = interp_weights(xyz, uvw, d=3)
    result = interpolate(source_values, vtx, weights, fill_value=None)

    # barycentric weights in each row should sum to 1
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)

    gd_rescale = griddata(xyz, source_values, uvw,
                          method='linear', rescale=True)
    gd_plain = griddata(xyz, source_values, uvw,
                        method='linear', rescale=False)
    valid = ~np.isnan(gd_rescale) & ~np.isnan(result)
    assert valid.sum() > 0

    # the rescaled interp_weights result must match griddata(rescale=True)
    np.testing.assert_allclose(result[valid], gd_rescale[valid],
                               rtol=1e-4, atol=1e-4)

    # verify that this is a meaningful assertion: for this
    # high-aspect-ratio cloud, rescaling actually changes the answer,
    # so matching rescale=True distinguishes the new code from the old
    # unscaled behavior.
    both = valid & ~np.isnan(gd_plain)
    assert np.nanmax(np.abs(gd_rescale[both] - gd_plain[both])) > 1e-3


def test_interp_weights_kdtree():
    """interp_weights_kdtree returns indices and normalized
    inverse-distance weights from the k nearest neighbors
    (default k = d + 1). Weights in each row must sum to 1, be
    non-negative, and index into the flattened source array.
    """
    rng = np.random.default_rng(3)
    nx, ny, nz = 10, 10, 5
    x = np.linspace(0, 40000, nx)
    y = np.linspace(0, 40000, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(float)

    n_dest = 100
    uvw = np.column_stack([
        rng.uniform(x[1], x[-2], n_dest),
        rng.uniform(y[1], y[-2], n_dest),
        rng.uniform(z[1], z[-2], n_dest),
    ])

    indices, weights = interp_weights_kdtree(xyz, uvw, d=3)

    # default k is d + 1
    assert indices.shape == (n_dest, 4)
    assert weights.shape == (n_dest, 4)
    # weights are a valid convex combination
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(weights >= 0)
    # indices address the flattened source array
    assert indices.min() >= 0
    assert indices.max() < len(xyz)


def test_interp_weights_kdtree_exact_match():
    """When a destination point coincides exactly with a source point
    (distance 0), that source point should receive all the weight so
    the interpolated value equals the source value there.
    """
    x = np.linspace(0, 5, 6)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 5, 6)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(float)

    def field(pts):
        return pts[:, 0] + 10 * pts[:, 1] + 100 * pts[:, 2]

    source_values = field(xyz)

    # destination points placed exactly on a subset of source points
    exact_idx = [0, 50, 123, len(xyz) - 1]
    uvw = xyz[exact_idx]

    indices, weights = interp_weights_kdtree(xyz, uvw, d=3)
    # the row weights still sum to 1
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    # each row should put weight 1 on exactly one neighbor
    assert np.allclose(weights.max(axis=1), 1.0)
    assert np.allclose((weights > 0).sum(axis=1), 1)

    result = interpolate(source_values, indices, weights, fill_value=None)
    np.testing.assert_allclose(result, source_values[exact_idx])


def test_interp_weights_kdtree_k1():
    """With k=1 the result is nearest-neighbor: a single index per
    destination point and a weight of 1.
    """
    rng = np.random.default_rng(5)
    xyz = rng.uniform(0, 100, size=(200, 3))

    def field(pts):
        return pts.sum(axis=1)

    source_values = field(xyz)
    uvw = rng.uniform(10, 90, size=(40, 3))

    indices, weights = interp_weights_kdtree(xyz, uvw, d=3, k=1)
    assert indices.shape == (40, 1)
    assert weights.shape == (40, 1)
    np.testing.assert_array_equal(weights, np.ones((40, 1)))

    # result should equal the value at the single nearest source point
    result = interpolate(source_values, indices, weights, fill_value=None)
    np.testing.assert_allclose(result, source_values[indices[:, 0]])


def test_interpolator_kdtree_weight_method():
    """The Interpolator routes to interp_weights_kdtree when
    weight_method='kdtree', caches the result, and interpolates.
    """
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    z = np.linspace(0, 10, 11)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(float)

    def field(pts):
        return 2.0 * pts[:, 0] + pts[:, 1] - pts[:, 2]

    source_values = field(xyz)

    rng = np.random.default_rng(9)
    uvw = np.column_stack([
        rng.uniform(1, 9, 60),
        rng.uniform(1, 9, 60),
        rng.uniform(1, 9, 60),
    ])

    interp = Interpolator(xyz, uvw, d=3, weight_method='kdtree', k=8)
    indices, weights = interp.interp_weights
    assert indices.shape == (60, 8)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    result = interp.interpolate(source_values, method='linear')
    np.testing.assert_allclose(result, field(uvw), rtol=0.1, atol=1.0)


def test_interpolator(dem_DataArray, modelgrid):

    da = dem_DataArray
    mg = modelgrid
    x = mg.xcellcenters.ravel()
    y = mg.ycellcenters.ravel()

    X, Y = np.meshgrid(da.x.values, da.y.values)
    X = X.ravel()
    Y = Y.ravel()
    xyz = np.array([X, Y]).transpose() # source data points are xarray datapoints
    uvw = np.array([x, y]).transpose()

    for method in 'linear', 'nearest':
        gdi = griddata(xyz, da.values.ravel(), uvw, method=method)

        interp = Interpolator(xyz, uvw, d=2)
        results = interp.interpolate(da.values.ravel(), method=method)

        assert np.allclose(gdi[~np.isnan(gdi)], results[~np.isnan(results)])


# even though test runs locally on Windows 10, and on Travis
@pytest.mark.xfail(os.environ.get('CI', 'False').lower() == 'true',
                   reason="")
def test_regrid_linear(pfl_nwt_with_grid):

    from mfsetup.interpolate import regrid
    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    arr = m.parent.dis.top.array

    # test basic regrid with no masking
    rg1 = m.regrid_from_parent(arr, method='linear')
    rg2 = regrid(arr, m.parent.modelgrid, m.modelgrid,
                 mask1=m.parent_mask,
                 method='linear')
    rg3 = regrid(arr, m.parent.modelgrid, m.modelgrid,
                 method='linear')
    err_msg = compare_float_arrays(rg1, rg2)
    np.testing.assert_allclose(rg1, rg2, atol=0.01, rtol=1e-4,
                               err_msg=err_msg)
    # check that the results from regridding using a window
    # are close to regridding from whole parent grid
    # results won't match exactly, presumably because the
    # simplexes created from the parent grid are unlikely to be the same.
    err_msg = compare_float_arrays(rg1, rg2)
    np.testing.assert_allclose(rg1.mean(), rg3.mean(), atol=0.01, rtol=1e-4)


def test_regrid_linear_with_mask(pfl_nwt_with_grid):

    from mfsetup.interpolate import regrid
    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    arr = m.parent.dis.top.array

    # pick out some pfl_nwt cells
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
    np.testing.assert_allclose(rg1, rg2)


def test_regrid_nearest(pfl_nwt_with_grid):

    from mfsetup.interpolate import regrid
    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    arr = m.parent.dis.top.array

    # test basic regrid with no masking
    rg1 = m.regrid_from_parent(arr, method='nearest')
    rg2 = regrid(arr, m.parent.modelgrid, m.modelgrid, method='nearest')
    np.testing.assert_allclose(rg1, rg2)
