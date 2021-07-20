import os

import numpy as np
import pytest
import rasterio
from scipy.interpolate import griddata, interpn

import xarray as xr
from mfsetup.grid import MFsetupGrid
from mfsetup.interpolate import Interpolator, get_source_dest_model_xys, interp_weights
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
