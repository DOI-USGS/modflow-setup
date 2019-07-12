import numpy as np
import xarray as xr
import rasterio
from scipy.interpolate import griddata, interpn
import pytest
from ..grid import MFsetupGrid


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


def test_interp(dem_DataArray, modelgrid):

    import matplotlib.pyplot as plt

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

