import itertools
import time

import flopy
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import qhull as qhull


def get_source_dest_model_xys(source_model, dest_model,
                              source_mask=None):
    """Get the xyz and uvw inputs to the interp_weights function.

    Parameters
    ----------
    source_model : flopy.modeflow.Modflow, flopy.mf6.MFModel, or MFsetupGrid instance
    dest_model : mfsetup.MFnwtModel, mfsetup.MF6model instance
    """
    source_modelgrid = source_model
    if isinstance(source_model, flopy.mbase.ModelInterface):
        source_modelgrid = source_model.modelgrid
    dest_modelgrid = dest_model.modelgrid

    if source_mask is None:
        if dest_model.parent_mask.shape == source_modelgrid.xcellcenters.shape:
            source_mask = dest_model.parent_mask
        else:
            source_mask = np.ones(source_modelgrid.xcellcenters.shape, dtype=bool)
    else:
        if source_mask.shape != source_modelgrid.xcellcenters.shape:
            msg = 'source mask of shape {} incompatible with source grid of shape {}'
            raise ValueError(msg.format(source_mask.shape,
                                        source_modelgrid.xcellcenters.shape))
    x = source_modelgrid.xcellcenters[source_mask].flatten()
    y = source_modelgrid.ycellcenters[source_mask].flatten()
    x2, y2 = dest_modelgrid.xcellcenters.ravel(), \
             dest_modelgrid.ycellcenters.ravel()
    source_model_xy = np.array([x, y]).transpose()
    dest_model_xy = np.array([x2, y2]).transpose()
    return source_model_xy, dest_model_xy


def interp_weights(xyz, uvw, d=2, mask=None):
    """Speed up interpolation vs scipy.interpolate.griddata (method='linear'),
    by only computing the weights once:
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    Parameters
    ----------
    xyz : ndarray or tuple
        x, y, z, ... locations of source data.
        (shape n source points x ndims)
    uvw : ndarray or tuple
        x, y, z, ... locations of where source data will be interpolated
        (shape n destination points x ndims)
    d : int
        Number of dimensions (2 for 2D, 3 for 3D, etc.)

    Returns
    -------
    indices : ndarray of shape n destination points x 3
        Index positions in flattened (1D) xyz array
    weights : ndarray of shape n destination points x 3
        Fractional weights for each row position
        in indices. Weights in each row sum to 1
        across the 3 columns.
    """
    print(f'Calculating {d}D interpolation weights...')
    # convert input to ndarrays of the right shape
    uvw = np.array(uvw)
    if uvw.shape[-1] != d:
        uvw = uvw.T
    xyz = np.array(xyz)
    if xyz.shape[-1] != d:
        xyz = xyz.T
    t0 = time.time()
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    # round the weights,
    # so that the weights for each simplex sum to 1
    # sums not exactly == 1 seem to cause spurious values
    weights = np.round(weights, 6)
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return vertices, weights


def interpolate(values, vtx, wts, fill_value='mean'):
    """Apply the interpolation weights to a set of values.

    Parameters
    ----------
    values : 1D array of length n source points (same as xyz in interp_weights)
    vtx : indices returned by interp_weights
    wts : weights returned by interp_weights
    fill_value : float
        Value used to fill in for requested points outside of the convex hull
        of the input points (i.e., those with at least one negative weight).
        If not provided, then the default is nan.
    Returns
    -------
    interpolated values
    """
    result = np.einsum('nj,nj->n', np.take(values, vtx), wts)

    # fill nans that might result from
    # child grid points that are outside of the convex hull of the parent grid
    # and for an unknown reason on the Appveyor Windows environment
    if fill_value == 'mean':
        fill_value = np.nanmean(result)
    if fill_value is not None:
        result[np.any(wts < 0, axis=1)] = fill_value
    return result


def regrid(arr, grid, grid2, mask1=None, mask2=None, method='linear'):
    """Interpolate array values from one model grid to another,
    using scipy.interpolate.griddata.

    Parameters
    ----------
    arr : 2D numpy array
        Source data
    grid : flopy.discretization.StructuredGrid instance
        Source grid
    grid2 : flopy.discretization.StructuredGrid instance
        Destination grid (to interpolate onto)
    mask1 : boolean array
        mask for source grid. Areas that are masked will be converted to
        nans, and not included in the interpolation.
    mask2 : boolean array
        mask denoting active area for destination grid.
        The mean value will be applied to inactive areas if linear interpolation
        is used (not for integer/categorical arrays).
    method : str
        interpolation method ('nearest', 'linear', or 'cubic')
    """
    try:
        from scipy.interpolate import griddata
    except:
        print('scipy not installed\ntry pip install scipy')
        return None

    arr = arr.copy()
    # only include points specified by mask
    x, y = grid.xcellcenters, grid.ycellcenters
    if mask1 is not None:
        mask1 = mask1.astype(bool)
        arr = arr[mask1]
        x = x[mask1]
        y = y[mask1]

    points = np.array([x.ravel(), y.ravel()]).transpose()

    arr2 = griddata(points, arr.flatten(),
                   (grid2.xcellcenters, grid2.ycellcenters),
                   method=method, fill_value=np.nan)

    # fill any areas that are nan
    # (new active area includes some areas not in uwsp model)
    fill = np.isnan(arr2)

    # if new active area is supplied, fill areas outside of that too
    if mask2 is not None:
        mask2 = mask2.astype(bool)
        fill = ~mask2 | fill

    # only fill with mean value if linear interpolation used
    # (floating point arrays)
    if method == 'linear':
        fill_value = np.nanmean(arr2)
        arr2[fill] = np.nanmean(arr2[~fill])
    #else:
    #    arr2[fill] = nodataval
    if arr2.min() < 0:
        j=2
    return arr2


def regrid3d(arr, grid, grid2, mask1=None, mask2=None, method='linear'):
    """Interpolate array values from one model grid to another,
    using scipy.interpolate.griddata.

    Parameters
    ----------
    arr : 3D numpy array
        Source data
    grid : flopy.discretization.StructuredGrid instance
        Source grid
    grid2 : flopy.discretization.StructuredGrid instance
        Destination grid (to interpolate onto)
    mask1 : boolean array
        mask for source grid. Areas that are masked will be converted to
        nans, and not included in the interpolation.
    mask2 : boolean array
        mask denoting active area for destination grid.
        The mean value will be applied to inactive areas if linear interpolation
        is used (not for integer/categorical arrays).
    method : str
        interpolation method ('nearest', 'linear', or 'cubic')

    Returns
    -------
    arr : 3D numpy array
        Interpolated values at the x, y, z locations in grid2.
    """
    try:
        from scipy.interpolate import griddata
    except:
        print('scipy not installed\ntry pip install scipy')
        return None

    assert len(arr.shape) == 3, "input array must be 3d"
    if grid2.botm is None:
        raise ValueError('regrid3d: grid2.botm is None; grid2 must have cell bottom elevations')

    # source model grid points
    px, py, pz = grid.xyzcellcenters

    # pad z cell centers to avoid nans
    # from dest cells that are above or below source cells
    # pad top by top layer thickness
    b1 = grid.top - grid.botm[0]
    top = pz[0] + b1
    # pad botm by botm layer thickness
    if grid.shape[0] > 1:
        b2 = -np.diff(grid.botm[-2:], axis=0)[0]
    else:
        b2 = b1
    botm = pz[-1] - b2
    pz = np.vstack([[top], pz, [botm]])
    nlay, nrow, ncol = pz.shape
    px = np.tile(px, (nlay, 1, 1))
    py = np.tile(py, (nlay, 1, 1))

    # pad the source array (and mask) on the top and bottom
    # so that dest cells above and below the top/bottom cell centers
    # will be within the interpolation space
    # (source x, y, z locations already contain this pad)
    arr = np.pad(arr, pad_width=1, mode='edge')[:, 1:-1, 1:-1]

    # apply the mask
    if mask1 is not None:
        mask1 = mask1.astype(bool)
        # tile the mask to nlay x nrow x ncol
        if len(mask1.shape) == 2:
            mask1 = np.tile(mask1, (nlay, 1, 1))
        # pad the mask vertically to match the source array
        elif (len(mask1.shape) == 3) and (mask1.shape[0] == (nlay - 2)):
            mask1 = np.pad(mask1, pad_width=1, mode='edge')[:, 1:-1, 1:-1]
        arr = arr[mask1]
        px = px[mask1]
        py = py[mask1]
        pz = pz[mask1]
    else:
        px = px.ravel()
        py = py.ravel()
        pz = pz.ravel()
        arr = arr.ravel()

    # dest modelgrid points
    x, y, z = grid2.xyzcellcenters
    try:
        nlay, nrow, ncol = z.shape
    except:
        j=2
    x = np.tile(x, (nlay, 1, 1))
    y = np.tile(y, (nlay, 1, 1))

    # interpolate inset boundary heads from 3D parent head solution
    arr2 = griddata((px, py, pz), arr,
                     (x, y, z), method='linear')
    # get the locations of any bad values
    bk, bi, bj = np.where(np.isnan(arr2))
    bx = x[bk, bi, bj]
    by = y[bk, bi, bj]
    bz = z[bk, bi, bj]
    # tweak the result slightly to resolve any apparent triangulation errors
    fixed = griddata((px, py, pz), arr,
                     (bx+0.0001, by+0.0001, bz+0.0001), method='linear')
    arr2[bk, bi, bj] = fixed

    # fill any remaining areas that are nan
    # (new active area includes some areas not in uwsp model)
    fill = np.isnan(arr2)

    # if new active area is supplied, fill areas outside of that too
    if mask2 is not None:
        mask2 = mask2.astype(bool)
        fill = ~mask2 | fill

    # only fill with mean value if linear interpolation used
    # (floating point arrays)
    if method == 'linear':
        arr2[fill] = np.nanmean(arr2[~fill])
    return arr2


class Interpolator:
    """Speed up barycentric interpolation similar to scipy.interpolate.griddata
    (method='linear'), by computing the weights once and then re-using them for
    successive interpolation with the same source and destination points.

    Parameters
    ----------
    xyz : ndarray or tuple
        x, y, z, ... locations of source data.
        (shape n source points x ndims)
    uvw : ndarray or tuple
        x, y, z, ... locations of where source data will be interpolated
        (shape n destination points x ndims)
    d : int
        Number of dimensions (2 for 2D, 3 for 3D, etc.)
    source_values_mask : boolean array
        Boolean array of same structure as the `source_values` array
        input to the :meth:`~mfsetup.interpolate.Interpolator.interpolate` method,
        with the same number of active values as the size of `xyz`.

    Notes
    -----
    The methods employed are based on this Stack Overflow post:
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    """
    def __init__(self, xyz, uvw, d=2, source_values_mask=None):

        self.xyz = xyz
        self.uvw = uvw
        self.d = d

        # properties
        self._interp_weights = None
        self._source_values_mask = None
        self.source_values_mask = source_values_mask

    @property
    def interp_weights(self):
        """Calculate the interpolation weights."""
        if self._interp_weights is None:
            self._interp_weights = interp_weights(self.xyz, self.uvw, self.d)
        return self._interp_weights

    @property
    def source_values_mask(self):
        return self._source_values_mask

    @source_values_mask.setter
    def source_values_mask(self, source_values_mask):
        if source_values_mask is not None and \
            np.sum(source_values_mask) != len(self.xyz[0]):
            raise ValueError('source_values_mask must contain the same number '
                             'of True (active) values as there are source (xyz) points')
        self._source_values_mask = source_values_mask

    def interpolate(self, source_values, method='linear'):
        """Interpolate values in source_values to the destination points in the *uvw* attribute.
        using modelgrid instances
        attached to the source and destination models.

        Parameters
        ----------
        source_values : ndarray
            Values to be interpolated to destination points. Array must be the same size as
            the number of source points, or the number of active points within source points,
            as defined by the `source_values_mask` array input to the :class:`~mfsetup.interpolate.Interpolator`.
        method : str ('linear', 'nearest')
            Interpolation method. With 'linear' a triangular mesh is discretized around
            the source points, and barycentric weights representing the influence of the *d* +1
            source points on each destination point (where *d* is the number of dimensions),
            are computed. With 'nearest', the input is simply passed to :meth:`scipy.interpolate.griddata`.

        Returns
        -------
        interpolated : 1D numpy array
            Array of interpolated values at the destination locations.
        """
        if self.source_values_mask is not None:
            source_values = source_values.flatten()[self.source_values_mask.flatten()]
        if method == 'linear':
            interpolated = interpolate(source_values, *self.interp_weights,
                                       fill_value=None)
        elif method == 'nearest':
            interpolated = griddata(self.xyz, source_values,
                                    self.uvw, method=method)
        return interpolated


if __name__ == '__main__':
    """Example from stack overflow. In this example, both
    xyz and uvw have points in 3 dimensions. (npoints x ndim)"""
    m, n, d = int(3.5e4), int(3e3), 3
    # make sure no new grid point is extrapolated
    bounding_cube = np.array(list(itertools.product([0, 1], repeat=d)))
    xyz = np.vstack((bounding_cube,
                     np.random.rand(int(m - len(bounding_cube)), d)))
    f = np.random.rand(m)
    g = np.random.rand(m)
    uvw = np.random.rand(n, d)

    vtx, wts = interp_weights(xyz, uvw)

    np.allclose(interpolate(f, vtx, wts), griddata(xyz, f, uvw))
