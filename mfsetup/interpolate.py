import time
import numpy as np
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools


def get_source_dest_model_xys(source_model, dest_model, source_mask=None):
    """Get the xyz and uvw inputs to the interp_weights function.

    Parameters
    ----------
    source_model : flopy.modeflow.Modflow or flopy.mf6.MFModel instance
    dest_model : mfsetup.MFnwtModel or mfsetup.MF6model instance
    """
    source_modelgrid = source_model.modelgrid
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
    x2, y2 = dest_model.modelgrid.xcellcenters.ravel(), \
             dest_model.modelgrid.ycellcenters.ravel()
    source_model_xy = np.array([x, y]).transpose()
    dest_model_xy = np.array([x2, y2]).transpose()
    return source_model_xy, dest_model_xy


def interp_weights(xyz, uvw, d=2):
    """Speed up interpolation vs scipy.interpolate.griddata (method='linear'),
    by only computing the weights once:
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    Parameters
    ----------
    xyz : ndarray of shape n source points x ndims
        x, y, (z) locations of source data.
    uvw : ndarray of shape n destination points x ndims
        x, y, (z) locations of where source data will be interpolated

    Returns
    -------
    indices : ndarray of shape n destination points x 3
        Index positions in flattened (1D) xyz array
    weights : ndarray of shape n destination points x 3
        Fractional weights for each row position
        in indices. Weights in each row sum to 1
        across the 3 columns.
    """
    print('Calculating interpolation weights...')
    t0 = time.time()
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts):
    """Apply the interpolation weights to a set of values.

    Parameters
    ----------
    values : 1D array of length n source points (same as xyz in interp_weights)
    vtx : indices returned by interp_weights
    wts : weights returned by interp_weights

    Returns
    -------
    interpolated values
    """
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


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
        #nodataval = arr[~mask1][0]
        #arr[~mask1] = np.nan
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
    return arr2

if __name__ == '__main__':
    """Exampmle from stack overflow. In this example, both
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

    np.allclose(interpolate(f, vtx, wts), spint.griddata(xyz, f, uvw))