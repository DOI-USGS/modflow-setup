import numpy as np

from mfsetup.grid import get_ij, national_hydrogeologic_grid_parameters


def compare_float_arrays(a1, a2):
    txt = ""
    for name, where_nan in {'array 1': np.where(np.isnan(a1)),
                            'array 2': np.where(np.isnan(a2))}.items():
        nvalues = len(where_nan)
        if nvalues > 0:
            txt += "{} nans in {}: {}\n".format(nvalues, name, where_nan)
    max_abs_diff = np.nanmax(np.abs(a2 - a1))
    txt += 'Max absolute difference: {}\n'.format(max_abs_diff)
    max_rel_diff = np.nanmax(rpd(a1, a2))
    txt += 'Max relative difference: {}\n'.format(max_rel_diff)
    txt += 'RMSE: {}\n'.format(rms_error(a1, a2))
    return txt


def compare_inset_parent_values(inset_array, parent_array,
                                inset_modelgrid, parent_modelgrid,
                                inset_parent_layer_mapping=None, nodata=-9999,
                                **kwargs):
    """Compare values on different model grids (for example, parent and inset models that overlap),
    by getting the closes parent cell at each inset cell location.

    todo: compare_inset_parent_values: add interpolation for more precise comparison

    Parameters
    ----------
    inset_array : inset model values (ndarray)
    parent_array : parent model values (ndarray)
    inset_modelgrid : flopy modelgrid for inset model
    parent_modelgrid : flopy modelgrid for parent model
    inset_parent_layer_mapping : dict
        Mapping between inset and parent model layers
        {inset model layer: parent model layer}
    nodata : float
        Exclude these values from comparison
    kwargs :
        kwargs to np.allclose

    Returns
    -------
    AssertionError if np.allclose evaluates to False for any layer

    """
    if len(inset_array.shape) < 3:
        inset_array = np.array([inset_array])
    if inset_parent_layer_mapping is None:
        nlay = inset_array.shape[0]
        inset_parent_layer_mapping = dict(zip(list(range(nlay)), list(range(nlay))))
    ix, iy = inset_modelgrid.xcellcenters.ravel(), inset_modelgrid.ycellcenters.ravel()
    pi, pj = get_ij(parent_modelgrid, ix, iy)
    for k, pk in inset_parent_layer_mapping.items():
        parent_vals = parent_array[pk, pi, pj]
        valid = (parent_vals != nodata) & (inset_array[k].ravel() != nodata)
        parent_vals = parent_vals[valid]
        inset_vals = inset_array[k].ravel()[valid]
        assert np.allclose(parent_vals, inset_vals, **kwargs)


def rms_error(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2))


def rpd(v1, v2):
    return np.abs(v1 - v2)/np.nanmean([v1, v2])


def dtypeisinteger(dtype):
    try:
        if np.issubdtype(dtype, np.integer):
            return True
    except:
        pass
    try:
        if isinstance(dtype, int):
            return True
    except:
        pass
    return False


def dtypeisfloat(dtype):
    try:
        if dtype == float:
            return True
    except:
        pass
    try:
        if issubclass(dtype, np.floating):
            return True
    except:
        return False


def issequence(object):
    try:
        iter(object)
    except TypeError as te:
        pass
    return True


def point_is_on_nhg(x, y, offset='edge'):
    """Check if a point is aligend with the National Hydrogeologic Grid
    (https://doi.org/10.5066/F7P84B24).

    Parameters
    ----------
    x : float
        x-coordinate of point
    y : float
        y-coordinate of point
    offset : {'edge', 'center'}
        Check if the point is aligned with an NHG cell edge (corner)
        or center.

    """
    xul = national_hydrogeologic_grid_parameters['xul']
    yul = national_hydrogeologic_grid_parameters['yul']
    dxy = national_hydrogeologic_grid_parameters['dx']
    if offset == 'center':
        xul += dxy/2
        yul -= dxy/2
    # verify that the spacing is a factor of 1000
    if not np.isscalar(x) and not np.allclose(1000 % np.diff(x), 0., rtol=0.001):
        return False
    if not np.isscalar(y) and not np.allclose(1000 % np.diff(y), 0., rtol=0.001):
        return False

    # verify that the first points are on an nhg point
    x0 = x if np.isscalar(x) else x[0]
    y0 = y if np.isscalar(y) else y[0]
    if not np.allclose((x0 - xul) % dxy, 0.) & np.allclose((y0 - yul) % dxy, 0.):
        return False
    return True
