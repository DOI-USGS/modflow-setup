import collections
import inspect
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from .gis import df2shp


def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out], thresh)
    return out

def update(d, u):
    """Recursively update a dictionary of varying depth
    d with items from u.
    from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    if d is None:
        d = {}
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_input_arguments(kwargs, function, warn=True):
    """Return subset of keyword arguments in kwargs dict
    that are valid parameters to a function or method.

    Parameters
    ----------
    kwargs : dict (parameter names, values)
    function : function of class method

    Returns
    -------
    input_kwargs : dict
    """
    print('\narguments to {}:'.format(function.__qualname__))
    params = inspect.signature(function)
    input_kwargs = {}
    not_arguments = {}
    for k, v in kwargs.items():
        if k in params.parameters:
            input_kwargs[k] = v
            print('{}: {}'.format(k, v))
        else:
            not_arguments[k] = v
    if warn:
        print('\nother arguments:')
        for k, v in not_arguments.items():
            print('{}: {}'.format(k, v))
    print('\n')
    return input_kwargs


def get_grid_bounding_box(modelgrid):
    """Get bounding box of potentially rotated modelgrid
    as a shapely Polygon object.

    Parameters
    ----------
    modelgrid : flopy.discretization.StructuredGrid instance
    """
    mg = modelgrid
    #x0 = mg.xedge[0]
    #x1 = mg.xedge[-1]
    #y0 = mg.yedge[0]
    #y1 = mg.yedge[-1]

    x0 = mg.xyedges[0][0]
    x1 = mg.xyedges[0][-1]
    y0 = mg.xyedges[1][0]
    y1 = mg.xyedges[1][-1]

    # upper left point
    #x0r, y0r = mg.transform(x0, y0)
    x0r, y0r = mg.get_coords(x0, y0)

    # upper right point
    #x1r, y1r = mg.transform(x1, y0)
    x1r, y1r = mg.get_coords(x1, y0)

    # lower right point
    #x2r, y2r = mg.transform(x1, y1)
    x2r, y2r = mg.get_coords(x1, y1)

    # lower left point
    #x3r, y3r = mg.transform(x0, y1)
    x3r, y3r = mg.get_coords(x0, y1)

    return Polygon([(x3r, y3r), (x0r, y0r),
                    (x1r, y1r), (x2r, y2r),
                    (x3r, y3r)])


def write_bbox_shapefile(modelgrid, outshp):
    outline = get_grid_bounding_box(modelgrid)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=modelgrid.epsg)


def regrid(arr, sr, sr2, mask1=None, mask2=None, method='linear'):
    """Interpolate array values from one model grid to another,
    using scipy.interpolate.griddata.

    Parameters
    ----------
    arr : 2D numpy array
        Source data
    sr : flopy.utils.SpatialReference instance
        Source grid
    sr2 : flopy.utils.SpatialReference instance
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
    x, y = sr.xcentergrid, sr.ycentergrid
    if mask1 is not None:
        mask1 = mask1.astype(bool)
        #nodataval = arr[~mask1][0]
        #arr[~mask1] = np.nan
        arr = arr[mask1]
        x = x[mask1]
        y = y[mask1]

    points = np.array([x.ravel(), y.ravel()]).transpose()

    arr2 = griddata(points, arr.flatten(),
                 (sr2.xcentergrid, sr2.ycentergrid),
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


def fill_layers(array):
    """Fill empty layers in a 3D array by linearly interpolating
    between the values above and below. Layers are defined
    as empty if they contain all nan values. In the example of
    model layer elevations, this would create equal layer thicknesses
    between layer surfaces with values.

    Parameters
    ----------
    array : 3D numpy.ndarray

    Returns
    -------
    filled : ndarray of same shape as array
    """
    def get_next_below(seq, value):
        for item in sorted(seq):
            if item > value:
                return item

    def get_next_above(seq, value):
        for item in sorted(seq[::-1]):
            if item < value:
                return item

    nlay = array.shape[0]
    layers_with_values = [k for k in range(nlay) if not np.all(np.isnan(array[k]), axis=(0, 1))]
    empty_layers = [k for k in range(nlay) if k not in layers_with_values]

    for k in empty_layers:
        nextabove = get_next_above(layers_with_values, k)
        nextbelow = get_next_below(layers_with_values, k)

        # linearly interpolate layer values between next layers
        # above and below that have values
        # (in terms of elevation
        n = nextbelow - nextabove
        diff = (array[nextbelow] - array[nextabove]) / n
        for i in range(k, nextbelow):
            array[i] = array[i - 1] + diff
        k = i
    return array
