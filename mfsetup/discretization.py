"""
Functions related to the Discretization Package.
"""
import time
from re import L

import flopy
import numpy as np
from flopy.mf6.data.mfdatalist import MFList
from scipy import ndimage
from scipy.signal import convolve2d


class ModflowGwfdis(flopy.mf6.ModflowGwfdis):
    def __init__(self, *args, **kwargs):
        flopy.mf6.ModflowGwfdis.__init__(self, *args, **kwargs)

    @property
    def thickness(self):
        return -1 * np.diff(np.stack([self.top.array] +
                                     [b for b in self.botm.array]), axis=0)


def adjust_layers(dis, minimum_thickness=1):
    """
    Adjust bottom layer elevations to maintain a minimum thickness.

    Parameters
    ----------
    dis : flopy.modflow.ModflowDis instance

    Returns
    -------
    new_layer_elevs : ndarray of shape (nlay, ncol, nrow)
        New layer bottom elevations
    """
    nrow, ncol, nlay, nper = dis.parent.nrow_ncol_nlay_nper
    new_layer_elevs = np.zeros((nlay+1, nrow, ncol))
    new_layer_elevs[0] = dis.top.array
    new_layer_elevs[1:] = dis.botm.array

    # constrain everything to model top
    for i in np.arange(1, nlay + 1):
        thicknesses = new_layer_elevs[0] - new_layer_elevs[i]
        too_thin = thicknesses < minimum_thickness * i
        new_layer_elevs[i, too_thin] = new_layer_elevs[0, too_thin] - minimum_thickness * i

    # constrain to underlying botms
    for i in np.arange(1, nlay)[::-1]:
        thicknesses = new_layer_elevs[i] - new_layer_elevs[i + 1]
        too_thin = thicknesses < minimum_thickness
        new_layer_elevs[i, too_thin] = new_layer_elevs[i + 1, too_thin] + minimum_thickness

    return new_layer_elevs[1:]


def deactivate_idomain_above(idomain, packagedata):
    """Sets ibound to 0 for all cells above active SFR cells.

    Parameters
    ----------
    packagedata : MFList, recarray or DataFrame
        SFR package reach data

    Notes
    -----
    This routine updates the ibound array of the flopy.model.ModflowBas6 instance. To produce a
    new BAS6 package file, model.write() or flopy.model.ModflowBas6.write()
    must be run.
    """
    if isinstance(packagedata, MFList):
        packagedata = packagedata.array
    idomain = idomain.copy()
    if isinstance(packagedata, np.recarray):
        packagedata.columns = packagedata.dtype.names
    if 'cellid' in packagedata.columns:
        k, i, j = cellids_to_kij(packagedata['cellid'])
    else:
        k, i, j = packagedata['k'], packagedata['i'], packagedata['j']
    deact_lays = [list(range(ki)) for ki in k]
    for ks, ci, cj in zip(deact_lays, i, j):
        for ck in ks:
            idomain[ck, ci, cj] = 0
    return idomain


def find_remove_isolated_cells(array, minimum_cluster_size=10):
    """Identify clusters of isolated cells in a binary array.
    Remove clusters less than a specified minimum cluster size.
    """
    if len(array.shape) == 2:
        arraylist = [array]
    else:
        arraylist = array

    # exclude diagonal connections
    structure = np.zeros((3, 3))
    structure[1, :] = 1
    structure[:, 1] = 1

    retained_arraylist = []
    for arr in arraylist:

        # for each cell in the binary array arr (i.e. representing active cells)
        # take the sum of the cell and 4 immediate neighbors (excluding diagonal connections)
        # values > 2 in the output array indicate cells with at least two connections
        convolved = convolve2d(arr, structure, mode='same')
        # taking union with (arr == 1) prevents inactive cells from being activated
        atleast_2_connections = (arr == 1) & (convolved > 2)

        # then apply connected component analysis
        # to identify small clusters of isolated cells to exclude
        labeled, ncomponents = ndimage.measurements.label(atleast_2_connections,
                                                          structure=structure)
        retain_areas = [c for c in range(1, ncomponents+1)
                        if (labeled == c).sum() >= minimum_cluster_size]
        retain = np.in1d(labeled.ravel(), retain_areas)
        retained = np.reshape(retain, arr.shape).astype(array.dtype)
        retained_arraylist.append(retained)
    if len(array.shape) == 3:
        return np.array(retained_arraylist, dtype=array.dtype)
    return retained_arraylist[0]


def cellids_to_kij(cellids, drop_inactive=True):
    """Unpack tuples of MODFLOW-6 cellids (k, i, j) to
    lists of k, i, j values; ignoring instances
    where cellid is None (unconnected cells).

    Parameters
    ----------
    cellids : sequence of (k, i, j) tuples
    drop_inactive : bool
        If True, drop cellids == 'none'. If False,
        distribute these to k, i, j.

    Returns
    -------
    k, i, j : 1D numpy arrays of integers
    """
    active = np.array(cellids) != 'none'
    if drop_inactive:
        k, i, j = map(np.array, zip(*cellids[active]))
    else:
        k = np.array([cid[0] if cid != 'none' else None for cid in cellids])
        i = np.array([cid[1] if cid != 'none' else None for cid in cellids])
        j = np.array([cid[2] if cid != 'none' else None for cid in cellids])
    return k, i, j


def create_vertical_pass_through_cells(idomain):
    """Replaces inactive cells with vertical pass-through cells at locations that have an active cell
    above and below by setting these cells to -1.

    Parameters
    ----------
    idomain : np.ndarray with 2 or 3 dimensions. 2D arrays are returned as-is.

    Returns
    -------
    revised : np.ndarray
        idomain with -1s added at locations that were previous <= 0
        that have an active cell (idomain=1) above and below.
    """
    if len(idomain.shape) == 2:
        return idomain
    revised = idomain.copy()
    for i in range(1, idomain.shape[0]-1):
        has_active_above = np.any(idomain[:i] > 0, axis=0)
        has_active_below = np.any(idomain[i+1:] > 0, axis=0)
        bounded = has_active_above & has_active_below
        pass_through = (idomain[i] <= 0) & bounded
        assert not np.any(revised[i][pass_through] > 0)
        revised[i][pass_through] = -1

        # scrub any pass through cells that aren't bounded by active cells
        revised[i][(idomain[i] <= 0) & ~bounded] = 0
    for i in (0, -1):
        revised[i][revised[i] < 0] = 0
    return revised


def fill_empty_layers(array):
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
        for item in sorted(seq)[::-1]:
            if item < value:
                return item

    array = array.copy()
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


def fill_cells_vertically(top, botm):
    """In MODFLOW 6, cells where idomain < 1 are excluded from the solution.
    However, in the botm array, values are needed in overlying cells to
    compute layer thickness (cells with idomain != 1 overlying cells with idomain >= 1 need
    values in botm). Given a 3D numpy array with nan values indicating excluded cells,
    fill in the nans with the overlying values. For example, given the column of cells
    [10, nan, 8, nan, nan, 5, nan, nan, nan, 1], fill the nan values to make
    [10, 10, 8, 8, 8, 5, 5, 5, 5], so that layers 2, 5, and 9 (zero-based)
    all have valid thicknesses (and all other layers have zero thicknesses).

    algorithm:
        * given a top and botm array (top of the model and layer bottom elevations),
          get the layer thicknesses (accounting for any nodata values) idomain != 1 cells in
          thickness array must be set to np.nan
        * set thickness to zero in nan cells take the cumulative sum of the thickness array
          along the 0th (depth) axis, from the bottom of the array to the top
          (going backwards in a depth-positive sense)
        * add the cumulative sum to the array bottom elevations. The backward difference in
          bottom elevations should be zero in inactive cells, and representative of the
          desired thickness in the active cells.
        * append the model bottom elevations (excluded in bottom-up difference)

    Parameters
    ----------
    top : 2D (nrow, ncol) array; model top elevations
    botm : 3D (nlay, nrow, ncol) array; model bottom elevations

    Returns
    -------
    botm : filled botm array
    """
    thickness = get_layer_thicknesses(top, botm)
    assert np.all(np.isnan(thickness[np.isnan(thickness)]))
    thickness[np.isnan(thickness)] = 0
    # cumulative sum from bottom to top
    filled = np.cumsum(thickness[::-1], axis=0)[::-1]
    # add in the model bottom elevations
    # use the minimum values instead of the bottom layer,
    # in case there are nans in the bottom layer
    # include the top, in case there are nans in all botms
    # introducing nans into the top can cause issues
    # with partical vertical LGR
    all_surfaces = np.stack([top] + [arr2d for arr2d in botm])
    filled += np.nanmin(all_surfaces, axis=0)  # botm[-1]
    # append the model bottom elevations
    filled = np.append(filled, [np.nanmin(all_surfaces, axis=0)], axis=0)
    return filled[1:].copy()


def fix_model_layer_conflicts(top_array, botm_array,
                              ibound_array=None,
                              minimum_thickness=3):
    """Compare model layer elevations; adjust layer bottoms downward
    as necessary to maintain a minimum thickness.

    Parameters
    ----------
    top_array : 2D numpy array (nrow * ncol)
        Model top elevations
    botm_array : 3D numpy array (nlay * nrow * ncol)
        Model bottom elevations
    minimum thickness : scalar
        Minimum layer thickness to enforce

    Returns
    -------
    new_botm_array : 3D numpy array of new layer bottom elevations
    """
    top = top_array.copy()
    botm = botm_array.copy()
    nlay, nrow, ncol = botm.shape
    if ibound_array is None:
        ibound_array = np.ones(botm.shape, dtype=int)
    # fix thin layers in the DIS package
    new_layer_elevs = np.empty((nlay + 1, nrow, ncol))
    new_layer_elevs[1:, :, :] = botm
    new_layer_elevs[0] = top
    for i in np.arange(1, nlay + 1):
        active = ibound_array[i - 1] > 0.
        thicknesses = new_layer_elevs[i - 1] - new_layer_elevs[i]
        with np.errstate(invalid='ignore'):
            too_thin = active & (thicknesses < minimum_thickness)
        new_layer_elevs[i, too_thin] = new_layer_elevs[i - 1, too_thin] - minimum_thickness * 1.001
    try:
        assert np.nanmax(np.diff(new_layer_elevs, axis=0)[ibound_array > 0]) * -1 >= minimum_thickness
    except:
        j=2
    return new_layer_elevs[1:]


def get_layer(botm_array, i, j, elev):
    """Return the layers for elevations at i, j locations.

    Parameters
    ----------
    botm_array : 3D numpy array of layer bottom elevations
    i : scaler or sequence
        row index (zero-based)
    j : scaler or sequence
        column index
    elev : scaler or sequence
        elevation (in same units as model)

    Returns
    -------
    k : np.ndarray (1-D) or scalar
        zero-based layer index
    """
# convert scalars to 1D numpy arrays
    def to_array(arg):
        if np.isscalar(arg):
            return np.array([arg])
        else: # if already an np.array, return as is
            return np.array(arg)

    i = to_array(i)
    j = to_array(j)
    nlay = botm_array.shape[0]
    elev = to_array(elev)
    botms = botm_array[:, i, j].tolist()
    layers = np.sum(((botms - elev) > 0), axis=0)
    # force elevations below model bottom into bottom layer
    layers[layers > nlay - 1] = nlay - 1
    layers = np.atleast_1d(np.squeeze(layers))
    if len(layers) == 1:
        layers = layers[0]
    return layers


def verify_minimum_layer_thickness(top, botm, isactive, minimum_layer_thickness):
    """Verify that model layer thickness is equal to or
    greater than a minimum thickness."""
    top = top.copy()
    botm = botm.copy()
    isactive = isactive.copy().astype(bool)
    nlay, nrow, ncol = botm.shape
    all_layers = np.zeros((nlay+1, nrow, ncol))
    all_layers[0] = top
    all_layers[1:] = botm
    isvalid = np.nanmax(np.diff(all_layers, axis=0)[isactive]) * -1 + 1e-4 >= \
              minimum_layer_thickness
    return isvalid


def make_ibound(top, botm, nodata=-9999,
                 minimum_layer_thickness=1,
                 drop_thin_cells=True, tol=1e-4):
    """Make the ibound array that specifies
    cells that will be excluded from the simulation. Cells are
    excluded based on:


    Parameters
    ----------
    model : mfsetup.MFnwtModel model instance

    Returns
    -------
    idomain : np.ndarray (int)

    """
    top = top.copy()
    botm = botm.copy()
    top[top == nodata] = np.nan
    botm[botm == nodata] = np.nan
    criteria = np.isnan(botm)

    # compute layer thicknesses, considering pinched cells (nans)
    b = get_layer_thicknesses(top, botm)
    all_cells_thin = np.all(b < minimum_layer_thickness + tol, axis=0)
    criteria = criteria | np.isnan(b)  # cells without thickness values

    if drop_thin_cells:
        criteria = criteria | all_cells_thin
        #all_layers = np.stack([top] + [b for b in botm])
        #min_layer_thickness = minimum_layer_thickness
        #isthin = np.diff(all_layers, axis=0) * -1 < min_layer_thickness + tol
        #criteria = criteria | isthin
    idomain = np.abs(~criteria).astype(int)
    return idomain


def make_lgr_idomain(parent_modelgrid, inset_modelgrid,
                     ncppl):
    """Inactivate cells in parent_modelgrid that coincide
    with area of inset_modelgrid."""
    if parent_modelgrid.rotation != inset_modelgrid.rotation:
        raise ValueError('LGR parent and inset models must have same rotation.'
                         f'\nParent rotation: {parent_modelgrid.rotation}'
                         f'\nInset rotation: {inset_modelgrid.rotation}'
                         )
    # upper left corner of inset model in parent model
    # use the cell centers, to avoid edge situation
    # where neighboring parent cell is accidentally selected
    x0 = inset_modelgrid.xcellcenters[0, 0]
    y0 = inset_modelgrid.ycellcenters[0, 0]
    pi0, pj0 = parent_modelgrid.intersect(x0, y0, forgive=True)
    # lower right corner of inset model
    x1 = inset_modelgrid.xcellcenters[-1, -1]
    y1 = inset_modelgrid.ycellcenters[-1, -1]
    pi1, pj1 = parent_modelgrid.intersect(x1, y1, forgive=True)
    idomain = np.ones(parent_modelgrid.shape, dtype=int)
    if any(np.isnan([pi0, pj0])):
        raise ValueError(f"LGR model upper left corner {pi0}, {pj0} "
                         "is outside of the parent model domain! "
                         "Check the grid offset and dimensions."
                         )
    if any(np.isnan([pi1, pj1])):
        raise ValueError(f"LGR model lower right corner {pi0}, {pj0} "
                         "is outside of the parent model domain! "
                         "Check the grid offset and dimensions."
                         )
    idomain[0:(np.array(ncppl) > 0).sum(),
            pi0:pi1+1, pj0:pj1+1] = 0
    return idomain


def make_idomain(top, botm, nodata=-9999,
                 minimum_layer_thickness=1,
                 drop_thin_cells=True, tol=1e-4):
    """Make the idomain array for MODFLOW 6 that specifies
    cells that will be excluded from the simulation. Cells are
    excluded based on:
    1) np.nans or nodata values in the botm array
    2) np.nans or nodata values in the top array (applies to the highest cells with valid botm elevations;
    in other words, these cells have no thicknesses)
    3) layer thicknesses less than the specified minimum thickness plus a tolerance (tol)

    Parameters
    ----------
    model : mfsetup.MF6model model instance

    Returns
    -------
    idomain : np.ndarray (int)

    """
    top = top.copy()
    botm = botm.copy()
    top[top == nodata] = np.nan
    botm[botm == nodata] = np.nan
    criteria = np.isnan(botm)

    # compute layer thicknesses, considering pinched cells (nans)
    b = get_layer_thicknesses(top, botm)
    criteria = criteria | np.isnan(b)  # cells without thickness values

    if drop_thin_cells:
        criteria = criteria | (b < minimum_layer_thickness + tol)
        #all_layers = np.stack([top] + [b for b in botm])
        #min_layer_thickness = minimum_layer_thickness
        #isthin = np.diff(all_layers, axis=0) * -1 < min_layer_thickness + tol
        #criteria = criteria | isthin
    idomain = np.abs(~criteria).astype(int)
    return idomain


def get_highest_active_layer(idomain, null_value=-9999):
    """Get the highest active model layer at each
    i, j location, accounting for inactive and
    vertical pass-through cells."""
    idm = idomain.copy()
    # reset all inactive/passthrough values to large positive value
    # for min calc
    idm[idm < 1] = 9999
    highest_active_layer = np.argmin(idm, axis=0)
    # set locations with all inactive cells to null values
    highest_active_layer[(idm == 9999).all(axis=0)] = null_value
    return highest_active_layer


def make_irch(idomain):
    """Make an irch array for the MODFLOW 6 Recharge Package,
    which specifies the highest active model layer at each
    i, j location, accounting for inactive and
    vertical pass-through cells. Set all i, j locations
    with no active layers to 1 (MODFLOW 6 only allows
    valid layer numbers in the irch array).
    """
    irch = get_highest_active_layer(idomain, null_value=-9999)
    # set locations where all layers are inactive back to 0
    irch[irch == -9999] = 0
    irch += 1 # set to one-based
    return irch


def get_layer_thicknesses(top, botm, idomain=None):
    """For each i, j location in the grid, get thicknesses
    between pairs of subsequent valid elevation values. Make
    a thickness array of the same shape as the model grid, assign the
    computed thicknesses for each pair of valid elevations to the
    position of the elevation representing the cell botm. For example,
    given the column of cells [nan nan  8. nan nan nan nan nan  2. nan],
    a thickness of 6 would be assigned to the second to last layer
    (position -2).

    Parameters
    ----------
    top : nrow x ncol array of model top elevations
    botm : nlay x nrow x ncol array of model botm elevations
    idomain : nlay x nrow x ncol array indicating cells to be
        included in the model solution. idomain=0 are converted to np.nans
        in the example column of cells above. (optional)
        If idomain is not specified, excluded cells are expected to be
        designated in the top and botm arrays as np.nans.

    Examples
    --------
    Make a fake model grid with 7 layers, but only top and two layer bottoms specified:
    >>> top = np.reshape([[10]]* 4, (2, 2))
    >>> botm = np.reshape([[np.nan,  8., np.nan, np.nan, np.nan,  2., np.nan]]*4, (2, 2, 7)).transpose(2, 0, 1)
    >>> result = get_layer_thicknesses(top, botm)
    >>> result[:, 0, 0]
    array([nan  2. nan nan nan  6. nan])

    example with all layer elevations specified
    note: this is the same result that np.diff(... axis=0) would produce;
    except positive in the direction of the zero axis
    >>> top = np.reshape([[10]] * 4, (2, 2))
    >>> botm = np.reshape([[9, 8., 8, 6, 3, 2., -10]] * 4, (2, 2, 7)).transpose(2, 0, 1)
    >>> result = get_layer_thicknesses(top, botm)
    array([1.,  1., 0., 2., 3.,  1., 12.])
    """
    print('computing cell thicknesses...')
    t0 = time.time()
    top = top.copy()
    botm = botm.copy()
    if idomain is not None:
        idomain = idomain >= 1
        top[~idomain[0]] = np.nan
        botm[~idomain] = np.nan
    all_layers = np.stack([top] + [b for b in botm])
    thicknesses = np.zeros_like(botm) * np.nan
    nrow, ncol = top.shape
    for i in range(nrow):
        for j in range(ncol):
            cells = all_layers[:, i, j]
            valid_b = list(-np.diff(cells[~np.isnan(cells)]))
            b_ij = np.zeros_like(cells[1:]) * np.nan
            has_top = False
            for k, elev in enumerate(cells):
                if not has_top and not np.isnan(elev):
                    has_top = True
                elif has_top and not np.isnan(elev):
                    b_ij[k-1] = valid_b.pop(0)
            thicknesses[:, i, j] = b_ij
    thicknesses[thicknesses == 0] = 0  # get rid of -0.
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return thicknesses


def weighted_average_between_layers(arr0, arr1, weight0=0.5):
    """"""
    weights = [weight0, 1-weight0]
    return np.average([arr0, arr1], axis=0, weights=weights)


def populate_values(values_dict, array_shape=None):
    """Given an input dictionary with non-consecutive keys,
    make a second dictionary with consecutive keys, with values
    that are linearly interpolated from the first dictionary,
    based on the key values. For example, given {0: 1.0, 2: 2.0},
    {0: 1.0, 1: 1.5, 2: 2.0} would be returned.

    Examples
    --------
    >>> populate_values({0: 1.0, 2: 2.0}, array_shape=None)
    {0: 1.0, 1: 1.5, 2: 2.0}
    >>> populate_values({0: 1.0, 2: 2.0}, array_shape=(2, 2))
    {0: array([[1., 1.],
               [1., 1.]]),
     1: array([[1.5, 1.5],
               [1.5, 1.5]]),
     2: array([[2., 2.],
               [2., 2.]])}
    """
    sorted_layers = sorted(list(values_dict.keys()))
    values = {}
    for i in range(len(sorted_layers[:-1])):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i+1]
        v1 = values_dict[l1]
        v2 = values_dict[l2]
        layers = np.arange(l1, l2+1)
        interp_values = dict(zip(layers, np.linspace(v1, v2, len(layers))))

        # if an array shape is given, fill an array of that shape
        # or reshape to that shape
        if array_shape is not None:
            for k, v in interp_values.items():
                if np.isscalar(v):
                    v = np.ones(array_shape, dtype=float) * v
                else:
                    v = np.reshape(v, array_shape)
                interp_values[k] = v
        values.update(interp_values)
    return values


def voxels_to_layers(voxel_array, z_edges, model_top=None, model_botm=None, no_data_value=0,
                     extend_top=True, extend_botm=False, tol=0.1,
                     minimum_frac_active_cells=0.01):
    """Combine a voxel array (voxel_array), with no-data values and either uniform or non-uniform top
    and bottom elevations, with land-surface elevations (model_top; to form the top of the grid), and
    additional elevation surfaces forming layering below the voxel grid (model_botm).

        * In places where the model_botm elevations are above the lowest voxel elevations,
          the voxels are given priority, and the model_botm elevations reset to equal the lowest voxel elevations
          (effectively giving the underlying layer zero-thickness).
        * Voxels with no_data_value(s) are also given zero-thickness. Typically these would be cells beyond a
          no-flow boundary, or below the depth of investigation (for example, in an airborne electromagnetic survey
          of aquifer electrical resisitivity). The vertical extent of the layering representing the voxel data then spans the highest and lowest valid voxels.
        * In places where the model_top (typically land-surface) elevations are higher than the highest valid voxel,
          the voxel layer can either be extended to the model_top (extend_top=True), or an additional layer
          can be created between the top edge of the highest voxel and model_top (extent_top=False).
        * Similarly, in places where elevations in model_botm are below the lowest valid voxel, the lowest voxel
          elevation can be extended to the highest underlying layer (extend_botm=True), or an additional layer can fill
          the gap between the lowest voxel and highest model_botm (extend_botm=False).

    Parameters
    ----------
    voxel_array : 3D numpy array
        3D array of voxel data- could be zones or actually aquifer properties. Empty voxels
        can be marked with a no_data_value. Voxels are assumed to have the same horizontal
        discretization as the model_top and model_botm layers.
    z_edges : 3D numpy array or sequence
        Top and bottom edges of the voxels (length is voxel_array.shape[0] + 1). A sequence
        can be used to specify uniform voxel edge elevations; non-uniform top and bottom
        elevations can be specified with a 3D numpy array (similar to the botm array in MODFLOW).
    model_top : 2D numpy array
        Top elevations of the model at each row/column location.
    model_botm : 2D or 3D numpy array
        Model layer(s) underlying the voxel grid.
    no_data_value : scalar, optional
        Indicates empty voxels in voxel_array.
    extend_top : bool, optional
        Option to extend the top voxel layer to the model_top, by default True.
    extend_botm : bool, optional
        Option to extend the bottom voxel layer to the next layer below in model_botm,
        by default False.
    tol : float, optional
        Depth tolerance used in comparing the voxel edges to model_top and model_botm.
        For example, if model_top - z_edges[0] is less than tol, the model_top and top voxel
        edge will be considered equal, and no additional layer will be added, regardless of extend_top.
        by default 0.1
    minimum_frac_active_cells : float
        Minimum fraction of cells with a thickness of > 0 for a layer to be retained,
        by default 0.01.

    Returns
    -------
    layers : 3D numpy array of shape (nlay +1, nrow, ncol)
        Model layer elevations (vertical edges of cells), including the model top.


    Raises
    ------
    ValueError
        If z_edges is not 1D or 3D
    """
    model_top = model_top.copy()
    model_botm = model_botm.copy()
    if len(model_botm.shape) == 2:
        model_botm = np.reshape(model_botm, (1, *model_botm.shape))
    if np.any(np.isnan(z_edges)):
        raise NotImplementedError("Nan values in z_edges array not allowed!")
    z_values = np.array(z_edges)[1:]

    # convert nodata values to nans
    hasdata = voxel_array.astype(float).copy()
    hasdata[hasdata == no_data_value] = np.nan
    hasdata[~np.isnan(hasdata)] = 1
    thicknesses = -np.diff(z_edges, axis=0)

    # apply nodata to thicknesses and botm elevations
    if len(z_values.shape) == 3:
        z = hasdata * z_values
        b = hasdata * thicknesses
    elif len(z_values.shape) == 1:
        z = (hasdata.transpose(1, 2, 0) * z_values).transpose(2, 0, 1)
        b = (hasdata.transpose(1, 2, 0) * thicknesses).transpose(2, 0, 1)
    else:
        msg = 'z_edges.shape = {}; z_edges must be a 3D or 1D numpy array'
        raise ValueError(msg.format(z_edges.shape))

    assert np.all(np.isnan(b[np.isnan(b)]))
    b[np.isnan(b)] = 0
    # cumulative sum from bottom to top
    layers = np.cumsum(b[::-1], axis=0)[::-1]
    # add in the model bottom elevations
    # use the minimum values instead of the bottom layer,
    # in case there are nans in the bottom layer
    layers += np.nanmin(z, axis=0)  # botm[-1]
    # append the model bottom elevations
    layers = np.append(layers, [np.nanmin(z, axis=0)], axis=0)

    # set all voxel edges greater than land surface to land surface
    k, i, j = np.where(layers > model_top)
    layers[k, i, j] = model_top[i, j]

    # reset model bottom to lowest valid voxels, where they are lower than model bottom
    lowest_valid_edges = np.nanmin(layers, axis=0)
    for i, layer_botm in enumerate(model_botm):
        loc = layer_botm > lowest_valid_edges
        model_botm[i][loc] = lowest_valid_edges[loc]

    # option to add another layer on top of voxel sequence,
    # if any part of the model top is above the highest valid voxel edges
    if np.any(layers[0] < model_top - tol) and not extend_top:
        layers = np.vstack([np.reshape(model_top, (1, *model_top.shape)), layers])
    # otherwise set the top edges of the voxel sequence to be consistent with model top
    else:
        layers[0] = model_top

    # option to add additional layers below the voxel sequence,
    # if any part of those layers in model botm array are below the lowest valid voxel edges
    if not extend_botm:
        new_botms = [layers]
        for layer_botm in model_botm:
            # get the percentage of active cells with > 0 thickness
            pct_cells = np.sum(layers[-1] > layer_botm + tol)/layers[-1].size
            if pct_cells > minimum_frac_active_cells:
                new_botms.append(np.reshape(layer_botm, (1, *layer_botm.shape)))
            layers = np.vstack(new_botms)
    # otherwise just set the lowest voxel edges to the highest layer in model botm
    # (model botm was already set to lowest valid voxels that were lower than the model botm;
    #  this extends any voxels that were above the model botm to the model botm)
    else:
        layers[-1] = model_botm[0]

    # finally, fill any remaining nans with next layer elevation (going upward)
    # might still have nans in areas where there are no voxel values, but model top and botm values
    botm = fill_cells_vertically(layers[0], layers[1:])
    layers = np.vstack([np.reshape(layers[0], (1, *layers[0].shape)), botm])
    return layers
