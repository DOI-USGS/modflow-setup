"""
Functions related to the Discretization Package.
"""
import time
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from flopy.mf6.data.mfdatalist import MFList

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
    cellids = list(packagedata['cellid'])
    deact_lays = [list(range(cellid[0])) for cellid in cellids]
    k, i, j = list(zip(*cellids))
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
        for item in sorted(seq[::-1]):
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
    """In MODFLOW 6, cells where idomain=0 are excluded from the solution.
    However, in the botm array, values are needed in overlying cells to
    compute layer thickness (cells with idomain=0 overlying cells with idomain=0 need
    values in botm). Given a 3D numpy array with nan values indicating excluded cells,
    fill in the nans with the overlying values. For example, given the column of cells
    [10, nan, 8, nan, nan, 5, nan, nan, nan, 1], fill the nan values to make
    [10, 10, 8, 8, 8, 5, 5, 5, 5], so that layers 2, 5, and 9 (zero-based)
    all have valid thicknesses (and all other layers have zero thicknesses).

    algorithm:
    * given a top and botm array (top of the model and layer bottom elevations),
      get the layer thicknesses (accounting for any nodata values)
      idomain=0 cells in thickness array must be set to np.nan
    * set thickness to zero in nan cells
      take the cumulative sum of the thickness array along the 0th (depth) axis,
      from the bottom of the array to the top (going backwards in a depth-positive sense)
    * add the cumulative sum to the array bottom elevations. The backward difference
      in bottom elevations should be zero in inactive cells, and representative of the
      desired thickness in the active cells.
    * append the model bottom elevations (excluded in bottom-up difference)

    Parameters
    ----------
    top : 2D numpy array; model top elevations
    botm : 3D (nlay, nrow, ncol) array; model bottom elevations

    Returns
    -------
    top, botm : filled top and botm arrays
    """
    thickness = get_layer_thicknesses(top, botm)
    assert np.all(np.isnan(thickness[np.isnan(thickness)]))
    thickness[np.isnan(thickness)] = 0
    # cumulative sum from bottom to top
    filled = np.cumsum(thickness[::-1], axis=0)[::-1]
    # add in the model bottom elevations
    filled += botm[-1]
    # append the model bottom elevations
    filled = np.append(filled, botm[-1:], axis=0)
    return filled[0].copy(), filled[1:].copy()


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
    assert np.nanmax(np.diff(new_layer_elevs, axis=0)[ibound_array > 0]) * -1 >= minimum_thickness
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
    def to_array(arg):
        if not isinstance(arg, np.ndarray):
            return np.array([arg])
        else:
            return arg

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


def make_idomain(top, botm, nodata=-9999,
                 minimum_layer_thickness=1,
                 drop_thin_cells=True, tol=1e-4):
    """Make the idomain array for MODFLOW 6 that specifies
    cells that will be excluded from the simulation. Cells are
    excluded based on:
    1) np.nans or nodata values in the botm array
    2) np.nans or nodata values in the top array
       (applies to the highest cells with valid botm elevations;
       in other words, these cells have no thicknesses)
    3) layer thicknesses less than the specified minimum thickness
       plus a tolerance (tol)

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
    """
    print('computing cell thicknesses...')
    t0 = time.time()
    top = top.copy()
    botm = botm.copy()
    if idomain is not None:
        idomain = idomain == 1
        top[~idomain[0]] = np.nan
        botm[~idomain] = np.nan
    all_layers = np.stack([top] + [b for b in botm])
    thicknesses = np.zeros_like(botm) * np.nan
    nrow, ncol = top.shape
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) == (2, 2):
                j=2
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
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return thicknesses




