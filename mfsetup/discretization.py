"""
Functions related to the Discretization Package.
"""
import numpy as np


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


def deactivate_idomain_above(idomain, reach_data):
    """Sets ibound to 0 for all cells above active SFR cells.

    Parameters
    ----------
    reach_data : recarray
        SFR package reach data

    Notes
    -----
    This routine updates the ibound array of the flopy.model.ModflowBas6 instance. To produce a
    new BAS6 package file, model.write() or flopy.model.ModflowBas6.write()
    must be run.
    """
    deact_lays = [list(range(i)) for i in reach_data['k']]
    for ks, i, j in zip(deact_lays, reach_data['i'], reach_data['j']):
        for k in ks:
            idomain[k, i, j] = 0
    return idomain


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
    top_array = top_array.copy()
    botm_array = botm_array.copy()
    nlay, nrow, ncol = botm_array.shape
    if ibound_array is None:
        ibound_array = np.ones(botm_array.shape, dtype=int)
    # fix thin layers in the DIS package
    new_layer_elevs = np.empty((nlay + 1, nrow, ncol))
    new_layer_elevs[1:, :, :] = botm_array
    new_layer_elevs[0] = top_array
    for i in np.arange(1, nlay + 1):
        active = ibound_array[i - 1] > 0.
        thicknesses = new_layer_elevs[i - 1] - new_layer_elevs[i]
        with np.errstate(invalid='ignore'):
            too_thin = active & (thicknesses < minimum_thickness)
        new_layer_elevs[i, too_thin] = new_layer_elevs[i - 1, too_thin] - minimum_thickness
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
    isvalid = np.nanmax(np.diff(all_layers, axis=0)[isactive]) * -1 >= minimum_layer_thickness
    return isvalid


