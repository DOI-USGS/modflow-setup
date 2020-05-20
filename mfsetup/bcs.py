"""
Functions for simple MODFLOW boundary conditions such as ghb, drain, etc.
"""
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from shapely.geometry import Polygon
import rasterio
from rasterstats import zonal_stats
from mfsetup.discretization import get_layer, cellids_to_kij
from mfsetup.grid import rasterize
from mfsetup.units import convert_length_units


def setup_ghb_data(model):

    m = model
    source_data = model.cfg['ghb'].get('source_data').copy()
    # get the GHB cells
    # todo: generalize more of the GHB setup code and move it somewhere else
    if 'shapefile' in source_data:
        shapefile_data = source_data['shapefile']
        key = [k for k in shapefile_data.keys() if 'filename' in k.lower()][0]
        shapefile_name = shapefile_data.pop(key)
        ghbcells = rasterize(shapefile_name, m.modelgrid, **shapefile_data)
    else:
        raise NotImplementedError('Only shapefile input supported for GHBs')

    cond = model.cfg['ghb'].get('cond')
    if cond is None:
        raise KeyError("key 'cond' not found in GHB yaml input. "
                       "Must supply conductance via this key for GHB setup.")

    # sample DEM for minimum elevation in each cell with a GHB
    # todo: GHB: allow time-varying bheads via csv input
    vertices = np.array(m.modelgrid.vertices)[ghbcells.flat > 0, :, :]
    polygons = [Polygon(vrts) for vrts in vertices]
    if 'dem' in source_data:
        key = [k for k in source_data['dem'].keys() if 'filename' in k.lower()][0]
        dem_filename = source_data['dem'].pop(key)
        with rasterio.open(dem_filename) as src:
            meta = src.meta
        all_touched = False
        if meta['transform'][0] > m.modelgrid.delr[0]:
            all_touched = True
        results = zonal_stats(polygons, dem_filename, stats='min',
                              all_touched=all_touched)
        min_elevs = np.ones((m.nrow * m.ncol), dtype=float) * np.nan
        min_elevs[ghbcells.flat > 0] = np.array([r['min'] for r in results])
        units_key = [k for k in source_data['dem'] if 'units' in k]
        if len(units_key) > 0:
            min_elevs *= convert_length_units(source_data['dem'][units_key[0]],
                                              model.length_units)
        min_elevs = np.reshape(min_elevs, (m.nrow, m.ncol))
    else:
        raise NotImplementedError('Must supply DEM to sample for GHB elevations\n'
                                  '(GHB: source_data: dem:)')

    # make a DataFrame with MODFLOW input
    i, j = np.indices((m.nrow, m.ncol))
    df = pd.DataFrame({'per': 0,
                       'k': 0,
                       'i': i.flat,
                       'j': j.flat,
                       'bhead': min_elevs.flat,
                       'cond': cond})
    df.dropna(axis=0, inplace=True)

    # assign layers so that bhead is above botms
    df['k'] = get_layer(model.dis.botm.array, df.i, df.j, df.bhead)
    # remove GHB cells from places where the specified head is below the model
    below_bottom_of_model = df.bhead < model.dis.botm.array[-1, df.i, df.j] + 0.01
    df = df.loc[~below_bottom_of_model].copy()

    # exclude inactive cells
    k, i, j = df.k, df.i, df.j
    if model.version == 'mf6':
        active_cells = model.idomain[k, i, j] >= 1
    else:
        active_cells = model.ibound[k, i, j] >= 1
    df = df.loc[active_cells]
    return df


def get_bc_package_cells(package, exclude_horizontal=True):
    """

    Parameters
    ----------
    package : flopy package instance for boundary condition

    Returns
    -------
    k, i, j : 1D numpy arrays of boundary condition package cell locations
    """
    if package.package_type == 'sfr':
        if package.parent.version == 'mf6':
            k, i, j = cellids_to_kij(package.packagedata.array['cellid'])
        else:
            rd = package.reach_data
            k, i, j = rd['k'], rd['i'], rd['j']
    elif package.package_type == 'lak':
        if package.parent.version == 'mf6':
            connectiondata = package.connectiondata.array
            if exclude_horizontal:
                connectiondata = connectiondata[connectiondata['claktype'] == 'vertical']
            k, i, j = map(np.array, zip(*connectiondata['cellid']))
        else:
            try:
                # todo: figure out why flopy sometimes can't read external files for lakarr
                k, i, j = np.where(package.lakarr.array[0, :, :, :] > 0)
            except:
                k, i, j = np.where(package.parent.lakarr > 0)
    else:
        df = mftransientlist_to_dataframe(package.stress_period_data,
                                          squeeze=True)
        k, i, j = df['k'].values, df['i'].values, df['j'].values
    return k, i, j


def mftransientlist_to_dataframe(mftransientlist, squeeze=True):
    """
    Cast a MFTransientList of stress period data
    into single dataframe containing all stress periods. Output data are
    aggregated (summed) to the model cell level, to avoid
    issues with non-unique row indices.

    Parameters
    ----------
    mftransientlist : flopy.mf6.data.mfdatalist.MFTransientList instance
    squeeze : bool
        Reduce number of columns in dataframe to only include
        stress periods where a variable changes.

    Returns
    -------
    df : dataframe
        Dataframe of shape nrow = ncells, ncol = nvar x nper. If
        the squeeze option is chosen, nper is the number of
        stress periods where at least one cell is different,
        otherwise it is equal to the number of keys in MfList.data.
    """

    data = mftransientlist
    names = ['cellid']
    if isinstance(data.package, flopy.mf6.modflow.ModflowGwfmaw):
        names += ['wellid']

    # monkey patch the mf6 version to behave like the mf2005 version
    #if isinstance(mftransientlist, flopy.mf6.data.mfdatalist.MFTransientList):
    #    mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}

    # find relevant variable names
    # may have to iterate over the first stress period
    #for per in range(data.model.nper):
    for per, spd in data.data.items():
        if spd is not None and hasattr(spd, 'dtype'):
            varnames = list([n for n in spd.dtype.names
                             if n not in ['k', 'i', 'j', 'cellid', 'boundname']])
            break

    # create list of dataframes for each stress period
    # each with index of k, i, j
    dfs = []
    for per, recs in data.data.items():

        if recs is None or recs is 0:
            # add an empty dataframe if a stress period is
            # set to 0 (e.g. no pumping during a predevelopment
            # period)
            columns = names + list(['{}{}'.format(c, per)
                                    for c in varnames])
            dfi = pd.DataFrame(data=None, columns=columns)
            dfi = dfi.set_index(names)
        else:
            dfi = pd.DataFrame.from_records(recs)
            if {'k', 'i', 'j'}.issubset(dfi.columns):
                dfi['cellid'] = list(zip(dfi.k, dfi.i, dfi.j))
                dfi.drop(['k', 'i', 'j'], axis=1, inplace=True)
            dfi = dfi.set_index(names)

            # aggregate (sum) data to model cells
            # because pd.concat can't handle a non-unique index
            # (and modflow input doesn't have a unique identifier at sub-cell level)
            dfg = dfi.groupby(names)
            dfi = dfg.sum()  # aggregate
            #dfi.columns = names + list(['{}{}'.format(c, per) for c in varnames])
            dfi.columns = ['{}{}'.format(c, per) if c in varnames else c for c in dfi.columns]
        dfs.append(dfi)
    df = pd.concat(dfs, axis=1)
    if squeeze:
        keep = []
        for var in varnames:
            diffcols = list([n for n in df.columns if var in n])
            squeezed = squeeze_columns(df[diffcols])
            keep.append(squeezed)
        df = pd.concat(keep, axis=1)
    data_cols = df.columns.tolist()
    df['cellid'] = df.index.tolist()
    idx_cols = ['cellid']
    if isinstance(df.index.values[0], tuple):
        df['k'], df['i'], df['j'] = list(zip(*df['cellid']))
        idx_cols += ['k', 'i', 'j']
    cols = idx_cols + data_cols
    df = df[cols]
    return df


def squeeze_columns(df, fillna=0.):
    """Drop columns where the forward difference
    (along axis 1, the column axis) is 0 in all rows.
    In other words, only retain columns where the data
    changed in at least one row.

    Parameters
    ----------
    df : DataFrame
        Containing homogenous data to be differenced (e.g.,
        just flux values, no id or other ancillary information)
    fillna : float
        Value for nan values in DataFrame
    Returns
    -------
    squeezed : DataFrame

    """
    df.fillna(fillna, inplace=True)
    diff = df.diff(axis=1)
    diff[diff.columns[0]] = 1  # always return the first stress period
    changed = diff.sum(axis=0) != 0
    squeezed = df.loc[:, changed.index[changed]]
    return squeezed