"""
Functions for simple MODFLOW boundary conditions such as ghb, drain, etc.
"""
import shutil

import flopy
import numpy as np
import pandas as pd

fm = flopy.modflow
import pyproj
import rasterio
from gisutils import project
from rasterstats import zonal_stats
from shapely.geometry import Polygon

from mfsetup.discretization import cellids_to_kij, get_layer
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

        # reproject the polygons to the dem crs if needed
        try:
            from gisutils import get_authority_crs
            dem_crs = get_authority_crs(src.crs)
        except:
            dem_crs = pyproj.crs.CRS.from_user_input(src.crs)
        if dem_crs != m.modelgrid.crs:
            polygons = project(polygons, m.modelgrid.crs, dem_crs)

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

        if recs is None or recs == 0:
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
            if 'cellid' in dfi.columns:
                dfi['cellid'] = dfi['cellid'].astype(str)
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
    df.index = [eval(s) for s in df.index]
    df['cellid'] = df.index.tolist()
    idx_cols = ['cellid']
    if isinstance(df.index.values[0], tuple):
        df['k'], df['i'], df['j'] = list(zip(*df['cellid']))
        idx_cols += ['k', 'i', 'j']
    cols = idx_cols + data_cols
    df = df[cols]
    return df


def remove_inactive_bcs(pckg):
    """Remove boundary conditions from cells that are inactive.

    Parameters
    ----------
    model : flopy model instance
    pckg : flopy package instance
    """
    model = pckg.parent
    if model.version == 'mf6':
        active = model.dis.idomain.array > 0
    else:
        active = model.bas6.ibound.array > 0
    spd = pckg.stress_period_data.data

    new_spd = {}
    for per, rec in spd.items():
        if 'cellid' in rec.dtype.names:
            k, i, j = zip(*rec['cellid'])
        else:
            k, i, j = zip(*rec[['k', 'i', 'j']])
        new_spd[per] = rec[active[k, i, j]]
    pckg.stress_period_data = new_spd


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


def setup_flopy_stress_period_data(model, package, data, flopy_package_class,
                                   variable_column='q', external_files=True,
                                   external_filename_fmt="{}_{:03d}.dat"):
    """Set up stress period data input for flopy, from a DataFrame
    of stress period data information.

    Parameters
    ----------
    package : str
        Flopy package abbreviation (e.g. 'chd')
    data : DataFrame
        Pandas DataFrame of stress period data with the following columns:

        ================= ==============================
        per               zero-based model stress period
        k                 zero-based model layer
        i                 zero-based model row
        j                 zero-based model column
        <variable_column> stress period input values
        boundname         modflow-6 boundname (optional)
        ================= ==============================

    external_files : bool
        Whether or not to set up external files
    external_filename_fmt : format str
        Format for external file names. For example, "{}_{:03d}.dat"
        would produce "wel_000.dat" for the package='wel' and stress period 0.

    Returns
    -------
    spd : dict
        If external_files=False, spd is populated with numpy recarrays of the
        stress period data. With external files, the data are written to external
        files, which are then passed to flopy via the model configuration (cfg)
        dictonary, and spd is empty.
    """

    df = data
    # set up stress_period_data
    if external_files:
        # get the file path (allowing for different external file locations, specified name format, etc.)
        filepaths = model.setup_external_filepaths(package, 'stress_period_data',
                                                   filename_format=external_filename_fmt,
                                                   file_numbers=sorted(df.per.unique().tolist()))
        # convert to one-based
        df.rename(columns={'k': '#k'}, inplace=True)
        df['#k'] += 1
        df['i'] += 1
        df['j'] += 1

    spd = {}
    period_groups = df.groupby('per')
    for kper in range(model.nper):
        if kper in period_groups.groups:
            group = period_groups.get_group(kper)
            group.drop('per', axis=1, inplace=True)
            if external_files:
                if model.version == 'mf6':
                    group.to_csv(filepaths[kper]['filename'], index=False, sep=' ', float_format='%g')
                    # make a copy for the intermediate data folder, for consistency with mf-2005
                    shutil.copy(filepaths[kper]['filename'], model.cfg['intermediate_data']['output_folder'])
                else:
                    group.to_csv(filepaths[kper], index=False, sep=' ', float_format='%g')

            else:
                if model.version == 'mf6':
                    kspd = flopy_package_class.stress_period_data.empty(model,
                                                                        len(group),
                                                                        boundnames=True)[0]
                    kspd['cellid'] = list(zip(group.k, group.i, group.j))
                    kspd[variable_column] = group[variable_column]
                    if 'boundname' in group.columns:
                        kspd['boundname'] = group['boundname']
                else:
                    kspd = flopy_package_class.get_empty(len(group))
                    kspd['k'] = group['k']
                    kspd['i'] = group['i']
                    kspd['j'] = group['j']

                    # special case of start and end values for MODFLOW-2005 CHDs
                    # assign starting and ending head values for each period
                    # starting chd is parent values for previous period
                    # ending chd is parent values for that period
                    if package.lower() == 'chd':
                        if kper == 0:
                            kspd['shead'] = group[variable_column]
                            kspd['ehead'] = group[variable_column]
                        else:
                            kspd['ehead'] = group[variable_column]
                            # populate sheads with eheads from the same cells
                            # if the cell didn't exist previously
                            # set shead == ehead
                            # dict of ending heads from last stress period, but (k,i,j) location
                            previous_inds = spd[kper - 1][['k', 'i', 'j']].tolist()
                            previous_ehead = dict(zip(previous_inds, spd[kper - 1]['ehead']))
                            current_inds = kspd[['k', 'i', 'j']].tolist()
                            sheads = np.array([previous_ehead.get((k, i, j), np.nan)
                                               for (k, i, j) in current_inds])
                            sheads[np.isnan(sheads)] = kspd['ehead'][np.isnan(sheads)]
                            kspd['shead'] = sheads
                    else:
                        kspd[variable_column] = group[variable_column]
                spd[kper] = kspd
        else:
            pass  # spd[kper] = None
    return spd
