"""
Functions for simple MODFLOW boundary conditions such as ghb, drain, etc.
"""
import numbers
import shutil
import warnings

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd

fm = flopy.modflow
import pyproj
import rasterio
from gisutils import project
from rasterstats import zonal_stats
from shapely.geometry import Polygon

from mfsetup.discretization import cellids_to_kij, get_highest_active_layer, get_layer
from mfsetup.grid import rasterize
from mfsetup.sourcedata import TransientTabularSourceData
from mfsetup.units import convert_length_units, convert_time_units


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def setup_basic_stress_data(model, shapefile=None, csvfile=None,
                            head=None, elev=None, bhead=None, stage=None,
                            cond=None, rbot=None, default_rbot_thickness=1,
                            all_touched=True,
                            **kwargs):

    m = model

    # basic stress package variables
    variables = {'head': head, 'elev': elev, 'bhead': bhead,
                 'stage': stage, 'cond': cond, 'rbot': rbot}

    # get the BC cells
    # todo: generalize more of the GHB setup code and move it somewhere else
    bc_cells = None
    if shapefile is not None:

        # rename some columns for interface change
        if 'boundname_col' in shapefile:
            warnings.warn(
                "The boundname_col: item will be deprecated and removed in "
                "version 0.4.0. Use boundname_column: instead.",
                PendingDeprecationWarning,
            )
        renames = {'boundname_col': 'boundname_column'}
        shapefile = shapefile.copy()
        shapefile = {renames.get(k, k): v for k, v in shapefile.items()}
        key = [k for k in shapefile.keys() if 'filename' in k.lower()]
        if key:
            shapefile_name = shapefile.pop(key[0])
            if 'all_touched' in shapefile:
                all_touched = shapefile['all_touched']
                if 'boundname_column' in shapefile:
                    shapefile['names_column'] = shapefile.pop('boundname_column')
            bc_cells = rasterize(shapefile_name, m.modelgrid, **shapefile)
            bc_cell_id_kwargs = shapefile.copy()
            bc_cell_id_kwargs['names_column'] = shapefile['id_column']
            bc_cell_ids = rasterize(shapefile_name, m.modelgrid, **bc_cell_id_kwargs)
            # fill unnamed feature names with id numbers (as strings)
            unnamed = bc_cells == 'nan'
            bc_cells[unnamed] = bc_cell_ids[unnamed]

    csv_input = None
    if csvfile is not None:
        csv_kwargs = csvfile.copy()
        csv_kwargs['filenames'] = [csv_kwargs.pop('filename')]
        data_columns = {v: k.split('_')[0] for k, v in csv_kwargs.items()
                                  if k.split('_')[0] in variables.keys()}
        csv_kwargs['data_columns'] = list(data_columns.keys())
        sd = TransientTabularSourceData(
                dest_model=m, **csv_kwargs)
        csv_input = sd.get_data()
        csv_input.index = csv_input[csv_kwargs['id_column']].astype(str)
        # rename the input data columns to their MODFLOW variable names
        csv_input.rename(columns=data_columns, inplace=True)

        #raise NotImplementedError('Time-varying (CSV) file input not yet supported for this package.')
    if bc_cells is None:
        return

    # create polygons of model grid cells
    if bc_cells.dtype == object:
        cells_with_bc = bc_cells.flat != ''
    else:
        cells_with_bc = bc_cells.flat > 0

    vertices = np.array(m.modelgrid.vertices)[cells_with_bc, :, :]
    polygons = [Polygon(vrts) for vrts in vertices]

    # setup DataFrame for MODFLOW input
    i, j = np.indices((m.nrow, m.ncol))
    # start with stress period 0 as a template
    df_0 = pd.DataFrame({'per': 0,
                       'k': 0,
                       'i': i.flat,
                       'j': j.flat})
    # add the boundnames
    if bc_cells.dtype == object:
        df_0['boundname'] = bc_cells.flat
        nan_boundnames = df_0.boundname.isna() | df_0.boundname.isin({'', 'nan'})
        df_0.loc[nan_boundnames, 'boundname'] = 'unnamed'
        # convert any numeric boundnames to strings
        # (otherwise MODFLOW 6 will mistake them for cell IDs)
        df_0['boundname'] = [f"feature-{bname}" if is_number(bname)
                             else bname for bname in df_0['boundname']]
    # add the feature ids
    # (for associating transient input with cells)
    df_0['feature_id'] = bc_cell_ids.flat
    df_0.index = df_0['feature_id']

    # cull to just the cells with bcs
    df_0 = df_0.loc[cells_with_bc].copy()
    df = gpd.GeoDataFrame(df_0, geometry=polygons, crs=model.modelgrid.crs)

    # collect input that may be mixed
    # between transient input supplied via csvfiles
    # and static input supplied via rasters or global values
    for var, entry in variables.items():
        # check for transient csv input for the variable
        if csv_input is not None and var in csv_input:
            df_0 = df.copy()
            dfs = []
            for per in csv_input['per'].unique():
                df_per = df_0.copy()
                df_per['per'] = per
                df_per[var] = csv_input.loc[csv_input['per'] == per, var]
                dfs.append(df_per)
            df = pd.concat(dfs, axis=0)
        # otherwise, check for static input
        elif entry is not None:
            # Raster of spatially varying values supplied
            if isinstance(entry, dict):
                filename_entries = [k for k in entry.keys() if 'filename' in k.lower()]
                if not any(filename_entries):
                    continue
                filename = entry[filename_entries[0]]
                with rasterio.open(filename) as src:
                    meta = src.meta

                # reproject the polygons to the dem crs if needed
                try:
                    from gisutils import get_authority_crs
                    raster_crs = get_authority_crs(src.crs)
                except:
                    raster_crs = pyproj.crs.CRS.from_user_input(src.crs)
                if raster_crs != m.modelgrid.crs:
                    polygons = project(df.geometry.tolist(), m.modelgrid.crs, raster_crs)
                else:
                    polygons = df.geometry.tolist()

                # all_touched arg for rasterstats.zonal_stats
                all_touched = False
                if meta['transform'][0] > m.modelgrid.delr[0]:
                    all_touched = True
                stat = entry['stat']
                # load raster and specify affine transform to avoid issues with proj
                with rasterio.open(filename) as src:
                    affine = src.transform
                    array = src.read(1)
                results = zonal_stats(polygons, array, affine=affine, stats=stat,
                                    all_touched=all_touched)
                #values = np.ones((m.nrow * m.ncol), dtype=float) * np.nan
                #values[cells_with_bc] = np.array([r[stat] for r in results])
                values = np.array([r[stat] for r in results])
                # cull to polygon statistics within model area
                valid = values != None
                values = values[valid]
                df = df.loc[valid].copy()

                # Convert units if they are specified
                # Note: this only works because the variables considered here
                # all have length in the numerator
                # and cond has time in the denominator
                if 'length_units' in entry:
                    values *= convert_length_units(entry['length_units'],
                                                    model.length_units)
                if var == 'cond' and 'time_units' in entry:
                    values /= convert_time_units(entry['time_units'],
                                                    model.time_units)

                # add the layer and the values to the Modflow input DataFrame
                # assign layers so that the elevation is above the cell bottoms
                if var in ['head', 'elev', 'bhead']:
                    df['k'] = get_layer(model.dis.botm.array, df.i, df.j, values)
                df[var] = values
            # single global value specified
            elif isinstance(entry, numbers.Number):
                df[var] = entry
            else:
                raise ValueError(f"Unrecognized input for {var}:\n{entry}. "
                                 "If this is from a YAML format configuration file, "
                                 "check that the number is formatted correctly "
                                 "(i.e. 1.e+3 for 1e3)")

    # drop cells that don't include this boundary condition
    df.dropna(axis=0, inplace=True)

    # special handling of rbot for RIV package
    if 'stage' in df.columns and 'rbot' not in df.columns:
        df['rbot'] = df['stage'] - default_rbot_thickness

    # exclude inactive cells
    #k, i, j = df.k, df.i, df.j
    if model.version == 'mf6':
        idomain = model.idomain
    else:
        idomain = model.ibound

    # remove BC cells from places where the specified head is below the model
    # set the layer according to the variable
    for var in ['head', 'elev', 'bhead', 'rbot']:
        if var in df.columns:
            below_bottom_of_model = df[var] < model.dis.botm.array[-1, df.i, df.j] + 0.01
            df = df.loc[~below_bottom_of_model].copy()
            df['k'] = get_layer(model.dis.botm.array,
                                df['i'], df['j'], df[var])
            # move any variable elevations in inactive cells
            # to the highest active layer below
            df['idomain'] = idomain[df['k'], df['i'], df['j']]
            if any(df['idomain'] < 1):
                idomain_slice = idomain[:, df['i'], df['j']]
                is_above = model.dis.botm.array[:, df['i'], df['j']] > \
                    [df[var].tolist()] * idomain.shape[0]
                idomain_slice[is_above] = 0
                highest_active_below = np.argmax(idomain_slice, axis=0)
                df.loc[df['idomain'] < 1, 'k'] = highest_active_below[df['idomain'] < 1]

    #active_cells = idomain[k, i, j] >= 1
    #df = df.loc[active_cells]

    # set layer to the highest active
    #highest_active = get_highest_active_layer(idomain)
    #df['k'] = highest_active[df['i'], df['j']]

    # drop locations that are completely inactive
    df = df.loc[df['k'].isin(range(idomain.shape[0]))]

    # for models with LGR:
    # drop non-well BCs within LGR areas
    # (assumes that CHD, DRN, GHB, and RIV cells
    # all represent surface or near-surface features
    # that are represented in the LGR models;
    # the Well Package is set up using other code in wells.py)
    if model.lgr:
        df['lgr_idomain'] = model._lgr_idomain2d[df['i'], df['j']]
        df = df.loc[df['lgr_idomain'] > 0].copy()

    # sort the columns
    col_order = ['per', 'k', 'i', 'j', 'head', 'elev', 'bhead', 'stage',
                 'cond', 'rbot', 'boundname']
    cols = [c for c in col_order if c in df.columns]
    df = df[cols].copy()
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

        if recs is None or len(recs) == 0:
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


def remove_inactive_bcs(pckg, external_files=False):
    """Remove boundary conditions from cells that are inactive.

    Parameters
    ----------
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

    if external_files:
        if model.version == 'mf6':
            spd_input = {}
            for per, filename in external_files.items():
                df = pd.DataFrame(new_spd[per])
                df['#k'], df['i'], df['j'] = zip(*df['cellid'])
                df[['#k', 'i', 'j']] += 1  # convert to 1-based for external file
                cols = ['#k', 'i', 'j'] + list(new_spd[per].dtype.names[1:])
                if isinstance(filename, dict):
                    filename = filename['filename']
                df[cols].to_csv(filename, index=False, sep=' ', float_format='%g')
                spd_input[per] = {'filename': filename}
                # make a copy for the intermediate data folder, for consistency with mf-2005
                #shutil.copy(file_entry['filename'], model.cfg['intermediate_data']['output_folder'])
            pckg.stress_period_data = spd_input
        else:
            raise NotImplementedError('External file input for MODFLOW-2005-style list-type data.')
    else:
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
                                   variable_columns=None, external_files=True,
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

    df = data.copy()
    missing_variables = set(variable_columns).difference(df.columns)
    if any(missing_variables):
        raise ValueError(f"{package.upper()} Package: missing input for variables: "
                         f"{', '.join(missing_variables)}")
    # set up stress_period_data
    if external_files and model.version == 'mf6':
        # get the file path (allowing for different external file locations, specified name format, etc.)
        filepaths = model.setup_external_filepaths(package, 'stress_period_data',
                                                   filename_format=external_filename_fmt,
                                                   file_numbers=sorted(df.per.unique().tolist()))
        # convert to one-based
        df.rename(columns={'k': '#k'}, inplace=True)
        df['#k'] += 1
        df['i'] += 1
        df['j'] += 1
        cols = ['per', '#k', 'i', 'j'] + variable_columns + ['boundname']
    else:
        cols = ['per', 'k', 'i', 'j'] + variable_columns + ['boundname']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    spd = {}
    period_groups = df.groupby('per')
    for kper in range(model.nper):
        if kper in period_groups.groups:
            group = period_groups.get_group(kper)
            group.drop('per', axis=1, inplace=True)
            if external_files and model.version == 'mf6':
                group.to_csv(filepaths[kper]['filename'], index=False, sep=' ', float_format='%g')
                # make a copy for the intermediate data folder, for consistency with mf-2005
                shutil.copy(filepaths[kper]['filename'], model.cfg['intermediate_data']['output_folder'])

                # external list or tabular type files not supported for MODFLOW-NWT
                # adding support for this may require changes to Flopy

            else:
                if model.version == 'mf6':
                    kspd = flopy_package_class.stress_period_data.empty(model,
                                                                        len(group),
                                                                        boundnames=True)[0]
                    kspd['cellid'] = list(zip(group.k, group.i, group.j))
                    for col in variable_columns:
                        kspd[col] = group[col]
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
                            kspd['shead'] = group['head']
                            kspd['ehead'] = group['head']
                        else:
                            kspd['ehead'] = group['head']
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
                        for col in variable_columns:
                            kspd[col] = group[col]
                        if 'boundname' in group.columns and 'boundname' in kspd.dtype.names:
                            kspd['boundname'] = group['boundname']
                spd[kper] = kspd
        else:
            pass  # spd[kper] = None
    return spd
