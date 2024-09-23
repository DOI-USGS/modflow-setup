import os
import time
import warnings
from pathlib import Path

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from shapely.geometry import Polygon

fm = flopy.modflow
from gisutils import project, shp2df

from mfsetup.evaporation import hamon_evaporation
from mfsetup.fileio import save_array
from mfsetup.grid import rasterize
from mfsetup.sourcedata import (
    SourceData,
    TabularSourceData,
    TransientSourceDataMixin,
    aggregate_dataframe_to_stress_period,
)
from mfsetup.units import convert_length_units, convert_temperature_units
from mfsetup.utils import get_input_arguments


def make_lakarr2d(grid, lakesdata,
                  include_ids=None, id_column='hydroid'):
    """
    Make a nrow x ncol array with lake package extent for each lake,
    using the numbers in the 'id' column in the lakes shapefile.
    """
    if isinstance(lakesdata, str):
        # implement automatic reprojection in gis-utils
        # maintaining backwards compatibility
        kwargs = {'dest_crs': grid.crs}
        kwargs = get_input_arguments(kwargs, shp2df)
        lakes = shp2df(lakesdata, **kwargs)
    elif isinstance(lakesdata, pd.DataFrame):
        lakes = lakesdata.copy()
    else:
        raise ValueError('unrecognized input for "lakesdata": {}'.format(lakesdata))
    id_column = id_column.lower()
    lakes.columns = [c.lower() for c in lakes.columns]
    lakes.index = lakes[id_column]
    if include_ids is not None:
        lakes = lakes.loc[include_ids]
    else:
        include_ids = lakes[id_column].tolist()
    lakes['lakid'] = np.arange(1, len(lakes) + 1)
    lakes['geometry'] = [Polygon(g.exterior) for g in lakes.geometry]
    arr = rasterize(lakes, grid=grid, id_column='lakid')

    # ensure that order of hydroids is unchanged
    # (used to match features to lake IDs in lake package)
    assert lakes[id_column].tolist() == include_ids
    return arr


def make_bdlknc_zones(grid, lakesshp, include_ids,
                      feat_id_column='feat_id',
                      lake_package_id_column='lak_id',
                      littoral_zone_buffer_width=20):
    """
    Make zones for populating with lakebed leakance values. Same as
    lakarr, but with a buffer around each lake so that horizontal
    connections have non-zero values of bdlknc, and near-shore
    areas can be assigend higher leakance values.
    """
    print('setting up lakebed leakance zones...')
    t0 = time.time()
    if isinstance(lakesshp, str):
        # implement automatic reprojection in gis-utils
        # maintaining backwards compatibility
        #kwargs = {'dest_crs': grid.crs}
        #kwargs = get_input_arguments(kwargs, shp2df)
        #lakes = shp2df(lakesshp, **kwargs)
        lakes = gpd.read_file(lakesshp)
    elif isinstance(lakesshp, pd.DataFrame):
        lakes = gpd.GeoDataFrame(lakesshp)
    else:
        raise ValueError('unrecognized input for "lakesshp": {}'.format(lakesshp))
    lakes.to_crs(grid.crs, inplace=True)

    # Exterior buffer
    id_column = feat_id_column.lower()
    lakes.columns = [c.lower() for c in lakes.columns]
    lakes.index = lakes[id_column]
    lakes = lakes.loc[include_ids]
    if lake_package_id_column not in lakes.columns:
        lakes[lake_package_id_column] = np.arange(1, len(lakes) + 1)
    # speed up buffer construction by getting exteriors once
    # and probably more importantly,
    # simplifying potentially complex geometries of lakes
    # set the exterior buffer to 1.5x the grid spacing
    exterior_buffer = grid.delr[0] * 1.5
    unbuffered_exteriors = [Polygon(g.exterior).simplify(5) for g in lakes.geometry]
    lakes['geometry'] = [g.buffer(exterior_buffer) for g in unbuffered_exteriors]
    arr = rasterize(lakes, grid=grid, id_column=lake_package_id_column)

    # Interior buffer for lower leakance
    interior_buffer = -littoral_zone_buffer_width
    lakes['geometry'] = [g.buffer(interior_buffer) for g in unbuffered_exteriors]
    if not lakes.is_empty.all():
        arr2 = rasterize(lakes, grid=grid, id_column=lake_package_id_column)
        arr2 = arr2 * 100  # Create new ids for the interior, as multiples of 10
        arr[arr2 > 0] = arr2[arr2 > 0]

    # ensure that order of hydroids is unchanged
    # (used to match features to lake IDs in lake package)
    assert lakes[id_column].tolist() == list(include_ids)
    print('finished in {:.2f}s'.format(time.time() - t0))
    return arr


def make_bdlknc2d(lakzones, littoral_leakance, profundal_leakance):
    """Make a lakebed leakance array using piecewise-constant zones
    and universal values for littoral and profundal leakance."""
    bdlknc = np.zeros(lakzones.shape, dtype=float)
    bdlknc[(lakzones > 0) & (lakzones < 100)] = littoral_leakance
    bdlknc[lakzones >= 100] = profundal_leakance
    return bdlknc


def get_littoral_profundal_zones(lakzones):
    """Make a version of the lakebed leakance zone
    array that just designates cells as littoral or profundal.
    """
    zones = np.zeros(lakzones.shape, dtype=object)
    zones[(lakzones > 0) & (lakzones < 100)] = 'littoral'
    zones[lakzones >= 100] = 'profundal'
    return zones


def get_flux_variable_from_config(variable, config, nlakes, nper):
    # MODFLOW 6 models
    if 'perioddata' in config:
        config = config['perioddata']
    data = config[variable]
    # copy to all stress periods
    # single scalar value
    if np.isscalar(data):
        values = [data] * nper * nlakes
    # flat list of global values by stress period
    elif isinstance(data, list) and np.isscalar(data[0]):
        values = data.copy()
        if len(data) < nper:
            for i in range(nper - len(data)):
                values.append(data[-1])
        # repeat stress period data for all lakes
        values = values * nlakes
    elif isinstance(data, dict):
        values = []
        lake_numbers = list(sorted(data.keys()))
        if not len(lake_numbers) == nlakes and all(np.diff(lake_numbers) == 1):
            raise ValueError(
                f"Lake Package {variable}: Dictionary-style input "
                "requires consecutive lake numbers and an entry for each lake")
        for lake_no in lake_numbers:
            lake_values = data[lake_no]
            if np.isscalar(lake_values):
                lake_values = [lake_values] * nper
            elif len(lake_values) < nper:
                for i in range(nper - len(lake_values)):
                    lake_values.append(lake_values[-1])
            values += lake_values
    # dict of lists, or nested dict of values by lake, stress_period
    else:
        raise ValueError(f"Invalid Lake Package {variable}: input:\n{config}")
    assert isinstance(values, list)
    return values


def setup_lake_info(model):

    # lake package must have a source_data block
    # (e.g. for supplying shapefile delineating lake extents)
    source_data = model.cfg.get('lak', {}).get('source_data')
    if source_data is None or 'lak' not in model.package_list:
        return
    kwargs = get_input_arguments(source_data['lakes_shapefile'], model.load_features)
    lakesdata = model.load_features(**kwargs)
    id_column = source_data['lakes_shapefile']['id_column'].lower()
    name_column = source_data['lakes_shapefile'].get('name_column', 'name').lower()
    nlakes = len(lakesdata)

    # make dataframe with lake IDs, names and locations
    centroids = project([g.centroid for g in lakesdata.geometry],
                        lakesdata.crs, 4269)
    # boundnames for lakes
    # from source shapefile
    lak_ids = np.arange(1, nlakes + 1)
    names = None
    if name_column in lakesdata.columns:
        names = lakesdata[name_column].values
        if len(set(names)) < len(names) or 'nan' in names:
            names = None
    # from configuration file (by feature id)
    elif 'boundnames' in source_data['lakes_shapefile']:
        names = [source_data['lakes_shapefile']['boundnames'][feat_id]
                 for feat_id in lakesdata[id_column].values]
    if names is None:  # default to names based on lake ID
        names = names = ['lake{}'.format(i) for i in lak_ids]
    df = gpd.GeoDataFrame({'lak_id': lak_ids,
                       'feat_id': lakesdata[id_column].values,
                       'name': names,
                       'latitude': [c.y for c in centroids],
                       'geometry': lakesdata['geometry']
                       }, crs=lakesdata.crs)

    # get starting stages from model top, for specifying ranges
    stages = []
    for lakid in df['lak_id']:
        loc = model._lakarr2d == lakid
        est_stage = model.dis.top.array[loc].min()
        stages.append(est_stage)
    df['strt'] = np.array(stages)

    # save a lookup file mapping lake ids to hydroids
    lookup_file = Path(
        model.cfg['lak']['output_files']['lookup_file'].format(model.name)).name
    out_table = Path(model._tables_path) / lookup_file
    df.drop('geometry', axis=1).to_csv(out_table)
    print(f'wrote {out_table}')
    poly_shp = Path(
        model.cfg['lak']['output_files']['lak_polygons_shapefile'].format(model.name)).name
    out_shapefile = Path(model._shapefiles_path) / poly_shp
    df.to_file(out_shapefile)
    print(f'wrote {out_shapefile}')

    # clean up names
    #df['name'].replace('nan', '', inplace=True)
    df['name'].replace(' ', '', inplace=True)
    return df


def setup_lake_fluxes(model, block='lak'):
    """Set up dataframe of fluxes by lake and stress period

    Parameters
    ----------
    model : modflow-setup model instance
    block : str, {'lak', 'high_k_lakes'}
        Location of input for setting up fluxes. If 'lak',
        input is read from the Lake Package ('lak') block
        in the configuration file. If 'high_k_lakes',
        input is read from the 'high_k_lakes' block.

    Returns
    -------

    """

    # setup empty dataframe
    variables = ['precipitation', 'evaporation', 'runoff', 'withdrawal']
    columns = ['per', 'lak_id'] + variables
    nlakes = len(model.lake_info['lak_id'])
    df = pd.DataFrame(np.zeros((nlakes * model.nper, len(columns)), dtype=float),
                      columns=columns)
    df['per'] = list(range(model.nper)) * nlakes
    # lake dataframe sorted by lake id and then period
    df['lak_id'] = sorted(model.lake_info['lak_id'].tolist() * model.nper)

    # option 1; precip and evaporation specified directly
    # values are assumed to be in model units
    for variable in variables:
        # MODFLOW 2005 models or MODFLOW 6 Lake Package
        if variable in model.cfg[block] or\
            variable in model.cfg[block].get('perioddata', {}):
            values = get_flux_variable_from_config(variable, model.cfg[block], nlakes, model.nper)
            df[variable] = values

    # option 2; precip and temp specified from PRISM output
    # compute evaporation from temp using Hamon method
    if 'climate' in model.cfg[block]['source_data']:
        cfg = model.cfg[block]['source_data']['climate']
        format = cfg.get('format', 'csv').lower()
        if format == 'prism':
            sd = PrismSourceData.from_config(cfg, dest_model=model)
            data = sd.get_data()
            # if PRISM source data were specified without any lake IDs
            # assign to all lakes
            if data['lake_id'].sum() == 0:
                dfs = []
                for lake_id in model.lake_info['lak_id']:
                    data_lake_id = data.copy()
                    data_lake_id['lak_id'] = lake_id
                    dfs.append(data_lake_id)
                data = pd.concat(dfs, axis=0)
            tmid = data['start_datetime'] + (data['end_datetime'] - data['start_datetime'])/2
            data['day_of_year'] = tmid.dt.dayofyear
            id_lookup = {'feat_id': dict(zip(model.lake_info['lak_id'],
                                             model.lake_info['feat_id']))}
            id_lookup['lak_id'] = {v:k for k, v in id_lookup['feat_id'].items()}
            for col in ['lak_id', 'feat_id']:
                if len(set(data.lake_id).intersection(model.lake_info[col])) > 0:
                    latitude = dict(zip(model.lake_info[col],
                                        model.lake_info['latitude']))
                    other_col = 'lak_id' if col == 'feat_id' else 'feat_id'
                    data[other_col] = [id_lookup[other_col][id] for id in data['lake_id']]
            data['latitude'] = [latitude[id] for id in data['lake_id']]
            data['evaporation'] = hamon_evaporation(data['day_of_year'],
                                             data['temp'],  # temp in C
                                             data['latitude'],  # DD
                                             dest_length_units=model.length_units)
            # update flux dataframe
            data.sort_values(by=['lak_id', 'per'], inplace=True)
            df.sort_values(by=['lak_id', 'per'], inplace=True)
            assert np.all(data[['lak_id', 'per']].values == \
                df[['lak_id', 'per']].values)
            data.reset_index(drop=True, inplace=True)
            df.update(data)

        else:
            # TODO: option 3; general csv input for lake fluxes
            raise NotImplementedError('General csv input for lake fluxes')
    # compute a value to use for high-k lake recharge, for each stress period
    if block == 'high_k_lakes' and model.cfg['high_k_lakes']['simulate_high_k_lakes']:
        per_means = df.groupby('per').mean()
        highk_lake_rech = dict(per_means['precipitation'] - per_means['evaporation'])
        df['highk_lake_rech'] = [highk_lake_rech[per] for per in df.per]
    return df


def setup_lake_tablefiles(model, cfg):

    print('setting up tabfiles...')
    sd = TabularSourceData.from_config(cfg)
    df = sd.get_data()

    lakes = df.groupby(sd.id_column)
    n_included_lakes = len(set(model.lake_info['feat_id']). \
                           intersection(set(lakes.groups.keys())))
    assert n_included_lakes == model.nlakes, "stage_area_volume (tableinput) option" \
                                       " requires info for each lake, " \
                                       "only these feature IDs found:\n{}".format(df[sd.id_column].tolist())
    tab_files = []
    for i, id in enumerate(model.lake_info['feat_id'].tolist()):
        dfl = lakes.get_group(id)
        tabfilename = '{}/{}/{}_stage_area_volume.dat'.format(model.model_ws,
                                                              model.external_path,
                                                              id)
        if model.version == 'mf6':
            with open(tabfilename, 'w', newline="") as dest:
                dest.write('begin dimensions\n')
                dest.write('nrow {}\n'.format(len(dfl)))
                dest.write('ncol {}\n'.format(df.shape[1]))
                dest.write('end dimensions\n')
                dest.write('\nbegin table\n')
                cols = ['stage', 'volume', 'area']
                dest.write('#{}\n'.format(' '.join(cols)))
                dfl[cols].to_csv(dest, index=False, header=False,
                                                        sep=' ', float_format='%.5e')
                dest.write('end table\n')

        else:
            assert len(dfl) == 151, "151 values required for each lake; " \
                                    "only {} for feature id {} in {}" \
                .format(len(dfl), id, cfg)
            dfl[['stage', 'volume', 'area']].to_csv(tabfilename, index=False, header=False,
                                                    sep=' ', float_format='%.5e')
        print('wrote {}'.format(tabfilename))
        tab_files.append(tabfilename)
    return tab_files


def setup_lake_connectiondata(model, for_external_file=True,
                              include_horizontal_connections=True):

    cfg = model.cfg['lak']

    # set up littoral and profundal zones
    if model.lake_info is None:
        model.lake_info = setup_lake_info(model)

    # zone numbers
    # littoral zones are the same as the one-based lake number
    # profundal zones are the one-based lake number times 100
    # for example, for lake 1, littoral zone is 1; profundal zone is 100.
    lakzones = make_bdlknc_zones(model.modelgrid, model.lake_info,
                                 include_ids=model.lake_info['feat_id'],
                                littoral_zone_buffer_width=cfg['source_data']['littoral_zone_buffer_width'])
    littoral_profundal_zones = get_littoral_profundal_zones(lakzones)

    model.setup_external_filepaths('lak', 'lakzones',
                                   cfg['{}_filename_fmt'.format('lakzones')])
    save_array(model.cfg['intermediate_data']['lakzones'][0], lakzones, fmt='%d')

    # make the (2D) areal footprint of lakebed leakance from the zones
    bdlknc = make_bdlknc2d(lakzones,
                           cfg['source_data']['littoral_leakance'],
                           cfg['source_data']['profundal_leakance'])

    # cell tops and bottoms
    layer_elevations = np.zeros((model.nlay + 1, model.nrow, model.ncol), dtype=float)
    layer_elevations[0] = model.dis.top.array
    layer_elevations[1:] = model.dis.botm.array

    lakeno = []
    cellid = []
    bedleak = []
    zone = []
    for lake_id in range(1, model.nlakes + 1):

        # get the vertical GWF connections for each lake
        k, i, j = np.where(model.lakarr == lake_id)
        # get just the unique i, j locations for each lake
        i, j = zip(*set(zip(i, j)))
        # assign vertical connections to the highest active layer
        k = np.argmax(model.idomain[:, i, j], axis=0)
        cellid += list(zip(k, i, j))
        lakeno += [lake_id] * len(k)
        bedleak += list(bdlknc[i, j])
        zone += list(littoral_profundal_zones[i, j])

    df = pd.DataFrame({'lakeno': lakeno,
            'cellid': cellid,
            'claktype': 'vertical',
            'bedleak': bedleak,
            'belev': 0.,
            'telev': 0.,
            'connlen': 0.,
            'connwidth': 0.,
            'zone': zone
            })

    if include_horizontal_connections:
        for lake_id in range(1, model.nlakes + 1):
            # make an array where the cells for the lake are zero,
            # and all other cells are one
            # get_horizontal_connections will find the cells == 1
            # that connect (share faces) with cells == 0
            lake_extent = model.lakarr != lake_id
            horizontal_connections = get_horizontal_connections(lake_extent,
                                                                connection_info=True,
                                                                layer_elevations=layer_elevations,
                                                                delr=model.dis.delr.array,
                                                                delc=model.dis.delc.array,
                                                                bdlknc=bdlknc)
            # drop horizontal connections to inactive cells
            k, i, j = horizontal_connections[['k', 'i', 'j']].T.values
            inactive = model.idomain[k, i, j] < 1
            horizontal_connections = horizontal_connections.loc[~inactive].copy()
            horizontal_connections['lakeno'] = lake_id
            k, i, j = horizontal_connections[['k', 'i', 'j']].T.values
            horizontal_connections['zone'] = littoral_profundal_zones[i, j]
            horizontal_connections['cellid'] = list(zip(k, i, j))
            horizontal_connections.drop(['k', 'i', 'j'], axis=1, inplace=True)
            df = pd.concat([df, horizontal_connections], axis=0)
    # assign iconn (connection number) values for each lake
    dfs = []
    for lakeno, group in df.groupby('lakeno'):
        group = group.copy()
        group['iconn'] = list(range(len(group)))
        dfs.append(group)
    df = pd.concat(dfs)

    connections_lookup_file = model.cfg['lak']['output_files']['connections_lookup_file'].format(model.name)
    connections_lookup_file = os.path.join(model._tables_path, os.path.split(connections_lookup_file)[1])
    model.cfg['lak']['output_files']['connections_lookup_file'] = connections_lookup_file
    df.to_csv(connections_lookup_file, index=False)
    # write shapefile version
    k, i, j = map(np.array, zip(*df['cellid']))
    df['k'] = k
    df['i'] = i
    df['j'] = j
    ncol = model.modelgrid.ncol
    gwf_nodes = i * ncol + j
    gdf = gpd.GeoDataFrame(df.drop('cellid', axis=1),
                          geometry=np.array(model.modelgrid.polygons)[gwf_nodes],
                          crs=model.modelgrid.crs)
    connections_shapefile = Path(
        model.cfg['lak']['output_files']['connections_shapefile'].format(model.name)).name
    out_shapefile = Path(model._shapefiles_path) / connections_shapefile
    gdf.to_file(out_shapefile)
    print(f'wrote {out_shapefile}')

    # convert to one-based and comment out header if df will be written straight to external file
    if for_external_file:
        df.rename(columns={'lakeno': '#lakeno'}, inplace=True)
        df['iconn'] += 1
        k, i, j = zip(*df['cellid'])
        df.drop('cellid', axis=1, inplace=True)
        df['k'] = np.array(k) + 1
        df['i'] = np.array(i) + 1
        df['j'] = np.array(j) + 1
    else:
        df['lakeno'] -= 1  # convert to zero-based for mf6
    return df


def setup_mf6_lake_obs(kwargs):

    packagedata = kwargs['packagedata']
    types = ['stage',
             #'ext-inflow',
             #'outlet-inflow',
             'inflow',
             #'from-mvr',
             'rainfall',
             'runoff',
             'lak',
             'withdrawal',
             'evaporation',
             #'ext-outflow',
             #'to-mvr',
             'storage',
             #'constant',
             #'outlet',
             'volume',
             'surface-area',
             'wetted-area',
             'conductance'
             ]
    obs_input = {}
    for rec in packagedata:
        lakeno = rec[0]
        boundname = rec[3]
        if boundname == '' or str(boundname) == 'nan':
            lakename = 'lake' + str(lakeno + 1)
            boundname = lakeno + 1  # convert to one-based
        else:
            lakename = rec[3]

        filename = '{}.obs.csv'.format(lakename)
        lake_obs = []
        for obstype in types:
            obsname = obstype  #'{}_{}'.format(lakename, obstype)
            entry = (obsname, obstype, boundname)
            lake_obs.append(entry)
        obs_input[filename] = lake_obs
    obs_input['digits'] = 10
    return obs_input


def get_lakeperioddata(lake_fluxes):
    """Convert lake_fluxes table to MODFLOW-6 lakeperioddata input.
    """
    lakeperioddata = {}
    periods = lake_fluxes.groupby('per')
    for per, group in periods:
        group = group.replace('ACTIVE', np.nan)
        group.rename(columns={'precipitation': 'rainfall'}, inplace=True)
        group.index = group.lak_id.values
        datacols = {'rainfall', 'evaporation', 'withdrawal', 'runoff', 'stage'}
        datacols = [c for c in group.columns if c in datacols]
        group = group.loc[:, datacols]
        data = group.stack().reset_index()
        data.rename(columns={'level_0': 'lakeno',
                             0: 'value'}, inplace=True)
        # data['laksetting'] = ['{} {}'.format(variable, value)
        #                      for variable, value in zip(data['level_1'], data['value'])]
        data['lakeno'] -= 1
        data = data[['lakeno', 'level_1', 'value']].values.tolist()
        lakeperioddata[per] = data
    return lakeperioddata


def get_horizontal_connections(extent, inside=False, connection_info=False,
                               layer_elevations=None, delr=None, delc=None,
                               bdlknc=None):
    """Get cells along the edge of an aerial feature (e.g. a lake or irregular model inset area),
    using the sobel filter method (for edge detection) in SciPy.

    see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
    https://en.wikipedia.org/wiki/Sobel_operator

    Parameters
    ----------
    extent : 2 or 3D array of ones and zeros
        With ones indicating an area interest. get_horizontal_connections
        will return connection information between cells == 1 and
        any neighboring cells == 0 that share a face (no diagonal connections).
        The resulting connections will be located within the area of cells == 1
        (inside of the perimeter between 1s and 0s). In practice, this means that
        for finding horizontal lake connections, the lake cells should be == 0,
        and the active model cells should be == 1. For finding perimeter boundary
        cells, the active model cells should be == 1; inactive areas beyond should
        be == 0.
    connection_info : bool
        Option to return the top and bottom elevation, length, width, and
        bdlknc value of each connection (i.e., as needed for the MODFLOW-6 lake package).
        By default, False.
    layer_elevations : np.ndarray
        Numpy array of cell top and bottom elevations.
        (shape = nlay + 1, nrow, ncol). Optional, only needed if connection_info == True
        (by default, None).
    delr : 1D array of cell spacings along a model row
        Optional, only needed if connection_info == True
        (by default, None).
    delc : 1D array of cell spacings along a model column
        Optional, only needed if connection_info == True
        (by default, None).
    bdlknc : 2D array
        Array of lakebed leakance values
        (optional; default=1)

    Returns
    -------
    df : DataFrame
        Table of horizontal cell connections
        Columns:
        k, i, j, cellface;
        optionally (if connection_info == True):
        claktype, bedleak, belev, telev, connlen, connwidth
        (see MODFLOW-6 io guide for an explanation or the Connectiondata
        input block)

    """
    if inside:
        warnings.warn(('The "inside" argument is deprecated. '
                       'Cell connections are now always located along the inside'
                      'edge of the perimeter of cells == 1 in the extent array.'))
    extent = extent.astype(float)
    if len(extent.shape) != 3:
        extent = np.expand_dims(extent, axis=0)
    if bdlknc is None:
        bdlknc = np.ones_like(extent[0], dtype=float)

    cellid = []
    bedleak = []
    belev = []
    telev = []
    connlen = []
    connwidth = []
    cellface = []
    for klay, extent_k in enumerate(extent):
        sobel_x = sobel(extent_k, axis=1, mode='reflect') #, cval=cval)
        sobel_x[extent_k == 0] = 10
        sobel_y = sobel(extent_k, axis=0, mode='reflect') #, cval=cval)
        sobel_y[extent_k == 0] = 10

        # right face connections
        # (i.e. through the right face of an interior cell)
        # orthagonal connections have a value of -2
        # diagonal connections have a value of -1
        # the sobel filter sums the connections for each cell
        # so a cell with 2 diagonal and 1 orthagonal connections will be -4;
        # cells with an orthagonal right-face connection will range from -2 to -4
        # https://en.wikipedia.org/wiki/Sobel_operator
        i, j = np.where((sobel_x <= -2) & (sobel_x >= -4))
        #if inside:
        #    j -= 1
        k = np.ones(len(i), dtype=int) * klay
        cellid += list(zip(k, i, j))
        if connection_info:
            bedleak += list(bdlknc[i, j])
            belev += list(layer_elevations[k + 1, i, j])
            telev += list(layer_elevations[k, i, j])
            connlen += list(0.5 * delr[j - 1] + 0.5 * delr[j])
            connwidth += list(delc[i])
        cellface += ['right'] * len(i)

        # left face connections
        # (i.e. through the left face of an interior cell)
        i, j = np.where((sobel_x >= 2) & (sobel_x <= 4))
        #if inside:
        #    j += 1
        k = np.ones(len(i), dtype=int) * klay
        cellid += list(zip(k, i, j))
        if connection_info:
            bedleak += list(bdlknc[i, j])
            belev += list(layer_elevations[k + 1, i, j])
            telev += list(layer_elevations[k, i, j])
            connlen += list(0.5 * delr[j + 1] + 0.5 * delr[j])
            connwidth += list(delc[i])
        cellface += ['left'] * len(i)

        # bottom face connections
        # (i.e. through the bottom face of an interior cell)
        i, j = np.where((sobel_y <= -2) & (sobel_y >= -4))
        #if inside:
        #    i -= 1
        k = np.ones(len(i), dtype=int) * klay
        cellid += list(zip(k, i, j))
        if connection_info:
            bedleak += list(bdlknc[i, j])
            belev += list(layer_elevations[k + 1, i, j])
            telev += list(layer_elevations[k, i, j])
            connlen += list(0.5 * delc[i-1] + 0.5 * delc[i])
            connwidth += list(delr[j])
        cellface += ['bottom'] * len(i)

        # top face connections
        i, j = np.where((sobel_y >= 2) & (sobel_y <= 4))
        #if inside:
        #    i += 1
        k = np.ones(len(i), dtype=int) * klay
        cellid += list(zip(k, i, j))
        if connection_info:
            bedleak += list(bdlknc[i, j])
            belev += list(layer_elevations[k + 1, i, j])
            telev += list(layer_elevations[k, i, j])
            connlen += list(0.5 * delc[i + 1] + 0.5 * delc[i])
            connwidth += list(delr[j])
        cellface += ['top'] * len(i)
    try:
        k, i, j = zip(*cellid)
    except:
        j=2
    data = {'k': k,
            'i': i,
            'j': j,
            'cellface': cellface
            }
    if connection_info:
        data.update({'claktype': 'horizontal',
                     'bedleak': bedleak,
                     'belev': belev,
                     'telev': telev,
                     'connlen': connlen,
                     'connwidth': connwidth,
                    })
    df = pd.DataFrame(data)
    return df


class PrismSourceData(SourceData, TransientSourceDataMixin):
    """Subclass for handling tabular source data that
    represents a time series."""

    def __init__(self, filenames, period_stats='mean', id_column='lake_id',
                 dest_temperature_units='celsius',
                 dest_model=None):
        SourceData.__init__(self, filenames=filenames,
                            dest_model=dest_model)
        TransientSourceDataMixin.__init__(self, period_stats=period_stats, dest_model=dest_model)


        self.id_column = id_column
        self.data_columns = ['precipitation', 'temp']
        self.datetime_column = 'datetime'
        self.dest_temperature_units = dest_temperature_units

    def parse_header(self, f):
        meta = {}
        with open(f) as src:
            for i, line in enumerate(src):
                if 'Location' in line:
                    _, _, lat, _, lon, _, _ = line.strip().split()
                if 'Date' in line:
                    names = line.strip().split(',')
                    meta['length_units'] = names[1].split()[1].strip('()')
                    meta['temperature_units'] = names[2].split()[-1].strip('()')
                    names = [self.datetime_column] + self.data_columns
                    break
        self.data_columns = names[1:]
        meta['latitude'] = float(lat)
        meta['longitude'] = float(lon)
        meta['column_names'] = names
        meta['skiprows'] = i + 1
        meta['temp_conversion'] = convert_temperature_units(meta['temperature_units'],
                                                            self.dest_temperature_units)
        return meta

    def get_data(self):

        # aggregate the data from multiple files
        dfs = []
        for lake_id, f in self.filenames.items():
            meta = self.parse_header(f)
            df = pd.read_csv(f, skiprows=meta['skiprows'],
                             header=None, names=meta['column_names'])
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
            df.index = df[self.datetime_column]
            df['start_datetime'] = df.index
            # check if data are monthly
            ndays0 = (df.index[1] - df.index[0]).days
            ismonthly = ndays0 >=28 & ndays0 <=31
            if ismonthly:
                ndays = df.index.days_in_month
                df['end_datetime'] = df['start_datetime'] + pd.to_timedelta(ndays, unit='D')
            elif ndays0 == 1:
                ndays = 1
                df['end_datetime'] = df['start_datetime']
            else:
                raise ValueError("Check {}; only monthly or daily values supported.")

            # convert precip to model units
            # assumes that precip is monthly values
            mult = convert_length_units(meta['length_units'],
                                        self.dest_model.length_units)
            df[meta['column_names'][1]] = df[meta['column_names'][1]] * mult/ndays

            # convert temperatures to C
            df[meta['column_names'][2]] = meta['temp_conversion'](df[meta['column_names'][2]])

            # record lake ID
            df[self.id_column] = lake_id
            dfs.append(df)
        df = pd.concat(dfs)

        # sample values to model stress periods
        #starttimes = self.dest_model.perioddata['start_datetime'].copy()
        #endtimes = self.dest_model.perioddata['end_datetime'].copy()

        # if period ends are specified as the same as the next starttime
        # need to subtract a day, otherwise
        # pandas will include the first day of the next period in slices
        #endtimes_equal_startimes = np.all(endtimes[:-1].values == starttimes[1:].values)
        #if endtimes_equal_startimes:
        #    endtimes -= pd.Timedelta(1, unit='d')

        period_data = []
        #current_stat = None
        #for kper, (start, end) in enumerate(zip(starttimes, endtimes)):
        for kper, period_stat in self.period_stats.items():
            if period_stat is None:
                continue
            # missing (period) keys default to 'mean';
            # 'none' to explicitly skip the stress period
            #period_stat = self.period_stats.get(kper, current_stat)
            #current_stat = period_stat
            aggregated = aggregate_dataframe_to_stress_period(df, id_column=self.id_column, data_column=self.data_columns,
                                                              datetime_column=self.datetime_column,
                                                              **period_stat)
            aggregated['per'] = kper
            period_data.append(aggregated)
        dfm = pd.concat(period_data)
        dfm.sort_values(by=['per', self.id_column], inplace=True)
        return dfm.reset_index(drop=True)
