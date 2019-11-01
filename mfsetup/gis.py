import os
import warnings
import collections
import time
from functools import partial
import fiona
from shapely.ops import transform
from shapely.geometry import shape, mapping
import pyproj
import numpy as np
import pandas as pd


def df2shp(dataframe, shpname, geo_column='geometry', index=False,
           retain_order=False,
           prj=None, epsg=None, proj4=None, crs=None):
    '''
    Write a DataFrame to a shapefile
    dataframe: dataframe to write to shapefile
    geo_column: optional column containing geometry to write - default is 'geometry'
    index: If true, write out the dataframe index as a column
    retain_order : boolean
        Retain column order in dataframe, using an OrderedDict. Shapefile will
        take about twice as long to write, since OrderedDict output is not
        supported by the pandas DataFrame object.

    --->there are four ways to specify the projection....choose one
    prj: <file>.prj filename (string)
    epsg: EPSG identifier (integer)
    proj4: pyproj style projection string definition
    crs: crs attribute (dictionary) as read by fiona
    '''
    # first check if output path exists
    if os.path.split(shpname)[0] != '' and not os.path.isdir(os.path.split(shpname)[0]):
        raise IOError("Output folder doesn't exist")

    # check for empty dataframe
    if len(dataframe) == 0:
        raise IndexError("DataFrame is empty!")

    df = dataframe.copy()  # make a copy so the supplied dataframe isn't edited

    # reassign geometry column if geo_column is special (e.g. something other than "geometry")
    if geo_column != 'geometry':
        df['geometry'] = df[geo_column]
        df.drop(geo_column, axis=1, inplace=True)

    # assign none for geometry, to write a dbf file from dataframe
    Type = None
    if 'geometry' not in df.columns:
        df['geometry'] = None
        Type = 'None'
        mapped = [None] * len(df)

    # reset the index to integer index to enforce ordering
    # retain index as attribute field if index=True
    df.reset_index(inplace=True, drop=not index)

    # enforce character limit for names! (otherwise fiona marks it zero)
    # somewhat kludgey, but should work for duplicates up to 99
    df.columns = list(map(str, df.columns))  # convert columns to strings in case some are ints
    overtheline = [(i, '{}{}'.format(c[:8], i)) for i, c in enumerate(df.columns) if len(c) > 10]

    newcolumns = list(df.columns)
    for i, c in overtheline:
        newcolumns[i] = c
    df.columns = newcolumns

    properties = shp_properties(df)
    del properties['geometry']

    # set projection (or use a prj file, which must be copied after shp is written)
    # alternatively, provide a crs in dictionary form as read using fiona
    # from a shapefile like fiona.open(inshpfile).crs

    if epsg is not None:
        from fiona.crs import from_epsg
        crs = from_epsg(int(epsg))
    elif proj4 is not None:
        from fiona.crs import from_string
        crs = from_string(proj4)
    elif crs is not None:
        pass
    else:
        pass

    if Type != 'None':
        for g in df.geometry:
            try:
                Type = g.type
            except:
                continue
        mapped = [mapping(g) for g in df.geometry]

    schema = {'geometry': Type, 'properties': properties}
    length = len(df)

    if not retain_order:
        props = df.drop('geometry', axis=1).astype(object).to_dict(orient='records')
    else:
        props = [collections.OrderedDict(r) for i, r in df.drop('geometry', axis=1).astype(object).iterrows()]
    print('writing {}...'.format(shpname))
    with fiona.collection(shpname, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as output:
        for i in range(length):
            output.write({'properties': props[i],
                          'geometry': mapped[i]})

    if prj is not None:
        """
        if 'epsg' in prj.lower():
            epsg = int(prj.split(':')[1])
            prjstr = getPRJwkt(epsg).replace('\n', '') # get rid of any EOL
            ofp = open("{}.prj".format(shpname[:-4]), 'w')
            ofp.write(prjstr)
            ofp.close()
        """
        try:
            print('copying {} --> {}...'.format(prj, "{}.prj".format(shpname[:-4])))
            shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))
        except IOError:
            print('Warning: could not find specified prj file. shp will not be projected.')


def shp2df(shplist, index=None, index_dtype=None, clipto=[], filter=None,
           true_values=None, false_values=None, layer=None,
           skip_empty_geom=True):
    """Read shapefile/DBF, list of shapefiles/DBFs, or File geodatabase (GDB)
     into pandas DataFrame.

    Parameters
    ----------
    shplist : string or list
        of shapefile/DBF name(s) or FileGDB
    index : string
        Column to use as index for dataframe
    index_dtype : dtype
        Enforces a datatype for the index column (for example, if the index field is supposed to be integer
        but pandas reads it as strings, converts to integer)
    clipto : list
        limit what is brought in to items in index of clipto (requires index)
    filter : tuple (xmin, ymin, xmax, ymax)
        bounding box to filter which records are read from the shapefile.
    true_values : list
        same as argument for pandas read_csv
    false_values : list
        same as argument for pandas read_csv
    layer : str
        Layer name to read (if opening FileGDB)
    skip_empty_geom : True/False, default True
        Drops shapefile entries with null geometries.
        DBF files (which specify null geometries in their schema) will still be read.

    Returns
    -------
    df : DataFrame
        with attribute fields as columns; feature geometries are stored as
    shapely geometry objects in the 'geometry' column.
    """
    if isinstance(shplist, str):
        shplist = [shplist]
    if not isinstance(true_values, list) and true_values is not None:
        true_values = [true_values]
    if not isinstance(false_values, list) and false_values is not None:
        false_values = [false_values]
    if len(clipto) > 0 and index:
        clip = True
    else:
        clip = False

    df = pd.DataFrame()
    for shp in shplist:
        print("\nreading {}...".format(shp))
        if not os.path.exists(shp):
            raise IOError("{} doesn't exist".format(shp))
        shp_obj = fiona.open(shp, 'r', layer=layer)

        if index is not None:
            # handle capitolization issues with index field name
            fields = list(shp_obj.schema['properties'].keys())
            index = [f for f in fields if index.lower() == f.lower()][0]

        attributes = []
        # for reading in shapefiles
        meta = shp_obj.meta
        if meta['schema']['geometry'] != 'None':
            if filter is not None:
                print('filtering on bounding box {}, {}, {}, {}...'.format(*filter))
            if clip:  # limit what is brought in to items in index of clipto
                for line in shp_obj.filter(bbox=filter):
                    props = line['properties']
                    if not props[index] in clipto:
                        continue
                    props['geometry'] = line.get('geometry', None)
                    attributes.append(props)
            else:
                for line in shp_obj.filter(bbox=filter):
                    props = line['properties']
                    props['geometry'] = line.get('geometry', None)
                    attributes.append(props)
            print('--> building dataframe... (may take a while for large shapefiles)')
            shp_df = pd.DataFrame(attributes)
            # reorder fields in the DataFrame to match the input shapefile
            if len(attributes) > 0:
                shp_df = shp_df[list(attributes[0].keys())]

            # handle null geometries
            if len(shp_df) == 0:
                print('Empty dataframe! No features were read.')
                if filter is not None:
                    print('Check filter {} for consistency \
with shapefile coordinate system'.format(filter))
            geoms = shp_df.geometry.tolist()
            if geoms.count(None) == 0:
                shp_df['geometry'] = [shape(g) for g in geoms]
            elif skip_empty_geom:
                null_geoms = [i for i, g in enumerate(geoms) if g is None]
                shp_df.drop(null_geoms, axis=0, inplace=True)
                shp_df['geometry'] = [shape(g) for g in shp_df.geometry.tolist()]
            else:
                shp_df['geometry'] = [shape(g) if g is not None else None
                                      for g in geoms]

        # for reading in DBF files (just like shps, but without geometry)
        else:
            if clip:  # limit what is brought in to items in index of clipto
                for line in shp_obj:
                    props = line['properties']
                    if not props[index] in clipto:
                        continue
                    attributes.append(props)
            else:
                for line in shp_obj:
                    attributes.append(line['properties'])
            print('--> building dataframe... (may take a while for large shapefiles)')
            shp_df = pd.DataFrame(attributes)
            # reorder fields in the DataFrame to match the input shapefile
            if len(attributes) > 0:
                shp_df = shp_df[list(attributes[0].keys())]

        shp_obj.close()
        if len(shp_df) == 0:
            continue
        # set the dataframe index from the index column
        if index is not None:
            if index_dtype is not None:
                shp_df[index] = shp_df[index].astype(index_dtype)
            shp_df.index = shp_df[index].values

        df = df.append(shp_df)

        # convert any t/f columns to numpy boolean data
        if true_values is not None or false_values is not None:
            replace_boolean = {}
            for t in true_values:
                replace_boolean[t] = True
            for f in false_values:
                replace_boolean[f] = False

            # only remap columns that have values to be replaced
            cols = [c for c in df.columns if c != 'geometry']
            for c in cols:
                if len(set(replace_boolean.keys()).intersection(set(df[c]))) > 0:
                    df[c] = df[c].map(replace_boolean)

    return df


def shp_properties(df):

    newdtypes = {'bool': 'str',
                 'object': 'str',
                 'datetime64[ns]': 'str'}

    # fiona/OGR doesn't like numpy ints
    # shapefile doesn't support 64 bit ints,
    # but apparently leaving the ints alone is more reliable
    # than intentionally downcasting them to 32 bit
    # pandas is smart enough to figure it out on .to_dict()?
    for c in df.columns:
        if c != 'geometry':
            df[c] = df[c].astype(newdtypes.get(df.dtypes[c].name,
                                               df.dtypes[c].name))
        if 'int' in df.dtypes[c].name:
            if np.max(np.abs(df[c])) > 2**31 -1:
                df[c] = df[c].astype(str)

    # strip dtypes to just 'float', 'int' or 'str'
    def stripandreplace(s):
        return ''.join([i for i in s
                        if not i.isdigit()]).replace('object', 'str')
    dtypes = [stripandreplace(df[c].dtype.name)
              if c != 'geometry'
              else df[c].dtype.name for c in df.columns]
    properties = collections.OrderedDict(list(zip(df.columns, dtypes)))
    return properties


def project(geom, projection1, projection2):
    """Reproject shapely geometry object(s) or scalar
    coodrinates to new coordinate system

    Parameters
    ----------
    geom: shapely geometry object, list of shapely geometry objects,
          list of (x, y) tuples, or (x, y) tuple.
    projection1: string
        Proj4 string specifying source projection
    projection2: string
        Proj4 string specifying destination projection
    """
    # pyproj 2 style
    # https://pyproj4.github.io/pyproj/dev/gotchas.html
    transformer = pyproj.Transformer.from_crs(projection1, projection2, always_xy=True)

    # check for x, y values instead of shapely objects
    if isinstance(geom, tuple):
        # tuple of scalar values
        if np.isscalar(geom[0]):
            return transformer.transform(*geom)
        elif isinstance(geom[0], collections.Iterable):
            return transformer.transform(*geom)
            #return np.squeeze([projectXY(geom[0], geom[1], projection1, projection2)])

    # sequence of tuples or shapely objects
    if isinstance(geom, collections.Iterable):
        geom = list(geom) # in case it's a generator
        geom0 = geom[0]
    else:
        geom0 = geom

    # sequence of tuples
    if isinstance(geom0, tuple):
        a = np.array(geom)
        x = a[:, 0]
        y = a[:, 1]
        return transformer.transform(x, y)

    # transform shapely objects
    # enforce strings
    projection1 = str(projection1)
    projection2 = str(projection2)

    # define projections
    #pr1 = pyproj.Proj(projection1, errcheck=True, preserve_units=True)
    #pr2 = pyproj.Proj(projection2, errcheck=True, preserve_units=True)



    # projection function
    # (see http://toblerity.org/shapely/shapely.html#module-shapely.ops)
    #project = partial(pyproj.transform, pr1, pr2)
    project = partial(transformer.transform)

    # do the transformation!
    if isinstance(geom, collections.Iterable):
        return [transform(project, g) for g in geom]
    return transform(project, geom)


def get_values_at_points(rasterfile, x=None, y=None,
                         points=None, out_of_bounds_errors='coerce'):
    """Get raster values single point or list of points.
    Points must be in same coordinate system as raster.

    Parameters
    ----------
    rasterfile : str
        Filename of raster.
    x : 1D array
        X coordinate locations
    y : 1D array
        Y coordinate locations
    points : list of tuples or 2D numpy array (npoints, (row, col))
        Points at which to sample raster.
    out_of_bounds_errors : {‘raise’, ‘coerce’}, default 'raise'
        * If 'raise', then x, y locations outside of the raster will raise an exception.
        * If 'coerce', then x, y locations outside of the raster will be set to NaN.
    Returns
    -------
    list of floats

    Notes
    -----
    requires gdal
    """
    from osgeo import gdal

    # read in sample points
    if x is not None and isinstance(x[0], tuple):
        x, y = np.array(x).transpose()
        warnings.warn(
            "new argument input for get_values_at_points is x, y, or points"
        )
    elif x is not None:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
    elif points is not None:
        if not isinstance(points, np.ndarray):
            x, y = np.array(points)
        else:
            x, y = points[:, 0], points[:, 1]
    else:
        print('Must supply x, y or list/array of points.')

    assert os.path.exists(rasterfile), "raster {} not found".format(rasterfile)
    t0 = time.time()

    print("reading data from {}...".format(rasterfile))
    if rasterfile.endswith('.asc'):
        data, meta = read_arc_ascii(rasterfile)
        xul, dx, rx, yul, ry, dy = meta['geotransform']
        nodata = meta['nodata_value']
    else:
        # open the raster
        gdata = gdal.Open(rasterfile)
        # get the location info
        xul, dx, rx, yul, ry, dy = gdata.GetGeoTransform()
        # read the array
        data = gdata.ReadAsArray().astype(np.float)
        nodata = gdata.GetRasterBand(1).GetNoDataValue()

    nrow, ncol = data.shape

    print("getting nearest values at points...")
    # find the closest row, col location for each point
    i = ((y - yul) / dy).astype(int)
    j = ((x - xul) / dx).astype(int)

    # mask row, col locations outside the raster
    within = (i > 0) & (i < nrow) & (j > 0) & (j < ncol)

    # get values at valid point locations
    results = np.ones(len(i), dtype=float) * np.nan
    results[within] = data[i[within], j[within]]
    if out_of_bounds_errors == 'raise' and np.any(np.isnan(results)):
        n_invalid = np.sum(np.isnan(results))
        raise ValueError("{} points outside of {} extent.".format(n_invalid, rasterfile))

    # convert nodata values to np.nans
    results[results == nodata] = np.nan
    print("finished in {:.2f}s".format(time.time() - t0))
    return results


def intersect(feature, grid, id_column=None,
              epsg=None,
              proj4=None, dtype=np.float32):
    """Intersect a feature with the model grid, using
    the rasterio.features.rasterize method. Features are intersected
    if they contain the cell center.

    Parameters
    ----------
    feature : str (shapefile path), list of shapely objects,
              or dataframe with geometry column
    id_column : str
        Column with unique integer identifying each feature; values
        from this column will be assigned to the output raster.
    grid : grid.StructuredGrid instance
    epsg : int
        EPSG code for feature coordinate reference system. Optional,
        but an epgs code or proj4 string must be supplied if feature
        isn't a shapefile, and isn't in the same CRS as the model.
    proj4 : str
        Proj4 string for feature CRS (optional)
    dtype : dtype
        Datatype for the output array

    Returns
    -------
    2D numpy array with intersected values

    """
    try:
        from rasterio import features
        from rasterio import Affine
    except:
        print('This method requires rasterio.')
        return

    #trans = Affine(sr.delr[0], 0., sr.xul,
    #               0., -sr.delc[0], sr.yul) * Affine.rotation(sr.rotation)
    trans = grid.transform

    if isinstance(feature, str):
        proj4 = get_proj4(feature)
        df = shp2df(feature)
    elif isinstance(feature, pd.DataFrame):
        df = feature.copy()
    elif isinstance(feature, collections.Iterable):
        df = pd.DataFrame({'geometry': feature})
    elif not isinstance(feature, collections.Iterable):
        df = pd.DataFrame({'geometry': [feature]})
    else:
        print('unrecognized feature input')
        return

    # handle shapefiles in different CRS than model grid
    reproject = False
    if proj4 is not None:
        if proj4 != grid.proj_str:
            reproject = True
    elif epsg is not None and grid.epsg is not None:
        if epsg != grid.epsg:
            reproject = True
            from fiona.crs import to_string, from_epsg
            proj4 = to_string(from_epsg(epsg))
    if reproject:
        df['geometry'] = project(df.geometry.values, proj4, grid.proj_str)

    # create list of GeoJSON features, with unique value for each feature
    if id_column is None:
        numbers = range(1, len(df)+1)
    # if IDs are strings, get a number for each one
    # pd.DataFrame.unique() generally preserves order
    elif isinstance(df[id_column].dtype, np.object):
        unique_values = df[id_column].unique()
        values = dict(zip(unique_values, range(1, len(unique_values) + 1)))
        numbers = [values[n] for n in df[id_column]]
    else:
        numbers = df[id_column].tolist()

    geoms = list(zip(df.geometry, numbers))
    result = features.rasterize(geoms,
                                out_shape=(grid.nrow, grid.ncol),
                                transform=trans)
    assert result.sum(axis=(0, 1)) != 0, "Nothing was intersected!"
    return result.astype(dtype)


def zonal_stats(feature, raster, out_shape=None,
                stats=['mean']):
    from rasterstats import zonal_stats

    if not isinstance(feature, str):
        feature_name = 'feature'
    else:
        feature_name = feature
    t0 = time.time()
    print('computing {} {} for zones in {}...'.format(raster,
                                                      ', '.join(stats),
                                                      feature_name
                                                      ))
    print(stats)
    results = zonal_stats(feature, raster, stats=stats)
    print(out_shape)
    if out_shape is None:
        out_shape = (len(results),)
    #print(results)
    #means = [r['mean'] for r in results]
    #means = np.asarray(means)
    #means = np.reshape(means, out_shape).astype(float)
    #results = means

    #results = np.reshape(results, out_shape)
    #results = np.reshape(results, out_shape).astype(float)
    results_dict = {}
    for stat in stats:
        res = [r[stat] for r in results]
        res = np.asarray(res)
        res = np.reshape(res, out_shape).astype(float)
        results_dict[stat] = res
    #print("finished in {:.2f}s".format(time.time() - t0))
    return results_dict


def get_proj4(prj):
    """Get proj4 string for a projection file

    Parameters
    ----------
    prj : string
        Shapefile or projection file

    Returns
    -------
    proj4 string (http://trac.osgeo.org/proj/)

    """
    '''
    Using fiona (couldn't figure out how to do this with just a prj file)
    from fiona.crs import to_string
    c = fiona.open(shp).crs
    proj4 = to_string(c)
    '''
    # using osgeo
    from osgeo import osr

    prjfile = prj[:-4] + '.prj' # allows shp or prj to be argued
    try:
        with open(prjfile) as src:
            prjtext = src.read()
        srs = osr.SpatialReference()
        srs.ImportFromESRI([prjtext])
        proj4 = srs.ExportToProj4()
        return proj4
    except:
        pass


def arc_ascii(array, filename, xll=0, yll=0, cellsize=1.,
              nodata=-9999, **kwargs):
    """Write numpy array to Arc Ascii grid.

    Parameters
    ----------
    kwargs: keyword arguments to np.savetxt
    """
    array = array.copy()
    array[np.isnan(array)] = nodata

    filename = '.'.join(filename.split('.')[:-1]) + '.asc'  # enforce .asc ending
    nrow, ncol = array.shape
    txt = 'ncols  {:d}\n'.format(ncol)
    txt += 'nrows  {:d}\n'.format(nrow)
    txt += 'xllcorner  {:f}\n'.format(xll)
    txt += 'yllcorner  {:f}\n'.format(yll)
    txt += 'cellsize  {}\n'.format(cellsize)
    txt += 'NODATA_value  {:.0f}\n'.format(nodata)
    with open(filename, 'w') as output:
        output.write(txt)
    with open(filename, 'ab') as output:
        np.savetxt(output, array, **kwargs)
    print('wrote {}'.format(filename))


def read_arc_ascii(filename, shape=None):
    with open(filename) as src:
        meta = {}
        for i in range(6):
            k, v = next(src).strip().split()
            v = float(v) if '.' in v else int(v)
            meta[k.lower()] = v

        # make a gdal-style geotransform
        dx = meta['cellsize']
        dy = meta['cellsize']
        xul = meta['xllcorner']
        yul = meta['yllcorner'] + dy * meta['nrows']
        rx, ry = 0, 0
        meta['geotransform'] = xul, dx, rx, yul, ry, -dy

        if shape is not None:
            assert (meta['nrow'], meta['ncol']) == shape, \
                "Data in {} are {}x{}, expected {}x{}".format(filename,
                                                              meta['nrows'],
                                                              meta['ncols'],
                                                              *shape)
        arr = np.loadtxt(src)
    return arr, meta
