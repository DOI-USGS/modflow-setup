"""
Grid stuff that flopy.discretization.StructuredGrid doesn't do and other grid-related functions
"""
import os
import time
import collections
import numpy as np
import pandas as pd
from rasterio import Affine
from shapely.geometry import Polygon, MultiPolygon
from flopy.discretization import StructuredGrid
from gisutils import df2shp, get_proj_str, project, shp2df
import mfsetup.fileio as fileio
from .units import convert_length_units
from .utils import get_input_arguments


class MFsetupGrid(StructuredGrid):

    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 lenuni=None, epsg=None, proj4=None, prj=None, xoff=0.0,
                 yoff=0.0, xul=None, yul=None, angrot=0.0):

        super(MFsetupGrid, self).__init__(np.array(delc), np.array(delr),
                                          top, botm, idomain,
                                          lenuni, epsg, proj4, prj, xoff,
                                          yoff, angrot)
        # properties
        self._vertices = None
        self._polygons = None

        # if no epsg, set from proj4 string if possible
        if epsg is None and proj4 is not None and 'epsg' in proj4.lower():
            self.epsg = int(proj4.split(':')[1])

        # in case the upper left corner is known but the lower left corner is not
        if xul is not None and yul is not None:
            xll = self._xul_to_xll(xul)
            yll = self._yul_to_yll(yul)
            self.set_coord_info(xoff=xll, yoff=yll, epsg=epsg, proj4=proj4)

    def __eq__(self, other):
        if not isinstance(other, StructuredGrid):
            return False
        if not np.allclose(other.xoffset, self.xoffset):
            return False
        if not np.allclose(other.yoffset, self.yoffset):
            return False
        if not np.allclose(other.angrot, self.angrot):
            return False
        if not other.proj_str == self.proj_str:
            return False
        if not np.array_equal(other.delr, self.delr):
            return False
        if not np.array_equal(other.delc, self.delc):
            return False
        return True

    @property
    def xul(self):
        x0 = self.xyedges[0][0]
        y0 = self.xyedges[1][0]
        x0r, y0r = self.get_coords(x0, y0)
        return x0r

    @property
    def yul(self):
        x0 = self.xyedges[0][0]
        y0 = self.xyedges[1][0]
        x0r, y0r = self.get_coords(x0, y0)
        return y0r

    @property
    def bbox(self):
        """Shapely polygon bounding box of the model grid."""
        return get_grid_bounding_box(self)

    @property
    def bounds(self):
        """Grid bounding box in order used by shapely.
        """
        x0, x1, y0, y1 = self.extent
        return x0, y0, x1, y1

    @property
    def transform(self):
        """Rasterio Affine object (same as transform attribute of rasters).
        """
        return Affine(self.delr[0], 0., self.xul,
                      0., -self.delc[0], self.yul) * \
               Affine.rotation(self.angrot)

    @property
    def proj_str(self):
        return self.proj4

    @property
    def vertices(self):
        """Vertices for grid cell polygons."""
        if self._vertices is None:
            self._set_vertices()
        return self._vertices

    @property
    def polygons(self):
        """Vertices for grid cell polygons."""
        if self._polygons is None:
            self._set_polygons()
        return self._polygons

    def write_shapefile(self, filename='grid.shp'):
        i, j = np.indices((self.nrow, self.ncol))
        df = pd.DataFrame({'node': list(range(len(self.polygons))),
                           'i': i.ravel(),
                           'j': j.ravel(),
                           'geometry': self.polygons
                           })
        df2shp(df, filename, epsg=self.epsg, proj_str=self.proj_str)

    def _set_polygons(self):
        """
        Create shapely polygon for each grid cell
        """
        print('creating shapely Polygons of grid cells...')
        t0 = time.time()
        self._polygons = [Polygon(verts) for verts in self.vertices]
        print("finished in {:.2f}s\n".format(time.time() - t0))

    # stuff to conform to sr
    @property
    def length_multiplier(self):
        """assume that """
        return convert_length_units(self.lenuni,
                                    2)

    @property
    def rotation(self):
        return self.angrot

    def get_vertices(self, i, j):
        """Get vertices for a single cell or sequence if i, j locations."""
        return self._cell_vert_list(i, j)

    def _set_vertices(self):
        """
        Populate vertices for the whole grid
        """
        jj, ii = np.meshgrid(range(self.ncol), range(self.nrow))
        jj, ii = jj.ravel(), ii.ravel()
        self._vertices = self._cell_vert_list(ii, jj)


def get_ij(grid, x, y, local=False, chunksize=100):
    """Return the row and column of a point or sequence of points
    in real-world coordinates.

    Parameters
    ----------
    grid : flopy.discretization.StructuredGrid instance
    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates
    local : bool
        Flag for returning real-world or model (local) coordinates.
        (default False)
    chunksize : int
        Because this function compares each x, y location to a vector
        of model grid cell locations, memory usage can quickly get
        out of hand, as it increases as the square of the number of locations.
        This can be avoided by breaking the x, y location vectors into
        chunks. Experimentation with approx. 5M points suggests
        that a chunksize of 100 provides close to optimal
        performance in terms of execution time. (default 100)

    Returns
    -------
    i : row or sequence of rows (zero-based)
    j : column or sequence of columns (zero-based)
    """
    x = np.array(x)
    y = np.array(y)
    if not local:
        xc, yc = grid.xcellcenters, grid.ycellcenters
    else:
        xc, yc = grid.xyzcellcenters()

    if np.isscalar(x):
        j = (np.abs(xc[0] - x)).argmin()
        i = (np.abs(yc[:, 0] - y)).argmin()
    else:
        print('getting i, j locations...')
        t0 = time.time()
        chunks = list(range(0, len(x), chunksize)) + [None]
        i = []
        j = []
        for c in range(len(chunks))[:-1]:
            chunk_slice = slice(chunks[c], chunks[c+1])
            xcp = np.array([xc[0]] * (len(x[chunk_slice])))
            ycp = np.array([yc[:, 0]] * (len(x[chunk_slice])))
            j += (np.abs(xcp.transpose() - x[chunk_slice])).argmin(axis=0).tolist()
            i += (np.abs(ycp.transpose() - y[chunk_slice])).argmin(axis=0).tolist()
        i = np.array(i)
        j = np.array(j)
        print("finished in {:.2f}s\n".format(time.time() - t0))
    return i, j


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

    return Polygon([(x0r, y0r),
                    (x1r, y1r),
                    (x2r, y2r),
                    (x3r, y3r),
                    (x0r, y0r)])


def get_point_on_national_hydrogeologic_grid(x, y):
    """Given an x, y location representing the upper left
    corner of a model grid, return the upper left corner
    of the cell in the National Hydrogeologic Grid that
    contains it."""
    # national grid parameters
    xul, yul = -2553045.0, 3907285.0 # upper left corner
    ngrows = 4000
    ngcols = 4980
    natCellsize = 1000

    # locations of left and top cell edges
    ngx = np.arange(ngcols) * natCellsize
    ngy = np.arange(ngrows) * -natCellsize

    # nearest left and top edge to upper left corner
    j = int(np.floor((x - xul) / natCellsize))
    i = int(np.floor((yul - y) / natCellsize))
    return ngx[j] + xul, ngy[i] + yul


def write_bbox_shapefile(modelgrid, outshp):
    outline = get_grid_bounding_box(modelgrid)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=modelgrid.epsg)


def rasterize(feature, grid, id_column=None,
              include_ids=None,
              epsg=None,
              proj4=None, dtype=np.float32):
    """Rasterize a feature onto the model grid, using
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
        proj4 = get_proj_str(feature)
        df = shp2df(feature)
    elif isinstance(feature, pd.DataFrame):
        df = feature.copy()
    elif isinstance(feature, collections.Iterable):
        # list of shapefiles
        if isinstance(feature[0], str):
            proj4 = get_proj_str(feature[0])
            df = shp2df(feature)
        else:
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

    # subset to include_ids
    if id_column is not None and include_ids is not None:
        df = df.loc[df[id_column].isin(include_ids)].copy()

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


def setup_structured_grid(xoff=None, yoff=None, xul=None, yul=None,
                          nrow=None, ncol=None,
                          dxy=None, delr=None, delc=None,
                          top=None, botm=None,
                          rotation=0.,
                          parent_model=None, snap_to_NHG=False,
                          features=None, features_shapefile=None,
                          id_column=None, include_ids=[],
                          buffer=1000,
                          epsg=None, model_length_units=None,
                          grid_file='grid.json',
                          bbox_shapefile=None, **kwargs):
    """"""
    print('setting up model grid...')
    t0 = time.time()

    # conversions for model/parent model units to meters
    # set regular flag for handling delc/delr
    to_meters_inset = convert_length_units(model_length_units, 'meters')
    regular = True
    if dxy is not None:
        delr_m = np.round(dxy * to_meters_inset, 4) # dxy is specified in model units
        delc_m = delr_m
    if delr is not None:
        delr_m = np.round(delr * to_meters_inset, 4)  # delr is specified in model units
        if not np.isscalar(delr_m):
            if (set(delr_m)) == 1:
                delr_m = delr_m[0]
            else:
                regular = False
    if delc is not None:
        delc_m = np.round(delc * to_meters_inset, 4) # delc is specified in model units
        if not np.isscalar(delc_m):
            if (set(delc_m)) == 1:
                delc_m = delc_m[0]
            else:
                regular = False
    if parent_model is not None:
        to_meters_parent = convert_length_units(parent_model.dis.lenuni, 'meters')
        # parent model grid spacing in meters
        parent_delr_m = np.round(parent_model.dis.delr.array[0] * to_meters_parent, 4)
        parent_delc_m = np.round(parent_model.dis.delc.array[0] * to_meters_parent, 4)

    if epsg is None and parent_model is not None:
        epsg = parent_model.modelgrid.epsg

    # option 1: make grid from xoff, yoff and specified dimensions
    if xoff is not None and yoff is not None:
        assert nrow is not None and ncol is not None, \
            "Need to specify nrow and ncol if specifying xoffset and yoffset."
        if regular:
            height_m = np.round(delc_m * nrow, 4)
            width_m = np.round(delr_m * ncol, 4)
        else:
            height_m = np.sum(delc_m)
            width_m = np.sum(delr_m)

        # optionally align grid with national hydrologic grid
        # grids snapping to NHD must have spacings that are a factor of 1 km
        if snap_to_NHG:
            assert regular and np.allclose(1000 % delc_m, 0, atol=1e-4)
            x, y = get_point_on_national_hydrogeologic_grid(xoff, yoff)
            xoff = x
            yoff = y
            rotation = 0.

        # need to specify xul, yul in case snapping to parent
        # todo: allow snapping to parent grid on xoff, yoff
        if rotation != 0:
            raise NotImplementedError('Rotated grids not supported.')
        xul = xoff
        yul = yoff + height_m

    # option 2: make grid using buffered feature bounding box
    else:
        if features is None and features_shapefile is not None:
            # Make sure shapefile and bbox filter are in dest (model) CRS
            # TODO: CRS wrangling could be added to shp2df as a feature
            features_proj_str = get_proj_str(features_shapefile)
            model_proj_str = "epsg:{}".format(epsg)
            filter = None
            if parent_model is not None:
                if features_proj_str.lower() != model_proj_str:
                    filter = project(parent_model.modelgrid.bbox,
                                     model_proj_str, features_proj_str).bounds
                else:
                    filter = parent_model.modelgrid.bbox.bounds
            df = shp2df(features_shapefile,
                        filter=filter)
            if features_proj_str.lower() != model_proj_str:
                df['geometry'] = project(df['geometry'], features_proj_str, model_proj_str)

            # subset shapefile data to specified features
            rows = df.loc[df[id_column].isin(include_ids)]
            features = rows.geometry.tolist()

            # convert multiple features to a MultiPolygon
            if isinstance(features, list):
                if len(features) > 1:
                    features = MultiPolygon(features)
                else:
                    features = features[0]

            # size the grid based on the bbox for features
            x1, y1, x2, y2 = features.bounds
            L = buffer  # distance from area of interest to boundary
            xul = x1 - L
            yul = y2 + L
            height_m = np.round(yul - (y1 - L), 4) # initial model height from buffer distance
            width_m = np.round((x2 + L) - xul, 4)
            rotation = 0.  # rotation not supported with this option

    # align model with parent grid if there is a parent model
    # (and not snapping to national hydrologic grid)
    if parent_model is not None and not snap_to_NHG:

        # get location of coinciding cell in parent model for upper left
        pi, pj = parent_model.modelgrid.intersect(xul, yul)
        verts = np.array(parent_model.modelgrid.get_cell_vertices(pi, pj))
        xul, yul = verts[:, 0].min(), verts[:, 1].max()

        # adjust the dimensions to align remaining corners
        def roundup(number, increment):
            return int(np.ceil(number / increment) * increment)
        height = roundup(height_m, parent_delr_m)
        width = roundup(width_m, parent_delc_m)

        # update nrow, ncol after snapping to parent grid
        if regular:
            nrow = int(height / delc_m) # h is in meters
            ncol = int(width / delr_m)

    # set the grid configuration dictionary
    # spacing is in meters (consistent with projected CRS)
    # (modelgrid object will be updated automatically from this dictionary)
    #if rotation == 0.:
    #    xll = xul
    #    yll = yul - model.height
    grid_cfg = {'nrow': int(nrow), 'ncol': int(ncol),
                'delr': delr_m, 'delc': delc_m,
                'xoff': xoff, 'yoff': yoff,
                'xul': xul, 'yul': yul,
                'rotation': rotation,
                'epsg': epsg,
                'lenuni': 2
                }
    if regular:
        grid_cfg['delr'] = np.ones(grid_cfg['ncol'], dtype=float) * grid_cfg['delr']
        grid_cfg['delc'] = np.ones(grid_cfg['nrow'], dtype=float) * grid_cfg['delc']
    grid_cfg['delr'] = grid_cfg['delr'].tolist()  # for serializing to json
    grid_cfg['delc'] = grid_cfg['delc'].tolist()

    # renames for flopy modelgrid
    renames = {'rotation': 'angrot'}
    for k, v in renames.items():
        if k in grid_cfg:
            grid_cfg[v] = grid_cfg.pop(k)

    # set up the model grid instance
    grid_cfg['top'] = top
    grid_cfg['botm'] = botm
    grid_cfg.update(kwargs)  # update with any kwargs from function call
    kwargs = get_input_arguments(grid_cfg, MFsetupGrid)
    modelgrid = MFsetupGrid(**kwargs)
    modelgrid.cfg = grid_cfg

    # write grid info to json, and shapefile of bbox
    # omit top and botm arrays from json represenation of grid
    # (just for horizontal disc.)
    del grid_cfg['top']
    del grid_cfg['botm']
    fileio.dump(grid_file, grid_cfg)
    if bbox_shapefile is not None:
        write_bbox_shapefile(modelgrid, bbox_shapefile)
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return modelgrid


