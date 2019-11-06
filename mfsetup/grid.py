"""
Grid stuff that flopy.discretization.StructuredGrid doesn't do and other grid-related functions
"""
import collections
import numpy as np
import pandas as pd
from rasterio import Affine
from shapely.geometry import Polygon
from flopy.discretization import StructuredGrid
from gisutils import df2shp, get_proj_str, project, shp2df
from .units import convert_length_units


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


def get_ij(grid, x, y, local=False):
    """Return the row and column of a point or sequence of points
    in real-world coordinates.

    Parameters
    ----------
    grid : flopy.discretization.StructuredGrid instance
    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates

    Returns
    -------
    i : row or sequence of rows (zero-based)
    j : column or sequence of columns (zero-based)
    """
    if not local:
        xc, yc = grid.xcellcenters, grid.ycellcenters
    else:
        xc, yc = grid.xyzcellcenters()

    if np.isscalar(x):
        j = (np.abs(xc[0] - x)).argmin()
        i = (np.abs(yc[:, 0] - y)).argmin()
    else:
        xcp = np.array([xc[0]] * (len(x)))
        ycp = np.array([yc[:, 0]] * (len(x)))
        j = (np.abs(xcp.transpose() - x)).argmin(axis=0)
        i = (np.abs(ycp.transpose() - y)).argmin(axis=0)
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


def write_bbox_shapefile(sr, outshp):
    outline = get_grid_bounding_box(sr)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=sr.epsg)


def rasterize(feature, grid, id_column=None,
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


