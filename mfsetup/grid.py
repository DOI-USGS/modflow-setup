"""
Grid stuff that flopy.discretization.StructuredGrid doesn't do and other grid-related functions
"""
import numpy as np
import pandas as pd
from rasterio import Affine
from shapely.geometry import Polygon
from flopy.discretization import StructuredGrid
from .gis import df2shp
from mfsetup.units import lenuni_text


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
        if not np.allclose(other.xoff, self.xoff):
            return False
        if not np.allclose(other.yoff, self.yoff):
            return False
        if not np.allclose(other.angrot, self.angrot):
            return False
        if not np.allclose(other.proj4, self.proj4):
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
        return 1.

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


def write_bbox_shapefile(sr, outshp):
    outline = get_grid_bounding_box(sr)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=sr.epsg)




