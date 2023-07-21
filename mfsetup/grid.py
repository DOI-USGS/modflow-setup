"""
Code for creating and working with regular (structured) grids. Focus is on the 2D representation of
the grid in the cartesian plane. For methods involving layering (in the vertical dimension), see
the discretization module.
"""
import collections
import time
import warnings
from pathlib import Path

import geopandas as gpd
import gisutils
import numpy as np
import pandas as pd
import pyproj
from flopy.discretization import StructuredGrid
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from geopandas.geodataframe import GeoDataFrame
from gisutils import df2shp, get_proj_str, project, shp2df
from packaging import version
from rasterio import Affine
from scipy import spatial
from shapely.geometry import MultiPolygon, Polygon

from mfsetup import fileio as fileio

from .mf5to6 import get_model_length_units
from .units import convert_length_units
from .utils import get_input_arguments


class MFsetupGrid(StructuredGrid):
    """Class representing a structured grid. Extends flopy.discretization.StructuredGrid
    to facilitate gis operations in a projected (real-word) coordinate reference system (CRS).

    Parameters
    ----------
    delc : ndarray
        1D numpy array of grid spacing along a column (len nrow), in CRS units.
    delr : ndarray
        1D numpy array of grid spacing along a row (len ncol), in CRS units.
    top : ndarray
        2D numpy array of model top elevations
    botm : ndarray
        3D numpy array of model bottom elevations
    idomain : ndarray
        3D numpy array of model idomain values
    laycbd : ndarray
        (Modflow 2005 and earlier style models only):
        LAYCBD—is a flag, with one value for each model layer,
        that indicates whether or not a layer has a Quasi-3D
        confining bed below it. 0 indicates no confining bed,
        and not zero indicates a confining bed.
        LAYCBD for the bottom layer must be 0.
    lenuni : int, optional
        MODFLOW length units variable. See
        `the Online Guide to MODFLOW <https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/index.html?beginners_guide_to_modflow.htm>`_
    epsg : int, optional
        EPSG code for the model CRS
    proj_str : str, optional
        PROJ string for model CRS. In general, a spatial reference ID
        (such as an EPSG code) or Well-Known Text (WKT) string is prefered
        over a PROJ string (see References)
    prj : str, optional
        Filepath for ESRI projection file (containing wkt) describing model CRS
    wkt : str, optional
        Well-known text string describing model CRS.
    crs : obj, optional
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to :meth:`pyproj.crs.CRS.from_user_input`
        Can be any of:

          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

    xoff, yoff : float, float, optional
        Model grid offset (location of lower left corner), by default 0.0, 0.0
    xul, yul : float, float, optional
        Model grid offset (location of upper left corner), by default 0.0, 0.0
    angrot : float, optional
        Rotation of the model grid, in degrees counter-clockwise about the lower left corner.
        Non-zero rotation values require input of xoff, yoff (xul, yul not supported).
        By default 0.0

    References
    ----------
    https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems

    """

    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 laycbd=None, lenuni=None, binary_grid_file=None,
                 epsg=None, proj_str=None, prj=None, wkt=None, crs=None,
                 xoff=0.0, yoff=0.0, xul=None, yul=None, angrot=0.0):
        super(MFsetupGrid, self).__init__(delc=np.array(delc), delr=np.array(delr),
                                          top=top, botm=botm, idomain=idomain,
                                          laycbd=laycbd, lenuni=lenuni,
                                          epsg=epsg, proj4=proj_str, prj=prj,
                                          xoff=xoff, yoff=yoff, angrot=angrot
                                          )

        # properties
        self._crs = None
        # pass all CRS representations through pyproj.CRS.from_user_input
        # to convert to pyproj.CRS instance
        self.crs = get_crs(crs=crs, epsg=epsg, prj=prj, wkt=wkt, proj_str=proj_str)

        # other CRS-related properties are set in the flopy Grid base class
        self._vertices = None
        self._polygons = None
        self._dataframe = None

        # MODFLOW 6 binary grid file, for getting intercell connections
        # (needed for reading cell budget files)
        self.binary_grid_file = binary_grid_file

        # if no epsg, set from proj4 string if possible
        #if epsg is None and proj_str is not None and 'epsg' in proj_str.lower():
        #    self.epsg = int(proj_str.split(':')[1])

        # in case the upper left corner is known but the lower left corner is not
        if xul is not None and yul is not None:
            xll = self._xul_to_xll(xul)
            yll = self._yul_to_yll(yul)
            self.set_coord_info(xoff=xll, yoff=yll, epsg=epsg, proj4=proj_str, angrot=angrot)

    def __eq__(self, other):
        if not isinstance(other, StructuredGrid):
            return False
        if not np.allclose(other.xoffset, self.xoffset):
            return False
        if not np.allclose(other.yoffset, self.yoffset):
            return False
        if not np.allclose(other.angrot, self.angrot):
            return False
        if not other.crs == self.crs:
            return False
        if not np.array_equal(other.delr, self.delr):
            return False
        if not np.array_equal(other.delc, self.delc):
            return False
        return True

    def __repr__(self):
        txt = ''
        if self.nlay is not None:
            txt += f'{self.nlay:d} layer(s), '
        txt += f'{self.nrow:d} row(s), {self.ncol:d} column(s)\n'
        txt += (f'delr: [{self.delr[0]:.2f}...{self.delr[-1]:.2f}]'
                f' {self.units}\n'
                f'delc: [{self.delc[0]:.2f}...{self.delc[-1]:.2f}]'
                f' {self.units}\n'
                )
        txt += f'CRS: {self.crs}\n'
        txt += f'length units: {self.length_units}\n'
        txt += f'xll: {self.xoffset}; yll: {self.yoffset}; rotation: {self.rotation}\n'
        txt += 'Bounds: {}\n'.format(self.extent)
        return txt

    def __str__(self):
        return StructuredGrid.__repr__(self)

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
    def size(self):
        if self.nlay is None:
            return self.nrow * self.ncol
        return self.nlay * self.nrow * self.ncol

    @property
    def transform(self):
        """Rasterio Affine object (same as transform attribute of rasters).
        """
        return get_transform(self)

    @property
    def crs(self):
        """pyproj.crs.CRS instance describing the coordinate reference system
        for the model grid.
        """
        return self._crs

    @crs.setter
    def crs(self, crs):
        """Get a pyproj CRS instance from various inputs
        (epsg, proj string, wkt, etc.).

        crs : obj, optional
            Coordinate reference system for model grid.
            A Python int, dict, str, or pyproj.crs.CRS instance
            passed to the pyproj.crs.from_user_input
            See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
            Can be any of:
              - PROJ string
              - Dictionary of PROJ parameters
              - PROJ keyword arguments for parameters
              - JSON string with PROJ parameters
              - CRS WKT string
              - An authority string [i.e. 'epsg:4326']
              - An EPSG integer code [i.e. 4326]
              - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
              - An object with a `to_wkt` method.
              - A :class:`pyproj.crs.CRS` class
        """
        crs = get_crs(crs=crs)
        self._crs = crs

    @property
    def proj_str(self):
        if self.crs is not None:
            return self.crs.to_proj4()

    @property
    def wkt(self):
        if self.crs is not None:
            return self.crs.to_wkt(pretty=True)

    @property
    def length_units(self):
        return get_crs_length_units(self.crs)

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

    @property
    def dataframe(self):
        """Pandas DataFrame of grid cell polygons
        with i, j locations."""
        if self._dataframe is None:
            self._dataframe = self.get_dataframe(layers=True)
        return self._dataframe

    @property
    def intercell_connections(self):
        """Pandas DataFrame of flow connections between grid cells."""
        if self._intercell_connections is None:
            self._intercell_connections = self.get_intercell_connections()
        return self._intercell_connections

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top):
        self._top = top

    @property
    def botm(self):
        return self._botm

    @botm.setter
    def botm(self, botm):
        if (self._StructuredGrid__nrow, self._StructuredGrid__ncol) != botm.shape[1:]:
            raise ValueError("botm array shape is inconsistent with the model grid")
        self._StructuredGrid__nlay = botm.shape[0]
        if self._laycbd.size != botm.shape[0]:
            self._laycbd = np.zeros(botm.shape[0], dtype=int)
        self._botm = botm

    def get_intercell_connections(self, binary_grid_file=None):
        """_summary_

        Parameters
        ----------
        binary_grid_file : str or pathlike
            MODFLOW 6 binary grid file

        Returns
        -------
        df : DataFrame
            Intercell connections, with the following columns:

            === =============================================================
            n   from zero-based node number
            kn  from zero-based layer
            in  from zero-based row
            jn  from zero-based column
            m   to zero-based node number
            kn  to zero-based layer
            in  to zero-based row
            jn  to zero-based column
            === =============================================================

        Raises
        ------
        ValueError
            _description_
        """
        if binary_grid_file is not None:
            self.binary_grid_file = binary_grid_file
        if self.binary_grid_file is None:
            raise ValueError("A MODFLOW 6 binary_grid_file "
                             "is needed to get intercell connections. "
                             "Either run get_intercell_connections or "
                             "re-instantiate the grid with a binary_grid_file argument.")
        self._intercell_connections = get_intercell_connections(self.binary_grid_file)
        return self._intercell_connections

    def get_dataframe(self, layers=True):
        """Get a pandas DataFrame of grid cell polygons
        with i, j locations.

        Parameters
        ----------
        layers : bool
            If True, return a row for each k, i, j location
            and a 'k' column; if False, only return i, j
            locations with no 'k' column. By default, True

        Returns
        -------
        layers : DataFrame
            Pandas Dataframe with k, i, j and geometry column
            with a shapely polygon representation of each model cell.
        """
        # get dataframe of model grid cells
        i, j = np.indices((self.nrow, self.ncol))
        geoms = self.polygons
        df = gpd.GeoDataFrame({'i': i.ravel(),
                               'j': j.ravel(),
                              'geometry': geoms}, crs=5070)
        if layers and self.nlay is not None:
            # add layer information
            dfs = []
            for k in range(self.nlay):
                layer_df = df.copy()
                layer_df['k'] = k
                dfs.append(layer_df)
            df = pd.concat(dfs)
            df = df[['k', 'i', 'j', 'geometry']].copy()
        return df

    def write_bbox_shapefile(self, filename='grid_bbox.shp'):
        write_bbox_shapefile(self, filename)

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


# definition of national hydrogeologic grid
national_hydrogeologic_grid_parameters = {
    'xul': -2553045.0,  # upper left corner
    'yul': 3907285.0,
    'height': 4000,
    'width': 4980,
    'dx': 1000,
    'dy': 1000,
    'rotation': 0.
}


def get_crs(crs=None, epsg=None, prj=None, wkt=None, proj_str=None):
    """Get a pyproj CRS instance from various CRS representations.
    """
    if crs is not None:
        crs = pyproj.CRS.from_user_input(crs)
    elif epsg is not None:
        crs = pyproj.CRS.from_epsg(epsg)
    elif prj is not None:
        with open(prj) as src:
            wkt = src.read()
            crs = pyproj.CRS.from_wkt(wkt)
    elif wkt is not None:
        crs = pyproj.CRS.from_wkt(wkt)
    elif proj_str is not None:
        crs = pyproj.CRS.from_string(proj_str)
    else: # crs is None
        return
    # if possible, have pyproj try to find the closest
    # authority name and code matching the crs
    # so that input from epsg codes, proj strings, and prjfiles
    # results in equal pyproj_crs instances
    authority = crs.to_authority()
    if authority is not None:
        crs = pyproj.CRS.from_user_input(crs.to_authority())
    return crs


def get_crs_length_units(crs):
    length_units = crs.axis_info[0].unit_name
    if 'foot' in length_units.lower() or 'feet' in length_units.lower():
        length_units = 'feet'
    elif 'metre' in length_units.lower() or 'meter' in length_units.lower():
        length_units = 'meters'
    return length_units


def get_ij(grid, x, y, local=False):
    """Return the row and column of a point or sequence of points
    in real-world coordinates.

    Parameters
    ----------
    grid : flopy.discretization.StructuredGrid instance
    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates
    local: bool (optional)
        If True, x and y are in local coordinates (defaults to False)

    Returns
    -------
    i : row or sequence of rows (zero-based)
    j : column or sequence of columns (zero-based)
    """
    xc, yc = grid.xcellcenters, grid.ycellcenters
    if local:
        x, y = grid.get_coords(x, y)
    print('getting i, j locations...')
    t0 = time.time()
    xyc = np.array([xc.ravel(), yc.ravel()]).transpose()
    pxy = np.array([x, y]).transpose()
    kdtree = spatial.KDTree(xyc)
    distance, loc = kdtree.query(pxy)
    i, j = np.unravel_index(loc, (grid.nrow, grid.ncol))
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return i, j


def get_kij_from_node3d(node3d, nrow, ncol):
    """For a consecutive cell number in row-major order
    (row, column, layer), get the zero-based row, column position.
    """
    node2d = node3d % (nrow * ncol)
    k = node3d // (nrow * ncol)
    i = node2d // ncol
    j = node2d % ncol
    return k, i, j


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


def get_nearest_point_on_grid(x, y, transform=None,
                              xul=None, yul=None,
                              dx=None, dy=None, rotation=0.,
                              offset='center', op=None):
    """

    Parameters
    ----------
    x : float
        x-coordinate of point
    y : float
        y-coordinate of point
    transform : Affine instance, optional
        Affine object instance describing grid
    xul : float
        x-coordinate of upper left corner of the grid
    yul : float
        y-coordinate of upper left corner of the grid
    dx : float
        grid spacing in the x-direction (along rows)
    dy : float
        grid spacing in the y-direction (along columns)
    rotation : float
        grid rotation about the upper left corner, in degrees clockwise from the x-axis
    offset : str, {'center', 'edge'}
        Whether the point on the grid represents a cell center or corner (edge). This
        argument is only used if xul, yul, dx, dy and rotation are supplied. If
        an Affine transform instance is supplied, it is assumed to already incorporate
        the offset.
    op : function, optional
        Function to convert fractional pixels to whole numbers (np.round, np.floor, np.ceiling).
        Defaults to np.round if offset == 'center'; otherwise defaults to np.floor.



    Returns
    -------
    x_nearest, y_nearest : float
        Coordinates of nearest grid cell center.

    """
    # get the closet (fractional) grid cell location
    # (in case the grid is rotated)
    if transform is None:
        transform = Affine(dx, 0., xul,
                           0., dy, yul) * \
                    Affine.rotation(rotation)
        if offset == 'center':
            transform *= Affine.translation(0.5, 0.5)
    x_raster, y_raster = ~transform * (x, y)

    if offset == 'center':
        op = np.round
    elif op is None:
        op = np.floor

    j = int(op(x_raster))
    i = int(op(y_raster))

    x_nearest, y_nearest = transform * (j, i)
    return x_nearest, y_nearest


def get_point_on_national_hydrogeologic_grid(x, y, offset='edge', **kwargs):
    """Given an x, y location representing the upper left
    corner of a model grid, return the upper left corner
    of the cell in the National Hydrogeologic Grid that
    contains it."""
    params = get_input_arguments(national_hydrogeologic_grid_parameters, get_nearest_point_on_grid)
    params.update(kwargs)
    return get_nearest_point_on_grid(x, y, offset=offset, **params)


def write_bbox_shapefile(modelgrid, outshp):
    outline = get_grid_bounding_box(modelgrid)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=modelgrid.epsg)


def rasterize(feature, grid, id_column=None,
              include_ids=None, names_column=None,
              crs=None,
              dtype=np.float32, **kwargs):
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
    include_ids : sequence
        Subset of IDs in id_column to include
    names_column : str, optional
        By default, the IDs in id_column, or sequential integers
        are returned. This option allows another column of strings
        to be specified (i.e. feature names); in which case
        an array of the strings will be returned.
    grid : grid.StructuredGrid instance
    crs : obj
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to :meth:`pyproj.crs.CRS.from_user_input`
        Can be any of:

          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

    dtype : dtype
        Datatype for the output array
    **kwargs : keyword arguments to rasterio.features.rasterize()
        https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html

    Returns
    -------
    2D numpy array with intersected values

    """
    try:
        from rasterio import Affine, features
    except:
        print('This method requires rasterio.')
        return

    if crs is not None:
        if version.parse(gisutils.__version__) < version.parse('0.2.0'):
            raise ValueError("The rasterize function requires gisutils >= 0.2")
        from gisutils import get_authority_crs
        crs = get_authority_crs(crs)

    trans = get_transform(grid)

    if isinstance(feature, str) or isinstance(feature, Path):
        df = gpd.read_file(feature)
    elif isinstance(feature, pd.DataFrame):
        df = feature.copy()
        df = gpd.GeoDataFrame(df, crs=crs)
    elif isinstance(feature, collections.abc.Iterable):
        # list of shapefiles
        if isinstance(feature[0], str) or isinstance(feature[0], Path):
            # use shp2df to read multiple shapefiles
            # then convert to gdf
            df = shp2df(feature, dest_crs=grid.crs)
            df = gpd.GeoDataFrame(df, crs=grid.crs)
        else:
            df = pd.DataFrame({'geometry': feature})
            df = gpd.GeoDataFrame(df, crs=crs)
    elif not isinstance(feature, collections.abc.Iterable):
        df = pd.DataFrame({'geometry': [feature]})
        df = gpd.GeoDataFrame(df, crs=crs)
    else:
        print('unrecognized feature input')
        return

    # reproject to grid crs
    if df.crs is not None:
        orig_crs = df.crs
        df.to_crs(grid.crs, inplace=True)
        if not df['geometry'].is_valid.all():
            df['geometry'] = [g.buffer(0) for g in df.geometry]
        geoms_are_valid = df['geometry'].is_valid.all() & \
            (not df.geometry.is_empty.any()) & \
                np.isfinite(df.geometry.bounds.sum().sum())
        if not geoms_are_valid:
            raise ValueError('Something went wrong with reprojecting '
                             f'the input features from\n{orig_crs}\nto\n{grid.crs}\n'
                             'Check the input feature and model grid projections'
                             'If you are on a network that requires special '
                             'SSL authentication, try running this operation '
                             'again off-network.'
                             )

    # subset to include_ids
    if id_column is not None and include_ids is not None:
        df = df.loc[df[id_column].isin(include_ids)].copy()

    # create list of GeoJSON features, with unique value for each feature
    if id_column is None:
        numbers = list(range(1, len(df)+1))
    # if IDs are strings, get a number for each one
    # pd.DataFrame.unique() generally preserves order
    elif df[id_column].dtype == object:
        unique_values = df[id_column].unique()
        values = dict(zip(unique_values, range(1, len(unique_values) + 1)))
        numbers = [values[n] for n in df[id_column]]
    else:
        numbers = df[id_column].values
        # add one if the lowest number is 0
        # (zero indicates non-intersected raster cells)
        if np.min(numbers) == 0:
            numbers += 1
        elif np.min(numbers) < 0:
            raise ValueError("id_column must have positive integers!")
        numbers = list(numbers)

    geoms = list(zip(df.geometry, numbers))
    result = features.rasterize(geoms,
                                out_shape=(grid.nrow, grid.ncol),
                                transform=trans, **kwargs)
    assert result.sum(axis=(0, 1)) != 0, "Nothing was intersected!"
    if names_column is not None:
        names_lookup = dict(zip(numbers, df[names_column]))
        result = [names_lookup.get(n, '') for n in result.flat]
        result = np.reshape(result, (grid.nrow, grid.ncol))
        dtype = object
    return result.astype(dtype)


def setup_structured_grid(xoff=None, yoff=None, xul=None, yul=None,
                          nrow=None, ncol=None, nlay=None,
                          dxy=None, delr=None, delc=None,
                          top=None, botm=None,
                          rotation=0.,
                          parent_model=None, snap_to_parent=True, snap_to_NHG=False,
                          features=None, features_shapefile=None,
                          id_column=None, include_ids=None,
                          buffer=1000,
                          crs=None, epsg=None, prj=None, wkt=None,
                          model_length_units=None,
                          grid_file='grid.json',
                          bbox_shapefile=None, **kwargs):
    """"""
    print('setting up model grid...')
    t0 = time.time()

    # make sure crs is populated, then get CRS units for the grid
    crs = get_crs(crs=crs, epsg=epsg, prj=prj, wkt=wkt)
    if crs is None and parent_model is not None:
        crs = parent_model.modelgrid.crs

    grid_units = get_crs_length_units(crs)
    if grid_units not in {'feet', 'meters'}:
        raise ValueError(f'unrecognized CRS units {grid_units}: CRS must be projected in feet or meters')

    # conversions for model/parent model units to meters
    # set regular flag for handling delc/delr
    to_grid_units_inset = convert_length_units(model_length_units, grid_units)

    regular = True
    if dxy is not None:
        delr_grid = np.round(dxy, 4) # dxy is specified in CRS units
        delc_grid = delr_grid
    if delr is not None:
        delr_grid = np.round(delr * to_grid_units_inset, 4)  # delr is specified in model units
        if not np.isscalar(delr_grid):
            if (set(delr_grid)) == 1:
                delr_grid = delr_grid[0]
            else:
                regular = False
    if delc is not None:
        delc_grid = np.round(delc * to_grid_units_inset, 4) # delc is specified in model units
        if not np.isscalar(delc_grid):
            if (set(delc_grid)) == 1:
                delc_grid = delc_grid[0]
            else:
                regular = False
    if parent_model is not None and snap_to_parent:
        to_grid_units_parent = convert_length_units(get_model_length_units(parent_model), grid_units)
        # parent model grid spacing in meters
        parent_delr_grid = np.round(parent_model.dis.delr.array[0] * to_grid_units_parent, 4)
        if not parent_delr_grid % delr_grid % parent_delr_grid == 0:
            raise ValueError('inset delr spacing of {} must be factor of parent spacing of {}'.format(delr_grid,
                                                                                                      parent_delr_grid))
        parent_delc_grid = np.round(parent_model.dis.delc.array[0] * to_grid_units_parent, 4)
        if not parent_delc_grid % delc_grid % parent_delc_grid == 0:
            raise ValueError('inset delc spacing of {} must be factor of parent spacing of {}'.format(delc_grid,
                                                                                                      parent_delc_grid))

    # option 1: make grid from xoff, yoff and specified dimensions
    if xoff is not None and yoff is not None:
        assert nrow is not None and ncol is not None, \
            "Need to specify nrow and ncol if specifying xoffset and yoffset."
        if regular:
            height_grid = np.round(delc_grid * nrow, 4)
            width_grid = np.round(delr_grid * ncol, 4)
        else:
            height_grid = np.sum(delc_grid)
            width_grid = np.sum(delr_grid)

        # optionally align grid with national hydrologic grid
        # grids snapping to NHD must have spacings that are a factor of 1 km
        if snap_to_NHG:
            assert regular and np.allclose(1000 % delc_grid, 0, atol=1e-4)
            x, y = get_point_on_national_hydrogeologic_grid(xoff, yoff,
                                                            offset='edge', op=np.floor)
            xoff = x
            yoff = y
            rotation = 0.

        # need to specify xul, yul in case snapping to parent
        # todo: allow snapping to parent grid on xoff, yoff
        if rotation != 0:
            rotation_rads = rotation * np.pi/180
            # note rotating around xoff,yoff not the origin!
            xul = xoff - (height_grid) * np.sin(rotation_rads)
            yul = yoff + (height_grid) * np.cos(rotation_rads)
        else:
            xul = xoff
            yul = yoff + height_grid

    # option 2: make grid using buffered feature bounding box
    else:
        if features is None and features_shapefile is not None:
            # Make sure shapefile and bbox filter are in dest (model) CRS
            # TODO: CRS wrangling could be added to shp2df as a feature
            reproject_filter = False
            try:
                from gisutils import get_shapefile_crs
                features_crs = get_shapefile_crs(features_shapefile)
                if features_crs != crs:
                    reproject_filter = True
            except:
                features_crs = get_proj_str(features_shapefile)
                reproject_filter = True
            bbox_filter = None
            if parent_model is not None:
                if reproject_filter:
                    bbox_filter = project(parent_model.modelgrid.bbox,
                                     parent_model.modelgrid.crs, features_crs).bounds
                else:
                    bbox_filter = parent_model.modelgrid.bbox.bounds
            shp2df_kwargs = {'dest_crs': crs}
            shp2df_kwargs = get_input_arguments(shp2df_kwargs, shp2df)
            df = shp2df(features_shapefile,
                        filter=bbox_filter, **shp2df_kwargs)

            # optionally subset shapefile data to specified features
            if id_column is not None and include_ids is not None:
                df = df.loc[df[id_column].isin(include_ids)]
            # use all features by default
            features = df.geometry.tolist()

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
            height_grid = np.round(yul - (y1 - L), 4) # initial model height from buffer distance
            width_grid = np.round((x2 + L) - xul, 4)
            rotation = 0.  # rotation not supported with this option
            nrow = int(np.ceil(height_grid / delc_grid))
            ncol = int(np.ceil(width_grid / delr_grid))

    # align model with parent grid if there is a parent model
    # (and not snapping to national hydrologic grid)
    if parent_model is not None and (snap_to_parent and not snap_to_NHG):

        # An alternative method for snapping to the parent grid
        # if the code below is causing issues
        # get the parent cell containing
        # the inset upper left corner
        # xul_in, yul_in = xul, yul
        #pi2, pj2 = get_ij(parent_model.modelgrid, xul_in, yul_in)
        # find the closest vertex
        #verts2 = np.array(parent_model.modelgrid.get_cell_vertices(pi2, pj2))
        #closest_pos = np.argmin(np.sqrt((xul-verts2[:,0])**2 +
        #                                (yul-verts2[:,0])**2))
        #xul, yul = verts2[closest_pos]

        # get location of coinciding cell in parent model for upper left
        # first make sure not sitting at the top of a cell (which can shift into wrong parent cell)
        # move to model coords
        xul_mod, yul_mod = parent_model.modelgrid.get_local_coords(xul, yul)
        # move away from the edge of a cell
        xul_mod += (delr_grid * 0.25)
        yul_mod -= (delc_grid * 0.25)
        # flip back to world coords
        xul, yul = parent_model.modelgrid.get_coords(xul_mod, yul_mod)
        # get corresponding cell
        pi, pj = parent_model.modelgrid.intersect(xul, yul)
        # find the vertices of that cell
        verts = np.array(parent_model.modelgrid.get_cell_vertices(pi, pj))
        # flip to model space to easily locate upper left corner
        verts_model_space = np.array([parent_model.modelgrid.get_local_coords(x,y) for x,y in verts])
        # finally, back to world space
        xul,yul = parent_model.modelgrid.get_coords(verts_model_space[:,0].min(),verts_model_space[:,1].max())

        # adjust the dimensions to align remaining corners
        def roundup(number, increment):
            return int(np.ceil(number / increment) * increment)
        height_grid = roundup(height_grid, parent_delr_grid)
        width_grid = roundup(width_grid, parent_delc_grid)

        # update nrow, ncol after snapping to parent grid
        if regular:
            nrow = int(height_grid / delc_grid) # h is in meters
            ncol = int(width_grid / delr_grid)

    if xoff is None:
        xoff = xul + (np.sin(np.radians(rotation)) * height_grid)
    if yoff is None:
        yoff = yul - (np.cos(np.radians(rotation)) * height_grid)
    # set the grid configuration dictionary
    # spacing is in meters (consistent with projected CRS)
    # (modelgrid object will be updated automatically from this dictionary)
    #if rotation == 0.:
    #    xll = xul
    #    yll = yul - model.height
    grid_cfg = {'nrow': int(nrow), 'ncol': int(ncol),
                'nlay': nlay,
                'delr': delr_grid, 'delc': delc_grid,
                'xoff': xoff, 'yoff': yoff,
                'xul': xul, 'yul': yul,
                'rotation': rotation,
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

    # add epsg or wkt if there isn't an epsg
    if crs is not None:
        grid_cfg['crs'] = crs
    elif epsg is not None:
        grid_cfg['epsg'] = epsg
    #elif crs is not None:
    #    if 'epsg' in crs.srs.lower():
    #        grid_cfg['epsg'] = int(crs.srs.split(':')[1])
    #    else:
    #        grid_cfg['wkt'] = crs.srs
    else:
        warnings.warn(("Coordinate Reference System information must be supplied via"
                      "the 'crs'' argument."))

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

    # crs needs to be cast to epsg or wkt to be serialized
    if isinstance(crs, pyproj.CRS):
        grid_cfg['epsg'] = grid_cfg['crs'].to_epsg()
        if grid_cfg['epsg'] is None:
            grid_cfg['wkt'] = grid_cfg['crs'].to_wkt()
        del grid_cfg['crs']

    fileio.dump(grid_file, grid_cfg)
    if bbox_shapefile is not None:
        write_bbox_shapefile(modelgrid, bbox_shapefile)
    print("finished in {:.2f}s\n".format(time.time() - t0))
    return modelgrid


def get_cellface_midpoint(grid, k, i, j, direction):
    """Return the midpoint of vertical cell face within a structured grid.
    For example, the midpoint for the right cell face is halfway between
    the upper and lower right corners of the cell, halfway between the
    top and bottom edges."""
    if np.isscalar(k):
        k = [k]
    if np.isscalar(i):
        i = [i]
    if np.isscalar(j):
        j = [j]
    k = np.array(k).astype(int)
    i = np.array(i).astype(int)
    j = np.array(j).astype(int)
    if isinstance(direction, str):
        direction = [direction] * len(k)
    x_edges_model = grid.xyedges[0]
    x_centers_model = grid.xycenters[0]
    y_edges_model = grid.xyedges[1]
    y_centers_model = grid.xycenters[1]
    model_x = []
    model_y = []
    for ii, jj, dn in zip(i, j, direction):
        if dn == 'right':
            x = x_edges_model[jj+1]
            y = y_centers_model[ii]
        elif dn == 'left':
            x = x_edges_model[jj]
            y = y_centers_model[ii]
        elif dn == 'top':
            x = x_centers_model[jj]
            y = y_edges_model[ii]
        elif dn == 'bottom':
            x = x_centers_model[jj]
            y = y_edges_model[ii+1]
        else:
            raise ValueError("direction needs to be right, left, top or bottom")
        model_x.append(x)
        model_y.append(y)
    x, y = grid.get_coords(model_x, model_y)
    z = grid.zcellcenters[k, i, j]
    return x, y, z


def get_intercell_connections(binary_grid_file):
    """Get all of the connections between cells in a
    MODFLOW 6 structured grid.

    Parameters
    ----------
    binary_grid_file : str or pathlike
        MODFLOW 6 binary grid file

    Returns
    -------
    df : DataFrame
        Intercell connections, with the following columns:

        ==== =============================================================
        n    from zero-based node number
        kn   from zero-based layer
        in   from zero-based row
        jn   from zero-based column
        m    to zero-based node number
        kn   to zero-based layer
        in   to zero-based row
        jn   to zero-based column
        qidx index position of flow in cell budget file
        ==== =============================================================

    Raises
    ------
    ValueError
        _description_
    """
    print('Getting intercell connections...')
    ta = time.time()
    bgf = MfGrdFile(binary_grid_file)
    nrow = bgf.nrow
    ncol = bgf.ncol
    # IA array maps cell number to connection number
    # (one-based index number of first connection at each cell)?
    # taking the forward difference then yields nconnections per cell
    ia = bgf._datadict['IA'] - 1
    # Connections in the JA array correspond directly with the
    # FLOW-JA-FACE record that is written to the budget file.
    ja = bgf._datadict['JA'] - 1  # cell connections

    all_n = []
    m = []
    qidx = []
    for n in range(len(ia)-1):
        for ipos in range(ia[n] + 1, ia[n+1]):
            all_n.append(n)
            m.append(ja[ipos])  # m is the cell that n connects to
            qidx.append(ipos)
    df = pd.DataFrame({'n': all_n, 'm': m, 'qidx': qidx})
    k, i, j = get_kij_from_node3d(df['n'].values, nrow, ncol)
    df['kn'], df['in'], df['jn'] = k, i, j
    k, i, j = get_kij_from_node3d(df['m'].values, nrow, ncol)
    df['km'], df['im'], df['jm'] = k, i, j
    df.reset_index()
    print(f"Getting intercell connections took {time.time() - ta:.2f}s\n")
    return df


def get_transform(modelgrid):
    """Get a rasterio Affine object from a Flopy modelgrid
    (same as transform attribute of rasters).
    """
    if not isinstance(modelgrid, StructuredGrid):
        raise ValueError(
            f"{type(modelgrid)}: Input needs to be a flopy.discretization.StructuredGrid")
    x0 = modelgrid.xyedges[0][0]
    y0 = modelgrid.xyedges[1][0]
    xul, yul = modelgrid.get_coords(x0, y0)
    return Affine(modelgrid.delr[0], 0., xul,
                    0., -modelgrid.delc[0], yul) * \
            Affine.rotation(-modelgrid.angrot)
