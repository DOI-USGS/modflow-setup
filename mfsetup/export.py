import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from flopy.utils import MfList
from flopy.export.utils import contour_array, export_contours
from .gis import df2shp

def export_array(modelgrid, filename, a, nodata=-9999,
                 fieldname='value',
                 **kwargs):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    modelgrid : MFsetupGrid instance
    filename : str
        Path of output file. Export format is determined by
        file extention.
        '.asc'  Arc Ascii grid
        '.tif'  GeoTIFF (requries rasterio package)
        '.shp'  Shapefile
    a : 2D numpy.ndarray
        Array to export
    nodata : scalar
        Value to assign to np.nan entries (default -9999)
    fieldname : str
        Attribute field name for array values (shapefile export only).
        (default 'values')
    kwargs:
        keyword arguments to np.savetxt (ascii)
        rasterio.open (GeoTIFF)
        or flopy.export.shapefile_utils.write_grid_shapefile2

    Notes
    -----
    Rotated grids will be either be unrotated prior to export,
    using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
    included in their transform property (GeoTiff format). In either case
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """

    if filename.lower().endswith(".asc"):
        if len(np.unique(modelgrid.delr)) != len(np.unique(modelgrid.delc)) != 1 \
                or modelgrid.delr[0] != modelgrid.delc[0]:
            raise ValueError('Arc ascii arrays require a uniform grid.')

        xoffset, yoffset = modelgrid.xoffset, modelgrid.yoffset
        cellsize = modelgrid.delr[0] # * self.length_multiplier
        fmt = kwargs.get('fmt', '%.18e')
        a = a.copy()
        a[np.isnan(a)] = nodata
        if modelgrid.angrot != 0:
            try:
                from scipy.ndimage import rotate
                a = rotate(a, modelgrid.angrot, cval=nodata)
                height_rot, width_rot = a.shape
                xmin, ymin, xmax, ymax = modelgrid.extent
                dx = (xmax - xmin) / width_rot
                dy = (ymax - ymin) / height_rot
                cellsize = np.max((dx, dy))
                xoffset, yoffset = xmin, ymin
            except ImportError:
                print('scipy package required to export rotated grid.')

        filename = '.'.join(
            filename.split('.')[:-1]) + '.asc'  # enforce .asc ending
        nrow, ncol = a.shape
        a[np.isnan(a)] = nodata
        txt = 'ncols  {:d}\n'.format(ncol)
        txt += 'nrows  {:d}\n'.format(nrow)
        txt += 'xllcorner  {:f}\n'.format(xoffset)
        txt += 'yllcorner  {:f}\n'.format(yoffset)
        txt += 'cellsize  {}\n'.format(cellsize)
        # ensure that nodata fmt consistent w values
        txt += 'NODATA_value  {}\n'.format(fmt) % (nodata)
        with open(filename, 'w') as output:
            output.write(txt)
        with open(filename, 'ab') as output:
            np.savetxt(output, a, **kwargs)
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".tif"):
        if len(np.unique(modelgrid.delr)) != len(np.unique(modelgrid.delc)) != 1 \
                or modelgrid.delr[0] != modelgrid.delc[0]:
            raise ValueError('GeoTIFF export require a uniform grid.')
        try:
            import rasterio
            from rasterio import Affine
        except ImportError:
            print('GeoTIFF export requires the rasterio package.')
            return
        dxdy = modelgrid.delc[0] # * self.length_multiplier
        trans = modelgrid.transform

        # third dimension is the number of bands
        a = a.copy()
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))
        if a.dtype.name == 'int64':
            a = a.astype('int32')
            dtype = rasterio.int32
        elif a.dtype.name == 'int32':
            dtype = rasterio.int32
        elif a.dtype.name == 'float64':
            dtype = rasterio.float64
        elif a.dtype.name == 'float32':
            dtype = rasterio.float32
        else:
            msg = 'ERROR: invalid dtype "{}"'.format(a.dtype.name)
            raise TypeError(msg)

        meta = {'count': a.shape[0],
                'width': a.shape[2],
                'height': a.shape[1],
                'nodata': nodata,
                'dtype': dtype,
                'driver': 'GTiff',
                'crs': modelgrid.proj4,
                'transform': trans
                }
        meta.update(kwargs)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".shp"):
        from flopy.export.shapefile_utils import write_grid_shapefile2
        epsg = kwargs.get('epsg', None)
        prj = kwargs.get('prj', None)
        if epsg is None and prj is None:
            epsg = modelgrid.epsg
        write_grid_shapefile2(filename, modelgrid, array_dict={fieldname: a},
                              nan_val=nodata,
                              epsg=epsg, prj=prj)


def contour_array(modelgrid, ax, a, **kwargs):
    """
    Create a QuadMesh plot of the specified array using pcolormesh

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        ax to add the contours

    a : np.ndarray
        array to contour

    Returns
    -------
    contour_set : ContourSet

    """
    try:
        import matplotlib.tri as tri
    except:
        tri = None
    plot_triplot = False
    if 'plot_triplot' in kwargs:
        plot_triplot = kwargs.pop('plot_triplot')
    if 'extent' in kwargs and tri is not None:
        extent = kwargs.pop('extent')
        idx = (modelgrid.xcellcenters >= extent[0]) & (
                modelgrid.xcellcenters <= extent[1]) & (
                      modelgrid.ycellcenters >= extent[2]) & (
                      modelgrid.ycellcenters <= extent[3])
        a = a[idx].flatten()
        xc = modelgrid.xcellcenters[idx].flatten()
        yc = modelgrid.ycellcenters[idx].flatten()
        triang = tri.Triangulation(xc, yc)
        try:
            amask = a.mask
            mask = [False for i in range(triang.triangles.shape[0])]
            for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                if amask[n0] or amask[n1] or amask[n2]:
                    mask[ipos] = True
            triang.set_mask(mask)
        except:
            mask = None
        contour_set = ax.tricontour(triang, a, **kwargs)
        if plot_triplot:
            ax.triplot(triang, color='black', marker='o', lw=0.75)
    else:
        contour_set = ax.contour(modelgrid.xcellcenters, modelgrid.ycellcenters,
                                 a, **kwargs)
    return contour_set


def export_array_contours(modelgrid, filename, a,
                          fieldname='level',
                          interval=None,
                          levels=None,
                          maxlevels=1000,
                          epsg=None,
                          prj=None,
                          **kwargs):
    """
    Contour an array using matplotlib; write shapefile of contours.

    Parameters
    ----------
    filename : str
        Path of output file with '.shp' extention.
    a : 2D numpy array
        Array to contour
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.
    **kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

    """
    import matplotlib.pyplot as plt

    if epsg is None:
        epsg = modelgrid.epsg
    if prj is None:
        prj = modelgrid.proj4

    if interval is not None:
        imin = np.nanmin(a)
        imax = np.nanmax(a)
        nlevels = np.round(np.abs(imax - imin) / interval, 2)
        msg = '{:.0f} levels at interval of {} > maxlevels={}'.format(
            nlevels,
            interval,
            maxlevels)
        assert nlevels < maxlevels, msg
        levels = np.arange(imin, imax, interval)
    ax = plt.subplots()[-1]
    ctr = contour_array(modelgrid, ax, a, levels=levels)
    export_contours(modelgrid, filename, ctr, fieldname, epsg, prj, **kwargs)
    plt.close()


def export_shapefile(filename, modelgrid, data, kper=None,
                     squeeze=True,
                     epsg=None, proj_str=None, prj=None):
    mfl = data
    if isinstance(data, MfList):
        df = mfl.get_dataframe(squeeze=squeeze)
    elif isinstance(data, np.recarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        pass

    if kper is not None:
        df = df.loc[df.per == kper]
        verts = np.array(modelgrid.get_cell_vertices(df.i, df.j))
    elif df is not None:
        verts = modelgrid.get_vertices(df.i.values, df.j.values)
    polys = np.array([Polygon(v) for v in verts])
    df['geometry'] = polys
    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str
    if prj is None:
        prj = modelgrid.prj
    df2shp(df, filename, epsg=epsg, proj4=proj_str, prj=prj)


def get_surface_bc_flux(cbbobj, txt, kstpkper=(0, 0), idx=0):
    """Read a flow component from MODFLOW binary cell budget output;

    Parameters
    ----------
    cbbobj : open file handle (instance of flopy.utils.binaryfile.CellBudgetFile
    txt : cell budget record to read (e.g. 'STREAM LEAKAGE')
    kstpkper : tuple
        (timestep, stress period) to read
    idx : index of list returned by cbbobj (usually 0)

    Returns
    -------
    arr : ndarray
    """
    nrow, ncol, nlay = cbbobj.nrow, cbbobj.ncol, cbbobj.nlay
    results = cbbobj.get_data(text=txt, kstpkper=kstpkper, idx=idx)
    # this logic needs some cleanup
    if len(results) > 0:
        results = results[0]
    else:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if isinstance(results, list) and txt == 'RECHARGE':
        results = results[1]
    if results.size == 0:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if results.shape == (nlay, nrow, ncol):
        return results
    elif results.shape == (1, nrow, ncol):
        return results[0]
    elif len(results.shape) == 1 and \
            len({'node', 'q'}.difference(set(results.dtype.names))) == 0:
        arr = np.zeros(nlay * nrow * ncol, dtype=float)
        arr[results.node - 1] = results.q
        arr = np.reshape(arr, (nlay, nrow, ncol))
        arr = arr.sum(axis=0)
        return arr
