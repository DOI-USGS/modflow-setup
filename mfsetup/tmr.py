import time
from pathlib import Path

import flopy
import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString

fm = flopy.modflow
from flopy.discretization import StructuredGrid
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from flopy.utils import binaryfile as bf

from mfsetup.discretization import find_remove_isolated_cells
from mfsetup.fileio import check_source_files
from mfsetup.grid import get_cellface_midpoint, get_ij, get_intercell_connections
from mfsetup.interpolate import Interpolator, interp_weights
from mfsetup.lakes import get_horizontal_connections


def get_qx_qy_qz(cell_budget_file, binary_grid_file=None,
                 cell_connections_df=None,
                 version='mf6',
                 kstpkper=(0, 0),
                 specific_discharge=False,
                 headfile=None,
                 modelgrid=None):
    """Get 2 or 3D arrays of cell by cell flows across the cell faces
    (for structured grid models).

    Parameters
    ----------
    cell_budget_file : str, pathlike, or instance of flopy.utils.binaryfile.CellBudgetFile
        File path or pointer to MODFLOW cell budget file.
    binary_grid_file : str or pathlike
        File path to MODFLOW 6 binary grid (``*.dis.grb``) file. Not needed for MFNWT
    cell_connections_df : DataFrame
        DataFrame of cell connections that can be provided as an alternative to bindary_grid_file,
        to avoid having to get the connections with each call to get_qx_qy_qz. This can
        be produced by the :meth:``mfsetup.grid.MFsetupGrid.intercell_connections`` method.
        Must have following columns:

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

    version : str
        MODFLOW version- 'mf6' or other. If not 'mf6', the cell budget output
        is assumed to be formatted similar to a MODFLOW 2005 style model.
    model_top : 2D numpy array of shape (nrow, ncol)
        Model top elevations (only needed for modflow 2005 style models without
        a binary grid file)
    model_bottom_array : 3D numpy array of shape (nlay, nrow, ncol)
        Model bottom elevations (only needed for modflow 2005 style models
        without a binary grid file)
    kstpkper : tuple
        zero-based (time step, stress period)
    specific_discharge : bool
        Option to return arrays of specific discharge (1D vector components)
        instead of volumetric fluxes.
        By default, False
    headfile : str, pathlike, or instance of flopy.utils.binaryfile.HeadFile
        File path or pointer to MODFLOW head file. Only required if
        specific_discharge=True
    modelgrid : instance of MFsetupGrid object
        Defaults to None, only required if specific_discharge=True


    Returns
    -------
    Qx, Qy, Qz : tuple of 2 or 3D numpy arrays
        Volumetric or specific discharge fluxes across cell faces.
    """
    msg = 'Getting discharge...'
    if specific_discharge:
        msg = 'Getting specific discharge...'
    print(msg)
    ta = time.time()
    if version == 'mf6':
        # get the cell connections
        if cell_connections_df is not None:
            df = cell_connections_df
        elif binary_grid_file is not None:
            df = get_intercell_connections(binary_grid_file)
        else:
            raise ValueError("Must specify a binary_grid_file or cell_connections_df.")

        # get the flows
        # this constitutes almost all of the execution time for this fn
        t1 = time.time()
        if isinstance(cell_budget_file, str) or isinstance(cell_budget_file, Path):
            cbb = bf.CellBudgetFile(cell_budget_file)
        else:
            cbb = cell_budget_file
        nlay, nrow, ncol = cbb.shape
        flowja = cbb.get_data(text='FLOW-JA-FACE', kstpkper=kstpkper)[0][0, 0, :]
        df['q'] = flowja[df['qidx']]
        print(f"getting flows from budget file took {time.time() - t1:.2f}s\n")

        # get arrays of flow through cell faces
        # Qx (right face; TODO: confirm direction)
        rfdf = df.loc[(df['jn'] < df['jm'])]
        nlay = rfdf['km'].max() + 1
        qx = np.zeros((nlay, nrow, ncol))
        qx[rfdf['kn'].values, rfdf['in'].values, rfdf['jn'].values] = -rfdf.q.values

        # Qy (front face; TODO: confirm direction)
        ffdf = df.loc[(df['in'] < df['im'])]
        qy = np.zeros((nlay, nrow, ncol))
        qy[ffdf['kn'].values, ffdf['in'].values, ffdf['jn'].values] = -ffdf.q.values

        # Qz (bottom face; TODO: confirm that this is downward positive)
        bfdf = df.loc[(df['kn'] < df['km'])]
        qz = np.zeros((nlay, nrow, ncol))
        qz[bfdf['kn'].values, bfdf['in'].values, bfdf['jn'].values] = -bfdf.q.values
    else:
        if isinstance(cell_budget_file, str) or isinstance(cell_budget_file, Path):
            cbb = bf.CellBudgetFile(cell_budget_file)
        else:
            cbb = cell_budget_file
        qx = cbb.get_data(text="flow right face", kstpkper=kstpkper)[0]
        qy = cbb.get_data(text="flow front face", kstpkper=kstpkper)[0]
        unique_rec_names = [bs.decode().strip().lower() for bs in cbb.get_unique_record_names()]
        if "flow lower face" in unique_rec_names:
            qz = cbb.get_data(text="flow lower face", kstpkper=kstpkper)[0]
        else:
            qz = np.zeros_like(qy)

    # optionally get specific discharge
    if specific_discharge:
        if modelgrid is None:
            raise Exception('specific discharge calculations require a modelgrid input')
        if headfile is None:
            print('No headfile object provided - thickness for specific discharge calculations\n' +
                'will be based on the model top rather than the water table')
            thickness = modelgrid.cell_thickness
        else:
            if isinstance(headfile, str) or isinstance(headfile, Path):
                hds = bf.HeadFile(headfile).get_data(kstpkper=kstpkper)
            else:
                hds = headfile.get_data(kstpkper=kstpkper)
            thickness = modelgrid.saturated_thickness(array=hds)

        delr_gridp, delc_gridp = np.meshgrid(modelgrid.delr,
                                            modelgrid.delc)
        nlay, nrow, ncol = modelgrid.shape

        # multiply average thickness by width (along rows or cols) to
        # obtain cross sectional area on the faces
        # https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/delrdelcillustration.png
        qy_face_areas = np.tile(delr_gridp[:-1,:], (nlay,1,1)) * \
                                ((thickness[:,:-1,:]+thickness[:,1:,:])/2)
        # the above calculation results in a missing dimension ( only internal faces are
        # calculated ) so we concatenate on a repetition of the final row or column
        qy_face_areas = np.concatenate([qy_face_areas,
                    np.expand_dims(qy_face_areas[:,-1,:], axis=1)], axis=1)

        qx_face_areas = np.tile(delc_gridp[:,:-1], (nlay,1,1)) * \
                                ((thickness[:,:,:-1]+thickness[:,:,1:])/2)
        qx_face_areas = np.concatenate([qx_face_areas,
                    np.expand_dims(qx_face_areas[:,:,-1], axis=2)], axis=2)

        # z direction is simply delr * delc across all layers
        qz_face_areas = np.tile(delr_gridp * delc_gridp, (nlay,1,1))

        # divide by the areas resulting in normalized, specific discharge
        qx /= qx_face_areas
        qy /= qy_face_areas
        qz /= qz_face_areas

    print(f"{msg} took {time.time() - ta:.2f}s\n")
    return qx, qy, qz

class Tmr:
    """
    Class for general telescopic mesh refinement of a MODFLOW model. Head or
    flux fields from parent model are interpolated to boundary cells of
    inset model, which may be in any configuration (jagged, rotated, etc.).

    Parameters
    ----------
    parent_model : flopy model instance instance of parent model
        Must have a valid, attached ``modelgrid`` attribute that is an instance of
        :class:`mfsetup.grid.MFsetupGrid`.
    inset_model : flopy model instance instance of inset model
        Must have a valid, attached ``modelgrid`` attribute that is an instance of
        :class:`mfsetup.grid.MFsetupGrid`.
    parent_head_file : filepath
        MODFLOW binary head output
    parent_cell_budget_file : filepath
        MODFLOW binary cell budget output
    parent_binary_grid_file : filepath
        MODFLOW 6 binary grid file (``*.grb``)
    define_connections : str, {'max_active_extent', 'by_layer'}
        Method for defining perimeter cells where the TMR boundary
        condition will be applied. If 'max_active_extent', the
        maximum footprint of the active area (including all cell
        locations with at least one layer that is active) will be used.
        If 'by_layer', the perimeter of the active area in each layer will be used
        (excluding any interior clusters of active cells). The 'by_layer'
        option is potentially problematic if some layers have substantial
        areas of pinched-out (idomain != 1) cells, which may result
        in perimeter boundary condition cells getting placed too close
        to the area of interest. By default, 'max_active_extent'.

    Notes
    -----
    """

    def __init__(self, parent_model, inset_model,
                 parent_head_file=None, parent_cell_budget_file=None,
                 parent_binary_grid_file=None,
                 boundary_type=None, inset_parent_period_mapping=None,
                 parent_start_date_time=None, source_mask=None,
                 define_connections_by='max_active_extent',
                 shapefile=None,
                 ):
        self.parent = parent_model
        self.inset = inset_model
        self.parent_head_file = parent_head_file
        self.parent_cell_budget_file = parent_cell_budget_file
        self.parent_binary_grid_file = parent_binary_grid_file
        self.define_connections_by = define_connections_by
        self.shapefile = shapefile
        self.boundary_type = boundary_type
        if boundary_type is None and parent_head_file is not None:
            self.boundary_type = 'head'
        elif boundary_type is None and parent_cell_budget_file is not None:
            self.boundary_type = 'flux'
        self.parent_start_date_time = parent_start_date_time

        # Path for writing auxilliary output tables
        # (boundary_cells.shp, etc.)
        if hasattr(self.inset, '_tables_path'):
            self._tables_path = Path(self.inset._tables_path)
        else:
            self._tables_path = Path(self.inset.model_ws) / 'tables'
        self._tables_path.mkdir(exist_ok=True, parents=True)

        # properties
        self._idomain = None
        self._inset_boundary_cells = None
        self._inset_parent_period_mapping = inset_parent_period_mapping
        self._interp_weights_heads = None
        self._interp_weights_flux = None
        self._source_mask = source_mask
        self._inset_zone_within_parent = None

    @property
    def idomain(self):
        """Active area of the inset model.
        """
        if self._idomain is None:
            if self.inset.version == 'mf6':
                idomain = self.inset.dis.idomain.array
                if idomain is None:
                    idomain = np.ones_like(self.inset.dis.botm.array, dtype=int)
            else:
                idomain = self.inset.bas6.ibound.array
            self._idomain = idomain
        return self._idomain

    @property
    def inset_boundary_cells(self):
        if self._inset_boundary_cells is None:
            by_layer = self.define_connections_by == 'by_layer'
            df = self.get_inset_boundary_cells(by_layer=by_layer)
            x, y, z = self.inset.modelgrid.xyzcellcenters
            df['x'] = x[df.i, df.j]
            df['y'] = y[df.i, df.j]
            df['z'] = z[df.k, df.i, df.j]
            self._inset_boundary_cells = df
            self._interp_weights = None
        return self._inset_boundary_cells

    @property
    def inset_parent_period_mapping(self):
        nper = self.inset.nper
        # if mapping between source and dest model periods isn't specified
        # assume one to one mapping of stress periods between models
        if self._inset_parent_period_mapping is None:
            parent_periods = list(range(self.parent.nper))
            self._inset_parent_period_mapping = {i: parent_periods[i]
            if i < self.parent.nper else parent_periods[-1] for i in range(nper)}
        return self._inset_parent_period_mapping

    @inset_parent_period_mapping.setter
    def inset_parent_period_mapping(self, inset_parent_period_mapping):
        self._inset_parent_period_mapping = inset_parent_period_mapping

    @property
    def interp_weights_flux(self):
        """For the two main directions of flux (i, j) and the four orientations of
        inset faces to interpolate to (right.left,top,bottom
        we can precalulate the interpolation weights of the combinations to speed up
        interpolation"""
        if self._interp_weights_flux is None:
            self._interp_weights_flux = dict() # we need four flux directions for the insets
            # x, y, z locations of parent model head values for i faces
            ipx, ipy, ipz = self.x_iface_parent, self.y_iface_parent, self.z_iface_parent
            # x, y, z locations of parent model head values for j faces
            jpx, jpy, jpz = self.x_jface_parent, self.y_jface_parent, self.z_jface_parent


            # these are the i-direction fluxes
            x,y,z = self.inset_boundary_cell_faces.loc[
                self.inset_boundary_cell_faces.cellface.isin(['top','bottom'])][['xface','yface','zface']].T.values
            self._interp_weights_flux['iface'] = interp_weights((ipx, ipy, ipz), (x, y, z), d=3)
            assert not np.any(np.isnan(self._interp_weights_flux['iface'][1]))

            # these are the j-direction fluxes
            x,y,z = self.inset_boundary_cell_faces.loc[
                self.inset_boundary_cell_faces.cellface.isin(['left','right'])][['xface','yface','zface']].T.values

            self._interp_weights_flux['jface'] = interp_weights((jpx, jpy, jpz), (x, y, z), d=3)
            assert not np.any(np.isnan(self._interp_weights_flux['jface'][1]))


        return self._interp_weights_flux

    @property
    def parent_xyzcellcenters(self):
        """Get x, y, z locations of parent cells in a buffered area
        (defined by the _source_grid_mask property) around the
        inset model."""
        px, py, pz = self.parent.modelgrid.xyzcellcenters

        # add an extra layer on the top and bottom
        # for inset model cells above or below
        # the last cell center in the vert. direction
        # pad top by top layer thickness
        b1 = self.parent.modelgrid.top - self.parent.modelgrid.botm[0]
        top = pz[0] + b1
        # pad botm by botm layer thickness
        if self.parent.modelgrid.shape[0] > 1:
            b2 = -np.diff(self.parent.modelgrid.botm[-2:], axis=0)[0]
        else:
            b2 = b1
        botm = pz[-1] - b2
        pz = np.vstack([[top], pz, [botm]])

        nlay, nrow, ncol = pz.shape
        px = np.tile(px, (nlay, 1, 1))
        py = np.tile(py, (nlay, 1, 1))
        mask = self._source_grid_mask
        # mask already has extra top/botm layers
        # (_source_grid_mask property)
        px = px[mask]
        py = py[mask]
        pz = pz[mask]
        return px, py, pz

    @property
    def parent_xyzcellfacecenters(self):
        """Get x, y, z locations of the centroids of the cell faces
        in the row and column directions in a buffered area
        (defined by the _source_grid_mask property) around the
        inset model. Analogous to parent_xyzcellcenters, but for
        interpolating parent model cell by cell fluxes that are located
        at the cell face centers (instead of heads that are located
        at the cell centers).
        """
        #px, py, pz = self.parent.modelgrid.xyzcellcenters
        k, i, j = np.indices(self.parent.modelgrid.shape)
        xyzcellfacecenters = {}
        for cellface in 'right', 'bottom':
            px, py, pz = get_cellface_midpoint(self.parent.modelgrid,
                                               k, i, j,
                                               cellface)
            px = np.reshape(px, self.parent.modelgrid.shape)
            py = np.reshape(py, self.parent.modelgrid.shape)
            pz = np.reshape(pz, self.parent.modelgrid.shape)
            # add an extra layer on the top and bottom
            # for inset model cells above or below
            # the last cell center in the vert. direction
            # pad top by top layer thickness
            b1 = self.parent.modelgrid.top - self.parent.modelgrid.botm[0]
            top = pz[0] + b1
            # pad botm by botm layer thickness
            if self.parent.modelgrid.shape[0] > 1:
                b2 = -np.diff(self.parent.modelgrid.botm[-2:], axis=0)[0]
            else:
                b2 = b1
            botm = pz[-1] - b2
            pz = np.vstack([[top], pz, [botm]])

            nlay, nrow, ncol = pz.shape
            px = np.tile(px, (nlay, 1, 1))
            py = np.tile(py, (nlay, 1, 1))
            mask = self._source_grid_mask
            # mask already has extra top/botm layers
            # (_source_grid_mask property)
            px = px[mask]
            py = py[mask]
            pz = pz[mask]

            xyzcellfacecenters[cellface] = px, py, pz
        return xyzcellfacecenters


    @property
    def _inset_max_active_area(self):
        """The maximum (2D) footprint of the active area within the inset
        model grid, where each i, j location has at least 1 active cell
        vertically, excluding any inactive holes that are surrounded by
        active cells.
        """
        # get the max footprint of active cells
        max_active_area = np.sum(self.idomain > 0, axis=0) > 0
        # fill any holes within the max footprint
        # including any LGR areas (that are inactive in this model)
        # set min cluster size to 1 greater than number of inactive cells
        # (to not allow any holes)
        minimum_cluster_size = np.sum(max_active_area == 0) + 1
        # find_remove_isolated_cells fills clusters of 1s with 0s
        # to fill holes, we want to look for clusters of 0s and fill with 1s
        to_fill = ~max_active_area
        # pad the array to fill so that exterior inactive cells
        # (outside the active area perimeter) aren't included
        to_fill = np.pad(to_fill, pad_width=1, mode='reflect')
        # invert the result to get True values for active cells and filled areas
        filled = ~find_remove_isolated_cells(to_fill, minimum_cluster_size)
        # de-pad the result
        filled = filled[1:-1, 1:-1]
        max_active_area = filled
        return max_active_area

    @property
    def inset_zone_within_parent(self):
        """The footprint of the inset model maximum active area footprint
        (``Tmr._inset_max_active_area``) within the parentmodel grid.
        In other words, all parent cells containing one or inset
        model cell centers within ``Tmr._inset_max_active_area`` (ones).
        Zeros indicate parent cells with no inset cells.
        """
        # get the locations of the inset model cells within _inset_max_active_area
        x, y, z = self.inset.modelgrid.xyzcellcenters
        x = x[self._inset_max_active_area]
        y = y[self._inset_max_active_area]
        pi, pj = get_ij(self.parent.modelgrid, x, y)
        inset_zone_within_parent = np.zeros((self.parent.modelgrid.nrow,
                                             self.parent.modelgrid.ncol), dtype=bool)
        inset_zone_within_parent[pi, pj] = True
        return inset_zone_within_parent


    @property
    def _source_grid_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the inset model domain. Used to speed up interpolation
        of parent grid values onto inset grid."""
        if self._source_mask is None:
            mask = np.zeros((self.parent.modelgrid.nrow,
                             self.parent.modelgrid.ncol), dtype=bool)
            if hasattr(self.inset, 'parent_mask') and \
                (self.inset.parent_mask.shape == self.parent.modelgrid.xcellcenters.shape):
                mask = self.inset.parent_mask
            else:
                #x, y = np.squeeze(self.inset.modelgrid.bbox.exterior.coords.xy)
                l, r, b, t = self.inset.modelgrid.extent
                x = np.array([r, r, l, l, r])
                y = np.array([b, t, t, b, b])
                pi, pj = get_ij(self.parent.modelgrid, x, y)
                pad = 3
                i0 = np.max([pi.min() - pad, 0])
                i1 = np.min([pi.max() + pad + 1, self.parent.modelgrid.nrow])
                j0 = np.max([pj.min() - pad, 0])
                j1 = np.min([pj.max() + pad + 1, self.parent.modelgrid.ncol])
                mask[i0:i1, j0:j1] = True
            # make the mask 3D
            # include extra layer for top and bottom edges of model
            mask3d = np.tile(mask, (self.parent.modelgrid.nlay + 2, 1, 1))
            self._source_mask = mask3d
        elif len(self._source_mask.shape) == 2:
            mask3d = np.tile(self._source_mask, (self.parent.modelgrid.nlay + 2, 1, 1))
            self._source_mask = mask3d
        return self._source_mask

    def get_inset_boundary_cells(self, by_layer=False, shapefile=None):
        """Get a dataframe of connection information for
        horizontal boundary cells.

        Parameters
        ----------
        by_layer : bool
            Controls how boundary cells will be defined. If True,
            the perimeter of the active area in each layer will be used
            (excluding any interior clusters of active cells). If
            False, the maximum footprint of the active area
            (including all cell locations with at least one layer that
            is active).
        """
        print('\ngetting perimeter cells...')
        t0 = time.time()
        if shapefile is None:
            shapefile = self.shapefile
        if shapefile:
            perimeter = gp.read_file(shapefile)
            perimeter = perimeter[['geometry']]
            # reproject the perimeter shapefile to the model CRS if needed
            if perimeter.crs != self.inset.modelgrid.crs:
                perimeter.to_crs(self.inset.modelgrid.crs, inplace=True)
            # convert polygons to linear rings
            # (so just the cells along the polygon exterior are selected)
            geoms = []
            for g in perimeter.geometry:
                if g.type == 'MultiPolygon':
                    g = MultiLineString([p.exterior for p in g.geoms])
                elif g.type == 'Polygon':
                    g = g.exterior
                geoms.append(g)
            # add a buffer of 1 cell width so that cells aren't missed
            # extra cells will get culled later
            # when only cells along the outer perimeter (max idomain extent)
            # are selected
            buffer_dist = np.mean([self.inset.modelgrid.delr.mean(),
                                   self.inset.modelgrid.delc.mean()])
            perimeter['geometry'] = [g.buffer(buffer_dist * 0.5) for g in geoms]
            grid_df = self.inset.modelgrid.get_dataframe(layers=False)
            df = gp.sjoin(grid_df, perimeter, predicate='intersects', how='inner')
            # add layers
            dfs = []
            for k in range(self.inset.modelgrid.nlay):
                kdf = df.copy()
                kdf['k'] = k
                dfs.append(kdf)
            specified_bcells = pd.concat(dfs)
            # get the active extent in each layer
            # and the cell faces along the edge
            # apply those cell faces to specified_bcells
            by_layer = True
        else:
            specified_bcells = None
        if not by_layer:

            # attached the filled array as an attribute
            max_active_area = self._inset_max_active_area

            # pad filled idomain array with zeros around the edge
            # so that perimeter connections are identified
            filled = np.pad(max_active_area, 1, constant_values=0)
            filled3d = np.tile(filled, (self.idomain.shape[0], 1, 1))
            df = get_horizontal_connections(filled3d, connection_info=False)
            # deincrement rows and columns
            # so that they reflect positions in the non-padded array
            df['i'] -= 1
            df['j'] -= 1
        else:
            dfs = []
            for k, layer_idomain in enumerate(self.idomain):

                # just get the perimeter of inactive cells
                # (exclude any interior active cells)
                # start by filling any interior active cells
                from scipy.ndimage import binary_fill_holes
                binary_idm = layer_idomain > 0
                filled = binary_fill_holes(binary_idm)
                # pad filled idomain array with zeros around the edge
                # so that perimeter connections are identified
                filled = np.pad(filled, 1, constant_values=0)
                # get the cells along the inside edge
                # of the model active area perimeter,
                # via a sobel filter
                df = get_horizontal_connections(filled, connection_info=False)
                df['k'] = k
                # deincrement rows and columns
                # so that they reflect positions in the non-padded array
                df['i'] -= 1
                df['j'] -= 1
                dfs.append(df)
            df = pd.concat(dfs)

            # cull the boundary cells identified above
            # with the sobel filter on the outer perimeter
            # to just the cells specified in the shapefile
            if specified_bcells is not None:
                df['cellid'] = list(zip(df.k, df.i, df.j))
                specified_bcells['cellid'] = list(zip(specified_bcells.k, specified_bcells.i, specified_bcells.j))
                df = df.loc[df.cellid.isin(specified_bcells.cellid)]

        # add layer top and bottom and idomain information
        layer_tops = np.stack([self.inset.dis.top.array] +
                              [l for l in self.inset.dis.botm.array])[:-1]
        df['top'] = layer_tops[df.k, df.i, df.j]
        df['botm'] = self.inset.dis.botm.array[df.k, df.i, df.j]
        df['idomain'] = 1
        if self.inset.version == 'mf6':
            df['idomain'] = self.idomain[df.k, df.i, df.j]
        elif 'BAS6' in self.inset.get_package_list():
            df['idomain'] = self.inset.bas6.ibound.array[df.k, df.i, df.j]
        df = df[['k', 'i', 'j', 'cellface', 'top', 'botm', 'idomain']]
        # drop inactive cells
        df = df.loc[df['idomain'] > 0]

        # get cell polygons from modelgrid
        # write shapefile of boundary cells with face information
        grid_df = self.inset.modelgrid.dataframe.copy()
        grid_df['cellid'] = list(zip(grid_df.k, grid_df.i, grid_df.j))
        geoms = dict(zip(grid_df['cellid'], grid_df['geometry']))
        if 'cellid' not in df.columns:
            df['cellid'] = list(zip(df.k, df.i, df.j))
        df['geometry'] = [geoms[cellid] for cellid in df.cellid]
        df = gp.GeoDataFrame(df, crs=self.inset.modelgrid.crs)
        outshp = Path(self._tables_path, 'boundary_cells.shp')
        df.drop('cellid', axis=1).to_file(outshp)
        print(f"wrote {outshp}")
        print("perimeter cells took {:.2f}s\n".format(time.time() - t0))
        return df

    def get_inset_boundary_values(self, for_external_files=False):

        if self.boundary_type == 'head':
            check_source_files([self.parent_head_file])
            hdsobj = bf.HeadFile(self.parent_head_file)  # , precision='single')
            all_kstpkper = hdsobj.get_kstpkper()

            last_steps = {kper: kstp for kstp, kper in all_kstpkper}

            # create an interpolator instance
            cell_centers_interp = Interpolator(self.parent_xyzcellcenters,
                                               self.inset_boundary_cells[['x', 'y', 'z']].T.values,
                                               d=3,
                                               source_values_mask=self._source_grid_mask)
            # compute the weights
            _ = cell_centers_interp.interp_weights

            print('\ngetting perimeter heads...')
            t0 = time.time()
            dfs = []
            parent_periods = []
            for inset_per, parent_per in self.inset_parent_period_mapping.items():
                print(f'for stress period {inset_per}', end=', ')
                t1 = time.time()
                # skip getting data if parent period is already represented
                # (heads will be reused)
                if parent_per in parent_periods:
                    continue
                else:
                    parent_periods.append(parent_per)
                parent_kstpkper = last_steps[parent_per], parent_per
                parent_heads = hdsobj.get_data(kstpkper=parent_kstpkper)
                # pad the parent heads on the top and bottom
                # so that inset cells above and below the top/bottom cell centers
                # will be within the interpolation space
                # (parent x, y, z locations already contain this pad; parent_xyzcellcenters)
                parent_heads = np.pad(parent_heads, pad_width=1, mode='edge')[:, 1:-1, 1:-1]

                # interpolate inset boundary heads from 3D parent head solution
                heads = cell_centers_interp.interpolate(parent_heads, method='linear')
                #heads = griddata((px, py, pz), parent_heads.ravel(),
                #                  (x, y, z), method='linear')

                # make a DataFrame of interpolated heads at perimeter cell locations
                df = self.inset_boundary_cells.copy()
                df['per'] = inset_per
                df['head'] = heads

                # boundary heads must be greater than the cell bottom
                # and idomain > 0
                loc = (df['head'] > df['botm']) & (df['idomain'] > 0)
                df = df.loc[loc]
                # drop invalid heads (most likely due to dry cells)
                valid = (df['head'] < 1e10) & (df['head'] > -1e10)
                df = df.loc[valid]
                dfs.append(df)
                print("took {:.2f}s".format(time.time() - t1))

            df = pd.concat(dfs)
            # drop duplicate cells (accounting for stress periods)
            # (that may have connections in the x and y directions,
            #  and therefore would be listed twice)
            df['cellid'] = list(zip(df.per, df.k, df.i, df.j))
            duplicates = df.duplicated(subset=['cellid'])
            df = df.loc[~duplicates, ['k', 'i', 'j', 'per', 'head']]
            print("getting perimeter heads took {:.2f}s\n".format(time.time() - t0))


        elif self.boundary_type == 'flux':
            check_source_files([self.parent_cell_budget_file])
            if self.parent.version == 'mf6':
                if self.parent_binary_grid_file is None:
                    raise ValueError('Specified flux perimeter boundary requires a parent_binary_grid_file if parent is MF6')
                else:
                    check_source_files([self.parent_binary_grid_file])
            fileobj = bf.CellBudgetFile(self.parent_cell_budget_file)  # , precision='single')
            all_kstpkper = fileobj.get_kstpkper()

            last_steps = {kper: kstp for kstp, kper in all_kstpkper}

            print('\ngetting perimeter fluxes...')
            t0 = time.time()
            dfs = []
            parent_periods = []

            # TODO: consider refactoring to move this into its own function
            # * handle vertical fluxes
            # * possibly handle rotated inset with differnt angle than parent - now assuming colinear
            # * Handle the geometry issues for the inset
            # * need to locate edge faces (x,y,z) based on which faces is out (e.g. left, right, up, down)

            # TODO: refactor self.inset_boundary_cells
            # it's probably not ideal to have self.inset_boundary_cells
            # be a 'public' attribute that gets modified every stress period
            # but without any information tying the current state of it
            # to a specific stress period. It should either have all stress periods
            # or the stress period-specific information
            # (the fluxes and cell thickness if we are considering sat. thickness)
            # pulled out into a separate container

            # make a dataframe to store these
            self.inset_boundary_cell_faces = self.inset_boundary_cells.copy()
            # get the locations of the boundary face midpoints
            x, y, z = get_cellface_midpoint(self.inset.modelgrid,
                    *self.inset_boundary_cells[['k', 'i', 'j', 'cellface']].T.values)
            self.inset_boundary_cell_faces['x'] = x
            self.inset_boundary_cell_faces['y'] = y
            self.inset_boundary_cell_faces['z'] = z
            # renaming columns to be clear now x,y,z, is for the outer cell face
            #self.inset_boundary_cell_faces.rename(columns={'x':'xface','y':'yface','z':'zface'}, inplace=True)
            # convert x,y coordinates to model coords from world coords
            #self.inset_boundary_cell_faces.xface, self.inset_boundary_cell_faces.yface = \
            #        self.inset.modelgrid.get_local_coords(self.inset_boundary_cell_faces.xface, self.inset_boundary_cell_faces.yface)
            # calculate the thickness to later get the area
            # TODO: consider saturated thickness instead, but this would require interpolating parent heads to inset cell locations

            self.inset_boundary_cell_faces['thickness'] = self.inset_boundary_cell_faces.top - self.inset_boundary_cell_faces.botm
            # populate cell face widths
            self.inset_boundary_cell_faces['width'] = np.nan
            left_right_faces = self.inset_boundary_cell_faces['cellface'].isin({'left', 'right'})
            # left and right faces are along columns
            rows = self.inset_boundary_cell_faces.loc[left_right_faces, 'i']
            self.inset_boundary_cell_faces.loc[left_right_faces, 'width'] = self.inset.modelgrid.delc[rows]
            # top and bottom faces are along rows
            top_bottom_faces = self.inset_boundary_cell_faces['cellface'].isin({'top', 'bottom'})
            columns = self.inset_boundary_cell_faces.loc[top_bottom_faces, 'j']
            self.inset_boundary_cell_faces.loc[top_bottom_faces, 'width'] = self.inset.modelgrid.delr[columns]
            assert not self.inset_boundary_cell_faces['width'].isna().any()

            self.inset_boundary_cell_faces['face_area'] = self.inset_boundary_cell_faces['width'] *\
                self.inset_boundary_cell_faces['thickness']
            # pre-seed the area as thickness to later mult by width
            #self.inset_boundary_cell_faces['face_area'] = self.inset_boundary_cell_faces['thickness'].values
            # placeholder for interpolated values
            self.inset_boundary_cell_faces['q_interp'] = np.nan
            # placeholder for flux to well package
            # self.inset_boundary_cell_faces['Q'] = np.nan

            # make a grid of the spacings
            #delr_gridi, delc_gridi = np.meshgrid(self.inset.modelgrid.delr, self.inset.modelgrid.delc)
            #
            #for cn in self.inset_boundary_cell_faces.cellface.unique():
            #    curri = self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn].i
            #    currj = self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn].j
            #    curr_delc = delc_gridi[curri, currj]
            #    curr_delr = delr_gridi[curri, currj]
            #    if cn == 'top':
            #        #self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'yface'] += curr_delc/2
            #        self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delr
            #    elif cn == 'bottom':
            #        #self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'yface'] -= curr_delc/2
            #        self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delr
            #    if cn == 'right':
            #        #self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'xface'] += curr_delr/2
            #        self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delc
            #    elif cn == 'left':
            #        #self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'xface'] -= curr_delr/2
            #        self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delc

            #
            # Now handle the geometry issues for the parent
            # first thicknesses (at cell centers)

            parent_thick = self.parent.modelgrid.cell_thickness

            # make matrices of the row and column spacings
            # NB --> trying to preserve the always seemingly
            # backwards delr/delc definitions
            # also note - for now, taking average thickness at a connected face

            # need XYZ locations of the center of each face for
            # iface and jface edges (faces)
            # NB edges are returned in model coordinates
            #xloc_edge, yloc_edge = self.parent.modelgrid.xyedges
            #nlay = self.parent.modelgrid.nlay
            #nrow = self.parent.modelgrid.nrow
            #ncol = self.parent.modelgrid.ncol
            ## throw out the left and top edges, respectively
            #xloc_edge=xloc_edge[1:]
            #yloc_edge=yloc_edge[1:]
            ## tile out to full dimensions of the grid
            #xloc_edge = np.tile(np.atleast_2d(xloc_edge),(nlay+2,nrow,1))
            #yloc_edge = np.tile(np.atleast_2d(yloc_edge).T,(nlay+2,1,ncol))
#
            ## TODO: implement vertical fluxes
            #''' parent_vface_areas  = np.tile(delc_grid, (nlay,1,1)) * \
            #                    np.tile(delr_grid, (nlay,1,1))
            #'''
            #xloc_center, yloc_center = self.parent.modelgrid.xycenters
#
            ## tile out to full dimensions of the grid
#
            #xloc_center = np.tile(np.atleast_2d(xloc_center),(nlay+2,nrow,1))
            #yloc_center = np.tile(np.atleast_2d(yloc_center).T,(nlay+2,1,ncol))
#
            ## get the vertical centroids initially at cell centroids
            #zloc = (self.parent.modelgrid.top_botm[:-1,:,:] +
            #    self.parent.modelgrid.top_botm[1:,:,:] ) / 2
#
            ## pad in the vertical above and below the model
            #zpadtop = np.expand_dims(self.parent.modelgrid.top_botm[0,:,:] + parent_thick[0], axis=0)
            #zpadbotm = np.expand_dims(self.parent.modelgrid.top_botm[-1,:,:] - parent_thick[-1], axis=0)
            #zloc=np.vstack([zpadtop,zloc,zpadbotm])
#
            ## for iface, all cols, nrow-1 rows
            #self.x_iface_parent = xloc_center[:,:-1,:].ravel()
            #self.y_iface_parent = yloc_edge[:,:,:-1].ravel()
            ## need to calculate the average z location along rows
            #self.z_iface_parent = ((zloc[:,:-1,:]+zloc[:,1:,:]) / 2).ravel()
            ## for jface, all rows, ncol-1 cols
            #self.x_jface_parent = xloc_edge[:,:-1,:].ravel()
            #self.y_jface_parent = yloc_center[:,:,:-1].ravel()
            ## need to calculate the average z location along columns
            #self.z_jface_parent = ((zloc[:,:,:-1]+zloc[:,:,1:]) / 2).ravel()
            ## for kface, all cols, all rows
            #self.x_kface_parent = xloc_center.ravel()
            #self.y_kface_parent = yloc_center.ravel()
            ##  for zlocations, -1 layers
            #self.z_kface_parent = zloc.ravel()
#
            #'''
            ## get the perimeter cells and calculate the weights
            #_ = self.interp_weights_flux
            #'''
            # interpolate parent face centers
            # (where the cell by cell flows and specific discharge values are located)
            # to inset face centers along the exterior sides of the boundary cells
            # (the edge of the inset model, where the boundary fluxes will be located)

            # interpolate parent y fluxes (column parallel)
            # to inset boundary cell face centers
            #px = self.x_iface_parent
            #py = self.y_iface_parent
            #pz = self.z_iface_parent
            px, py, pz = self.parent_xyzcellcenters
            #px, py, pz = self.parent_xyzcellfacecenters['bottom']
            iface_interp = Interpolator((px, py, pz),
                                        #self.inset_boundary_cell_faces[['x', 'y', 'z']].T.values,
                                        self.inset_boundary_cells[['x', 'y', 'z']].T.values,
                                        d=3, source_values_mask=self._source_grid_mask
                                        )
            _ = iface_interp.interp_weights
            # interpolate parent x fluxes (row parallel)
            # to inset boundary cell face centers
            #px = self.x_jface_parent
            #py = self.y_jface_parent
            #pz = self.z_jface_parent
            #px, py, pz = self.parent_xyzcellfacecenters['right']
            #jface_interp = Interpolator((px, py, pz),
            #                            #self.inset_boundary_cell_faces[['x', 'y', 'z']].T.values,
            #                            self.inset_boundary_cells[['x', 'y', 'z']].T.values,
            #                            d=3, source_values_mask=self._source_grid_mask
            #                            )
            #_ = jface_interp.interp_weights

            #kface_interp = Interpolator((self.x_kface_parent, self.y_kface_parent, self.z_kface_parent),
            #                            self.inset_boundary_cells[['x', 'y', 'z']].T.values,
            #                            d=3)
            #_ = kface_interp.interp_weights

            # get a dataframe of cell connections
            # (that can be reused with subsequent stress periods)
            cell_connections_df = None
            if self.parent.version == 'mf6':
                cell_connections_df = get_intercell_connections(self.parent_binary_grid_file)

            for inset_per, parent_per in self.inset_parent_period_mapping.items():
                print(f'for stress period {inset_per}', end=', ')
                t1 = time.time()
                # skip getting data if parent period is already represented
                # (heads will be reused)
                if parent_per in parent_periods:
                    continue
                else:
                    parent_periods.append(parent_per)
                parent_kstpkper = last_steps[parent_per], parent_per

                # get parent specific discharge for inset area
                qx, qy, qz = get_qx_qy_qz(self.parent_cell_budget_file,
                                          cell_connections_df=cell_connections_df,
                                          version=self.parent.version,
                                          kstpkper=parent_kstpkper,
                                          specific_discharge=True,
                                          modelgrid=self.parent.modelgrid,
                                          headfile=self.parent_head_file)

                # pad the two parent flux arrays on the top and bottom
                # so that inset cells above and below the top/bottom cell centers
                # will be within the interpolation space
                qx = np.pad(qx, pad_width=1, mode='edge')[:, 1:-1, 1:-1]
                qy = np.pad(qy, pad_width=1, mode='edge')[:, 1:-1, 1:-1]
                qz = np.pad(qz, pad_width=1, mode='edge')[:, 1:-1, 1:-1]


                # TODO: consider padding or not on top, left, and "top (row-wise)"
                # (parent x, y, z locations already contain this pad - see zloc above)
                #q_iface = np.pad(q_iface, pad_width=1, mode='edge')[:, 1:-1, 1:-1].ravel()
                #q_jface = np.pad(q_jface, pad_width=1, mode='edge')[:, 1:-1, 1:-1].ravel()


                # TODO: refactor interpolation to use the new interpolator object - DONE: see above
                # interpolate q at the four different face orientations (e.g. fluxdir)

                # interpolate inset boundary heads from 3D parent head solution
                t2 = time.time()
                y_flux = iface_interp.interpolate(qy, method='linear')
                x_flux = iface_interp.interpolate(qx, method='linear')
                # v_flux = kface_interp.interpolate(qz, method='linear')
                f"interpolation took {time.time() - t2:.2f}s"

                t2 = time.time()
                self.inset_boundary_cell_faces = self.inset_boundary_cell_faces.assign(
                    qx_interp=x_flux,
                    qy_interp=y_flux)#,
                    #qz_interp=v_flux)

                # assign q values and flip the sign for flux counter to the CBB convention directions of right and bottom
                top_faces = self.inset_boundary_cell_faces.cellface == 'top'
                self.inset_boundary_cell_faces.loc[top_faces, 'q_interp'] = self.inset_boundary_cell_faces.loc[top_faces, 'qy_interp']
                bottom_faces = self.inset_boundary_cell_faces.cellface == 'bottom'
                self.inset_boundary_cell_faces.loc[bottom_faces, 'q_interp'] = -self.inset_boundary_cell_faces.loc[bottom_faces, 'qy_interp']
                left_faces = self.inset_boundary_cell_faces.cellface == 'left'
                self.inset_boundary_cell_faces.loc[left_faces, 'q_interp'] = self.inset_boundary_cell_faces.loc[left_faces, 'qx_interp']
                right_faces = self.inset_boundary_cell_faces.cellface == 'right'
                self.inset_boundary_cell_faces.loc[right_faces, 'q_interp'] = -self.inset_boundary_cell_faces.loc[right_faces, 'qx_interp']

                # convert specific discharge in inset cells to Q -- flux for well package
                self.inset_boundary_cell_faces['q'] = \
                    self.inset_boundary_cell_faces['q_interp'] * self.inset_boundary_cell_faces['face_area']


                # make a DataFrame of boundary fluxes at perimeter cell locations
                df = self.inset_boundary_cell_faces[['k','i','j','idomain','q']].copy()
                # aggregate fluxes by cell
                # so that we are accurately compare to the WELL package budget in the listing file
                #by_cell = df.groupby('cellid').first()
                #by_cell['q'] = df.groupby('cellid').sum()['q']
                ## drop the cellid index
                #by_cell.reset_index(drop=True, inplace=True)
                df['per'] = inset_per

                # boundary fluxes must be in active cells
                # corresponding parent cells must be active too,
                # otherwise a nan flux will be produced
                # drop nan fluxes, which will revert these boundary cells to the
                # default no-flow condition in MODFLOW
                # (consistent with parent model cell being inactive)
                keep = (df['idomain'] > 0) & ~df['q'].isna()
                dfs.append(df.loc[keep].copy())
                f"assigning face fluxes took {time.time() - t2:.2f}s"
                print(f"took {time.time() - t1:.2f}s total")

            df = pd.concat(dfs)
            # drop duplicate cells (accounting for stress periods)
            # (that may have connections in the x and y directions,
            #  and therefore would be listed twice)
            #df['cellid'] = list(zip(df.per, df.k, df.i, df.j))
            #duplicates = df.duplicated(subset=['cellid'])
            #df = df.loc[~duplicates, ['k', 'i', 'j', 'per', 'q']]
            print("getting perimeter fluxes took {:.2f}s\n".format(time.time() - t0))

        # convert to one-based and comment out header if df will be written straight to external file
        if for_external_files:
            df.rename(columns={'k': '#k'}, inplace=True)
            df['#k'] += 1
            df['i'] += 1
            df['j'] += 1
        return df
