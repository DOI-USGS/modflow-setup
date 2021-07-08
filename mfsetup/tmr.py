import os
import time
from pathlib import Path

import flopy
import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString

fm = flopy.modflow
from flopy.discretization import StructuredGrid
from flopy.utils import binaryfile as bf
from flopy.utils.mfgrdfile import MfGrdFile
from flopy.utils.postprocessing import get_water_table
from scipy.interpolate import griddata

from mfsetup.discretization import (
    find_remove_isolated_cells,
    weighted_average_between_layers,
)
from mfsetup.fileio import check_source_files
from mfsetup.grid import get_ij
from mfsetup.interpolate import (
    get_source_dest_model_xys,
    interp_weights,
    interpolate,
    regrid,
)
from mfsetup.lakes import get_horizontal_connections
from mfsetup.sourcedata import ArraySourceData
from mfsetup.units import convert_length_units


class Tmr:
    """
    Class for basic telescopic mesh refinement of a MODFLOW model.
    Handles the case where the pfl_nwt grid is a rectangle exactly aligned with
    the parent grid.

    Parameters
    ----------
    parent_model : flopy.modflow.Modflow instance of parent model
        Must have a valid, attached ModelGrid (modelgrid) attribute.
    inset_model : flopy.modflow.Modflow instance of pfl_nwt model
        Must have a valid, attached ModelGrid (modelgrid) attribute.
        ModelGrid of pfl_nwt and parent models is used to determine cell
        connections.
    parent_head_file : filepath
        MODFLOW binary head output
    parent_cell_budget_file : filepath
        MODFLOW binary cell budget output


    Notes
    -----
    Assumptions:
    * Uniform parent and pfl_nwt grids, with equal delr and delc spacing.
    * Inset model upper right corner coincides with an upper right corner of a cell
      in the parent model
    * Inset cell spacing is a factor of the parent cell spacing
      (so that each pfl_nwt cell is only connected horizontally to one parent cell).
    * Inset model row/col dimensions are multiples of parent model cells
      (no parent cells that only partially overlap the pfl_nwt model)
    * Horizontally, fluxes are uniformly distributed to child cells within a parent cell. The
    * Vertically, fluxes are distributed based on transmissivity (sat. thickness x Kh) of
    pfl_nwt model layers.
    * The pfl_nwt model is fully penetrating. Total flux through each column of parent cells
    is equal to the total flux through the corresponding columns of connected pfl_nwt model cells.
    The get_inset_boundary_flux_side verifies this with an assertion statement.

    """
    flow_component = {'top': 'fff', 'bottom': 'fff',
                      'left': 'frf', 'right': 'frf'}
    flow_sign = {'top': 1, 'bottom': -1,
                 'left': 1, 'right': -1}

    def __init__(self, parent_model, inset_model,
                 parent_head_file=None, parent_cell_budget_file=None,
                 parent_length_units=None, inset_length_units=None,
                 inset_parent_layer_mapping=None,
                 inset_parent_period_mapping=None,
                 ):

        self.inset = inset_model
        self.parent = parent_model
        self.inset._set_parent_modelgrid()
        self.cbc = None
        self._inset_parent_layer_mapping = inset_parent_layer_mapping
        self._source_mask = None
        self._inset_parent_period_mapping = inset_parent_period_mapping
        self.hpth = None  # path to parent heads output file
        self.cpth = None  # path to parent cell budget output file

        self.pi0 = None
        self.pj0 = None
        self.pi1 = None
        self.pj1 = None
        self.pi_list = None
        self.pj_list = None

        if parent_length_units is None:
            parent_length_units = self.inset.cfg['parent']['length_units']
        if inset_length_units is None:
            inset_length_units = self.inset.length_units
        self.length_unit_conversion = convert_length_units(parent_length_units, inset_length_units)

        if parent_head_file is None:
            parent_head_file = os.path.join(self.parent.model_ws,
                                     '{}.hds'.format(self.parent.name))
            if os.path.exists(parent_head_file):
                self.hpth = parent_cell_budget_file
        else:
            self.hpth = parent_head_file
        if parent_cell_budget_file is None:
            for extension in 'cbc', 'cbb':
                parent_cell_budget_file = os.path.join(self.parent.model_ws,
                                         '{}.{}'.format(self.parent.name, extension))
                if os.path.exists(parent_cell_budget_file):
                    self.cpth = parent_cell_budget_file
                    break
        else:
            self.cpth = parent_cell_budget_file

        if self.hpth is None and self.cpth is None:
            raise ValueError("No head or cell budget output files found for parent model {}".format(self.parent.name))

        # get bounding cells in parent model for pfl_nwt model
        irregular_domain = False

        # see if irregular domain
        irregbound_cfg = self.inset.cfg['perimeter_boundary'].get('source_data',{}).get('irregular_boundary')
        if irregbound_cfg is not None:
            irregular_domain = True
            irregbound_cfg['variable'] = 'perimeter_boundary'
            irregbound_cfg['dest_model'] = self.inset


            sd = ArraySourceData.from_config(irregbound_cfg)
            data = sd.get_data()
            idm_outline = data[0]
            connections = get_horizontal_connections(idm_outline, connection_info=False,
                                             layer_elevations=1,
                                             delr=1, delc=1, inside=True)
            self.pi_list, self.pj_list = connections.i.to_list(), connections.j.to_list()
        # otherwise just get the corners of the inset if rectangular domain
        else:
            self.pi0, self.pj0 = get_ij(self.parent.modelgrid,
                                        self.inset.modelgrid.xcellcenters[0, 0],
                                        self.inset.modelgrid.ycellcenters[0, 0])
            self.pi1, self.pj1 = get_ij(self.parent.modelgrid,
                                        self.inset.modelgrid.xcellcenters[-1, -1],
                                        self.inset.modelgrid.ycellcenters[-1, -1])
            self.parent_nrow_in_inset = self.pi1 - self.pi0 + 1
            self.parent_ncol_in_inset = self.pj1 - self.pj0 + 1

        # check for an even number of inset cells per parent cell in x and y directions
        x_refinement = self.parent.modelgrid.delr[0] / self.inset.modelgrid.delr[0]
        y_refinement = self.parent.modelgrid.delc[0] / self.inset.modelgrid.delc[0]
        msg = "inset {0} of {1:.2f} {2} must be factor of parent {0} of {3:.2f} {4}"
        if not int(x_refinement) == np.round(x_refinement, 2):
            raise ValueError(msg.format('delr', self.inset.modelgrid.delr[0], self.inset.modelgrid.length_units,
                                        self.parent.modelgrid.delr[0], self.parent.modelgrid.length_units))
        if not int(y_refinement) == np.round(y_refinement, 2):
            raise ValueError(msg.format('delc', self.inset.modelgrid.delc[0], self.inset.modelgrid.length_units,
                                        self.parent.modelgrid.delc[0], self.parent.modelgrid.length_units))
        if not np.allclose(x_refinement, y_refinement):
            raise ValueError("grid must have same x and y discretization")
        self.refinement = int(x_refinement)

    @property
    def inset_parent_layer_mapping(self):
        nlay = self.inset.nlay
        # if mapping between source and dest model layers isn't specified
        # use property from dest model
        # this will be the DIS package layer mapping if specified
        # otherwise same layering is assumed for both models
        if self._inset_parent_layer_mapping is None:
            return self.inset.parent_layers
        elif self._inset_parent_layer_mapping is not None:
            nspecified = len(self._inset_parent_layer_mapping)
            if nspecified != nlay:
                raise Exception("Variable should have {} layers "
                                "but only {} are specified: {}"
                                .format(nlay, nspecified, self._inset_parent_layer_mapping))
            return self._inset_parent_layer_mapping

    @property
    def inset_parent_period_mapping(self):
        nper = self.inset.nper
        # if mapping between source and dest model periods isn't specified
        # assume one to one mapping of stress periods between models
        if self._inset_parent_period_mapping is None:
            parent_periods = list(range(self.parent.nper))
            self._inset_parent_period_mapping = {i: parent_periods[i]
            if i < self.parent.nper else parent_periods[-1] for i in range(nper)}
        else:
            return self._inset_parent_period_mapping

    @inset_parent_period_mapping.setter
    def inset_parent_period_mapping(self, inset_parent_period_mapping):
        self._inset_parent_period_mapping = inset_parent_period_mapping

    @property
    def _source_grid_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the pfl_nwt model domain. Used to speed up interpolation
        of parent grid values onto pfl_nwt grid."""
        if self._source_mask is None:
            mask = np.zeros((self.parent.modelgrid.nrow,
                             self.parent.modelgrid.ncol), dtype=bool)
            if self.inset.parent_mask.shape == self.parent.modelgrid.xcellcenters.shape:
                mask = self.inset.parent_mask
            else:
                x, y = np.squeeze(self.inset.bbox.exterior.coords.xy)
                pi, pj = get_ij(self.parent.modelgrid, x, y)
                pad = 3
                i0, i1 = pi.min() - pad, pi.max() + pad
                j0, j1 = pj.min() - pad, pj.max() + pad
                mask[i0:i1, j0:j1] = True
            self._source_mask = mask
        return self._source_mask

    @property
    def interp_weights(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to pfl_nwt."""
        if self._interp_weights is None:
            source_xy, dest_xy = get_source_dest_model_xys(self.parent.modelgrid,
                                                           self.inset.modelgrid,
                                                           source_mask=self._source_grid_mask)
            self._interp_weights = interp_weights(source_xy, dest_xy)
        return self._interp_weights

    def regrid_from_parent(self, source_array,
                                 mask=None,
                                 method='linear'):
        """Interpolate values in source array onto
        the destination model grid, using SpatialReference instances
        attached to the source and destination models.

        Parameters
        ----------
        source_array : ndarray
            Values from source model to be interpolated to destination grid.
            1 or 2-D numpy array of same sizes as a
            layer of the source model.
        mask : ndarray (bool)
            1 or 2-D numpy array of same sizes as a
            layer of the source model. True values
            indicate cells to include in interpolation,
            False values indicate cells that will be
            dropped.
        method : str ('linear', 'nearest')
            Interpolation method.
        """
        if mask is not None:
            return regrid(source_array, self.parent.modelgrid, self.inset.modelgrid,
                          mask1=mask,
                          method=method)
        if method == 'linear':
            parent_values = source_array.flatten()[self._source_grid_mask.flatten()]
            regridded = interpolate(parent_values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = regrid(source_array, self.parent.modelgrid, self.inset.modelgrid,
                               method='nearest')
        regridded = np.reshape(regridded, (self.inset.modelgrid.nrow,
                                           self.inset.modelgrid.ncol))
        return regridded

    def get_parent_cells(self, side='top'):
        """
        Get i, j locations in parent model along boundary of pfl_nwt model.

        Parameters
        ----------
        pi0, pj0 : ints
            Parent cell coinciding with origin (0, 0) cell of pfl_nwt model
        pi1, pj1 : ints
            Parent cell coinciding with lower right corner of pfl_nwt model
            (location nrow, ncol)
        side : str
            Side of pfl_nwt model ('left', 'bottom', 'right', or 'top')

        Returns
        -------
        i, j : 1D arrays of ints
            i, j locations of parent cells along pfl_nwt model boundary
        """
        pi0, pj0 = self.pi0, self.pj0
        pi1, pj1 = self.pi1 + 1, self.pj1 + 1

        # Add a plus 1 because rounded to the nearest 10 for the rows and columns above.
        parent_height = pi1 - pi0  # +1
        parent_width = pj1 - pj0  # +1

        if side == 'top':
            return np.ones(parent_width, dtype=int) * pi0-1, \
                   np.arange(pj0, pj1)
        elif side == 'left':
            return np.arange(pi0, pi1), \
                   np.ones(parent_height, dtype=int) * pj0-1
        elif side == 'bottom':
            return np.ones(parent_width, dtype=int) * pi1-1, \
                   np.arange(pj0, pj1)
        elif side == 'right':
            return np.arange(pi0, pi1), \
                   np.ones(parent_height, dtype=int) * pj1-1

    def get_inset_cells(self, i, j,
                        side='top'):
        """
        Get boundary cells in pfl_nwt model corresponding to parent cells i, j.

        Parameters
        ----------
        i, j : int
            Cell in parent model connected to boundary of pfl_nwt model.
        pi0, pj0 : int
            Parent cell coinciding with origin (0, 0) cell of pfl_nwt model
        refinement : int
            Refinement level (i.e. 10 if there are 10 pfl_nwt cells for every parent cell).
        side : str
            Side of pfl_nwt model ('left', 'bottom', 'right', or 'top')

        Returns
        -------
        i, j : 1D arrays of ints
            Corresponding i, j locations along boundary of pfl_nwt grid
        """
        pi0, pj0 = self.pi0, self.pj0
        refinement = self.refinement

        if side == 'top':
            ij0 = (j - pj0) * refinement
            ij1 = np.min([ij0 + refinement,
                          self.inset.ncol])
            ij = np.arange(ij0, ij1)
            ii = np.array([0] * len(ij))
        elif side == 'left':
            ii0 = (i - pi0) * refinement
            ii1 = np.min([ii0 + refinement,
                          self.inset.nrow])
            ii = np.arange(ii0, ii1)
            ij = np.array([0] * len(ii))
        elif side == 'right':
            ii0 = (i - pi0) * refinement
            ii1 = np.min([ii0 + refinement,
                          self.inset.nrow])
            ii = np.arange(ii0, ii1)
            ij0 = np.min([(j - pj0 + 1) * refinement,
                          self.inset.ncol]) - 1
            ij = np.array([ij0] * len(ii))
        elif side == 'bottom':
            # Needed to adjust
            ij0 = (j - pj0) * refinement
            ij1 = np.min([ij0 + refinement,
                          self.inset.ncol + 1])
            ij = np.arange(ij0, ij1)
            ii0 = np.min([(i - pi0 + 1) * refinement,
                          self.inset.nrow]) - 1
            ii = np.array([ii0] * len(ij))
        return ii, ij

    def get_inset_boundary_flux_side(self, side):
        """
        Compute fluxes between parent and pfl_nwt models on a side;
        assuming that flux to among connecting child cells
        is horizontally uniform within a parent cell, but can vary
        vertically based on transmissivity.

        Parameters
        ----------
        side : str
            Side of pfl_nwt model (top, bottom, right, left)

        Returns
        -------
        df : DataFrame
            Columns k, i, j, Q; describing locations and boundary flux
            quantities for the pfl_nwt model side.
        """
        parent_cells = self.get_parent_cells(side=side)
        nlay_inset = self.inset.nlay

        Qside = []  # boundary fluxes
        kside = []  # k locations of boundary fluxes
        iside = []  # i locations ...
        jside = []
        for i, j in zip(*parent_cells):

            # get the pfl_nwt model cells
            ii, jj = self.get_inset_cells(i, j, side=side)

            # parent model flow and layer bottoms
            Q_parent = self.cbc[self.flow_component[side]][:, i, j] * self.flow_sign[side]
            botm_parent = self.parent.dis.botm.array[:, i, j]

            # pfl_nwt model bottoms, and K
            # assume equal transmissivity for child cell to a parent cell, within each layer
            # (use average child cell k and thickness for each layer)
            # These are the layer bottoms for the pfl_nwt
            botm_inset = self.inset.dis.botm.array[:, ii, jj].mean(axis=1, dtype=np.float64)
            # These are the ks from the pfl_nwt model
            kh_inset = self.inset.upw.hk.array[:, ii, jj].mean(axis=1, dtype=np.float64)

            # determine aquifer top
            water_table_parent = self.wt[i, j]
            top_parent = self.parent.dis.top.array[i, j]

            Q_inset_ij = distribute_parent_fluxes_to_inset(Q_parent=Q_parent,
                                                           botm_parent=botm_parent,
                                                           top_parent=top_parent,
                                                           botm_inset=botm_inset,
                                                           kh_inset=kh_inset,
                                                           water_table_parent=water_table_parent)
            assert len(ii) == self.refinement # no partial parent cells
            Qside += np.array(list(Q_inset_ij / self.refinement) * len(ii)).ravel().tolist()
            kside += list(range(0, nlay_inset)) * len(ii)
            iside += sorted(ii.tolist() * nlay_inset)
            jside += sorted(jj.tolist() * nlay_inset)

        # check that fluxes for the side match the parent
        Qparent_side = self.get_parent_boundary_fluxes_side(parent_cells[0],
                                                            parent_cells[1],
                                                            side=side)
        tol = 0.01
        assert np.abs(Qparent_side.sum() - np.sum(Qside)) < tol

        return pd.DataFrame({'k': kside,
                             'i': iside,
                             'j': jside,
                             'flux': Qside})

    def get_inset_boundary_fluxes(self, kstpkper=(0, 0)):
        """Get all boundary fluxes for a stress period.

        Parameters
        ----------
        kstpkper : tuple or list of tuples
            zero-based (timestep, stress period)

        Returns
        -------
        df : DataFrame of all pfl_nwt model boundary fluxes
            With columns k, i, j, flux, and per
        """
        assert 'UPW' in self.inset.get_package_list(), "need UPW package to get boundary fluxes"
        assert 'DIS' in self.inset.get_package_list(), "need DIS package to get boundary fluxes"

        if not isinstance(kstpkper, list):
            kstpkper = [kstpkper]
        t0 = time.time()
        print('getting boundary fluxes from {}...'.format(self.cpth))
        dfs = []
        for kp in kstpkper:
            hdsobj = bf.HeadFile(self.hpth)
            hds = hdsobj.get_data(kstpkper=kp)
            hdry = -9999
            self.wt = get_water_table(hds, nodata=hdry)

            self.read_parent_cbc_per(kstpkper=kp)

            for side in ['top', 'left', 'bottom', 'right']:
                print(side)
                Qside = self.get_inset_boundary_flux_side(side)
                Qside['per'] = kp[1]
                dfs.append(Qside)

        df = pd.concat(dfs)

        # check that Qnet out of the parent model equals
        # the derived fluxes on the pfl_nwt side
        tol = 0.01
        for per, dfp in df.groupby('per'):

            Qnet_parent = self.get_parent_boundary_net_flux(kstpkper=per)
            Qnet_inset = dfp.flux.sum()
            assert np.abs(Qnet_parent - Qnet_inset) < tol

        print("finished in {:.2f}s\n".format(time.time() - t0))
        return df

    def read_parent_cbc_per(self, kstpkper=(0, 0)):
        cbbobj = bf.CellBudgetFile(self.cpth)
        text = {'FLOW RIGHT FACE': 'frf',
                'FLOW FRONT FACE': 'fff'}
        self.cbc = {}
        for fulltxt, shorttxt in text.items():
            self.cbc[shorttxt] = get_surface_bc_flux(cbbobj, fulltxt,
                                                     kstpkper=kstpkper, idx=0)

    def get_parent_boundary_fluxes_side(self, i, j, side, kstpkper=(0, 0)):
        """Get boundary fluxes at a sequence of i, j locations
        in the parent model, for a specified side of the pfl_nwt model,
        for a given stress period.

        Parameters
        ----------
        i : sequence of i locations
        j : sequence of j locations
        side : str
            left, right, top or bottom
        kstpkper : tuple
            (timestep, Stress Period)

        Returns
        -------
        Qside_parent : 2D array
            Boundary fluxes through parent cells, along side of pfl_nwt model.
            Signed with respect to pfl_nwt model (i.e., for flow through the
            left face of the parent cells, into the right side of the
            pfl_nwt model, the sign is positive (flow into the pfl_nwt model),
            even though MODFLOW fluxes are right-positive.
            Shape: (n parent layers, len(i, j))
        """
        if self.cbc is None:
            self.read_parent_cbc_per(kstpkper=kstpkper)
        Qside_parent = self.cbc[self.flow_component[side]][:, i, j] * self.flow_sign[side]
        #Qside_inset = self.get_inset_boundary_flux_side(side)

        return Qside_parent

    def get_parent_boundary_net_flux(self, kstpkper=(0, 0)):
        """

        Parameters
        ----------
        kstpkper : int, Stress Period

        Returns
        -------
        Qnet_parent : float
            Net flux from parent model, summed from parent cell by cell flow results.
        """
        Qnet_parent = 0
        for side, flow_sign in self.flow_sign.items():
            parent_cells = self.get_parent_cells(side=side)
            Qnet_parent += self.get_parent_boundary_fluxes_side(parent_cells[0],
                                                                parent_cells[1],
                                                                side=side,
                                                                kstpkper=kstpkper).sum()
        return Qnet_parent

    def compare_specified_flux_budgets(self, kstpkper=(0, 0), outfile=None):

        kstp, per = kstpkper
        from collections import defaultdict
        components = defaultdict(dict)
        # get pfl_nwt boundary fluxes from scratch, or attached wel package
        if 'WEL' not in self.inset.get_package_list():
            df = self.get_inset_boundary_fluxes(kstpkper=(0, kstpkper))
            components['Boundary flux']['pfl_nwt'] = df.flux.sum()
        else:
            spd = self.inset.wel.stress_period_data[per]
            rowsides = (spd['i'] == 0) | (spd['i'] == self.inset.nrow-1)
            # only count the corners onces
            colsides = ((spd['j'] == 0) | (spd['j'] == self.inset.ncol-1)) & \
                       (spd['i'] > 0) & \
                       (spd['i'] < self.inset.nrow-1)
            isboundary = rowsides | colsides
            components['Boundary flux (WEL)']['pfl_nwt'] = spd[isboundary]['flux'].sum()
            components['Boundary flux (WEL)']['parent'] = self.get_parent_boundary_net_flux(kstpkper=kstpkper)
            # (wells besides boundary flux wells)
            components['Pumping (WEL)']['pfl_nwt'] = spd[~isboundary]['flux'].sum()

        if 'WEL' in self.parent.get_package_list():
            spd = self.parent.wel.stress_period_data[per]
            in_inset = (spd['i'] >= self.pi0) & \
                       (spd['i'] <= self.pi1) & \
                       (spd['j'] >= self.pj0) & \
                       (spd['j'] <= self.pj1)
            components['Pumping (WEL)']['parent'] = spd[in_inset]['flux'].sum()

        # compare attached recharge packages
        r_parent = self.parent.rch.rech.array[per].sum(axis=0)
        r_parent_in_inset = r_parent[self.pi0:self.pi1 + 1,
                                     self.pj0:self.pj1 + 1]
        rsum_parent_in_inset = r_parent_in_inset.sum(axis=(0, 1)) * \
                               self.parent.dis.delr[0]**2
        rsum_inset = self.inset.rch.rech.array[per].sum(axis=(0, 1, 2)) * \
                     self.inset.dis.delr[0]**2

        components['Recharge']['parent'] = rsum_parent_in_inset
        components['Recharge']['pfl_nwt'] = rsum_inset

        for k, v in components.items():
            components[k]['rpd'] = 100 * v['pfl_nwt']/v['parent']
        if outfile is not None:
            with open(outfile, 'w') as dest:
                dest.write('component parent pfl_nwt rpd\n')
                for k, v in components.items():
                    dest.write('{} {parent} {inset} {rpd:.3f}\n'.format(k, **v))

        print('component parent pfl_nwt rpd')
        for k, v in components.items():
            print('{} {parent} {inset}'.format(k, **v))

    def get_inset_boundary_heads(self, for_external_files=True):

        # source data
        headfile = self.hpth
        vmin, vmax = -1e30, 1e30,
        check_source_files([headfile])
        hdsobj = bf.HeadFile(headfile) #, precision='single')
        all_kstpkper = hdsobj.get_kstpkper()

        # get the last timestep in each stress period if there are more than one
        #kstpkper = []
        #unique_kper = []
        #for (kstp, kper) in all_kstpkper:
        #    if kper not in unique_kper:
        #        kstpkper.append((kstp, kper))
        #        unique_kper.append(kper)
        last_steps = {kper: kstp for kstp, kper in all_kstpkper}

        #assert len(unique_kper) == len(set(self.copy_stress_periods)), \
        #"read {} from {},\nexpected stress periods: {}".format(kstpkper,
        #                                                       headfile,
        #                                                       sorted(list(set(self.copy_stress_periods)))
        #                                                       )

        # get active cells along model edge
        if self.pi_list is None and self.pj_list is None:
            k, i, j = self.inset.get_boundary_cells(exclude_inactive=True)
        else:
            ktmp =[]
            for clay in range(self.inset.nlay):
                ktmp += list(clay*np.ones(len(self.pi_list)).astype(int))
            itmp = self.inset.nlay * self.pi_list
            jtmp = self.inset.nlay * self.pj_list

            # get rid of cells that are inactive
            wh = np.where(self.inset.dis.idomain.array >0)
            activecells = set([(i,j,k) for i,j,k in zip(wh[0],wh[1],wh[2])])
            chdcells = set([(kk,ii,jj) for ii,jj,kk in zip(itmp,jtmp,ktmp)])
            active_chd_cells = list(set(chdcells).intersection(activecells))

            # unpack back to lists, then convert to numpy arrays
            k, i, j = zip(*active_chd_cells)
            k = np.array(k)
            i = np.array(i)
            j = np.array(j)
            # get heads from parent model
        # TODO: generalize head extraction from parent model using 3D interpolation

        dfs = []
        parent_periods = []
        for inset_per, parent_per in self.inset_parent_period_mapping.items():
            # skip getting data if parent period is already represented
            # (heads will be reused)
            if parent_per in parent_periods:
                continue
            else:
                parent_periods.append(parent_per)
            parent_kstpkper = last_steps[parent_per], parent_per
            hds = hdsobj.get_data(kstpkper=parent_kstpkper)

            regridded = np.zeros((self.inset.nlay, self.inset.nrow, self.inset.ncol))
            for dest_k, source_k in self.inset_parent_layer_mapping.items():

                # destination model layers copied from source model layers
                if source_k <= 0:
                    arr = hds[0]
                elif np.round(source_k, 4) in range(hds.shape[0]):
                    source_k = int(np.round(source_k, 4))
                    arr = hds[source_k]
                # destination model layers that are a weighted average
                # of consecutive source model layers
                else:
                    weight0 = source_k - np.floor(source_k)
                    source_k0 = int(np.floor(source_k))
                    # first layer in the average can't be negative
                    source_k0 = 0 if source_k0 < 0 else source_k0
                    source_k1 = int(np.ceil(source_k))
                    arr = weighted_average_between_layers(hds[source_k0],
                                                          hds[source_k1],
                                                          weight0=weight0)
                # interpolate from source model using source model grid
                # exclude invalid values in interpolation from parent model
                mask = self._source_grid_mask & (arr > vmin) & (arr < vmax)

                regriddedk = self.regrid_from_parent(arr, mask=mask, method='linear')

                assert regriddedk.shape == self.inset.modelgrid.shape[1:]
                regridded[dest_k] = regriddedk * self.length_unit_conversion

            # drop heads in dry cells, but only in mf6
            # too much trouble with interpolated heads in mf2005
            head = regridded[k, i, j]
            if self.inset.version == 'mf6':
                wet = head > self.inset.dis.botm.array[k, i, j]
            else:
                wet = np.ones(len(head)).astype(bool)

            # make a DataFrame of regridded heads at perimeter cell locations
            df = pd.DataFrame({'per': inset_per,
                               'k': k[wet],
                               'i': i[wet],
                               'j': j[wet],
                               'head': head[wet]
                               })
            dfs.append(df)
        df = pd.concat(dfs)

        # convert to one-based and comment out header if df will be written straight to external file
        if for_external_files:
            df.rename(columns={'k': '#k'}, inplace=True)
            df['#k'] += 1
            df['i'] += 1
            df['j'] += 1
        return df


def distribute_parent_fluxes_to_inset(Q_parent, botm_parent, top_parent,
                                      botm_inset, kh_inset, water_table_parent,
                                      phiramp=0.05):
    """Redistributes a vertical column of parent model fluxes at a single
    location i, j in the parent model, to the corresponding layers in the
    pfl_nwt model, based on pfl_nwt model layer transmissivities, accounting for the
    position of the water table in the parent model.

    Parameters
    ----------
    Q_parent : 1D array,
        Vertical column of horizontal fluxes through a cell face
        at a location at a location i, j in the parent model.
        (Length is n parent layers)
    botm_parent : 1D array
        Layer bottom elevations at location i, j in parent model.
        (Length is n parent layers)
    top_parent : float
        Top elevation of parent model at location i, j
    botm_inset : 1D array
        Mean elevation of pfl_nwt cells along the boundary face, by layer.
        (Length is n pfl_nwt layers)
    kh_inset : 1D array
        Mean hydraulic conductivity of pfl_nwt cells along the boundary face, by layer.
        (Length is n pfl_nwt layers)
    water_table_parent : float
        Water table elevation in parent model.
    phiramp : float
        Fluxes in layers with saturated thickness fraction (sat thickness/total cell thickness)
        below this threshold will be assigned to the next underlying layer with a
        saturated thickness fraction above this threshold. (default 0.01)

    Returns
    -------
    Q_inset : 1D array
        Vertical column of horizontal fluxes through each layer of the pfl_nwt
        model, for the group of pfl_nwt model cells corresponding to parent
        location i, j (represents the sum of horizontal flux through the
        boundary face of the pfl_nwt model cells in each layer).
        (Length is n pfl_nwt layers).

    """

    # check dimensions
    txt = "Length of {0} {1} is {2}; " \
          "length of {0} botm elevation is {3}"
    assert len(Q_parent) == len(botm_parent), \
        txt.format('parent', 'fluxes', len(Q_parent), len(botm_parent))
    assert len(botm_inset) == len(kh_inset), \
        txt.format('pfl_nwt', 'kh_inset', len(kh_inset), len(botm_inset))

    # rename variables
    Q1 = Q_parent
    botm1 = botm_parent
    botm2 = botm_inset
    kh2 = kh_inset
    aqtop = water_table_parent if water_table_parent < top_parent \
        else top_parent  # top of the aquifer

    # Replace nans with 0s bc these are where cells are dry
    Q1[np.isnan(Q1)] = 0
    # In parent model cells with sat thickness fraction less than phiramp,
    # Distribute flux to next layer with sat thickness frac > phiramp
    b_parent = -np.diff(np.array([top_parent] + list(botm_parent)))
    sthick = aqtop - botm_parent
    confined = (sthick - b_parent) > 0
    sthick[confined] = b_parent[confined]
    stfrac = sthick/b_parent
    q_excess = 0.
    for k, stfk in enumerate(stfrac):
        if stfk < phiramp:
            q_excess += Q1[k]
            Q1[k] = 0.
            continue
        Q1[k] = Q1[k] + q_excess
        q_excess = 0.

    kh2 = np.append(kh2, [0])  # for any layers below bottom

    nlay1 = len(botm1)
    nlay2 = len(botm2)

    # all botms in both models, in reverse order (layer-positive)
    allbotms = np.sort(np.unique(np.hstack([botm1, botm2])))[::-1]

    # list layer numbers in parent and child model;
    # for each flux connection between them
    k1 = 0
    k2 = 0
    l1 = []  # layer in parent model
    l2 = []  # layer in child model
    for botm in allbotms:
        l1.append(k1)
        l2.append(k2)
        if botm in botm1:
            k1 += 1
        if botm in botm2:
            k2 += 1

    l1 = np.array(l1) # parent cell connections for pfl_nwt cells
    l2 = np.array(l2) # pfl_nwt cell connections for parent cells

    # if bottom of pfl_nwt hangs below bottom of parent;
    # last layer will >= nlay. Assign T=0 to these intervals.
    l2[l2 >= nlay2] = nlay2
    # include any part of parent model hanging below pfl_nwt
    # with the lowest layer in the transmissivity calculation
    l1[l1 >= nlay1] = nlay1 - 1

    # thickness of all layer connections between
    # parent and child models
    # (assign 0 for connections above the water table)
    b = np.diff(sorted([aqtop] + allbotms.tolist()), axis=0)[::-1]
    b[allbotms > aqtop] = 0

    # get transmissivities
    T2 = kh2[l2] * b
    T1 = []
    for k in range(nlay1):
        T1.append(np.sum(T2[l1 == k]))

    # get transmissivity fractions (weights)
    tfrac = []
    # for each parent/pfl_nwt connection
    for i2, i1 in enumerate(l1):
        # compute transmissivity fraction  (of parent cell)
        itfrac = T2[i2] / T1[i1] if T2[i2] > 0 else 0
        tfrac.append(itfrac)
    tfrac = np.array(tfrac)

    # assign incoming flux to each pfl_nwt/parent connection
    # multiply by weight
    Qs = Q1[l1] * tfrac

    # Where nan, make 0
    Qs[np.isnan(Qs)] = 0
    np.savetxt('../qs.dat', Qs)
    # sum fluxes by pfl_nwt model layer
    Q_inset = []
    for k in range(nlay2):
        Q_inset.append(Qs[l2 == k].sum())

    # check that total flux through column of cells
    # matches for pfl_nwt layers and parent layers
    assert np.abs(np.abs(np.sum(Q_parent)) - np.abs(np.sum(Q_inset))) < 1e-3
    return np.array(Q_inset)


def get_kij_from_node3d(node3d, nrow, ncol):
    """For a consecutive cell number in row-major order
    (row, column, layer), get the zero-based row, column position.
    """
    node2d = node3d % (nrow * ncol)
    k = node3d // (nrow * ncol)
    i = node2d // ncol
    j = node2d % ncol
    return k, i, j


def get_intercell_connections(ia, ja, flowja):
    print('Making DataFrame of intercell connections...')
    ta = time.time()
    all_n = []
    m = []
    q = []
    for n in range(len(ia)-1):
        for ipos in range(ia[n] + 1, ia[n+1]):
            all_n.append(n)
            m.append(ja[ipos])  # m is the cell that n connects to
            q.append(flowja[ipos])  # flow across the connection
    df = pd.DataFrame({'n': all_n, 'm': m, 'q': q})
    et = time.time() - ta
    print("finished in {:.2f}s\n".format(et))
    return df


def get_flowja_face(cell_budget_file, binary_grid_file, kstpkper=(0, 0)):
    """Get FLOW-JA-FACE (cell by cell flows) from MODFLOW 6 budget
    output and binary grid file.
    TODO: need test for extracted flowja fluxes
    """
    cbb = cell_budget_file
    if binary_grid_file is None:
        print("Couldn't get FLOW-JA-FACE, need binary grid file for connection information.")
        return
    bgf = MfGrdFile(binary_grid_file)
    # IA array maps cell number to connection number
    # (one-based index number of first connection at each cell)?
    # taking the forward difference then yields nconnections per cell
    ia = bgf._datadict['IA'] - 1
    # Connections in the JA array correspond directly with the
    # FLOW-JA-FACE record that is written to the budget file.
    ja = bgf._datadict['JA'] - 1  # cell connections
    flowja = cbb.get_data(text='FLOW-JA-FACE', kstpkper=kstpkper)[0][0, 0, :]
    df = get_intercell_connections(ia, ja, flowja)
    cols = ['n', 'm', 'q']

    # get the k, i, j locations for plotting the connections
    if isinstance(bgf.mg, StructuredGrid):
        nlay, nrow, ncol = bgf.mg.nlay, bgf.mg.nrow, bgf.mg.ncol
        k, i, j = get_kij_from_node3d(df['n'].values, nrow, ncol)
        df['kn'], df['in'], df['jn'] = k, i, j
        k, i, j = get_kij_from_node3d(df['m'].values, nrow, ncol)
        df['km'], df['im'], df['jm'] = k, i, j
        df.reset_index()
        cols += ['kn', 'in', 'jn', 'km', 'im', 'jm']
    return df[cols].copy()

class TmrNew:
    """
    Class for general telescopic mesh refinement of a MODFLOW model. Head or
    flux fields from parent model are interpolated to boundary cells of
    inset model, which may be in any configuration (jagged, rotated, etc.).

    Parameters
    ----------
    parent_model : flopy model instance instance of parent model
        Must have a valid, attached ModelGrid (modelgrid) attribute.
    inset_model : flopy model instance instance of inset model
        Must have a valid, attached ModelGrid (modelgrid) attribute.
    parent_head_file : filepath
        MODFLOW binary head output
    parent_cell_budget_file : filepath
        MODFLOW binary cell budget output
    parent_binary_grid_file : filepath
        MODFLOW 6 binary grid file (*.grb)
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
                self._idomain = self.inset.dis.idomain.array
            else:
                self._idomain = self.inset.bas6.ibound.array
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
    def interp_weights_heads(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to pfl_nwt."""
        if self._interp_weights_heads is None:

            # x, y, z locations of parent model head values
            px, py, pz = self.parent_xyzcellcenters

            # x, y, z locations of inset model boundary cells
            x, y, z = self.inset_boundary_cells[['x', 'y', 'z']].T.values

            self._interp_weights_heads = interp_weights((px, py, pz), (x, y, z), d=3)
            assert not np.any(np.isnan(self._interp_weights_heads[1]))
        return self._interp_weights_heads

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

            for fluxdir in ['left', 'right', 'top', 'bottom']:
                x,y,z = self.inset_boundary_cell_faces.loc[
                    self.inset_boundary_cell_faces.cellface==fluxdir][['xface','yface','zface']].T.values
                if fluxdir in ['top','bottom']:
                    # these are the i-direction fluxes
                    self._interp_weights_flux[fluxdir] = interp_weights((ipx, ipy, ipz), (x, y, z), d=3)
                if fluxdir in ['left','right']:
                    # these are the i-direction fluxes
                    self._interp_weights_flux[fluxdir] = interp_weights((jpx, jpy, jpz), (x, y, z), d=3)


                assert not np.any(np.isnan(self._interp_weights_flux[fluxdir][1]))
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
        (``TmrNew._inset_max_active_area``) within the parentmodel grid.
        In other words, all parent cells containing one or inset
        model cell centers within ``TmrNew._inset_max_active_area`` (ones).
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
            if self.inset.parent_mask.shape == self.parent.modelgrid.xcellcenters.shape:
                mask = self.inset.parent_mask
            else:
                x, y = np.squeeze(self.inset.bbox.exterior.coords.xy)
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
            df = gp.sjoin(grid_df, perimeter, op='intersects', how='inner')
            # add layers
            dfs = []
            for k in range(self.inset.nlay):
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
            df['idomain'] = self.inset.dis.idomain.array[df.k, df.i, df.j]
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
        outshp = Path(self.inset._tables_path, 'boundary_cells.shp')
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

            # get the perimeter cells and calculate the weights
            _ = self.interp_weights_heads

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
                heads = self.interpolate_values(parent_heads, method='linear')
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

            #
            # Handle the geometry issues for the inset
            #
            # need to locate edge faces (x,y,z) based on which faces is out (e.g. left, right, up, down)

            # make a dataframe to store these
            self.inset_boundary_cell_faces = self.inset_boundary_cells.copy()
            # renaming columns to be clear now x,y,z, is for the outer cell face
            self.inset_boundary_cell_faces.rename(columns={'x':'xface','y':'yface','z':'zface'}, inplace=True)
            # convert x,y coordinates to model coords from world coords
            self.inset_boundary_cell_faces.xface, self.inset_boundary_cell_faces.yface = \
                    self.inset.modelgrid.get_local_coords(self.inset_boundary_cell_faces.xface, self.inset_boundary_cell_faces.yface)
            # calculate the thickness to later get the area
            self.inset_boundary_cell_faces['thickness'] = self.inset_boundary_cell_faces.top - self.inset_boundary_cell_faces.botm
            # pre-seed the area as thickness to later mult by width
            self.inset_boundary_cell_faces['face_area'] = self.inset_boundary_cell_faces['thickness'].values
            # placeholder for interpolated values
            self.inset_boundary_cell_faces['q_interp'] = np.nan
            # placeholder for flux to well package
            self.inset_boundary_cell_faces['Q'] = np.nan

            # make a grid of the spacings
            delr_gridi, delc_gridi = np.meshgrid(self.inset.modelgrid.delr, self.inset.modelgrid.delc)

            for cn in self.inset_boundary_cell_faces.cellface.unique():
                curri = self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn].i
                currj = self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn].j
                curr_delc = delc_gridi[curri, currj]
                curr_delr = delr_gridi[curri, currj]
                if cn == 'top':
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'yface'] += curr_delc/2
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delr
                elif cn == 'bottom':
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'yface'] -= curr_delc/2
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delr
                if cn == 'right':
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'xface'] += curr_delr/2
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delc
                elif cn == 'left':
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'xface'] -= curr_delr/2
                    self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface==cn, 'face_area'] *= curr_delc

            #
            # Now handle the geometry issues for the parent
            #
            # first thicknesses (at cell centers)
            parent_thick = -np.diff(self.parent.modelgrid.top_botm, axis=0)
            # TODO: refactor to use updated modelgrid object sat thickness calcs

            # make matrices of the row and column spacings
            # NB --> trying to preserve the always seemingly
            # backwards delr/delc definitions
            # also note - for now, taking average thickness at a connected face
            # TODO: confirm thickness averaging is a valid approach
            delr_gridp, delc_gridp = np.meshgrid(self.parent.modelgrid.delr,
                                                self.parent.modelgrid.delc)

            nlay, nrow, ncol = self.parent.modelgrid.shape

            parent_iface_areas = np.tile(delc_gridp[:-1,:], (nlay,1,1)) * \
                                    ((parent_thick[:,:-1,:]+parent_thick[:,1:,:])/2)
            parent_jface_areas = np.tile(delr_gridp[:,:-1], (nlay,1,1)) * \
                                    ((parent_thick[:,:,:-1]+parent_thick[:,:,1:])/2)

            # TODO: implement vertical fluxes
            '''
            parent_vface_areas  = np.tile(delc_grid, (nlay,1,1)) * \
                                    np.tile(delr_grid, (nlay,1,1))
            '''
            # need XYZ locations of the center of each face for
            # iface and jface edges (faces)
            # NB edges are returned in model coordinates
            xloc_edge, yloc_edge = self.parent.modelgrid.xyedges

            # throw out the left and top edges, respectively
            xloc_edge=xloc_edge[1:]
            yloc_edge=yloc_edge[1:]
            # tile out to full dimensions of the grid
            xloc_edge = np.tile(np.atleast_2d(xloc_edge),(nlay+2,nrow,1))
            yloc_edge = np.tile(np.atleast_2d(yloc_edge).T,(nlay+2,1,ncol))

            # need XYZ locations of the center of each cell
            # iface and jface centroids
            xloc_center, yloc_center = self.parent.modelgrid.xycenters

            # tile out to full dimensions of the grid
            xloc_center = np.tile(np.atleast_2d(xloc_center),(nlay+2,nrow,1))
            yloc_center = np.tile(np.atleast_2d(yloc_center).T,(nlay+2,1,ncol))

            # get the vertical centroids initially at cell centroids
            zloc = (self.parent.modelgrid.top_botm[:-1,:,:] +
                self.parent.modelgrid.top_botm[1:,:,:] ) / 2

            # pad in the vertical above and below the model
            zpadtop = np.expand_dims(self.parent.modelgrid.top_botm[0,:,:] + parent_thick[0], axis=0)
            zpadbotm = np.expand_dims(self.parent.modelgrid.top_botm[-1,:,:] - parent_thick[-1], axis=0)
            zloc=np.vstack([zpadtop,zloc,zpadbotm])


            # for iface, all cols, nrow-1 rows
            self.x_iface_parent = xloc_center[:,:-1,:].ravel()
            self.y_iface_parent = yloc_edge[:,:-1,:].ravel()
            # need to calculate the average z location along rows
            self.z_iface_parent = ((zloc[:,:-1,:]+zloc[:,1:,:]) / 2).ravel()

            # for jface, all rows, ncol-1 cols
            self.x_jface_parent = xloc_edge[:,:,:-1].ravel()
            self.y_jface_parent = yloc_center[:,:,:-1].ravel()
            # need to calculate the average z location along columns
            self.z_jface_parent = ((zloc[:,:,:-1]+zloc[:,:,1:]) / 2).ravel()

            # get the perimeter cells and calculate the weights
            _ = self.interp_weights_flux


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

                if self.parent.version == 'mf6':
                    df = get_flowja_face(fileobj,
                                         binary_grid_file=self.parent_binary_grid_file,
                                         kstpkper=parent_kstpkper)
                    if df is None:
                        raise ValueError('No fluxes returned by get_flowja_face')

                    # TODO: flux BCs
                    # subset df to boundary cells  -- not possible a priori
                    # get x and y direction fluxes separately -- DONE
                    # do same for vertical fluxes -- DONE
                    # * normalize by cell face area to make specific discharge -- DONE
                    # * use meshgrid to locate all the cell face locations in the parent (hello xyedges from grid!) -- DONE
                    # * need to set up inset xyz locations to interpolate to -- DONE
                    # * branch the geometry stuff above for MF6 vs. MF2005 parent (does it matter for modelgrid object??)
                    # * interpolate using meshgrid-derived lox and arrays of fluxes to inset correct faces  -- DONE
                    # * consider correct face interpolation weights precalculation -- DONE
                    # * multiply by inset face area -- DONE
                    # * ---- verify direciton of q coming from CBC file (e.g. always m --> n???) -- yes, it's n-centric. e.g. + is into n
                    # * verify the interpolation scheme - getting NaNs
                    # * verify that flipping sign of q_interp below is correct (e.g. only flip left and top?)
                    # * verify that all the xy locating works with rotated grid (!) -- DONE (working only in model coords)
                    # for MF-2005 case, would slice arrays returned by flopy binary utility to boundary cells
                    #    (so that mf6 and mf2005 come out the same)
                    # *  refactor to use updated modelgrid object sat thickness calcs

                    #
                    # TODO: implement vertical fluxes
                    # Get the vertical fluxes
                    '''if 'kn' in df.columns and np.any(df['kn'] < df['km']):
                        vflux = df.loc[(df['kn'] < df['km'])]
                        vflux_array = np.zeros((vflux['km'].max(), nrow, ncol))
                        vflux_array[vflux['kn'].values,
                                    vflux['in'].values,
                                    vflux['jn'].values] = vflux.q.values
                    '''
                    # get modelgrid row-wise (i-direction) fluxes
                    if 'in' in df.columns and np.any(df['in'] < df['im']):
                        iflux = df.loc[(df['in'] < df['im'])]
                        iflux_array = np.zeros((nlay, nrow-1, ncol))
                        iflux_array[iflux['kn'].values,
                                    iflux['in'].values,
                                    iflux['jn'].values] = iflux.q.values

                    # get modelgrid column-wise (j-direction) fluxes
                    if 'jn' in df.columns and np.any(df['jn'] < df['jm']):
                        jflux = df.loc[(df['jn'] < df['jm'])]
                        jflux_array = np.zeros((nlay, nrow, ncol-1))
                        jflux_array[jflux['kn'].values,
                                    jflux['in'].values,
                                    jflux['jn'].values] = jflux.q.values


                    # divide the flux by the area to find specific discharge along faces
                    # NB --> padding on the top and left top ensure zeros surround
                    q_iface = (iflux_array / parent_iface_areas)
                    q_jface = (jflux_array / parent_jface_areas)

                else:
                    raise NotImplementedError('MODFLOW-2005 fluxes not yet supported')
                    # TODO: implement MF2005
                    #  *create i, j, and v face xyzq vectors as with MF6 above
                    #   x_iface, y_iface, z_iface, q_iface .... etc.

                # pad the two parent flux arrays on the top and bottom
                # so that inset cells above and below the top/bottom cell centers
                # will be within the interpolation space
                # (parent x, y, z locations already contain this pad - see zloc above)
                q_iface = np.pad(q_iface, pad_width=1, mode='edge')[:, 1:-1, 1:-1].ravel()
                q_jface = np.pad(q_jface, pad_width=1, mode='edge')[:, 1:-1, 1:-1].ravel()

                # interpolate q at the four different face orientations (e.g. fluxdir)
                for fluxdir in ['top','bottom','left','right']:
                    if fluxdir in ['top','bottom']:
                        self.inset_boundary_cell_faces.loc[ self.inset_boundary_cell_faces.cellface==fluxdir, 'q_interp'] = \
                            self.interpolate_flux_values(q_iface, fluxdir)
                    if fluxdir in ['left','right']:
                        self.inset_boundary_cell_faces.loc[ self.inset_boundary_cell_faces.cellface==fluxdir, 'q_interp'] = \
                            self.interpolate_flux_values(q_jface, fluxdir)

                # flip the sign for flux counter to the CBB convention directions of right and bottom
                self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface=='left', 'q_interp'] -= 1
                self.inset_boundary_cell_faces.loc[self.inset_boundary_cell_faces.cellface=='top', 'q_interp'] -= 1

                # convert specific discharge in inset cells to Q
                self.inset_boundary_cell_faces['q'] = \
                    self.inset_boundary_cell_faces['q_interp'] * self.inset_boundary_cell_faces['face_area']


                # make a DataFrame of interpolated heads at perimeter cell locations
                df = self.inset_boundary_cell_faces[['k','i','j','idomain','q']].copy()
                df['per'] = inset_per

                # boundary heads must be greater than the cell bottom
                # and idomain > 0
                loc = (df['q'].abs() > 0) & (df['idomain'] > 0)
                df = df.loc[loc]
                dfs.append(df)
                print("took {:.2f}s".format(time.time() - t1))

            df = pd.concat(dfs)
            # drop duplicate cells (accounting for stress periods)
            # (that may have connections in the x and y directions,
            #  and therefore would be listed twice)
            df['cellid'] = list(zip(df.per, df.k, df.i, df.j))
            duplicates = df.duplicated(subset=['cellid'])
            df = df.loc[~duplicates, ['k', 'i', 'j', 'per', 'q']]
            print("getting perimeter fluxes took {:.2f}s\n".format(time.time() - t0))

        # convert to one-based and comment out header if df will be written straight to external file
        if for_external_files:
            df.rename(columns={'k': '#k'}, inplace=True)
            df['#k'] += 1
            df['i'] += 1
            df['j'] += 1
        return df

    def interpolate_values(self, source_array, method='linear'):
        """Interpolate values in source array onto
        the destination model grid, using modelgrid instances
        attached to the source and destination models.

        Parameters
        ----------
        source_array : ndarray
            Values from source model to be interpolated to destination grid.
            3D numpy array of same shape as the source model.
        method : str ('linear', 'nearest')
            Interpolation method.

        Returns
        -------
        interpolated : ndarray
            3D array of interpolated values at the inset model grid locations.
        """
        parent_values = source_array.flatten()[self._source_grid_mask.flatten()]
        if method == 'linear':
            interpolated = interpolate(parent_values, *self.interp_weights_heads,
                                       fill_value=None)
        elif method == 'nearest':
            # x, y, z locations of parent model head values
            px, py, pz = self.parent_xyzcellcenters
            # x, y, z locations of inset model boundary cells
            x, y, z = self.inset_boundary_cells[['x', 'y', 'z']].T.values
            interpolated = griddata((px, py, pz), parent_values,
                                    (x, y, z), method=method)
        return interpolated

    def interpolate_flux_values(self, source_array, fluxdir, method='linear'):
        """Interpolate values in source array onto
        the destination model grid, using modelgrid instances
        attached to the source and destination models.

        Parameters
        ----------
        source_array : 1d-array
            Flux values from parent model to be interpolated to destination grid.
            1D numpy array of same shape as the Tmr properties of parent xyz
        fluxdir: str ('top','bottom','left','right')
            inset face at which flux is applied
        method : str ('linear', 'nearest')
            Interpolation method.

        Returns
        -------
        interpolated : ndarray
            3D array of interpolated values at the inset model grid locations.
        """


        if method == 'linear':
            interpolated = interpolate(source_array, *self.interp_weights_flux[fluxdir],
                                       fill_value=None)

        elif method == 'nearest':
            # x, y, z locations of inset model boundary cells
            x, y, z = self.inset_boundary_cell_faces.loc[
                self.inset_boundary_cell_faces.cellface== fluxdir][['xface', 'yface', 'zface']].T.values
            if fluxdir in ['top','bottom']:
                # x, y, z locations of parent model head values
                px, py, pz = self.x_iface_parent, self.y_iface_parent, self.z_iface_parent
            elif fluxdir == ['left','right']:
                # x, y, z locations of parent model head values
                px, py, pz = self.x_jface_parent, self.y_jface_parent, self.z_jface_parent
                # x, y, z locations of inset model boundary cells

            interpolated = griddata((px, py, pz), source_array,
                                        (x, y, z), method=method)
        return interpolated
