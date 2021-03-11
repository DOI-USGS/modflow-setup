import os
import time

import flopy
import pandas as pd

fm = flopy.modflow
import numpy as np
from flopy.utils import binaryfile as bf
from flopy.utils.postprocessing import get_water_table

from mfsetup.discretization import weighted_average_between_layers
#from mfsetup.export import get_surface_bc_flux
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
            bhead = regridded[k, i, j]
            if self.inset.version == 'mf6':
                wet = bhead > self.inset.dis.botm.array[k, i, j]
            else:
                wet = np.ones(len(bhead)).astype(bool)

            # make a DataFrame of regridded heads at perimeter cell locations
            df = pd.DataFrame({'per': inset_per,
                               'k': k[wet],
                               'i': i[wet],
                               'j': j[wet],
                               'bhead': bhead[wet]
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


if __name__ == '__main__':
    parent_head_file = '../csls100_nwt/csls100.hds'
    parent_cell_budget_file = '../csls100_nwt/csls100.cbc'
    kstpkper = (0, 0)

    parent = fm.Modflow.load('csls100.nam', model_ws='../csls100_nwt/',
                             load_only=['dis'], check=False)
    inset = fm.Modflow.load('pfl10.nam', model_ws='../plainfield_inset/',
                            load_only=['dis', 'upw'], check=False)
    tmr = Tmr(parent, inset)
    df = tmr.get_inset_boundary_fluxes(kstpkper=kstpkper)
