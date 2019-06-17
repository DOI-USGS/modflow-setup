import os
import time
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from flopy.utils import binaryfile as bf
from flopy.utils.postprocessing import get_water_table
from .export import get_surface_bc_flux
from mfsetup.grid import get_ij
import numpy as np


class Tmr:
    """
    Class for basic telescopic mesh refinement of a MODFLOW model.
    Handles the case where the inset grid is a rectangle exactly aligned with
    the parent grid.

    Parameters
    ----------
    parent_model : flopy.modflow.Modflow instance of parent model
        Must have a valid, attached SpatialReference (sr) attribute.
    inset_model : flopy.modflow.Modflow instance of inset model
        Must have a valid, attached SpatialReference (sr) attribute.
        SpatialReference of inset and parent models is used to determine cell
        connections.
    parent_head_file : filepath
        MODFLOW binary head output
    parent_cell_budget_file : filepath
        MODFLOW binary cell budget output

    Notes
    -----
    Assumptions:
    * Uniform parent and inset grids, with equal delr and delc spacing.
    * Inset model upper right corner coincides with an upper right corner of a cell
      in the parent model
    * Inset cell spacing is a factor of the parent cell spacing
      (so that each inset cell is only connected horizontally to one parent cell).
    * Inset model row/col dimensions are multiples of parent model cells
      (no parent cells that only partially overlap the inset model)
    * Horizontally, fluxes are uniformly distributed to child cells within a parent cell. The
    * Vertically, fluxes are distributed based on transmissivity (sat. thickness x Kh) of
    inset model layers.
    * The inset model is fully penetrating. Total flux through each column of parent cells
    is equal to the total flux through the corresponding columns of connected inset model cells.
    The get_inset_boundary_flux_side verifies this with an assertion statement.

    """
    flow_component = {'top': 'fff', 'bottom': 'fff',
                      'left': 'frf', 'right': 'frf'}
    flow_sign = {'top': 1, 'bottom': -1,
                 'left': 1, 'right': -1}

    def __init__(self, parent_model, inset_model,
                 parent_head_file=None, parent_cell_budget_file=None):

        self.inset = inset_model
        self.parent = parent_model
        self.inset._set_parent_modelgrid()
        self.cbc = None

        if parent_head_file is None:
            self.hpth = os.path.join(self.parent.model_ws,
                                     '{}.{}'.format(self.parent.name,
                                                    self.parent.hext))
            assert os.path.exists(self.hpth), '{} not found.'.format(self.hpth)
        else:
            self.hpth = parent_head_file
        if parent_cell_budget_file is None:
            self.cpth = os.path.join(self.parent.model_ws,
                                     '{}.{}'.format(self.parent.name,
                                                    self.parent.cext))
            assert os.path.exists(self.cpth), '{} not found.'.format(self.cpth)
        else:
            self.cpth = parent_cell_budget_file

        # get bounding cells in parent model for inset model
        self.pi0, self.pj0 = get_ij(self.parent.modelgrid,
                                    self.inset.modelgrid.xcellcenters[0, 0],
                                    self.inset.modelgrid.ycellcenters[0, 0])
        self.pi1, self.pj1 = get_ij(self.parent.modelgrid,
                                    self.inset.modelgrid.xcellcenters[-1, -1],
                                    self.inset.modelgrid.ycellcenters[-1, -1])
        self.parent_nrow_in_inset = self.pi1 - self.pi0 + 1
        self.parent_ncol_in_inset = self.pj1 - self.pj0 + 1

        # check for an even number of inset cells per parent cell in x and y directions
        x_refinment = self.parent.modelgrid.delr[0] / self.inset.modelgrid.delr[0]
        y_refinment = self.parent.modelgrid.delc[0] / self.inset.modelgrid.delc[0]
        assert int(x_refinment) == x_refinment, "inset delr must be factor of parent delr"
        assert int(y_refinment) == y_refinment, "inset delc must be factor of parent delc"
        assert x_refinment == y_refinment, "grid must have same x and y discretization"
        self.refinement = int(x_refinment)

    def get_parent_cells(self, side='top'):
        """
        Get i, j locations in parent model along boundary of inset model.

        Parameters
        ----------
        pi0, pj0 : ints
            Parent cell coinciding with origin (0, 0) cell of inset model
        pi1, pj1 : ints
            Parent cell coinciding with lower right corner of inset model
            (location nrow, ncol)
        side : str
            Side of inset model ('left', 'bottom', 'right', or 'top')

        Returns
        -------
        i, j : 1D arrays of ints
            i, j locations of parent cells along inset model boundary
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
        Get boundary cells in inset model corresponding to parent cells i, j.

        Parameters
        ----------
        i, j : int
            Cell in parent model connected to boundary of inset model.
        pi0, pj0 : int
            Parent cell coinciding with origin (0, 0) cell of inset model
        refinement : int
            Refinement level (i.e. 10 if there are 10 inset cells for every parent cell).
        side : str
            Side of inset model ('left', 'bottom', 'right', or 'top')

        Returns
        -------
        i, j : 1D arrays of ints
            Corresponding i, j locations along boundary of inset grid
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
        Compute fluxes between parent and inset models on a side;
        assuming that flux to among connecting child cells
        is horizontally uniform within a parent cell, but can vary
        vertically based on transmissivity.

        Parameters
        ----------
        side : str
            Side of inset model (top, bottom, right, left)

        Returns
        -------
        df : DataFrame
            Columns k, i, j, Q; describing locations and boundary flux
            quantities for the inset model side.
        """
        parent_cells = self.get_parent_cells(side=side)
        nlay_inset = self.inset.nlay

        Qside = []  # boundary fluxes
        kside = []  # k locations of boundary fluxes
        iside = []  # i locations ...
        jside = []
        for i, j in zip(*parent_cells):

            # get the inset model cells
            ii, jj = self.get_inset_cells(i, j, side=side)

            # parent model flow and layer bottoms
            Q_parent = self.cbc[self.flow_component[side]][:, i, j] * self.flow_sign[side]
            botm_parent = self.parent.dis.botm.array[:, i, j]

            # inset model bottoms, and K
            # assume equal transmissivity for child cell to a parent cell, within each layer
            # (use average child cell k and thickness for each layer)
            # These are the layer bottoms for the inset
            botm_inset = self.inset.dis.botm.array[:, ii, jj].mean(axis=1, dtype=np.float64)
            # These are the ks from the inset model
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
        df : DataFrame of all inset model boundary fluxes
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
        # the derived fluxes on the inset side
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
        in the parent model, for a specified side of the inset model,
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
            Boundary fluxes through parent cells, along side of inset model.
            Signed with respect to inset model (i.e., for flow through the
            left face of the parent cells, into the right side of the
            inset model, the sign is positive (flow into the inset model),
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
        # get inset boundary fluxes from scratch, or attached wel package
        if 'WEL' not in self.inset.get_package_list():
            df = self.get_inset_boundary_fluxes(kstpkper=(0, kstpkper))
            components['Boundary flux']['inset'] = df.flux.sum()
        else:
            spd = self.inset.wel.stress_period_data[per]
            rowsides = (spd['i'] == 0) | (spd['i'] == self.inset.nrow-1)
            # only count the corners onces
            colsides = ((spd['j'] == 0) | (spd['j'] == self.inset.ncol-1)) & \
                       (spd['i'] > 0) & \
                       (spd['i'] < self.inset.nrow-1)
            isboundary = rowsides | colsides
            components['Boundary flux (WEL)']['inset'] = spd[isboundary]['flux'].sum()
            components['Boundary flux (WEL)']['parent'] = self.get_parent_boundary_net_flux(kstpkper=kstpkper)
            # (wells besides boundary flux wells)
            components['Pumping (WEL)']['inset'] = spd[~isboundary]['flux'].sum()

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
        components['Recharge']['inset'] = rsum_inset

        for k, v in components.items():
            components[k]['rpd'] = 100 * v['inset']/v['parent']
        if outfile is not None:
            with open(outfile, 'w') as dest:
                dest.write('component parent inset rpd\n')
                for k, v in components.items():
                    dest.write('{} {parent} {inset} {rpd:.3f}\n'.format(k, **v))

        print('component parent inset rpd')
        for k, v in components.items():
            print('{} {parent} {inset}'.format(k, **v))



def distribute_parent_fluxes_to_inset(Q_parent, botm_parent, top_parent,
                                      botm_inset, kh_inset, water_table_parent,
                                      phiramp=0.05):
    """Redistributes a vertical column of parent model fluxes at a single
    location i, j in the parent model, to the corresponding layers in the
    inset model, based on inset model layer transmissivities, accounting for the
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
        Mean elevation of inset cells along the boundary face, by layer.
        (Length is n inset layers)
    kh_inset : 1D array
        Mean hydraulic conductivity of inset cells along the boundary face, by layer.
        (Length is n inset layers)
    water_table_parent : float
        Water table elevation in parent model.
    phiramp : float
        Fluxes in layers with saturated thickness fraction (sat thickness/total cell thickness)
        below this threshold will be assigned to the next underlying layer with a
        saturated thickness fraction above this threshold. (default 0.01)

    Returns
    -------
    Q_inset : 1D array
        Vertical column of horizontal fluxes through each layer of the inset
        model, for the group of inset model cells corresponding to parent
        location i, j (represents the sum of horizontal flux through the
        boundary face of the inset model cells in each layer).
        (Length is n inset layers).

    """

    # check dimensions
    txt = "Length of {0} {1} is {2}; " \
          "length of {0} botm elevation is {3}"
    assert len(Q_parent) == len(botm_parent), \
        txt.format('parent', 'fluxes', len(Q_parent), len(botm_parent))
    assert len(botm_inset) == len(kh_inset), \
        txt.format('inset', 'kh_inset', len(kh_inset), len(botm_inset))

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

    l1 = np.array(l1) # parent cell connections for inset cells
    l2 = np.array(l2) # inset cell connections for parent cells

    # if bottom of inset hangs below bottom of parent;
    # last layer will >= nlay. Assign T=0 to these intervals.
    l2[l2 >= nlay2] = nlay2
    # include any part of parent model hanging below inset
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
    # for each parent/inset connection
    for i2, i1 in enumerate(l1):
        # compute transmissivity fraction  (of parent cell)
        itfrac = T2[i2] / T1[i1] if T2[i2] > 0 else 0
        tfrac.append(itfrac)
    tfrac = np.array(tfrac)

    # assign incoming flux to each inset/parent connection
    # multiply by weight
    Qs = Q1[l1] * tfrac

    # Where nan, make 0
    Qs[np.isnan(Qs)] = 0
    np.savetxt('../qs.dat', Qs)
    # sum fluxes by inset model layer
    Q_inset = []
    for k in range(nlay2):
        Q_inset.append(Qs[l2 == k].sum())

    # check that total flux through column of cells
    # matches for inset layers and parent layers
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
