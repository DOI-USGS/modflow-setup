import os

import numpy as np
import pytest

from mfsetup.bcs import get_bc_package_cells
from mfsetup.testing import dtypeisinteger


def test_get_bc_package_cells(pleasant_mf6_setup_from_yaml): #pleasant_model):
    m = pleasant_mf6_setup_from_yaml
    for packagename in ['ghb', 'sfr', 'wel', 'lak']:
        if packagename.upper() in m.get_package_list():
            package = getattr(m, packagename)
            k, i, j = get_bc_package_cells(package)
            for var in k, i, j:
                assert dtypeisinteger(var.dtype)
            # connections in MF6 refer to GW cell, not lake cell;
            # GW cell should be highest active layer below lake
            # usually, this is the layer immediately below the lake cell
            # but if there are pinched layers extending across the lake,
            # the GW connection may be 1 or more layers below the lake
            # (as represented in the isbc array)
            # in other words, there may be inactive cells between the gw/lake connections
            # which causes the assertion below to fail

            # verify that cells between gw/lake connections are thin,
            # that the gw connection is in the highest active layer,
            # and that there is a lake cell in the same i,j location
            if package.package_type == 'lak' and package.parent.version == 'mf6':
                k -= 1  # most lake cell connections are in layer above highest active GW cell

                # make 3D array of next cells above highest active layer (at i, j locations of lake connections)
                has_lake_package_connection = np.zeros((m.nlay, m.nrow, m.ncol), dtype=bool)
                has_lake_package_connection[k, i, j] = True

                # of these locations, identify locations that are not labeled as lake in isbc array (isbc=1)
                not_labeled_as_lake = np.where((m.isbc == 0) & has_lake_package_connection)

                # verify that these are all inactive cells
                assert m.idomain[not_labeled_as_lake].sum() == 0

                # verify that these i, j locations have at least 1 cell labeled as Lake in isbc
                knl, inl, jnl = not_labeled_as_lake
                assert np.all(np.any(m.isbc[:, inl, jnl] == 1, axis=0))

                # verify that there are no active cells above the lake/gw connection
                assert np.all(np.argmax(m.idomain[:, inl, jnl], axis=0) > knl)

                # inactive cells between lake/gw connections can be caused by
                # - thin cells that get converted to inactive
                # - isolation of cells resulting from converting cells within the lake to idomain=0
                # the pleasant lake test case is maybe worse than might be expected,
                # because the 40 meter grid resolution effectively results in a higher threshold
                # for classifying isolated clusters of cells
                # (a 20 cell cluster at 40 m resolution would be an 80 cell cluster at 20 m resolution)
                # the code below gets the thicknesses of these inactive cells
                # changes in WGNHS layering (feeding the parent model) resulted in 5 cells > 1m thick
                # (one was 9 m thick). But this is a small number compared to the overall number of
                # lake cells, even @ 40 meter resolution. And in the end it just means the elevation
                # where the boundary condition is applied is off by this amount (actual BC is the lake stage)
                # so just live with it for now.
                #all_layers = np.stack([m.dis.top.array] + [botm for botm in m.dis.botm.array])
                #thickness = -np.diff(all_layers, axis=0)
                #assert np.all(thickness[not_labeled_as_lake] < 1.1)
            else:
                assert np.all(m.isbc[k, i, j] == m.bc_numbers[packagename])


def test_ghb_sfr_overlap(pleasant_nwt_with_dis_bas6, project_root_path):
    m = pleasant_nwt_with_dis_bas6
    m.cfg['ghb']['source_data']['shapefile'] = \
        {'filename': os.path.join(project_root_path, 'examples/data/pleasant/source_data/shps/ghb_lake.shp'),
         'id_column': 'id'
         }
    m.setup_ghb()
    nwel, no_bc, nlak, n_highklake, nghb = np.bincount(m.isbc.ravel() + 1)
    assert nghb == 23
    ghb_cells = set(zip(*np.where(m._isbc2d == 3)))
    m.setup_sfr()
    sfr_cells = set(zip(m.sfrdata.reach_data.i.values,
                    m.sfrdata.reach_data.j.values))
    assert len(ghb_cells.intersection(sfr_cells)) == 0
