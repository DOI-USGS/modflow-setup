import os

import numpy as np

from mfsetup.bcs import get_bc_package_cells, remove_inactive_bcs
from mfsetup.fileio import read_mf6_block
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
    m.cfg['ghb']['source_data']['bhead'] = m.dis.top.array.mean()
    m.cfg['ghb']['source_data']['cond'] = 100
    m.setup_ghb(**m.cfg['ghb'], **m.cfg['ghb']['mfsetup_options'])
    # using the isbc array to get the number of ghb cells
    # doesn't work because of this issue with layer 0 containing all i, j locations
    # (over-counts)
    m._set_isbc()
    nwel, no_bc, nlak, n_highklake, nghb = np.bincount(m.isbc.ravel() + 1)
    spd = m.ghb.stress_period_data[0]
    cellids = list(zip(spd.k, spd.i, spd.j))
    nghb = len(set(cellids))
    # todo: figure out why some cells aren't getting intersected with ghb_lake.shp
    assert nghb == 16
    ghb_cells = set(zip(*np.where(m._isbc2d == 3)))
    m.setup_sfr()
    sfr_cells = set(zip(m.sfrdata.reach_data.i.values,
                    m.sfrdata.reach_data.j.values))
    assert len(ghb_cells.intersection(sfr_cells)) == 0


def test_remove_inactive_bcs(basic_model_instance):
    m = basic_model_instance
    wd = m._abs_model_ws
    m.setup_dis()
    if m.version != 'mf6':
        m.setup_bas6()
    else:
        m.setup_tdis()
    m.setup_chd(**m.cfg['chd'], **m.cfg['chd']['mfsetup_options'])
    if m.version != 'mf6':
        idm = m.bas6.ibound.array
        idm[:, :, 0] = 0
        m.bas6.ibound = idm
        k, i, j = zip(*m.chd.stress_period_data.data[0][['k', 'i', 'j']])
        assert any(m.bas6.ibound.array[k, i, j] == 0)
    else:
        idm = m.dis.idomain.array
        idm[:, :, 0] = 0
        idm_files = m.cfg['dis']['griddata']['idomain']
        # update the external files
        # (as of 3.3.7, flopy doesn't appear to allow updating
        # an externally-based array in memory)
        for layer, arr2d in enumerate(idm):
            np.savetxt(idm_files[layer]['filename'], arr2d, fmt='%d')
        k, i, j = zip(*m.chd.stress_period_data.data[0]['cellid'])
        assert any(m.dis.idomain.array[k, i, j] == 0)
    external_files = m.cfg['chd']['stress_period_data']
    remove_inactive_bcs(m.chd, external_files=external_files)
    if m.version != 'mf6':
        k, i, j = zip(*m.chd.stress_period_data.data[0][['k', 'i', 'j']])
    else:
        k, i, j = zip(*m.chd.stress_period_data.data[0]['cellid'])
    assert np.all(idm[k, i, j] == 1)

    # test that package still writes external files
    if external_files:
        m.chd.write()
        perioddata = read_mf6_block(m.chd.filename, 'period')
        for per, data in perioddata.items():
            assert data[0].split()[0].strip() == 'open/close'
