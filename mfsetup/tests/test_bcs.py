import numpy as np
import pytest
from mfsetup.bcs import get_bc_package_cells
from mfsetup.testing import dtypeisinteger


def test_get_bc_package_cells(pleasant_model):
    m = pleasant_model
    for packagename in ['ghb', 'sfr', 'wel', 'lak']:
        if packagename.upper() in m.get_package_list():
            package = getattr(m, packagename)
            k, i, j = get_bc_package_cells(package)
            for var in k, i, j:
                assert dtypeisinteger(var.dtype)
            assert np.all(m.isbc[k, i, j] == m.bc_numbers[packagename])


def test_ghb_sfr_overlap(pleasant_nwt_with_dis_bas6):
    m = pleasant_nwt_with_dis_bas6
    m.cfg['ghb']['source_data']['shapefile'] = \
        {'filename': '../../data/pleasant/source_data/shps/ghb_lake.shp',
         'id_column': 'id'
         }
    m.setup_ghb()
    nwel, no_bc, nlak, n_highklake, nghb = np.bincount(m.isbc.ravel() + 1)
    assert nghb == 16
    ghb_cells = set(zip(*np.where(m._isbc2d == 3)))
    m.setup_sfr()
    sfr_cells = set(zip(m.sfrdata.reach_data.i.values,
                    m.sfrdata.reach_data.j.values))
    assert len(ghb_cells.intersection(sfr_cells)) == 0
