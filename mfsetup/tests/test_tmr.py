from pathlib import Path

import numpy as np
import pytest
from flopy.utils import binaryfile as bf

from mfsetup.discretization import get_layer
from mfsetup.grid import get_ij
from mfsetup.tmr import Tmr, TmrNew


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['get_pleasant_mf6_with_dis',
                        'pleasant_nwt_with_dis_bas6'])
def pleasant_model(request,
                   get_pleasant_mf6_with_dis,
                   pleasant_nwt_with_dis_bas6):
    return {'get_pleasant_mf6_with_dis': get_pleasant_mf6_with_dis,
            'pleasant_nwt_with_dis_bas6': pleasant_nwt_with_dis_bas6}[request.param]


@pytest.fixture(scope='function')
def tmr(pleasant_model):
    m = pleasant_model
    tmr = Tmr(m.parent, m,
              parent_head_file=m.cfg['parent']['headfile'],
              inset_parent_layer_mapping=m.parent_layers,
              inset_parent_period_mapping=m.parent_stress_periods)
    return tmr


@pytest.fixture(scope='function')
def parent_heads(tmr):
    headfile = tmr.hpth
    hdsobj = bf.HeadFile(headfile)
    yield hdsobj  # provide the fixture value
    # code below yield statement is executed after test function finishes
    print("closing the heads file")
    hdsobj.close()


def test_get_inset_boundary_heads(tmr, parent_heads):
    """Verify that inset model specified head boundary accurately
    reflects parent model head solution, including when cells
    are dry or missing (e.g. pinched out cells in MF6).
    """
    bheads_df = tmr.get_inset_boundary_heads(for_external_files=False)
    groups = bheads_df.groupby('per')
    all_kstpkper = parent_heads.get_kstpkper()
    kstpkper_list = [all_kstpkper[0], all_kstpkper[-1]]
    for kstp, kper in kstpkper_list:
        hds = parent_heads.get_data(kstpkper=(kstp, kper))
        df = groups.get_group(kper)
        df['cellid'] = list(zip(df.k, df.i, df.j))
        # check for duplicate locations (esp. corners)
        # in mf2005, duplicate chd heads will be summed
        assert not df.cellid.duplicated().any()

        # x, y, z locations of inset model boundary cells
        ix = tmr.inset.modelgrid.xcellcenters[df.i, df.j]
        iy = tmr.inset.modelgrid.ycellcenters[df.i, df.j]
        iz = tmr.inset.modelgrid.zcellcenters[df.k, df.i, df.j]

        # parent model grid cells associated with inset boundary cells
        i, j = get_ij(tmr.parent.modelgrid, ix, iy)
        k = get_layer(tmr.parent.dis.botm.array, i, j, iz)

        # error between parent heads and inset heads
        # todo: interpolate parent head solution to inset points for comparison


def test_tmr_new(pleasant_model):
    m = pleasant_model
    parent_headfile = Path(m.cfg['parent']['headfile'])
    parent_cellbudgetfile = parent_headfile.with_suffix('.cbc')

    tmr = TmrNew(m.parent, m,
                 parent_head_file=parent_headfile)

    results = tmr.get_inset_boundary_values(for_external_files=False)
    assert np.all(results.columns ==
                  ['k', 'i', 'j', 'per', 'bhead'])
    # indices should be zero-based
    assert results['k'].min() == 0
    # non NaN heads
    assert not results.bhead.isna().any()
    # no heads below cell bottoms
    cell_botms = m.dis.botm.array[results['k'], results['i'], results['j']]
    assert not np.any(results['bhead'] < cell_botms)
    # no duplicate heads
    results['cellid'] = list(zip(results.per, results.k, results.i, results.j))
    assert not results.cellid.duplicated().any()

    # test external files case
    # and with connections defined by layer
    tmr.define_connections_by = 'by_layer'
    tmr._inset_boundary_cells = None  # reset property
    results = tmr.get_inset_boundary_values(for_external_files=True)
    # '#k' required for header row
    assert np.all(results.columns ==
                  ['#k', 'i', 'j', 'per', 'bhead'])
    # indices should be one-based (written directly to external files)
    assert results['#k'].min() == 1
