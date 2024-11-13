"""Test initial conditions setup functionality
"""
import copy
from pathlib import Path

import flopy
import numpy as np
import pytest

from mfsetup import MF6model
from mfsetup.fileio import exe_exists, load, load_array, load_cfg, read_mf6_block
from mfsetup.tests.test_lgr import (
    pleasant_vertical_lgr_cfg,
    pleasant_vertical_lgr_test_cfg_path,
)
from mfsetup.utils import get_input_arguments


@pytest.mark.parametrize('ic_config,expected', (
     ({'griddata': {
         'strt': 19.
     }}, np.ones((2, 3, 2))*19),
    ({'griddata': {
         'strt': [19., 18.]
     }}, np.ones((2, 3, 2))*np.array([[[19]], [[18]]])),
 ))
def test_ic_direct_input(cfg_2x2x3_with_dis, ic_config, expected):
    """Very basic test of direct input to IC Package."""
    cfg = copy.deepcopy(cfg_2x2x3_with_dis)
    cfg['ic'].update(ic_config)
    m = MF6model(cfg=cfg)
    m.setup_dis()
    m.setup_ic()
    assert np.allclose(m.ic.strt.array, expected)


@pytest.mark.parametrize('ic_cfg,default_source_data,expected', (
    ('no ic: block', False, 'top'),
    ('no ic: block', True, 'parent starting heads'),
    (None, False, 'top'),
    ({'source_data': {
        'strt': {
            'from_parent': {
                'binaryfile': '../../../examples/data/pleasant/pleasant.hds',
                'stress_period': 0}}}}, True, 'parent heads'),
    ({'source_data': {
        'strt': 'from_model_top'}}, True, 'top'),
    ({'mfsetup_options': {},
     'griddata': {},
     'source_data': {},
     'source_data_config': {},
     'strt_filename_fmt': {},
     'filename_fmt': {}}, True, 'parent starting heads'),
    ({'source_data': {
        'strt': 'from_parent'}}, True, 'parent starting heads'),
))
def test_pleasant_vertical_lgr_ic_strt(pleasant_vertical_lgr_cfg, ic_cfg, default_source_data, expected,
                                       project_root_path):
    """More advanced example of IC Package input in an LGR context where a MODFLOW 6 model is inset
    within a MODFLOW-NWT model (one-way coupled with specified boundaries); and the MODFLOW 6 model
    contains a second inset model that is dyanmically coupled using the local grid refinement (LGR) capability.
    """
    cfg = copy.deepcopy(pleasant_vertical_lgr_cfg)
    if ic_cfg == 'no ic: block':
        del cfg['ic']
    else:
        cfg['ic'] = ic_cfg
    if not default_source_data:
        cfg['parent']['default_source_data'] = False
    m = MF6model(cfg=cfg)
    m.setup_dis()
    m.setup_ic()
    assert m.ic.strt.array.shape == m.modelgrid.shape
    if expected == 'top':
        top3d = np.array([m.dis.top.array] * m.modelgrid.nlay)
        assert np.allclose(m.ic.strt.array, top3d)
    elif expected == 'parent starting heads':
        np.allclose(m.ic.strt.array.mean(),
                    m.parent.bas6.strt.array.mean(), rtol=0.01)
    else:
        # check LGR parent heads against parent heads
        parent_headsfile = str(project_root_path / 'examples/data/pleasant/pleasant.hds')
        hds = flopy.utils.binaryfile.HeadFile(parent_headsfile).get_data(kstpkper=(0, 0))
        assert np.allclose(m.ic.strt.array.mean(), hds.mean(), rtol=0.01)

        # check for consistency between MFBinaryArraySourceData
        # and regridded results
        from mfsetup.interpolate import regrid3d
        resampled_parent_heads = regrid3d(hds,
                            m.parent.modelgrid,
                            m.modelgrid,
                            mask1=None, mask2=None, method='linear')

        assert np.allclose(m.ic.strt.array,
                           np.round(resampled_parent_heads, 2))

        from mfsetup.sourcedata import MFBinaryArraySourceData
        sd = MFBinaryArraySourceData(variable='strt', filename=parent_headsfile,
                                     datatype='array3d',
                                     dest_model=m,
                                     source_modelgrid=m.parent.modelgrid,
                                     from_source_model_layers=None,
                                     length_units=m.length_units,
                                     time_units=m.time_units,
                                     resample_method='linear', stress_period=0,
                                     )
        data = sd.get_data()
        assert np.allclose(m.ic.strt.array,
                           np.round(np.array(list(data.values())), 2))

        # check LGR inset heads against parent heads
        m.inset['plsnt_lgr_inset'].setup_dis()
        m.inset['plsnt_lgr_inset'].setup_ic()
        resampled_parent_heads_lgr_inset = regrid3d(hds,
                            m.parent.modelgrid,
                            m.inset['plsnt_lgr_inset'].modelgrid,
                            mask1=None, mask2=None, method='linear')
        diff = m.inset['plsnt_lgr_inset'].ic.strt.array -\
            np.round(resampled_parent_heads_lgr_inset, 2)

        # a small percentage of cells are appreciably different
        # unclear why
        assert np.sum(np.abs(diff) > 0.01)/diff.size <= 0.0005
