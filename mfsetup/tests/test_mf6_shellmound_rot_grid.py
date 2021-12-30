import os
from copy import deepcopy

import numpy as np
import pytest
from flopy import mf6
from gisutils import get_values_at_points
from scipy.interpolate import griddata

from mfsetup import MF6model
from mfsetup.fileio import exe_exists
from mfsetup.utils import get_input_arguments


@pytest.fixture()
def rotated_parent(shellmound_cfg, tmpdir, mf6_exe):
    cfg = deepcopy(shellmound_cfg)
    cfg['simulation']['sim_ws'] = os.path.join(tmpdir, 'shellmound_rotated_parent')
    kwargs = cfg['simulation'].copy()
    kwargs.update(cfg['simulation']['options'])

    kwargs = get_input_arguments(kwargs, mf6.MFSimulation)

    sim = mf6.MFSimulation(**kwargs)

    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = sim
    cfg['setup_grid']['snap_to_NHG'] = False
    cfg['setup_grid']['rotation'] = 18.
    cfg['setup_grid']['xoff'] = 509405
    cfg['setup_grid']['yoff'] = 1175835
    cfg['dis']['dimensions']['nrow'] = 20
    cfg['dis']['dimensions']['ncol'] = 25

    m = MF6model.setup_from_cfg(cfg)
    m.write_input()
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


def test_rotated_tmr(rotated_parent, shellmound_cfg, tmpdir, test_data_path):
    cfg = deepcopy(shellmound_cfg)
    cfg['simulation']['sim_ws'] = os.path.join(tmpdir, 'shellmound_rotated_tmr')
    kwargs = cfg['simulation'].copy()
    kwargs.update(cfg['simulation']['options'])

    kwargs = get_input_arguments(kwargs, mf6.MFSimulation)

    sim = mf6.MFSimulation(**kwargs)
    cfg['model']['simulation'] = sim

    cfg['setup_grid']['snap_to_NHG'] = False
    cfg['setup_grid']['rotation'] = 18.
    cfg['setup_grid']['xoff'] = 517425
    cfg['setup_grid']['yoff'] = 1178441
    cfg['setup_grid']['dxy'] = 250
    cfg['dis']['dimensions']['nrow'] = 15
    cfg['dis']['dimensions']['ncol'] = 15

    # make parent block
    cfg['parent'] = {}
    cfg['parent']['namefile'] = 'shellmound.nam'
    cfg['parent']['model_ws'] = '../tmp/shellmound_rotated_parent'
    cfg['parent']['version'] = 'mf6'
    cfg['parent']['headfile'] = '../tmp/shellmound_rotated_parent/shellmound.hds'  # needed for the perimeter boundary setup
    cfg['parent']['default_source_data'] = True  # if True, packages and variables that are omitted will be pulled from this model
    cfg['parent']['copy_stress_periods'] = 'all'
    cfg['parent']['start_date_time'] = '1998-04-01'
    # inset_layer_mapping assumed to be 1:1 if not entered
    cfg['parent']['length_units'] = 'meters'
    cfg['parent']['time_units'] = 'days'
    # parent model lower left corner location and CRS
    # (overrides any information in name file)
    cfg['parent']['SpatialReference'] = {}
    cfg['parent']['SpatialReference']['xoff'] = rotated_parent.modelgrid.xoffset
    cfg['parent']['SpatialReference']['yoff'] = rotated_parent.modelgrid.yoffset
    cfg['parent']['SpatialReference']['epsg'] = rotated_parent.modelgrid.epsg
    cfg['parent']['SpatialReference']['rotation'] = rotated_parent.modelgrid.angrot

    m = MF6model.setup_from_cfg(cfg)
    # interpolate the parent model values for the last layer bottom
    # compare to last layer bottom in tmr inset
    # the values for the inset were sampled independently from the source raster
    # so if the values compare, it means the rotations in the parent and inset model grid are consistent
    parent_model_values = griddata((m.parent.modelgrid.xcellcenters.ravel(),
                                    m.parent.modelgrid.ycellcenters.ravel()),
                                    m.parent.dis.botm.array[-1].ravel(),
                                   (m.modelgrid.xcellcenters, m.modelgrid.ycellcenters)
                                    )
    assert np.allclose(parent_model_values, m.dis.botm.array[-1], rtol=0.01)
    # check the last layer bottom for consistency with source raster
    # at the inset model cell centers
    rpath = test_data_path / 'shellmound/rasters/mdwy_surf.tif'
    source_raster_values = get_values_at_points(rpath,
                                                m.modelgrid.xcellcenters.ravel(),
                                                m.modelgrid.ycellcenters.ravel(),
                                                points_crs=5070, method='linear')
    assert np.allclose(m.dis.botm.array[-1].ravel(), source_raster_values * .3048, rtol=0.01)


def test_rotated_grid(shellmound_cfg, shellmound_simulation, mf6_exe):
    cfg = deepcopy(shellmound_cfg)
    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = shellmound_simulation
    cfg['setup_grid']['snap_to_NHG'] = False
    nrow, ncol = 20, 25
    xoff, yoff = 509405, 1175835
    rotation = 18
    cfg['setup_grid']['xoff'] = xoff
    cfg['setup_grid']['yoff'] = yoff
    cfg['setup_grid']['rotation'] = rotation
    cfg['dis']['dimensions']['nrow'] = nrow
    cfg['dis']['dimensions']['ncol'] = ncol

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()

    # check that the rotation and lower left corner are consistent
    assert m.modelgrid.angrot == 18.
    assert m.modelgrid.xoffset == xoff
    assert m.modelgrid.yoffset == yoff

    m.setup_dis()

    # rotation should be positive in the counter-clockwise direction
    # check that the model grid lower right corner is in the right place
    xlr = m.modelgrid.xyzvertices[0][-1, -1]
    ylr = m.modelgrid.xyzvertices[1][-1, -1]
    expected_x = xoff + np.cos(np.radians(rotation)) * ncol * 1000
    expected_y = yoff + np.sin(np.radians(rotation)) * ncol * 1000
    assert np.allclose(xlr, expected_x)
    assert np.allclose(ylr, expected_y)
