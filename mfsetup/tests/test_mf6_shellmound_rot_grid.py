import os
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from flopy import mf6
from gisutils import get_values_at_points
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon, box

from mfsetup import MF6model
from mfsetup.fileio import exe_exists, load
from mfsetup.utils import get_input_arguments, update


@pytest.fixture(scope='module')
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

    del cfg['sfr']['sfrmaker_options']['to_riv']

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


@pytest.fixture()
def rotated_parent_small(shellmound_cfg, tmpdir):
    cfg = deepcopy(shellmound_cfg)
    cfg['simulation']['sim_ws'] = os.path.join(tmpdir, 'shellmound_rotated_parent')
    kwargs = cfg['simulation'].copy()
    kwargs.update(cfg['simulation']['options'])

    kwargs = get_input_arguments(kwargs, mf6.MFSimulation)

    sim = mf6.MFSimulation(**kwargs)

    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = sim
    cfg['model']['packages'] = ['dis', 'sto']
    cfg['setup_grid'] = {
        'rotation': 30,
        'crs': 5070,
        'xoff': 519483,
        'yoff': 1184365,
        'snap_to_NHG': False
    }
    cfg['dis']['dimensions']['nrow'] = 10
    cfg['dis']['dimensions']['ncol'] = 12

    del cfg['sfr']['sfrmaker_options']['to_riv']

    m = MF6model.setup_from_cfg(cfg)
    m.write_input()
    return m


@pytest.mark.parametrize(
    'specified_tmr_parent_rotation,lgr_parent_rotation,lgr_inset_rotation,from_point',
    [
    # test parent model SpatialReference configuration
    (30., 30., 30., False),  # specified parent rotation that is consistent with DIS package
    (30., 30., None, False),  # LGR inset rotation not specified in config
    # make LGR inset grid from a buffer around a specified point
    # to test that buffer is maintained after snapping the model to the parent grid
    (30., 30., 30., True),
    # specified parent rotation that is inconsistent with parent model DIS package
    pytest.param(-99., 30., 30., False, marks=pytest.mark.xfail(reason='inconsistent sr')),
    # LGR grid with different rotation than LGR parent
    pytest.param(30., 30., 0, False, marks=pytest.mark.xfail(
        reason='LGR grid must have same rotation as LGR parent'))
])
def test_rotated_lgr_grid_setup(rotated_parent_small,
                                specified_tmr_parent_rotation,
                                lgr_parent_rotation, lgr_inset_rotation,
                                from_point,
                                project_root_path, tmpdir, test_data_path):
    """Test construction of an LGR model with a rotated TMR parent. LGR parent model
    can have different rotation from TMR parent, but LGR inset (refined)
    model must have same rotation as LGR parent. The specified SpatialReference
    for the TMR parent must be consistent with the DIS package spatial reference,
    unless override_dis_package_input: True is included in the configuration file.
    """
    tmr_parent = deepcopy(rotated_parent_small)
    tmr_parent_rotation = tmr_parent.modelgrid.angrot

    parent_sr = {
            'xoff': tmr_parent.cfg['setup_grid']['xoff'],
            'yoff': tmr_parent.cfg['setup_grid']['yoff'],
            'crs': tmr_parent.cfg['setup_grid']['crs'],
            'rotation': specified_tmr_parent_rotation
        }

    tmrp_nlay, tmrp_nrow, tmrp_ncol = tmr_parent.modelgrid.shape
    project_root_path = Path(project_root_path)
    default_cfg = load(project_root_path / 'mfsetup/mf6_defaults.yml')
    cfg = deepcopy(default_cfg)
    lgr_inset_cfg = deepcopy(default_cfg)

    if from_point:
        #gdf = gpd.read_file(test_data_path / 'shellmound/shps/waterbodies.shp')
        #feature = gdf.loc[gdf['COMID'].astype(str) == '18047154']
        #feature['geometry'] = feature['geometry'].centroid
        feature = gpd.GeoDataFrame({
            'COMID': ['18047154'],
            'geometry': [Point(522824.3, 1192240.6)]
            }, crs=5070)
        features_shapefile = Path(tmpdir, 'point.shp')
        feature.to_file(features_shapefile, index=False)
    else:
        features_shapefile = test_data_path / 'shellmound/shps/waterbodies.shp'
    specified_lgr_inset_cfg = {
        'simulation': {
        'sim_name': 'rotated_lgr_30',
        'version': 'mf6',
        'sim_ws': '.',
        'options': {}
        },
        'model': {
        'simulation': 'rotated_lgr_30',
        'modelname': 'rt-lgr30-inset',
        'packages': ['dis'],
        },
        'setup_grid': {
        'source_data': {
            'features_shapefile': {
            'filename': features_shapefile,
            'id_column': 'COMID',
            'include_ids': ['18047154']
            },

        },
        'buffer': 2000,
        'crs': 5070,
        'delr': tmr_parent.dis.delr[0]/2,
        'delc': tmr_parent.dis.delc[0]/2,
        'rotation': lgr_inset_rotation
        },
    }
    if lgr_inset_rotation is None:
        del specified_lgr_inset_cfg['setup_grid']['rotation']
    update(lgr_inset_cfg, specified_lgr_inset_cfg)

    specified_cfg = {
        'simulation': {
        'sim_name': 'rotated_lgr_30',
        'version': 'mf6',
        'sim_ws': '.',
        'options': {}
        },
        'model': {
        'simulation': 'rotated_lgr_30',
        'modelname': 'rt-lgr30-parent',
        'packages': ['dis'],
        },
        'parent': {
        'namefile': tmr_parent.name_file.filename,
        'model_ws': '.',
        'version': 'mf6',
        'default_source_data': True,  # if True, packages and variables that are omitted will be pulled from this model
        'copy_stress_periods': 'all',
        'start_date_time': tmr_parent.simulation.tdis.start_date_time.array,
        'length_units': tmr_parent.dis.length_units.array,
        'time_units': 'days',
        'SpatialReference': parent_sr
        },
        # use the same dimensions as parent for now
        # to better test grid extent
        'setup_grid': {
        'xoff': tmr_parent.cfg['setup_grid']['xoff'],
        'yoff': tmr_parent.cfg['setup_grid']['yoff'],
        'epsg': tmr_parent.cfg['setup_grid']['crs'],
        'nrow': tmrp_nrow,
        'ncol': tmrp_ncol,
        'rotation': lgr_parent_rotation,
        'nlay': tmrp_nlay,
        'delr': tmr_parent.dis.delr[0],
        'delc': tmr_parent.dis.delc[0],
        'lgr': {
            'rt_lgr30_inset': {
                'cfg': lgr_inset_cfg,
                'layer_refinement': 1
            }
    }
    }}
    update(cfg, specified_cfg)
    m = MF6model(cfg=cfg)
    m.setup_grid()

    # the parent model rotation shouldn't change
    assert m.parent.modelgrid.angrot == tmr_parent_rotation
    # LGR parent and TMR parent grids should be the same size
    # (because we specified them that way)
    if specified_tmr_parent_rotation == lgr_parent_rotation:
        assert np.allclose(m.parent.modelgrid.extent, m.modelgrid.extent)
    # another test for whether inset grid is centered around feature
    # (distance from feature to grid edge is >= buffer on all sides)
    inset_grid_cfg = cfg['setup_grid']['lgr']['rt_lgr30_inset']['cfg']['setup_grid']
    feature_cfg = inset_grid_cfg['source_data']['features_shapefile']
    gdf = gpd.read_file(feature_cfg['filename'])
    id_col = feature_cfg['id_column']
    include_ids = feature_cfg['include_ids']
    feature = gdf.loc[gdf[id_col].astype(str).isin(include_ids)]
    lgr_inset_grid = m.inset['rt-lgr30-inset'].modelgrid
    feature.to_crs(m.inset['rt-lgr30-inset'].modelgrid.crs, inplace=True)
    buffered = feature.buffer(inset_grid_cfg['buffer'])
    l, r, b, t = m.inset['rt-lgr30-inset'].modelgrid.extent
    perimeter = [
        (lgr_inset_grid.xyedges[0][0], lgr_inset_grid.xyedges[1][0]),
        (lgr_inset_grid.xyedges[0][-1], lgr_inset_grid.xyedges[1][0]),
        (lgr_inset_grid.xyedges[0][-1], lgr_inset_grid.xyedges[1][-1]),
        (lgr_inset_grid.xyedges[0][0], lgr_inset_grid.xyedges[1][-1])
        ]
    perimeter = Polygon([lgr_inset_grid.get_coords(model_x, model_y)
                 for model_x, model_y in perimeter])
    assert all(buffered.within(perimeter))

    import flopy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    pmv = flopy.plot.PlotMapView(m, ax=ax)
    pmv2 = flopy.plot.PlotMapView(m.inset['rt-lgr30-inset'], ax=ax)
    lc = pmv.plot_grid()
    lc2 = pmv2.plot_grid()
    feature.plot(ax=ax, color='b')
    buffered.plot(ax=ax, fc='none', ec='b')
    gpd.GeoSeries(perimeter).plot(ax=ax, fc='none', ec='b', zorder=10)
    plt.pause(1)
    # write out the shapefiles to visually inspect results
    m.parent.modelgrid.write_shapefile('tmr_parent_grid.shp')
    m.modelgrid.write_shapefile('lgr_parent_grid.shp')
    m.inset['rt-lgr30-inset'].modelgrid.write_shapefile('lgr_inset_grid.shp')


@pytest.mark.parametrize('inset_rotation,parent_rotation,override_dis', [
    (18., 18., False),  # same rotation as the parent model
    # different rotation from the parent
    (-30, 18., False),
    # option to override DIS package parent model spatial reference
    pytest.param(-99, 18., True, marks=pytest.mark.xfail(reason='inset model origin outside of parent model domain')),
    pytest.param(18., 10., True)
    ])
def test_rotated_tmr(rotated_parent,
                     inset_rotation, parent_rotation, override_dis,
                     shellmound_cfg, tmpdir, test_data_path):
    """Test making an inset model on a rotated grid with
    * the same or different rotation as the parent model
    * a rotation that places the upper left corner (reference point)
      outside of the parent model domain
    * a specified rotation for the parent model that is inconsistent
      with what is specified in the parent model DIS package input
      (only allowed with override_dis_package_input: True option)

    Note: this only tests model grid and DIS package setup,
    not the model solution.
    """
    cfg = deepcopy(shellmound_cfg)
    cfg['simulation']['sim_ws'] = os.path.join(tmpdir, 'shellmound_rotated_tmr')
    kwargs = cfg['simulation'].copy()
    kwargs.update(cfg['simulation']['options'])

    kwargs = get_input_arguments(kwargs, mf6.MFSimulation)

    sim = mf6.MFSimulation(**kwargs)
    cfg['model']['simulation'] = sim
    cfg['model']['packages'] = ['dis']

    cfg['setup_grid']['snap_to_NHG'] = False
    cfg['setup_grid']['rotation'] = inset_rotation
    cfg['setup_grid']['xoff'] = 517425
    cfg['setup_grid']['yoff'] = 1178441
    cfg['setup_grid']['dxy'] = 250
    cfg['dis']['dimensions']['nrow'] = 15
    cfg['dis']['dimensions']['ncol'] = 15

    cfg['chd'] = {'perimeter_boundary': {
        'parent_head_file':  '../tmp/shellmound_rotated_parent/shellmound.hds'
    }
        }
    # make parent block
    cfg['parent'] = {}
    cfg['parent']['namefile'] = 'shellmound.nam'
    cfg['parent']['model_ws'] = '../tmp/shellmound_rotated_parent'
    cfg['parent']['version'] = 'mf6'
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
    if override_dis:
        cfg['parent']['SpatialReference']['override_dis_package_input'] = True
    cfg['parent']['SpatialReference']['rotation'] = parent_rotation

    del cfg['sfr']['sfrmaker_options']['to_riv']

    m = MF6model.setup_from_cfg(cfg)

    assert m.modelgrid.rotation == inset_rotation
    assert m.parent.modelgrid.rotation == parent_rotation
    assert m.dis.nrow.array == cfg['dis']['dimensions']['nrow']
    assert m.dis.ncol.array == cfg['dis']['dimensions']['ncol']

    if inset_rotation != parent_rotation:
        # no snapping to parent model;
        #inset origin should match input
        assert np.allclose(m.modelgrid.xoffset, cfg['setup_grid']['xoff'])
        assert np.allclose(m.modelgrid.yoffset, cfg['setup_grid']['yoff'])
    else:
        # otherwise, snap_to_parent = True by default;
        # corner distance should be < square root of parent cell area
        corner_distance = np.sqrt((m.modelgrid.xoffset - cfg['setup_grid']['xoff'])**2 +\
                            (m.modelgrid.yoffset - cfg['setup_grid']['yoff'])**2)
        assert corner_distance < np.sqrt(m.parent.modelgrid.delc[0]**2 +\
            m.parent.modelgrid.delr[0]**2)

    # interpolate the parent model values for the last layer bottom
    # compare to last layer bottom in tmr inset
    # the values for the inset were sampled independently from the source raster
    # so if the values compare, it means the rotations in the parent and inset model grid are consistent
    parent_model_values = griddata((m.parent.modelgrid.xcellcenters.ravel(),
                                    m.parent.modelgrid.ycellcenters.ravel()),
                                    m.parent.dis.botm.array[-1].ravel(),
                                   (m.modelgrid.xcellcenters, m.modelgrid.ycellcenters)
                                    )
    parent_model_values = np.ma.masked_array(parent_model_values, mask=np.isnan(parent_model_values))
    inset_model_values = np.ma.masked_array(m.dis.botm.array[-1], mask=m.dis.idomain.array[-1] == 0)
    # this comparison is going to be sloppy
    # since we are comparing the downsampled model values to the original (fine res) raster
    # comparing values sample from the parent model will only work
    # if parent model rotation is specified correctly;
    # (otherwise wrong locations in parent model will be sample)
    if parent_rotation == rotated_parent.modelgrid.rotation:
        assert np.allclose(parent_model_values, inset_model_values, rtol=0.01)
    # check the last layer bottom for consistency with source raster
    # at the inset model cell centers
    rpath = test_data_path / 'shellmound/rasters/mdwy_surf.tif'
    source_raster_values = get_values_at_points(rpath,
                                                m.modelgrid.xcellcenters.ravel(),
                                                m.modelgrid.ycellcenters.ravel(),
                                                points_crs=5070, method='linear')
    source_raster_values = np.reshape(source_raster_values, m.modelgrid.shape[1:])
    source_raster_values = np.ma.masked_array(source_raster_values, mask=m.dis.idomain.array[-1] == 0)
    assert np.allclose(inset_model_values, source_raster_values * .3048, rtol=0.01)


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
