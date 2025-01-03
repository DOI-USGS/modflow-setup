import copy
import glob
import os
from copy import deepcopy
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pytest

mf6 = flopy.mf6
fm = flopy.modflow
from flopy.utils import binaryfile as bf

from mfsetup import MF6model
from mfsetup.discretization import make_lgr_idomain
from mfsetup.fileio import dump, exe_exists, load, load_cfg, read_mf6_block
from mfsetup.grid import get_ij
from mfsetup.mover import get_sfr_package_connections
from mfsetup.testing import compare_inset_parent_values
from mfsetup.utils import get_input_arguments, update


@pytest.fixture(scope="session")
def pleasant_lgr_test_cfg_path(project_root_path):
    return project_root_path / 'examples/pleasant_lgr_parent.yml'

@pytest.fixture(scope="session")
def pleasant_vertical_lgr_test_cfg_path(test_data_path):
    return test_data_path / 'pleasant_vertical_lgr_parent.yml'

@pytest.fixture(scope="function")
def pleasant_lgr_cfg(pleasant_lgr_test_cfg_path):
    cfg = load_cfg(pleasant_lgr_test_cfg_path,
                   default_file='mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg

@pytest.fixture(scope="function")
def pleasant_vertical_lgr_cfg(pleasant_vertical_lgr_test_cfg_path):
    cfg = load_cfg(pleasant_vertical_lgr_test_cfg_path,
                   default_file='mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg

@pytest.fixture(scope="function")
def pleasant_simulation(pleasant_lgr_cfg):
    cfg = pleasant_lgr_cfg.copy()
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    return sim


@pytest.fixture(scope="function")
def get_pleasant_lgr_parent(pleasant_lgr_cfg, pleasant_simulation):
    print('creating Pleasant Lake MF6model instance from cfgfile...')
    cfg = pleasant_lgr_cfg.copy()
    cfg['model']['simulation'] = pleasant_simulation
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def get_pleasant_lgr_parent_with_grid(get_pleasant_lgr_parent):
    print('creating Pleasant Lake MF6model instance with grid...')
    m = copy.deepcopy(get_pleasant_lgr_parent)
    m.setup_grid()
    return m


@pytest.fixture(scope="function")
def pleasant_lgr_setup_from_yaml(pleasant_lgr_cfg):
    m = MF6model.setup_from_cfg(pleasant_lgr_cfg)
    m.write_input()
    return m

@pytest.fixture(scope="function")
def pleasant_vertical_lgr_setup_from_yaml(pleasant_vertical_lgr_cfg):
    m = MF6model.setup_from_cfg(pleasant_vertical_lgr_cfg)
    m.write_input()
    return m

@pytest.fixture(scope="function")
def pleasant_lgr_stand_alone_parent(pleasant_lgr_test_cfg_path, tmpdir):
    """Stand-alone version of lgr parent model for comparing with LGR results.
    """
    # Edit the configuration file before the file paths within it are converted to absolute
    # (model.load_cfg converts the file paths)
    cfg = load(pleasant_lgr_test_cfg_path)
    del cfg['setup_grid']['lgr']
    cfg['simulation']['sim_ws'] = os.path.join(tmpdir, 'pleasant_lgr_just_parent')

    # save out the edited configuration file
    path, fname = os.path.split(pleasant_lgr_test_cfg_path)
    new_file = os.path.join(path, 'pleasant_lgr_just_parent.yml')
    dump(new_file, cfg)

    # load in the edited configuration file, converting the paths to absolute
    cfg = MF6model.load_cfg(new_file)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')

    m = MF6model.setup_from_cfg(cfg)
    m.write_input()
    return m


def test_make_lgr_idomain(get_pleasant_lgr_parent_with_grid):
    m = get_pleasant_lgr_parent_with_grid
    inset_model = m.inset['plsnt_lgr_inset']
    idomain = make_lgr_idomain(m.modelgrid, inset_model.modelgrid,
                               m.lgr['plsnt_lgr_inset'].ncppl)
    assert idomain.shape == m.modelgrid.shape
    l, b, r, t = inset_model.modelgrid.bounds
    isinset = (m.modelgrid.xcellcenters > l) & \
              (m.modelgrid.xcellcenters < r) & \
              (m.modelgrid.ycellcenters > b) & \
              (m.modelgrid.ycellcenters < t)
    assert idomain[:, isinset].sum() == 0
    assert np.all(idomain[:, ~isinset] >= 1)


@pytest.mark.parametrize(
    'lgr_grid_spacing,layer_refinement_input,inset_nlay', [
    # parent layers 0 through 3
    # are specified in pleasant_lgr_parent.yml
    (40, 'use configuration', 5),
    (40, [1, 1, 1, 1, 0], 4),
    (40, [2, 1, 1, 1, 0], 5),
    # dictionary input option
    (40, {0:1, 1:1, 2:1}, 3),
    # integer layer refinement input option
    (40, 1, 5),
    # special case to test setup
    # with no layer refinement specified
    (40, None, 5),
    pytest.param(35, None, 5, marks=pytest.mark.xfail(
        reason='inset spacing not a factor of parent spacing')),
    pytest.param(40, None, 4, marks=pytest.mark.xfail(
        reason='inset nlay inconsistent with parent layers')),
    pytest.param(40, [1, 1, 1, 1], 5, marks=pytest.mark.xfail(
        reason='List layer_refinement input must include a value for each layer')),
    pytest.param(40, [0, 1, 1, 1, 1], 5, marks=pytest.mark.xfail(
        reason='LGR child grid must start at model top')),
    pytest.param(40, {1: 1}, 5, marks=pytest.mark.xfail(
        reason='LGR child grid must start at model top')),
    pytest.param(40, [1, 1, 0, 1, 1], 5, marks=pytest.mark.xfail(
        reason='LGR child grid must be contiguous')),
    ]
    )
def test_lgr_grid_setup(lgr_grid_spacing, layer_refinement_input,
                        inset_nlay,
                        pleasant_lgr_cfg, pleasant_simulation,
                        project_root_path):

    # apply test parameters to inset/parent configurations
    inset_cfg_path = project_root_path / 'examples/pleasant_lgr_inset.yml'
    inset_cfg = load_cfg(inset_cfg_path,
                         default_file='mf6_defaults.yml')
    inset_cfg['setup_grid']['dxy'] = lgr_grid_spacing
    inset_cfg['dis']['dimensions']['nlay'] = inset_nlay

    cfg = pleasant_lgr_cfg.copy()
    lgr_cfg = cfg['setup_grid']['lgr']['pleasant_lgr_inset']
    lgr_cfg['cfg'] = inset_cfg
    del lgr_cfg['filename']
    if layer_refinement_input is None:
        del lgr_cfg['layer_refinement']
        layer_refinement = [1] * cfg['dis']['dimensions']['nlay']
    elif layer_refinement_input == 'use configuration':
        layer_refinement = lgr_cfg['layer_refinement']
    else:
        lgr_cfg['layer_refinement'] = layer_refinement_input
        layer_refinement = layer_refinement_input

    cfg['model']['simulation'] = pleasant_simulation
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()

    inset_model = m.inset['plsnt_lgr_inset']
    assert isinstance(inset_model, MF6model)
    assert inset_model.parent is m
    assert isinstance(m.lgr[inset_model.name], flopy.utils.lgrutil.Lgr)
    if os.environ.get('CI', 'false').lower() != 'true':
        outfolder = Path(m._shapefiles_path)
        output_shapefile = 'pleasant_lgr_parent_grid.shp'
        m.modelgrid.write_shapefile(outfolder / 'pleasant_lgr_parent_grid.shp')
        inset_model.modelgrid.write_shapefile(outfolder / 'pleasant_lgr_inset_grid.shp')

    # convert layer_refinement to a list
    # (mimicking what is done internally in modflow-setup)
    if np.isscalar(layer_refinement):
        layer_refinement = np.array([1] * m.modelgrid.nlay)
    elif isinstance(layer_refinement, dict):
        layer_refinement = [layer_refinement.get(i, 0) for i in range(m.modelgrid.nlay)]
    layer_refinement = np.array(layer_refinement)
    # verify that lgr area was removed from parent idomain
    lgr_idomain = make_lgr_idomain(m.modelgrid, inset_model.modelgrid,
                                   layer_refinement)
    idomain = m.idomain
    assert idomain[lgr_idomain == 0].sum() == 0
    inset_cells_per_layer = inset_model.modelgrid.shape[1] *\
        inset_model.modelgrid.shape[2]
    refinement_factor = int(cfg['setup_grid']['dxy'] / lgr_grid_spacing)
    nparent_cells_within_lgr_per_layer = inset_cells_per_layer / refinement_factor**2
    # for each layer where lgr is specified,
    # there should be at least enough inactive cells to cover the lrg grid area
    layers_with_lgr = (lgr_idomain == 0).sum(axis=(1, 2)) >=\
        nparent_cells_within_lgr_per_layer
    assert all(layers_with_lgr[layer_refinement > 0])
    # layers outside of the specified lgr range should have
    # less inactive cells than the size of the lgr grid area
    # (specific to this test case with a large LGR extent relative to the total grid)
    assert not any(layers_with_lgr[layer_refinement == 0])

    # make a plot for the docs
    if layer_refinement_input == 'use configuration':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        parent_mv = flopy.plot.PlotMapView(model=m, ax=ax, layer=0)
        inset_mv = flopy.plot.PlotMapView(model=inset_model, ax=ax, layer=0)
        l, r, b, t = m.modelgrid.extent
        lcp = parent_mv.plot_grid(lw=0.5, ax=ax)
        lci = inset_mv.plot_grid(lw=0.5)
        ax.set_ylim(b, t)
        ax.set_xlim(l, r)
        ax.set_aspect(1)
        plt.savefig(project_root_path / 'docs/source/_static/pleasant_lgr.png',
                    bbox_inches='tight')


def test_setup_mover(pleasant_lgr_setup_from_yaml):
    m = pleasant_lgr_setup_from_yaml
    assert isinstance(m.simulation.mvr, mf6.ModflowMvr)
    assert os.path.exists(m.simulation.mvr.filename)
    perioddata = read_mf6_block(m.simulation.mvr.filename, 'period')
    assert len(perioddata[1]) == 2
    for model in m, m.inset['plsnt_lgr_inset']:
        options = read_mf6_block(model.sfr.filename, 'options')
        assert 'mover' in options
    # verify that the mover file is referenced in the gwfgwf file
    gwfgwf_file = Path(m._abs_model_ws, m.simulation.gwfgwf.filename)
    with open(gwfgwf_file) as src:
        contents = src.read()
        assert m.simulation.mvr.filename in contents


def test_lgr_parent_bcs_in_lgr_area(pleasant_vertical_lgr_setup_from_yaml):
    """Test that boundary conditions specified in the parent model
    inside of the LGR area don't get created
    (these should be represented in the LGR inset/child model instead."""
    m = pleasant_vertical_lgr_setup_from_yaml
    ghb_source_data = {
        'shapefile': { # pond near pleasant lake
            'filename': '../../../../examples/data/pleasant/source_data/shps/all_lakes.shp',
            'id_column': 'HYDROID',
            'include_ids': [600059161],
            'all_touched': True
            },
        'bhead': {
            'filename': '../../../../examples/data/pleasant/source_data/rasters/dem40m.tif',
            'elevation_units': 'meters',
            'stat': 'mean'
            },
        'cond': 9,  # m2/d
    }
    m.setup_ghb(source_data=ghb_source_data)
    # there should be no GHB Package, because this feature is inside of the LGR area
    # (and should therefore be represented in the LGR inset/child model)
    assert not hasattr(m, 'ghb')
    wel_source_data = {
        'wells': {
            # well with screen mostly in LGR area (should not be represented)
            'well1': {'per': 1, 'x': 555910.8, 'y': 389618.3,
                      'screen_top': 300, 'screen_botm': 260, 'q': -2000},
            # well with screen mostly in parent model below LGR area (should be represented)
            'well2': {'per': 1, 'x': 555910.8, 'y': 389618.3,
                      'screen_top': 250, 'screen_botm': 200, 'q': -2000}
        }
    }
    added_wells = m.setup_wel(source_data=wel_source_data)
    # well1 should not be in the parent model (open interval within LGR area)
    assert 'well1' not in added_wells.stress_period_data.data[1]['boundname']
    assert 'well2' in added_wells.stress_period_data.data[1]['boundname']
    inset = m.inset['plsnt_lgr_inset']
    inset.setup_ghb(source_data=ghb_source_data)
    assert hasattr(inset, 'ghb')
    inset_added_wells = inset.setup_wel(source_data=wel_source_data)
    assert 'well1' in inset_added_wells.stress_period_data.data[1]['boundname']
    assert 'well2' not in inset_added_wells.stress_period_data.data[1]['boundname']


def test_mover_get_sfr_package_connections(pleasant_lgr_setup_from_yaml):
    m = pleasant_lgr_setup_from_yaml
    inset_model = m.inset['plsnt_lgr_inset']

    # check that packagedata for both models was written to external files
    assert Path(m.external_path, f"{m.name}_packagedata.dat").exists()
    assert Path(inset_model.external_path, f"{inset_model.name}_packagedata.dat").exists()

    parent_reach_data = m.sfrdata.reach_data
    inset_reach_data = inset_model.sfrdata.reach_data
    gwfgwf_exchangedata = m.simulation.gwfgwf.exchangedata.array
    to_inset, to_parent = get_sfr_package_connections(
        gwfgwf_exchangedata,
        parent_reach_data, inset_reach_data, distance_threshold=200)
    assert len(to_inset) == 0
    # verify that the last reaches in the two segments are keys
    last_reaches = inset_model.sfrdata.reach_data.groupby('iseg').last().rno
    assert not any(set(to_parent.keys()).difference(last_reaches))
    # verify that the first reaches are headwaters
    outreaches = set(m.sfrdata.reach_data.outreach)
    assert not any(set(to_parent.values()).intersection(outreaches))
    # the to_parent == {inset_reach: parent_reach, ...}
    # will break if the sfr package (inset extent, grid spacing, etc.) changes
    # if the grid changes and breaks this test
    # to figure out if to_parent is right,
    # run the test, then from the project root folder, go to
    # examples/pleasant_lgr/postproc/shps
    # plot the shapefiles in a GIS environment to verify the connections in to_parent
    # m.modelgrid.write_shapefile('inset_model_grid.shp')
    # m.parent.modelgrid.write_shapefile('parent_model_grid.shp')
    # {inset_reach: parent_reach, ...}
    assert to_parent == {29: 13, 41: 1}

    # test for no circular connections when two outlets are connected
    # and distance_threshold is large
    parent_reach_data.loc[parent_reach_data['rno'] == list(to_parent.values())[0], 'outreach'] = 0
    to_inset, to_parent = get_sfr_package_connections(
    gwfgwf_exchangedata,
    parent_reach_data, inset_reach_data, distance_threshold=1e4)
    assert not any(to_inset)


def test_meandering_sfr_connections(shellmound_cfg, project_root_path, tmpdir):
    """Test for SFR routing continuity in LGR cases
    where SFR streams meander back and forth
    between parent and child models."""
    cfg = deepcopy(shellmound_cfg)
    default_cfg = load(project_root_path / 'mfsetup/mf6_defaults.yml')
    del cfg['dis']['source_data']['idomain']
    cfg['dis']['griddata']['idomain'] = 1
    lgr_inset_cfg = deepcopy(default_cfg)
    del cfg['sfr']['sfrmaker_options']['to_riv']
    inset_sfr_config = deepcopy(cfg['sfr'])
    del inset_sfr_config['sfrmaker_options']['add_outlets']
    specified_lgr_inset_cfg = {
        'simulation': {
        'sim_name': 'shellmound',
        'version': 'mf6',
        'sim_ws': tmpdir / 'shellmound_lgr',
        'options': {}
        },
        'model': {
        'simulation': 'shellmound',
        'modelname': 'shellmound_lgr',
        'packages': ['dis', 'sfr'],
        'list_filename_fmt': '{}.list'
        },
        'setup_grid': {
            'xoff': 526958.2, # lower left x-coordinate
            'yoff': 1184284.0, # lower left y-coordinate
            'rotation': 0.,
            'dxy': 500,  # in CRS units of meters
            'crs': 5070
        },
        'dis': {
            'dimensions': {
                # if nrow and ncol are not specified here, the entries above in setup_grid are used
                'nlay': cfg['dis']['dimensions']['nlay'],
                'nrow': 4,
                'ncol': 16
            },
            'griddata': {
                'delr': 500,
                'delc': 500,
                'idomain': 1
            },
            'source_data': deepcopy(cfg['dis']['source_data'])

        },
        'sfr': inset_sfr_config

    }
    update(lgr_inset_cfg, specified_lgr_inset_cfg)
    cfg['simulation']['sim_ws'] = tmpdir / 'shellmound_lgr'
    cfg['setup_grid']['lgr'] = {
        'shellmound_lgr': {
            'cfg': lgr_inset_cfg
    }}
    m = MF6model(cfg=cfg)
    m.setup_grid()
    m.setup_tdis()
    m.inset['shellmound_lgr'].modelgrid.write_shapefile()
    m.setup_dis()
    m.setup_sfr()
    for k, v in m.inset.items():
        if v._is_lgr:
            v.setup_dis()
            v.setup_sfr()
    gwfgwf = m.setup_lgr_exchanges()

    # just test the connections for period 0
    exchangedata = pd.DataFrame(m.simulation.mvr.perioddata.array[0])
    # the reach numbers for these are liable to change
    # check SFR package shapefile output in test output folder
    # to verify that these are correct
    expected_connections = {
    ('shellmound', 167, 'shellmound_lgr', 7),
    ('shellmound', 229, 'shellmound_lgr', 0),
    ('shellmound', 288, 'shellmound_lgr', 5),
    ('shellmound_lgr', 4, 'shellmound', 252),
    ('shellmound_lgr', 14, 'shellmound', 180),
    ('shellmound_lgr', 17, 'shellmound', 164)
    }
    assert set(exchangedata[['mname1', 'id1', 'mname2', 'id2']].
               itertuples(index=False, name=None)) == expected_connections
    for modelname in 'shellmound', 'shellmound_lgr':
        model = m.simulation.get_model(modelname)
        cd1 = pd.DataFrame(m.simulation.get_model(modelname).sfr.connectiondata.array)
        cd1.index = cd1['ifno']
        # None of the reaches should have downstream connections
        # within their SFR Package
        # (no negative numbers indicates an outlet condition)
        connections_to_other_model = [cn[1] for cn in expected_connections
                                    if cn[0] == modelname]
        assert not any(cd1.loc[connections_to_other_model].iloc[:, 1] < 0)
        rd1 = model.sfrdata.reach_data
        assert all(rd1.loc[rd1['rno'].isin(np.array(connections_to_other_model)+1), 'outreach'] == 0)


def test_lgr_bottom_elevations(pleasant_vertical_lgr_setup_from_yaml, mf6_exe):
    """Test that boundary elevations specified in the LGR area
    are the same as the top of the underlying active area in
    the parent model (so there aren't gaps or overlap in the
    numerical grid)."""
    m = pleasant_vertical_lgr_setup_from_yaml
    inset = m.inset['plsnt_lgr_inset']
    x = inset.modelgrid.xcellcenters
    y = inset.modelgrid.ycellcenters
    pi, pj = get_ij(m.modelgrid, x.ravel(), y.ravel(),
                    local=False)
    pi = np.reshape(pi, x.shape)
    pj = np.reshape(pj, x.shape)
    parent_model_top = np.reshape(m.dis.top.array[pi, pj], x.shape)

    # LGR child bottom and parent top should be aligned
    assert np.allclose(inset.dis.botm.array[-1],
                       parent_model_top)

    # check exchange data
    exchange_data = pd.DataFrame(m.simulation.gwfgwf.exchangedata.array)
    k1, i1, j1 = zip(*exchange_data['cellidm1'])
    k2, i2, j2 = zip(*exchange_data['cellidm2'])
    exchange_data = exchange_data.assign(
        k1=k1, i1=i1, j1=j1, k2=k2, i2=i2, j2=j2)
    vertical_connections = exchange_data.loc[exchange_data['ihc'] == 0]

    lgr = m.lgr[inset.name] # Flopy Lgr utility instance
    active_cells = np.sum(inset.dis.idomain.array > 0, axis=0) > 0
    has_vertical_connection = np.zeros_like(active_cells)
    has_vertical_connection[vertical_connections['i2'], vertical_connections['j2']] = 1
    # all active cells should have a vertical connection
    assert all(has_vertical_connection[active_cells] == 1)
    # each active horizontal location should only have 1 vertical connection
    assert np.sum(has_vertical_connection[active_cells] == 1) == len(vertical_connections)
    # check connection distances
    all_layers = np.stack([m.dis.top.array] + [arr for arr in m.dis.botm.array])
    cell_thickness1 = -np.diff(all_layers, axis=0)
    all_layers2 = np.stack([inset.dis.top.array] + [arr for arr in inset.dis.botm.array])
    cell_thickness2 = -np.diff(all_layers2, axis=0)
    vertical_connections['layer_thick1'] = cell_thickness1[vertical_connections['k1'],
                                                           vertical_connections['i1'],
                                                           vertical_connections['j1']]
    vertical_connections['layer_thick2'] = cell_thickness2[vertical_connections['k2'],
                                                    vertical_connections['i2'],
                                                    vertical_connections['j2']]
    #  cl1 should be 0.5 x parent cell thickness
    assert np.allclose(vertical_connections['cl1'], vertical_connections['layer_thick1']/2)
    #  cl2 should be 0.5 x inset/child cell thickness
    assert np.allclose(vertical_connections['cl2'], vertical_connections['layer_thick2']/2)
    # hwva should equal inset cell bottom face area
    inset_cell_areas = inset.dis.delc.array[vertical_connections['i2']] *\
        inset.dis.delr.array[vertical_connections['j2']]
    assert np.allclose(vertical_connections['hwva'], inset_cell_areas)

    # test the horizontal connections too
    horizontal_connections_ns = exchange_data.loc[(exchange_data['ihc'] > 0) & exchange_data['angldegx'].isin([0., 180.])]
    horizontal_connections_ew = exchange_data.loc[(exchange_data['ihc'] > 0) & exchange_data['angldegx'].isin([90., 270.])]
    # cl1 should be 0.5 x parent cell width
    assert np.allclose(horizontal_connections_ns['cl1'], m.dis.delc.array[horizontal_connections_ns['i1']]/2)
    assert np.allclose(horizontal_connections_ew['cl1'], m.dis.delr.array[horizontal_connections_ew['j1']]/2)
    #cl2—is the distance between the center of cell 2 and the its shared face with cell 1.
    assert np.allclose(horizontal_connections_ns['cl2'], inset.dis.delc.array[horizontal_connections_ns['i2']]/2)
    assert np.allclose(horizontal_connections_ew['cl2'], inset.dis.delr.array[horizontal_connections_ew['j2']]/2)
    #hwva is the parent cell width perpendicular to the connection
    assert np.allclose(horizontal_connections_ns['hwva'], inset.dis.delr.array[horizontal_connections_ns['j2']])
    assert np.allclose(horizontal_connections_ew['hwva'], inset.dis.delc.array[horizontal_connections_ew['i2']])

    # verify that the model runs
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

    # make a cross section plot for the documentation
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    # plot cross section through an area prone to mis-patch
    # because of inconsistencies in input raster layer surfaces,
    # and sub-200m variability in topography
    #xs_line = [(553000, 390900), (558000, 390900)]
    # line thru the lake
    xs_line = [(553000, 390200), (558000, 390200)]
    xs = flopy.plot.PlotCrossSection(model=m,
                                    line={"line": xs_line}, ax=ax,
                                    geographic_coords=True)
    lc = xs.plot_grid(zorder=4)
    xs2 = flopy.plot.PlotCrossSection(model=inset,
                                    line={"line": xs_line}, ax=ax,
                                    geographic_coords=True)
    lc = xs2.plot_grid(zorder=4)
    #xs2.plot_surface(inset.ic.strt.array, c='b')
    #xs.plot_surface(m.ic.strt.array, c='b')
    plt.savefig('../../../../docs/source/_static/pleasant_vlgr_xsection.png',
                bbox_inches='tight')
    plt.close()


def test_lgr_model_setup(pleasant_lgr_setup_from_yaml, tmpdir):
    m = pleasant_lgr_setup_from_yaml
    assert isinstance(m.inset, dict)
    assert len(m.simulation._models) > 1
    for k, v in m.inset.items():
        # verify that the inset model is part of the same simulation
        # (same memory address)
        assert v.simulation is m.simulation
        assert v.name in m.simulation._models

        # read the options block in the inset name file
        # verify that all of the specified options are there
        name_options = read_mf6_block(v.name_file.filename, 'options')
        specified_options = {'list', 'print_input', 'save_flows', 'newton'}
        assert not any(specified_options.difference(name_options.keys()))
        path, fname = os.path.split(name_options['list'][0])
        assert os.path.abspath(m.model_ws).lower() == os.path.abspath(path).lower()
        assert name_options['newton'][0] == 'under_relaxation'

    # check that the model names were included in the external files
    external_files = glob.glob(os.path.join(m.model_ws, m.external_path, '*'))
    for f in external_files:
        if 'stage_area_volume' in f:
            continue
        assert m.name in f or 'plsnt_lgr_inset' in f

    top3d = np.array([m.dis.top.array] * m.modelgrid.nlay)
    assert np.allclose(m.ic.strt.array, top3d)


@pytest.mark.skip('need to add lake to stand-alone parent model')
def test_lgr_model_run(pleasant_lgr_stand_alone_parent, pleasant_lgr_setup_from_yaml,
                       tmpdir, mf6_exe):
    """Build a MODFLOW-6 version of Pleasant test case
    with LGR around the lake.

    Notes
    -----
    This effectively tests for gwf exchange connections involving inactive
    cells; Pleasant case has many due to layer pinchouts.
    """
    m1 = pleasant_lgr_stand_alone_parent
    m1.simulation.exe_name = mf6_exe

    m2 = pleasant_lgr_setup_from_yaml
    m2.simulation.exe_name = mf6_exe

    # run stand-alone parent and lgr version
    for model in m1, m2:
        success = False
        if exe_exists(mf6_exe):
            success, buff = model.simulation.run_simulation()
            if not success:
                list_file = model.name_file.list.array
                with open(list_file) as src:
                    list_output = src.read()
        assert success, 'model run did not terminate successfully:\n{}'.format(list_output)

    # compare heads from lgr model to stand-alone parent
    kstpkper = (0, 0)
    parent_hdsobj = bf.HeadFile(os.path.join(tmpdir,  'pleasant_lgr_just_parent',
                                                   'plsnt_lgr_parent.hds'))
    parent_heads = parent_hdsobj.get_data(kstpkper=kstpkper)
    inset_hdsobj = bf.HeadFile(os.path.join(tmpdir, 'pleasant_lgr', 'plsnt_lgr_inset.hds'))
    inset_heads = inset_hdsobj.get_data(kstpkper=kstpkper)
    compare_inset_parent_values(inset_heads, parent_heads,
                                m2.modelgrid, m1.modelgrid,
                                nodata=1e30,
                                rtol=0.05
                                )


def test_lgr_load(pleasant_lgr_setup_from_yaml,
                  pleasant_lgr_test_cfg_path):
    m = pleasant_lgr_setup_from_yaml  #deepcopy(pfl_nwt_setup_from_yaml)
    m2 = MF6model.load_from_config(pleasant_lgr_test_cfg_path)
    assert m2.inset['plsnt_lgr_inset'].simulation is m2.simulation

    assert set(m2.get_package_list()).difference(m.get_package_list()) == {'WEL_OBS', 'SFR_OBS', 'CHD_OBS'}
    # can't compare equality if sfr obs was added by SFRmaker, because it won't be listed in m.get_package_list()
    # but will be listed in m2.get_package_list()
    #assert m == m2
