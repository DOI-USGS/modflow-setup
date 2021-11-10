import copy
import glob
import os
import shutil

import flopy
import numpy as np
import pytest

mf6 = flopy.mf6
fm = flopy.modflow
from flopy.utils import binaryfile as bf

from mfsetup import MF6model
from mfsetup.discretization import make_lgr_idomain
from mfsetup.fileio import dump, exe_exists, load, load_cfg, read_mf6_block
from mfsetup.mover import get_sfr_package_connections
from mfsetup.testing import compare_inset_parent_values
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pleasant_lgr_test_cfg_path(project_root_path):
    return project_root_path + '/examples/pleasant_lgr_parent.yml'


@pytest.fixture(scope="function")
def pleasant_lgr_cfg(pleasant_lgr_test_cfg_path):
    cfg = load_cfg(pleasant_lgr_test_cfg_path,
                   default_file='/mf6_defaults.yml')
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
    idomain = make_lgr_idomain(m.modelgrid, inset_model.modelgrid)
    assert idomain.shape == m.modelgrid.shape
    l, b, r, t = inset_model.modelgrid.bounds
    isinset = (m.modelgrid.xcellcenters > l) & \
              (m.modelgrid.xcellcenters < r) & \
              (m.modelgrid.ycellcenters > b) & \
              (m.modelgrid.ycellcenters < t)
    assert idomain[:, isinset].sum() == 0
    assert np.all(idomain[:, ~isinset] >= 1)


def test_lgr_grid_setup(get_pleasant_lgr_parent_with_grid):
    m = get_pleasant_lgr_parent_with_grid
    inset_model = m.inset['plsnt_lgr_inset']
    assert isinstance(inset_model, MF6model)
    assert inset_model.parent is m
    assert isinstance(m.lgr[inset_model.name], flopy.utils.lgrutil.Lgr)
    if os.environ.get('CI', 'false').lower() != 'true':
        m.modelgrid.write_shapefile('../../../modflow-setup-dirty/pleasant_mf6_postproc/shps/pleasant_lgr_parent_grid.shp')
        inset_model.modelgrid.write_shapefile('../../../modflow-setup-dirty/pleasant_mf6_postproc/shps/pleasant_lgr_inset_grid.shp')

    # verify that lgr area was removed from parent idomain
    lgr_idomain = make_lgr_idomain(m.modelgrid, inset_model.modelgrid)
    idomain = m.idomain
    assert idomain[lgr_idomain == 0].sum() == 0

    # todo: add test that grids are aligned


def test_setup_mover(pleasant_lgr_setup_from_yaml):
    m = pleasant_lgr_setup_from_yaml
    assert isinstance(m.simulation.mvr, mf6.ModflowMvr)
    assert os.path.exists(m.simulation.mvr.filename)
    perioddata = read_mf6_block(m.simulation.mvr.filename, 'period')
    assert len(perioddata[1]) == 2
    for model in m, m.inset['plsnt_lgr_inset']:
        options = read_mf6_block(model.sfr.filename, 'options')
        assert 'mover' in options


def test_mover_get_sfr_package_connections(pleasant_lgr_setup_from_yaml):
    m = pleasant_lgr_setup_from_yaml
    parent_reach_data = m.sfrdata.reach_data
    inset_reach_data = m.inset['plsnt_lgr_inset'].sfrdata.reach_data
    to_inset, to_parent = get_sfr_package_connections(parent_reach_data, inset_reach_data,
                                                      distance_threshold=200)
    assert len(to_inset) == 0
    # verify that the last reaches in the two segments are keys
    last_reaches = m.inset['plsnt_lgr_inset'].sfrdata.reach_data.groupby('iseg').last().rno
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
    # {inset_reach: parent_reach, ...}
    assert to_parent == {23: 10, 25: 1}


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

    # todo: test_lgr_model_setup could use some more tests; although many potential issues will be tested by test_lgr_model_run


#def test_stand_alone_parent(pleasant_lgr_stand_alone_parent):
#    # todo: move test_stand_alone_parent test to test_lgr_model_run
#    j=2


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

    assert set(m2.get_package_list()).difference(m.get_package_list()) == {'SFR_OBS', 'CHD_OBS'}
    # can't compare equality if sfr obs was added by SFRmaker, because it won't be listed in m.get_package_list()
    # but will be listed in m2.get_package_list()
    #assert m == m2
