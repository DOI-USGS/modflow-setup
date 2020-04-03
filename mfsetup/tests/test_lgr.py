import copy
import os
import numpy as np
import pytest
import flopy
mf6 = flopy.mf6
fm = flopy.modflow
from mfsetup import MF6model
from mfsetup.discretization import make_lgr_idomain
from mfsetup.fileio import load_cfg, exe_exists, load, dump, read_mf6_block
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def pleasant_lgr_test_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/pleasant_lgr_parent.yml'


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
    sim = mf6.MFSimulation(**cfg['simulation'])
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
    for model in m, m.inset['plsnt_lgr_inset']:
        if hasattr(model, 'sfr'):
            sfr_package_filename = os.path.join(model.model_ws, model.sfr.filename)
            model.sfrdata.write_package(sfr_package_filename,
                                        version='mf6'
                                        )
    return m


@pytest.fixture(scope="function")
def pleasant_lgr_stand_alone_parent(pleasant_lgr_test_cfg_path):
    """Stand-alone version of lgr parent model for comparing with LGR results.
    """
    # Edit the configuration file before the file paths within it are converted to absolute
    # (model.load_cfg converts the file paths)
    cfg = load(pleasant_lgr_test_cfg_path)
    del cfg['setup_grid']['lgr']
    cfg['simulation']['sim_ws'] = 'pleasant_lgr_just_parent'

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
    if hasattr(m, 'sfr'):
        sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
        m.sfrdata.write_package(sfr_package_filename,
                                    version='mf6'
                                    )
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
    assert np.all(idomain[:, ~isinset] == 1)


def test_lgr_grid_setup(get_pleasant_lgr_parent_with_grid):
    m = get_pleasant_lgr_parent_with_grid
    inset_model = m.inset['plsnt_lgr_inset']
    assert isinstance(inset_model, MF6model)
    assert inset_model.parent is m
    assert isinstance(m.lgr[inset_model.name], flopy.utils.lgrutil.Lgr)
    if os.environ.get('CI', 'false').lower() != 'true':
        m.modelgrid.write_shapefile('../../../../../modflow-setup-dirty/pleasant_mf6_postproc/shps/pleasant_lgr_parent_grid.shp')
        inset_model.modelgrid.write_shapefile('../../../../../modflow-setup-dirty/pleasant_mf6_postproc/shps/pleasant_lgr_inset_grid.shp')

    # verify that lgr area was removed from parent idomain
    lgr_idomain = make_lgr_idomain(m.modelgrid, inset_model.modelgrid)
    idomain = m.idomain
    assert idomain[lgr_idomain == 0].sum() == 0

    # todo: add test that grids are aligned


def test_lgr_model_setup(pleasant_lgr_setup_from_yaml):
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
        assert os.path.abspath(m.model_ws).lower() == path.lower()
        assert name_options['newton'][0] == 'under_relaxation'
    # todo: test_lgr_model_setup could use some more tests; although many potential issues will be tested by test_lgr_model_run


def test_stand_alone_parent(pleasant_lgr_stand_alone_parent):
    # todo: move test_stand_alone_parent test to test_lgr_model_run
    j=2


def test_lgr_model_run(pleasant_lgr_setup_from_yaml, mf6_exe):
    """Build a MODFLOW-6 version of Pleasant test case
    with LGR around the lake.

    Notes
    -----
    This effectively tests for gwf exchange connections involving inactive
    cells; Pleasant case has many due to layer pinchouts.
    """
    m = pleasant_lgr_setup_from_yaml
    m.simulation.exe_name = mf6_exe

    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m