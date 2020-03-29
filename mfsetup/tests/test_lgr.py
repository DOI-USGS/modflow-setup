import copy
import os
import numpy as np
import pytest
import flopy
mf6 = flopy.mf6
fm = flopy.modflow
from mfsetup import MF6model
from mfsetup.discretization import make_lgr_idomain
from mfsetup.fileio import load_cfg, load_array, exe_exists
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
        assert v.name in m.simulation._models
    # todo: test_lgr_model_setup could use some more tests; although many potential issues will be tested by test_lgr_model_run


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