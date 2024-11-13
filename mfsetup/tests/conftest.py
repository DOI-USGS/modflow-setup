import copy
import os
import platform
import shutil
from pathlib import Path

import flopy
import pytest

fm = flopy.modflow
mf6 = flopy.mf6
from mfsetup import MF6model, MFnwtModel
from mfsetup.fileio import exe_exists, load, load_cfg
from mfsetup.tests.test_pleasant_mf6_inset import (
    get_pleasant_mf6,
    get_pleasant_mf6_with_dis,
    get_pleasant_mf6_with_grid,
    pleasant_mf6_cfg,
    pleasant_mf6_setup_from_yaml,
    pleasant_mf6_test_cfg_path,
    pleasant_simulation,
)
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def project_root_path():
    filepath = os.path.split(os.path.abspath(__file__))[0]
    filepath = os.path.normpath(os.path.join(filepath, '../../'))
    return Path(filepath)


@pytest.fixture(scope="session")
def test_data_path(project_root_path):
    """Root folder for the project (with pyproject.toml),
    two levels up from the location of this file.
    """
    return Path(project_root_path, 'mfsetup', 'tests', 'data')


@pytest.fixture(scope="session", autouse=True)
def tmpdir(project_root_path):
    folder = project_root_path / 'mfsetup/tests/tmp'
    reset = True
    if reset:
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
    return Path(folder)


def get_model(model):
    """Fetch a fresh copy of a model from a fixture;
    updating the workspace if needed."""
    m = copy.deepcopy(model)
    model_ws = m._abs_model_ws
    cwd = os.path.normpath(os.path.abspath(os.getcwd()))
    if cwd != model_ws:
        os.chdir(model_ws)
        print('changing workspace from {} to {}'.format(cwd, model_ws))
    return m


@pytest.fixture(scope="session")
def bin_path(project_root_path):
    bin_path = os.path.join(project_root_path, "bin")
    platform_info = platform.platform().lower()
    if "linux" in platform_info:
        bin_path = os.path.join(bin_path, "linux")
    elif "mac" in platform_info or "darwin" in platform_info:
        bin_path = os.path.join(bin_path, "mac")
    else:
        bin_path = os.path.join(bin_path, "win")
    return Path(bin_path)


@pytest.fixture(scope="session")
def mf6_exe(bin_path):
    _, version = os.path.split(bin_path)
    exe_name = 'mf6'
    if version == "win":
        exe_name += '.exe'
    return os.path.join(bin_path, exe_name)


@pytest.fixture(scope="session")
def zbud6_exe(bin_path):
    _, version = os.path.split(bin_path)
    exe_name = 'zbud6'
    if version == "win":
        exe_name += '.exe'
    return os.path.join(bin_path, exe_name)


@pytest.fixture(scope="session")
def mfnwt_exe(bin_path):
    _, version = os.path.split(bin_path)
    exe_name = 'mfnwt'
    if version == "win":
        exe_name += '.exe'
    return os.path.join(bin_path, exe_name)


@pytest.fixture(scope="session")
def demfile(project_root_path):
    return project_root_path / 'mfsetup/tests/data/shellmound/rasters/meras_100m_dem.tif'


@pytest.fixture(scope="session")
def pfl_nwt_test_cfg_path(project_root_path):
    return project_root_path / 'mfsetup/tests/data/pfl_nwt_test.yml'


@pytest.fixture(scope="function")
def pfl_nwt_cfg(pfl_nwt_test_cfg_path):
    cfg = load_cfg(pfl_nwt_test_cfg_path,
                   default_file='mfnwt_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['model']['model_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def pfl_nwt(pfl_nwt_cfg):
    print('pytest fixture pfl_nwt')
    cfg = pfl_nwt_cfg.copy()
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="function")
def pfl_nwt_with_grid(pfl_nwt):
    print('pytest fixture pfl_nwt_with_grid')
    m = pfl_nwt  #deepcopy(pfl_nwt)
    m.setup_grid()
    return pfl_nwt


@pytest.fixture(scope="function")
def pfl_nwt_with_dis(pfl_nwt_with_grid):
    print('pytest fixture pfl_nwt_with_dis')
    m = pfl_nwt_with_grid  #deepcopy(pfl_nwt_with_grid)
    m.cfg['dis']['remake_arrays'] = True
    m.cfg['dis']['regrid_top_from_dem'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="function")
def pfl_nwt_with_dis_bas6(pfl_nwt_with_dis):
    print('pytest fixture pfl_nwt_with_dis_bas6')
    bas = pfl_nwt_with_dis.setup_bas6()
    return pfl_nwt_with_dis


@pytest.fixture(scope="session")
def shellmound_cfg_path(project_root_path):
    return project_root_path / 'mfsetup/tests/data/shellmound.yml'


@pytest.fixture(scope="function")
def shellmound_datapath(shellmound_cfg_path):
    return os.path.join(os.path.split(shellmound_cfg_path)[0], 'shellmound')


@pytest.fixture(scope="module")
def shellmound_cfg(shellmound_cfg_path):
    cfg = load_cfg(shellmound_cfg_path, default_file='mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def shellmound_simulation(shellmound_cfg):
    cfg = shellmound_cfg.copy()
    kwargs = shellmound_cfg['simulation'].copy()
    kwargs.update(cfg['simulation']['options'])
    kwargs = get_input_arguments(kwargs, mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    return sim


@pytest.fixture(scope="function")
def shellmound_model(shellmound_cfg, shellmound_simulation):
    cfg = shellmound_cfg.copy()
    cfg['model']['simulation'] = shellmound_simulation
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def shellmound_model_with_grid(shellmound_model):
    model = shellmound_model  #deepcopy(shellmound_model)
    model.setup_grid()
    return model


@pytest.fixture(scope="function")
def shellmound_model_with_dis(shellmound_model_with_grid):
    print('pytest fixture model_with_grid')
    m = shellmound_model_with_grid  #deepcopy(pfl_nwt_with_grid)
    m.setup_tdis()
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="session")
def pleasant_nwt_test_cfg_path(project_root_path):
    return project_root_path / 'mfsetup/tests/data/pleasant_nwt_test.yml'


@pytest.fixture(scope="session")
def pleasant_nwt_cfg(pleasant_nwt_test_cfg_path):
    cfg = load_cfg(pleasant_nwt_test_cfg_path, default_file='mfnwt_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['model']['model_ws'], 'gis')
    return cfg


@pytest.fixture(scope="session")
def get_pleasant_nwt(pleasant_nwt_cfg):
    print('creating Pleasant Lake MFnwtModel instance from cfgfile...')
    cfg = pleasant_nwt_cfg.copy()
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="session")
def get_pleasant_nwt_with_grid(get_pleasant_nwt):
    print('creating Pleasant Lake MFnwtModel instance with grid...')
    m = copy.deepcopy(get_pleasant_nwt)
    m.setup_grid()
    return m


@pytest.fixture(scope="session")
def get_pleasant_nwt_with_dis(get_pleasant_nwt_with_grid):
    print('creating Pleasant Lake MFnwtModel instance with dis package...')
    m = copy.deepcopy(get_pleasant_nwt_with_grid)  #deepcopy(pleasant_nwt_with_grid)
    m.cfg['dis']['remake_arrays'] = True
    m.cfg['dis']['regrid_top_from_dem'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="function")
def get_pleasant_nwt_with_dis_bas6(get_pleasant_nwt_with_dis):
    print('creating Pleasant Lake MFnwtModel instance with dis and bas6 packages...')
    #m = copy.deepcopy(get_pleasant_nwt_with_dis)
    m = get_model(get_pleasant_nwt_with_dis)
    bas = m.setup_bas6()
    return m


@pytest.fixture(scope="function")
def pleasant_nwt_setup_from_yaml(pleasant_nwt_test_cfg_path):
    m = MFnwtModel.setup_from_yaml(pleasant_nwt_test_cfg_path)
    m.write_input()
    # verify that observation data were added and written
    sfr_package_filename = os.path.join(m.model_ws, m.sfr.file_name[0])
    #m.sfrdata.write_package(sfr_package_filename)
    return m


@pytest.fixture(scope="function")
def pleasant_nwt_model_run(pleasant_nwt_setup_from_yaml, mfnwt_exe):
    m = pleasant_nwt_setup_from_yaml
    m.exe_name = mfnwt_exe
    success = False
    if exe_exists(mfnwt_exe):
        success, buff = m.run_model(silent=False)
        if not success:
            list_file = m.lst.fn_path
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


@pytest.fixture(scope="function")
def pleasant_nwt(get_pleasant_nwt):
    return get_model(get_pleasant_nwt)


@pytest.fixture(scope="function")
def pleasant_nwt_with_grid(get_pleasant_nwt_with_grid):
    return get_model(get_pleasant_nwt_with_grid)


@pytest.fixture(scope="function")
def pleasant_nwt_with_dis(get_pleasant_nwt_with_dis):
    return get_model(get_pleasant_nwt_with_dis)


@pytest.fixture(scope="function")
def pleasant_nwt_with_dis_bas6(get_pleasant_nwt_with_dis_bas6):
    return get_model(get_pleasant_nwt_with_dis_bas6)


@pytest.fixture(scope="function")
def full_pleasant_nwt(pleasant_nwt_setup_from_yaml):
    m = get_model(pleasant_nwt_setup_from_yaml)
    #m.write_input()
    return m


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['shellmound_model_with_dis',
                        'pfl_nwt_with_dis_bas6'])
def models_with_dis(request,
                    shellmound_model_with_dis,
                    pfl_nwt_with_dis_bas6):
    return {'shellmound_model_with_dis': shellmound_model_with_dis,
            'pfl_nwt_with_dis_bas6': pfl_nwt_with_dis_bas6}[request.param]


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['mfnwt_exe',
                        'mf6_exe'])
def modflow_executable(request, mfnwt_exe, mf6_exe):
    return {'mfnwt_exe': mfnwt_exe,
            'mf6_exe': mf6_exe}[request.param]


# fixture to feed multiple model fixtures to a test
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['full_pleasant_nwt',
                        'pleasant_mf6_setup_from_yaml'])
def pleasant_model(request,
                   full_pleasant_nwt,
                   pleasant_mf6_setup_from_yaml):
    """MODFLOW-NWT and MODFLOW-6 versions of Pleasant Lake
    test case with all of the packages"""
    return {'full_pleasant_nwt': full_pleasant_nwt,
            'pleasant_mf6_setup_from_yaml': pleasant_mf6_setup_from_yaml}[request.param]


@pytest.fixture(params=['pfl_nwt',
                        'pleasant_nwt',
                        'get_pleasant_mf6'
                        ])
def basic_model_instance(request,
                   pfl_nwt,
                   pleasant_nwt,
                   get_pleasant_mf6
                   ):
    """Just some model instances with no packages.
    """
    return {'pfl_nwt': pfl_nwt,
            'pleasant_nwt': pleasant_nwt,
            'get_pleasant_mf6': get_pleasant_mf6
            }[request.param]


@pytest.fixture
def cfg_2x2x3_with_dis(project_root_path):
    cfg = load(Path(project_root_path) / 'mfsetup/mf6_defaults.yml')
    specified_cfg = {
        'setup_grid': {
            'xoff': 0.,
            'yoff': 0.,
            'nrow': 3,
            'ncol': 2,
            'dxy': 100.,
            'rotation': 0.,
            'crs': 26915
                    },
        'dis': {
            'options': {
                'length_units': 'meters'
            },
            'dimensions': {
                'nlay': 2,
                'nrow': 3,
                'ncol': 2
            },
            'griddata': {
                'delr': 100.,
                'delc': 100.,
                'top': 20.,
                'botm': [15., 0.]
            },
            }}
    cfg.update(specified_cfg)
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    cfg['model']['simulation'] = sim
    return cfg
