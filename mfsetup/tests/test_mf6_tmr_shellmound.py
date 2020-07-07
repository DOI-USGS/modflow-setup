import os

import flopy
import numpy as np
import pandas as pd
import pytest
from flopy import mf6 as mf6

from mfsetup import MF6model
from mfsetup.fileio import exe_exists, load_array, load_cfg
from mfsetup.grid import MFsetupGrid
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def shellmound_tmr_cfg_path(project_root_path):
    return project_root_path + '/mfsetup/tests/data/shellmound_tmr_inset.yml'


@pytest.fixture(scope="function")
def shellmound_tmr_datapath(shellmound_tmr_cfg_path):
    return os.path.join(os.path.split(shellmound_tmr_cfg_path)[0], 'shellmound')


@pytest.fixture(scope="module")
def shellmound_tmr_cfg(shellmound_tmr_cfg_path):
    cfg = load_cfg(shellmound_tmr_cfg_path, default_file='/mf6_defaults.yml')
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['simulation']['sim_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def shellmound_tmr_simulation(shellmound_tmr_cfg):
    cfg = shellmound_tmr_cfg.copy()
    kwargs = get_input_arguments(cfg['simulation'], mf6.MFSimulation)
    sim = mf6.MFSimulation(**kwargs)
    return sim


@pytest.fixture(scope="function")
def shellmound_tmr_model(shellmound_tmr_cfg, shellmound_tmr_simulation):
    cfg = shellmound_tmr_cfg.copy()
    cfg['model']['simulation'] = shellmound_tmr_simulation
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    return m


@pytest.fixture(scope="function")
def shellmound_tmr_model_with_grid(shellmound_tmr_model):
    model = shellmound_tmr_model  #deepcopy(shellmound_tmr_model)
    model.setup_grid()
    return model


@pytest.fixture(scope="function")
def shellmound_tmr_model_with_dis(shellmound_tmr_model_with_grid):
    print('pytest fixture model_with_grid')
    m = shellmound_tmr_model_with_grid  #deepcopy(pfl_nwt_with_grid)
    m.setup_tdis()
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="module")
def shellmound_tmr_model_setup(shellmound_tmr_cfg_path):
    m = MF6model.setup_from_yaml(shellmound_tmr_cfg_path)
    m.write_input()
    if hasattr(m, 'sfr'):
        sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
        m.sfrdata.write_package(sfr_package_filename,
                                    version='mf6',
                                    options=['save_flows',
                                             'BUDGET FILEOUT {}.sfr.cbc'.format(m.name),
                                             'STAGE FILEOUT {}.sfr.stage.bin'.format(m.name),
                                             # 'OBS6 FILEIN {}'.format(sfr_obs_filename)
                                             # location of obs6 file relative to sfr package file (same folder)
                                             ]
                                    )
    return m


@pytest.fixture(scope="module")
def shellmound_tmr_model_setup_and_run(shellmound_tmr_model_setup, mf6_exe):
    m = shellmound_tmr_model_setup
    m.simulation.exe_name = mf6_exe

    dis_idomain = m.dis.idomain.array.copy()
    for i, d in enumerate(m.cfg['dis']['griddata']['idomain']):
        arr = load_array(d['filename'])
        assert np.array_equal(m.idomain[i], arr)
        assert np.array_equal(dis_idomain[i], arr)
    success = False
    if exe_exists(mf6_exe):
        success, buff = m.simulation.run_simulation()
        if not success:
            list_file = m.name_file.list.array
            with open(list_file) as src:
                list_output = src.read()
    assert success, 'model run did not terminate successfully:\n{}'.format(list_output)
    return m


def test_irregular_perimeter_boundary(shellmound_tmr_model_with_dis):
    m = shellmound_tmr_model_with_dis
    chd = m.setup_perimeter_boundary()
    j=2


def test_set_parent_model(shellmound_tmr_model_with_dis):
    m = shellmound_tmr_model_with_dis
    assert isinstance(m.parent, mf6.MFModel)
    assert isinstance(m.parent.perioddata, pd.DataFrame)
    assert isinstance(m.parent.modelgrid, MFsetupGrid)
    assert m.parent.modelgrid.nrow == m.parent.dis.nrow.array
    assert m.parent.modelgrid.ncol == m.parent.dis.ncol.array
    assert m.parent.modelgrid.nlay == m.parent.dis.nlay.array


def test_sfr_riv_setup(shellmound_tmr_model_with_dis):
    m = shellmound_tmr_model_with_dis
    m.setup_sfr()
    assert isinstance(m.riv, mf6.ModflowGwfriv)
    rivdata_file = m.cfg['riv']['output_files']['rivdata_file'].format(m.name)
    rivdata = pd.read_csv(rivdata_file)
    for line_id in m.cfg['sfr']['to_riv']:
        assert line_id not in m.sfrdata.reach_data.line_id.values
        assert line_id in rivdata.line_id.values
    assert 'Yazoo River' in rivdata.name.unique()


def test_model_setup(shellmound_tmr_model_setup):
    m = shellmound_tmr_model_setup
    specified_packages = m.cfg['model']['packages']
    for pckg in specified_packages:
        package = getattr(m, pckg)
        assert isinstance(package, flopy.pakbase.PackageInterface)


def test_model_setup_and_run(shellmound_tmr_model_setup_and_run):
    m = shellmound_tmr_model_setup_and_run
    # todo: add test comparing shellmound parent heads to tmr heads
