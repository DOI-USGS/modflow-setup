import copy
import os
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pytest
from flopy import mf6 as mf6
from flopy.utils import binaryfile as bf
from scipy.interpolate import griddata

from mfsetup import MF6model
from mfsetup.fileio import exe_exists, load_array, load_cfg
from mfsetup.grid import MFsetupGrid
from mfsetup.interpolate import Interpolator
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope="session")
def shellmound_tmr_cfg_path(project_root_path):
    return project_root_path / 'mfsetup/tests/data/shellmound_tmr_inset.yml'


@pytest.fixture(scope="function")
def shellmound_tmr_datapath(shellmound_tmr_cfg_path):
    return os.path.join(os.path.split(shellmound_tmr_cfg_path)[0], 'shellmound')


@pytest.fixture(scope="module")
def shellmound_tmr_cfg(shellmound_tmr_cfg_path):
    cfg = load_cfg(shellmound_tmr_cfg_path, default_file='mf6_defaults.yml')
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
def shellmound_tmr_model_with_refined_dis(shellmound_tmr_cfg, shellmound_tmr_simulation):
    print('pytest fixture model_with_refined_dis')
    cfg = shellmound_tmr_cfg.copy()
    cfg['model']['simulation'] = shellmound_tmr_simulation
    # cut the delr and delc in half
    cfg['dis']['griddata']['delr'] /= 2
    cfg['dis']['griddata']['delc'] /= 2
    cfg['dis']['dimensions']['nrow'] *= 2
    cfg['dis']['dimensions']['ncol'] *= 2

    # use all SFR, like parent model (for accurate comparison)
    del cfg['sfr']['sfrmaker_options']['to_riv']

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()
    m.setup_tdis()
    m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


@pytest.fixture(scope="function")
def shellmound_tmr_small_rectangular_inset(shellmound_tmr_cfg, shellmound_tmr_simulation):
    cfg = shellmound_tmr_cfg.copy()
    cfg['model']['simulation'] = shellmound_tmr_simulation

    # create small rectangular domain in middle of parent model
    # that exactly aligns with parent cells
    spacing = 50
    # pad the inset model so that constant head cells are outside of the window in the parent
    cfg['setup_grid']['xoff'] = 521955.0 #- spacing
    cfg['setup_grid']['yoff'] = 1187285 #- spacing
    cfg['dis']['griddata']['delr'] = spacing
    cfg['dis']['griddata']['delc'] = spacing
    # pad the inset model so that constant head cells are outside of the window in the parent
    cfg['dis']['dimensions']['nrow'] = int(1000 * 5 / spacing) #+ 2
    cfg['dis']['dimensions']['ncol'] = int(1000 * 5 / spacing) #+ 2

    # use all SFR, like parent model (for accurate comparison)
    del cfg['sfr']['sfrmaker_options']['to_riv']
    # remove existing perimeter CHD setup
    del cfg['chd']

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf, exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()
    m.setup_tdis()
    #m.cfg['dis']['remake_top'] = True
    dis = m.setup_dis()
    return m


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
    #if hasattr(m, 'sfr'):
    #    sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
    #    m.sfrdata.write_package(sfr_package_filename,
    #                                version='mf6',
    #                                options=['save_flows',
    #                                         'BUDGET FILEOUT {}.sfr.cbc'.format(m.name),
    #                                         'STAGE FILEOUT {}.sfr.stage.bin'.format(m.name),
    #                                         # 'OBS6 FILEIN {}'.format(sfr_obs_filename)
    #                                         # location of obs6 file relative to sfr package file (same folder)
    #                                         ]
    #                                )
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


@pytest.mark.parametrize('from_binary', (False, True))
def test_ic_setup(shellmound_tmr_model_with_dis, from_binary):
    """Test starting heads setup from model top or parent model head solution
    (MODFLOW binary output)."""
    m = copy.deepcopy(shellmound_tmr_model_with_dis)
    parent_ws = Path(m.cfg['parent']['model_ws'])
    binaryfile = str(parent_ws / f'{m.parent.name}.hds')
    if from_binary:
        config = {'strt': {'from_parent': {'binaryfile': binaryfile,
                                           'period': 0
                                           }}}
        m.cfg['ic']['source_data'] = config
    ic = m.setup_ic()
    ic.write()
    assert os.path.exists(os.path.join(m.model_ws, ic.filename))
    assert isinstance(ic, mf6.ModflowGwfic)
    assert ic.strt.array.shape == m.dis.botm.array.shape

    assert m.ic.strt.array[m.dis.idomain.array > 0].min() > 0
    assert m.ic.strt.array[m.dis.idomain.array > 0].max() < 50


def test_predefined_stress_period_data(shellmound_tmr_model, test_data_path):
    perioddata = shellmound_tmr_model.perioddata
    expected_spd_file = test_data_path /\
        shellmound_tmr_model.cfg['tdis']['perioddata']['csvfile']['filename']
    expected_spd = pd.read_csv(expected_spd_file)
    assert np.all(perioddata[expected_spd.columns[:-1]] == expected_spd.iloc[:, :-1])


def test_irregular_perimeter_head_boundary(shellmound_tmr_model_with_dis, test_data_path, tmpdir):
    m = shellmound_tmr_model_with_dis

    if 'wel' in m.cfg:
        del m.cfg['wel']
    head_cfg = {
        'perimeter_boundary': {
            'shapefile': test_data_path / 'shellmound/tmr_parent/gis/irregular_boundary.shp',
            'parent_head_file': test_data_path / 'shellmound/tmr_parent/shellmound.hds'
            },
        'mfsetup_options': {
            'external_files': True,
            'external_filename_fmt': 'chd_{:03d}.dat'
            }
    }
    chd = m.setup_chd(**head_cfg, **head_cfg['mfsetup_options'])

    ra = chd.stress_period_data.array[0]
    kh, ih, jh = zip(*ra['cellid'])
    # all specified heads should be active
    assert np.all(m.idomain[kh, ih, jh] > 0)
    assert len(set(ra['cellid'])) == len(ra)

    bcells = m.tmr.inset_boundary_cells.copy()
    k, i, j = bcells.k.values, bcells.i.values, bcells.j.values
    bcells['idomain'] = m.idomain[k, i, j]
    bcells['botm'] = m.dis.botm.array[k, i, j]

    # get the parent head values
    hdsobj = bf.HeadFile(m.tmr.parent_head_file, precision='double')
    parent_heads = hdsobj.get_data(kstpkper=(0, 0))

    # pad the parent heads on the top and bottom
    # (as in tmr.get_inset_boundary_values())
    parent_heads = np.pad(parent_heads, pad_width=1, mode='edge')[:, 1:-1, 1:-1]

    # interpolate inset boundary heads using interpolate method in Tmr class
    # apparently we can't just use griddata to do this because
    # 'linear' leaves out some values (presumably due to weights that
    # don't exactly sum to 1 because of floating point error)
    # and 'nearest' leaves in too many values (presumably due to extrapolation)
    # todo: should probably look into a method that is friendlier
    #  to interpolating data from regular grids (the parent model),
    # such as interpn (which xarray uses) although regular grid methods
    # wouldn't work for 3 or 4D interpolation because z is irregular

    # create an interpolator instance
    cell_centers_interp = Interpolator(m.tmr.parent_xyzcellcenters,
                                       m.tmr.inset_boundary_cells[['x', 'y', 'z']].T.values,
                                       d=3,
                                       source_values_mask=m.tmr._source_grid_mask)

    #bheads_tmr = m.tmr.interpolate_values(parent_heads.ravel(), method='linear')
    bheads_tmr = cell_centers_interp.interpolate(parent_heads, method='linear')

    # x, y, z locations of parent model head values
    px, py, pz = m.tmr.parent_xyzcellcenters

    # x, y, z locations of inset model boundary cells
    x, y, z = bcells[['x', 'y', 'z']].T.values

    bcells['bhead_tmr'] = bheads_tmr
    # only include valid heads, for cells that are active and not above the water table
    valid_tmr = (bcells['bhead_tmr'] < 1e10) & (bcells['bhead_tmr'] > -1e10) & \
            (bcells['idomain'] > 0) & (bcells['bhead_tmr'] > bcells['botm'])

    # valid bcells derived above should have same collection of cell numbers
    # as recarray in constant head package
    assert len(set(bcells.loc[valid_tmr, 'cellid'])) == len(ra)

    # additional code to generate layers for visual comparison in a GIS environment
    export_layers = False
    if export_layers:
        from mfexport import export, export_array
        export(m, m.modelgrid, 'chd', pdfs=False, output_path=tmpdir)
        export(m, m.modelgrid, 'dis', 'idomain', pdfs=False, output_path=tmpdir)
        max_extent = np.sum(m.idomain == 1, axis=0) > 0
        rpath = Path(tmpdir, 'shellmound_tmr_inset/rasters')
        rpath.mkdir(parents=True, exist_ok=True)
        export_array(rpath / 'max_idm_extent.tif',
                     max_extent, modelgrid=m.modelgrid)
        parent_max_extent = np.sum(m.parent.dis.idomain.array == 1, axis=0) > 0
        export_array(rpath / 'parent_max_idm_extent.tif',
                     parent_max_extent, modelgrid=m.parent.modelgrid)

        # and for comparison plot of different interpolation results
        # nearest neighbor (can extrapolate)
        bheads_nearest = griddata((px, py, pz), parent_heads.ravel(),
                                  (x, y, z), method='nearest')
        bheads_nearest[(bheads_nearest > 1e10) | (bheads_nearest < -1e10)] = np.nan
        # linear method with griddata
        # (apparently prone to some spurious values that are rectified
        # by computing the weights manually and then rounding them)
        bheads_griddata_linear = griddata((px, py, pz), parent_heads.ravel(),
                                          (x, y, z), method='linear')
        bheads_griddata_linear[(bheads_griddata_linear > 1e10) | \
                               (bheads_griddata_linear < -1e10)] = np.nan
        bheads_tmr[(bheads_tmr > 1e10) | (bheads_tmr < -1e10)] = np.nan
        from matplotlib import pyplot as plt
        plt.plot(bheads_nearest, label='bheads_nearest')
        plt.plot(bheads_tmr, label='bheads_tmr')
        plt.plot(bheads_griddata_linear, label='bheads_linear')
        ax = plt.gca()
        #ax.set_ylim(33, 33.6); ax.set_xlim(0, 350)
        ax.legend()


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
    # BCs array should include 4 (SFR) and 5 (RIV)
    assert set(list(np.unique(m.isbc))) == {0, 4, 5}
    # riv cells should all be in isbc array as 5s
    k, i, j = zip(*m.riv.stress_period_data.array[0]['cellid'])
    assert set(m.isbc[k, i, j]) == {5}


@pytest.mark.skip(reason="still working on this one")
def test_perimeter_boundary(shellmound_tmr_model_with_dis):
    m = shellmound_tmr_model_with_dis
    m.setup_chd(**m.cfg['chd'], **m.cfg['chd']['mfsetup_options'])


def test_model_setup(shellmound_tmr_model_setup):
    m = shellmound_tmr_model_setup
    specified_packages = m.cfg['model']['packages']
    for pckg in specified_packages:
        package = getattr(m, pckg)
        assert isinstance(package, flopy.pakbase.PackageInterface)


def test_model_setup_and_run(shellmound_tmr_model_setup_and_run):
    m = shellmound_tmr_model_setup_and_run
    # todo: add test comparing shellmound parent heads to tmr heads
    plot_figure = False
    if plot_figure:
        from matplotlib import pyplot as plt
        parent_headfile = Path(m.parent.model_ws) / f"{m.parent.name}.hds"
        parent_hds = bf.HeadFile(parent_headfile)
        parent_heads = parent_hds.get_data(kstpkper=(0, 0))
        parent_heads = np.ma.masked_array(parent_heads, mask=parent_heads > 1e5)
        inset_hds = bf.HeadFile(Path(m.model_ws).absolute() / f"{m.name}.hds")
        inset_heads = inset_hds.get_data(kstpkper=(0, 0))
        inset_heads = np.ma.masked_array(inset_heads, mask=inset_heads > 1e5)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        pmv = flopy.plot.PlotMapView(model=m.parent, ax=ax)
        arr = pmv.plot_array(parent_heads[3])
        contours = pmv.contour_array(parent_heads[3], colors="white", levels=np.linspace(30, 38, 9))
        ax.clabel(contours, fmt="%2.2f")
        plt.colorbar(arr, shrink=0.5, ax=ax)
        ax.set_title("Simulated Heads")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        pmv = flopy.plot.PlotMapView(model=m, ax=ax)
        arr = pmv.plot_array(inset_heads[3], vmin=parent_heads.min(), vmax=parent_heads.max())
        contours = pmv.contour_array(inset_heads[3], colors="red", levels=np.linspace(30, 38, 9))
        ax.clabel(contours, fmt="%2.2f")
        plt.colorbar(arr, shrink=0.5, ax=ax)
        ax.set_title("Simulated Heads")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.savefig(m.model_ws / 'head_comp.pdf')
