import copy
import os

import fiona
import geopandas as gpd
import numpy as np
import pyproj
import pytest
from flopy import mf6
from flopy.utils import binaryfile as bf
from flopy.utils.geometry import rotate
from flopy.utils.mfreadnam import attribs_from_namfile_header
from gisutils import get_authority_crs, shp2df

from mfsetup import MF6model
from mfsetup.fileio import dump, load, load_modelgrid
from mfsetup.grid import (
    MFsetupGrid,
    get_cellface_midpoint,
    get_ij,
    get_nearest_point_on_grid,
    get_point_on_national_hydrogeologic_grid,
    rasterize,
)
from mfsetup.testing import point_is_on_nhg
from mfsetup.tests.test_mf6_tmr_shellmound import (
    shellmound_tmr_cfg,
    shellmound_tmr_cfg_path,
)
from mfsetup.units import convert_length_units
from mfsetup.utils import get_input_arguments


@pytest.fixture(scope='module')
def modelgrid():
    return MFsetupGrid(xoff=100., yoff=200., angrot=20.,
                       proj_str='epsg:3070',
                       delr=np.ones(10), delc=np.ones(2))


def test_grid_eq(modelgrid):
    grid2 = copy.deepcopy(modelgrid)
    assert modelgrid == grid2


def test_grid_init():
    ncol = 394
    nrow = 414
    kwargs = {
    "delc": np.ones(nrow) * 5280.0 * .3048,
    "delr": np.ones(ncol) * 5280.0 * .3048,
    "epsg": 5070,
    "proj_str": "+init=epsg:5070",
    "angrot": 0.0,
    "xul": 178389.0,
    "yul": 1604780.4160000002,
    "lenuni": 1
    }
    grid = MFsetupGrid(**kwargs)
    assert np.allclose(grid.yoffset, grid.yul - grid.nrow * 5280 * .3048)
    assert isinstance(grid.crs, pyproj.crs.CRS)
    assert grid.crs.srs == 'EPSG:5070'

    assert np.all(grid.dataframe.columns == ['i', 'j', 'geometry'])
    kwargs['botm'] = np.ones((1, nrow, ncol)) * 1
    kwargs['top'] = np.ones((nrow, ncol)) * 2
    grid = MFsetupGrid(**kwargs)
    assert np.all(grid.dataframe.columns == ['k', 'i', 'j', 'geometry'])


def test_grid_write_shapefile(modelgrid, tmpdir):
    # test writing the grid cells
    filename = os.path.join(tmpdir, 'grid.shp')
    modelgrid.write_shapefile(filename)
    with fiona.open(filename) as src:
        assert src.crs['init'] == 'epsg:3070'
        assert np.allclose(src.bounds, modelgrid.bounds)
    df = shp2df(filename)
    i, j = np.indices((modelgrid.nrow, modelgrid.ncol))
    assert np.array_equal(np.arange(len(df), dtype=int), df.node.values)
    assert np.array_equal(i.ravel(), df.i.values)
    assert np.array_equal(j.ravel(), df.j.values)

    # test writing the bounding box
    bbox_filename = os.path.join(tmpdir, 'grid_bbox.shp')
    modelgrid.write_shapefile(bbox_filename)
    with fiona.open(filename) as src:
        assert src.crs['init'] == 'epsg:3070'
        assert np.allclose(src.bounds, modelgrid.bounds)


@pytest.mark.parametrize('xul,yul,height,width,dx,dy,rotation,input,expected', ((0, 0, 3, 2, 1, -1, 45,
                                                                                 (0, -0.7),
                                                                                 (0, -np.sqrt(2)/2)),
                                                                                (0, 0, 2, 2, 5, -10, 0,
                                                                                 (4.9, -9),
                                                                                 (2.5, -5)),
                                                                                (0, 0, 3, 2, 5, -10, 0,
                                                                                 (5.1, -21),
                                                                                 (7.5, -25)))
                         )
def test_get_nearest_point_on_grid(xul, yul, height, width, dx, dy, rotation, input, expected):
    x, y = input
    result = get_nearest_point_on_grid(x, y, transform=None,
                                       xul=xul, yul=yul,
                                       dx=dx, dy=dy, rotation=rotation,
                                       offset='center')
    assert np.allclose(result, expected)


@pytest.mark.parametrize('offset', ('center', 'edge'))
def test_get_point_on_national_hydrogeologic_grid(offset):
    x, y = 178389, 938512
    x_nhg, y_nhg = get_point_on_national_hydrogeologic_grid(x, y, offset=offset)
    assert point_is_on_nhg(x_nhg, y_nhg, offset=offset)


def test_load_modelgrid(tmpdir):
    cfg = {'xoff': 100, 'yoff': 100, 'angrot': 20.,
           'proj_str': 'epsg:3070',
           'delr': np.ones(10).tolist(), 'delc': np.ones(2).tolist()}
    grid1 = MFsetupGrid(**cfg)
    grid_file = os.path.join(tmpdir, 'test_grid.json')
    dump(grid_file, cfg)

    grid2 = load_modelgrid(grid_file)
    assert grid1 == grid2


@pytest.mark.parametrize('rotation', (0, 18,))
def test_get_ij(rotation):
    modelgrid = MFsetupGrid(delc=np.ones(20) * 1000,
                            delr=np.ones(25) * 1000,
                            xoff=509405.0, yoff=1175835.0,
                            angrot=18,
                            crs=5070,
                            )
    x = np.array([513614.26519224, 523124.83035519, 526215.00029894,
                  516704.43513599, 513614.26519224])
    y = np.array([1189077.77462862, 1192167.94457237, 1182657.37940942,
                  1179567.20946567, 1189077.77462862])
    expected_pi, expected_pj = [], []
    for xx, yy in zip(x, y):
       tmpi, tmpj = modelgrid.intersect(xx, yy)
       expected_pi.append(tmpi)
       expected_pj.append(tmpj)
    pi, pj = get_ij(modelgrid, x, y)
    assert np.array_equal(pi, expected_pi)
    assert np.array_equal(pj, expected_pj)

    pi0, pj0 = get_ij(modelgrid, x[0], y[0])
    assert np.isscalar(pi0)
    assert np.isscalar(pj0)
    assert pi0 == pi[0]
    assert pj0 == pj[0]


@pytest.mark.parametrize('model_units', ('meters', 'feet'))
@pytest.mark.parametrize('crs,expected_crs_units', ((3696, 'feet'),
                                                    (3070, 'meters'),
                                                    ))
def test_grid_crs_units(crs, model_units, expected_crs_units,
                        pleasant_mf6_cfg):
    cfg = {}
    cfg['simulation'] = pleasant_mf6_cfg['simulation'].copy()
    cfg['model'] = pleasant_mf6_cfg['model'].copy()
    cfg['setup_grid'] = pleasant_mf6_cfg['setup_grid'].copy()
    cfg['setup_grid']['buffer'] = cfg['setup_grid']['buffer'] * \
                                  convert_length_units('meters', expected_crs_units)
    cfg['setup_grid']['crs'] = crs
    # if the CRS is the same as the parent,
    # set the parent up to so that the DIS package can be set up
    # and conversion of delr/delc between feet/meters can be tested
    # for now, parent model in different CRS not supported
    if crs == 3070:
        cfg['parent'] = pleasant_mf6_cfg['parent'].copy()
        cfg['dis'] = pleasant_mf6_cfg['dis'].copy()
    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()
    # this also tests whether crs argument
    # overrides epsg argument (in this case, 3070)
    assert m.modelgrid.crs == get_authority_crs(crs)
    assert m.modelgrid.length_units == expected_crs_units
    # Note: for this test case, need parent to set up DIS
    # can't set up parent in different CRS, so can only test delr for 3070
    if crs == 3070:
        m.setup_dis()
        to_model_units = convert_length_units(m.modelgrid.length_units, m.length_units)
        assert np.allclose(m.modelgrid.delr * to_model_units, m.dis.delr.array)
        assert np.allclose(m.modelgrid.delc * to_model_units, m.dis.delc.array)


def check_grid(m):
    xll = m.modelgrid.xyzvertices[0][-1][0]
    xul = m.modelgrid.xyzvertices[0][0][0]
    yll = m.modelgrid.xyzvertices[1][-1][0]
    yul = m.modelgrid.xyzvertices[1][0][0]
    if m.modelgrid.rotation == 0:
        l, b, r, t = m.modelgrid.bounds
        assert xll == l == xul
        assert yll == b
        assert yul == t
        assert m.modelgrid.xyzvertices[0][-1][-1] == r
    assert m.modelgrid.xoffset == xll
    assert m.modelgrid.yoffset == yll
    assert m.modelgrid.xul == xul
    assert m.modelgrid.yul == yul
    gridjson = load(m.cfg['setup_grid']['grid_file'])
    assert gridjson['xoff'] == xll
    assert gridjson['yoff'] == yll
    assert gridjson['xul'] == xul
    assert gridjson['yul'] == yul
    fp_info = attribs_from_namfile_header(m.namefile)
    if fp_info['xll'] is not None:
        assert fp_info['xll'] == xll
        assert fp_info['yll'] == yll
    if fp_info['xul'] is not None:
        assert fp_info['xul'] == xul
        assert fp_info['yul'] == yul
    # upper left corner should be consistent with
    # the number of rows and cell spacings
    assert np.allclose(np.sqrt((yul - yll)**2 + (xul - xll)**2),
                       m.modelgrid.delc.sum())


@pytest.mark.parametrize('snap_to_parent', (True, False))
def test_grid_corners(basic_model_instance, snap_to_parent, project_root_path):
    """Test grid corner locations with plainfield nwt,
    pleasant nwt and pleasnt mf6 test cases. All have model
    grids based on a buffer around feature of interest,
    snapped to parent model grid.
    """
    m = basic_model_instance
    m.cfg['setup_grid']['snap_to_parent'] = snap_to_parent
    os.chdir(m._abs_model_ws)
    m.setup_grid()
    if m.version == 'mf6':
        m.setup_tdis()
        m.write_input()
    else:
        m.write_name_file()

    # get offset of model origin to closest parent cell corner
    px = m.parent.modelgrid.xyzvertices[0]
    py = m.parent.modelgrid.xyzvertices[1]
    offset = np.min(np.sqrt((m.modelgrid.xoffset - px)**2 +\
        (m.modelgrid.yoffset - py)**2))
    if snap_to_parent:
        check_grid(m)
        # model origin should align with a parent cell corner
        assert offset == 0
    else:
        # model origin should not align with a parent cell corner
        assert offset > 0
    os.chdir(project_root_path)


def test_grid_corners_sm(shellmound_model_with_grid):
    """Test model grid corner locations for shellmound model,
    which is different from basic_model_instance models in that
    it has no parent, grid is specified from xoff, yoff
    and is snapped to NHG.
    """
    m = shellmound_model_with_grid
    m.setup_tdis()
    m.write_input()
    check_grid(m)


@pytest.mark.parametrize('snap_to_parent', (True, False))
def test_grid_corners_sm_tmr(shellmound_tmr_cfg, snap_to_parent):
    """Test model grid corner locations for a shellmound tmr model,
    which is specified from xoff, yoff
    and is snapped to a parent model.
    """
    cfg = copy.deepcopy(shellmound_tmr_cfg)
    cfg['setup_grid'] = {
            'xoff': 526958.20, # lower left x-coordinate
            'yoff': 1183288.00, # lower left y-coordinate
            'rotation': 0.,
            'dxy': 500,  # in CRS units of meters
            'crs': 5070,
            'snap_to_parent': snap_to_parent
        }
    cfg['dis'] = {
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
            'source_data': copy.deepcopy(cfg['dis']['source_data'])}
    m = MF6model(cfg=cfg)
    m.setup_grid()
    m.setup_tdis()
    m.write_input()
    check_grid(m)


def test_get_cellface_midpoint():
    delc = [10, 10]
    delr = [10, 10]
    grid = MFsetupGrid(delc=delc, delr=delr,
                       top=np.ones((2, 2)) * 10,
                       botm=np.ones((1, 2, 2)) * 0,
                       xoff=10, yoff=20,
                       angrot=45)
    k, i, j = 0, 0, 0
    x, y, z = get_cellface_midpoint(grid, k, i, j, 'right')
    x_expected, y_expected = rotate(10, 15, 0, 0, np.radians(grid.angrot))
    x_expected += grid.xoffset
    y_expected += grid.yoffset
    assert np.allclose(x, x_expected)
    assert np.allclose(y, y_expected)
    assert np.allclose(z, 5)
    x, y, z = get_cellface_midpoint(grid, k, i, j, 'left')
    x_expected, y_expected = rotate(0, 15, 0, 0, np.radians(grid.angrot))
    x_expected += grid.xoffset
    y_expected += grid.yoffset
    assert np.allclose(x, x_expected)
    assert np.allclose(y, y_expected)
    assert np.allclose(z, 5)
    x, y, z = get_cellface_midpoint(grid, k, i, j, 'top')
    x_expected, y_expected = rotate(5, 20, 0, 0, np.radians(grid.angrot))
    x_expected += grid.xoffset
    y_expected += grid.yoffset
    assert np.allclose(x, x_expected)
    assert np.allclose(y, y_expected)
    assert np.allclose(z, 5)
    x, y, z = get_cellface_midpoint(grid, k, i, j, 'bottom')
    x_expected, y_expected = rotate(5, 10, 0, 0, np.radians(grid.angrot))
    x_expected += grid.xoffset
    y_expected += grid.yoffset
    assert np.allclose(x, x_expected)
    assert np.allclose(y, y_expected)
    assert np.allclose(z, 5)

    # plot a visual of the cell midpoint
    #import matplotlib.pyplot as plt
    #import flopy
    #fig, ax = plt.subplots()
    #pmv = flopy.plot.PlotMapView(modelgrid=grid)
    #pmv.plot_grid()
    #plt.scatter(x, y)


def test_set_botm_nlay(test_data_path):
    grid = load_modelgrid(test_data_path / 'shellmound/tmr_parent/shellmound_grid.json')
    botm = []
    for i in range(13):
        botm.append(np.loadtxt(test_data_path /\
            f'shellmound/tmr_parent/external/botm{i}.dat'))
    botm = np.array(botm)
    grid.botm = botm
    assert grid.nlay is not None

def test_get_intercell_connections(test_data_path):
    grid = load_modelgrid(test_data_path / 'shellmound/tmr_parent/shellmound_grid.json')
    binary_grid_file = test_data_path / 'shellmound/tmr_parent/shellmound.dis.grb'
    cell_budget_file = test_data_path / 'shellmound/tmr_parent/shellmound.cbc'
    cbb = bf.CellBudgetFile(cell_budget_file)
    flowja = cbb.get_data(text='FLOW-JA-FACE', kstpkper=[0, 0])[0][0, 0, :]
    cn = grid.get_intercell_connections(binary_grid_file=binary_grid_file)
    q = flowja[cn['qidx']]
    assert len(cn) == len(q)

@pytest.mark.parametrize('id',
                         (0,
                          75004400017127.0,
                          ))
@pytest.mark.parametrize('id_col_dtype',
                         (None, str, float, int))
def test_rasterize(shellmound_model,
                   id_col_dtype, id,
                   test_data_path):
    gdf = gpd.read_file(test_data_path / 'shellmound/shps/active_area.shp')
    if id_col_dtype is None:
        id_column = None
        pass
    else:
        id_column = 'id'
        gdf[id_column] = id_col_dtype(id)
    results = rasterize(gdf,
                        shellmound_model.modelgrid,
                        id_column=id_column
                        )
    j=2
