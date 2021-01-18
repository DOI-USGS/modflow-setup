from copy import deepcopy

from flopy import mf6

from mfsetup import MF6model
from mfsetup.grid import write_bbox_shapefile
from mfsetup.utils import get_input_arguments


def test_rotated_grid(shellmound_cfg, shellmound_simulation):
    cfg = deepcopy(shellmound_cfg)
    #simulation = deepcopy(simulation)
    cfg['model']['simulation'] = shellmound_simulation
    cfg['setup_grid']['snap_to_NHG'] = False
    cfg['setup_grid']['rotation'] = 18.
    cfg['setup_grid']['xoff'] += 8000
    cfg['dis']['dimensions']['nrow'] = 20
    cfg['dis']['dimensions']['ncol'] = 25

    cfg = MF6model._parse_model_kwargs(cfg)
    kwargs = get_input_arguments(cfg['model'], mf6.ModflowGwf,
                                 exclude='packages')
    m = MF6model(cfg=cfg, **kwargs)
    m.setup_grid()

    assert m.modelgrid.angrot == 18.
    assert m.modelgrid.xoffset == cfg['setup_grid']['xoff']
    assert m.modelgrid.yoffset == cfg['setup_grid']['yoff']

    m.setup_dis()
    #m.setup_tdis()
    #m.setup_solver()
    #m.setup_packages(reset_existing=False)
    #m.write_input()
    j=2
