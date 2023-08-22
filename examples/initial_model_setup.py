import os

from mfsetup import MF6model


def setup_grid(cfg_file):
    """Just set up (a shapefile of) the model grid.
    For trying different grid configurations."""
    cwd = os.getcwd()
    m = MF6model(cfg=cfg_file)
    m.setup_grid()
    m.modelgrid.write_shapefile('postproc/shps/grid.shp')
    # Modflow-setup changes the working directory
    # to the model workspace; change it back
    os.chdir(cwd)


def setup_model(cfg_file):
    """Set up the whole model."""
    cwd = os.getcwd()
    m = MF6model.setup_from_yaml(cfg_file)
    m.write_input()
    os.chdir(cwd)
    return m


if __name__ == '__main__':

    #setup_grid('initial_config_poly.yaml')
    setup_model('initial_config_full.yaml')
