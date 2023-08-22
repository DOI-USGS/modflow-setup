from mfsetup import MF6model


def setup_grid(cfg_file):
    """Just set up (a shapefile of) the model grid.
    For trying different grid configurations."""
    m = MF6model(cfg=cfg_file)
    m.setup_grid()
    m.modelgrid.write_shapefile('postproc/shps/grid.shp')

if __name__ == '__main__':

    setup_grid('initial_config_poly.yaml')
