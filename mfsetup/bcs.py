"""
Functions for simple MODFLOW boundary conditions such as ghb, drain, etc.
"""
import numpy as np
import pandas as pd
import flopy
fm = flopy.modflow
from shapely.geometry import Polygon
import rasterio
from rasterstats import zonal_stats
from mfsetup.discretization import get_layer
from mfsetup.grid import rasterize
from mfsetup.units import convert_length_units


def setup_ghb_data(model):

    m = model
    source_data = model.cfg['ghb'].get('source_data').copy()
    # get the GHB cells
    # todo: generalize more of the GHB setup code and move it somewhere else
    if 'shapefile' in source_data:
        shapefile_data = source_data['shapefile']
        key = [k for k in shapefile_data.keys() if 'filename' in k.lower()][0]
        shapefile_name = shapefile_data.pop(key)
        ghbcells = rasterize(shapefile_name, m.modelgrid, **shapefile_data)
    else:
        raise NotImplementedError('Only shapefile input supported for GHBs')

    cond = model.cfg['ghb'].get('cond')
    if cond is None:
        raise KeyError("key 'cond' not found in GHB yaml input. "
                       "Must supply conductance via this key for GHB setup.")

    # sample DEM for minimum elevation in each cell with a GHB
    # todo: GHB: allow time-varying bheads via csv input
    vertices = np.array(m.modelgrid.vertices)[ghbcells.flat > 0, :, :]
    polygons = [Polygon(vrts) for vrts in vertices]
    if 'dem' in source_data:
        key = [k for k in source_data['dem'].keys() if 'filename' in k.lower()][0]
        dem_filename = source_data['dem'].pop(key)
        with rasterio.open(dem_filename) as src:
            meta = src.meta
        all_touched = False
        if meta['transform'][0] > m.modelgrid.delr[0]:
            all_touched = True
        results = zonal_stats(polygons, dem_filename, stats='min',
                              all_touched=all_touched)
        min_elevs = np.ones((m.nrow * m.ncol), dtype=float) * np.nan
        min_elevs[ghbcells.flat > 0] = np.array([r['min'] for r in results])
        units_key = [k for k in source_data['dem'] if 'units' in k]
        if len(units_key) > 0:
            min_elevs *= convert_length_units(source_data['dem'][units_key[0]],
                                              model.length_units)
        min_elevs = np.reshape(min_elevs, (m.nrow, m.ncol))
    else:
        raise NotImplementedError('Must supply DEM to sample for GHB elevations\n'
                                  '(GHB: source_data: dem:)')

    # make a DataFrame with MODFLOW input
    i, j = np.indices((m.nrow, m.ncol))
    df = pd.DataFrame({'per': 0,
                       'k': 0,
                       'i': i.flat,
                       'j': j.flat,
                       'bhead': min_elevs.flat,
                       'cond': cond})
    df.dropna(axis=0, inplace=True)

    # assign layers so that bhead is above botms
    df['k'] = get_layer(model.dis.botm.array, df.i, df.j, df.bhead)
    # remove GHB cells from places where the specified head is below the model
    below_bottom_of_model = df.bhead < model.dis.botm.array[-1, df.i, df.j] + 0.01
    df = df.loc[~below_bottom_of_model].copy()

    # exclude inactive cells
    k, i, j = df.k, df.i, df.j
    if model.version == 'mf6':
        active_cells = model.idomain[k, i, j] >= 1
    else:
        active_cells = model.ibound[k, i, j] >= 1
    df = df.loc[active_cells]
    return df