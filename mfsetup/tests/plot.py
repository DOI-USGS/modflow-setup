import glob
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mfsetup import load_modelgrid


def make_lake_xsections(model, i_range, j_range,
                        bathymetry_raster, datum, outpdf):

    grid = model.modelgrid
    top = model.dis.top.array
    botm = model.dis.botm.array
    idomain = model.dis.idomain.array

    i0, i1 = i_range
    j0, j1 = j_range

    with PdfPages(outpdf) as pdf:
        for i in range(i0, i1, 10):
            j_values = slice(j0, j1)
            j_edges = slice(j0, j1 + 1)
            x = grid.xcellcenters[i, j_edges]
            y = grid.ycellcenters[i, j_edges]

            with rasterio.open(bathymetry_raster) as src:
                bathy = np.squeeze(list(src.sample(zip(x, y))))
                bathy[(bathy == src.nodata) | (bathy == 0)] = np.nan
                bathy = datum - bathy

            x_edges = grid.xcellcenters[i, j_edges]
            z_edges = np.vstack([top[i, j_edges]] + [b for b in botm[:, i, j_edges]])

            plt.figure()
            plt.pcolormesh(x_edges, z_edges, idomain[:, i, j_values], cmap='Blues_r', shading='flat', edgecolors='k',
                           lw=0.1)
            plt.plot(x_edges, bathy, color='r', label=bathymetry_raster)
            plt.title(f'Row {i}')
            plt.legend()
            pdf.savefig()
            plt.close()

        for j in range(j0, j1, 10):
            i_values = slice(i0, i1)
            i_edges = slice(i0, i1 + 1)
            x = grid.xcellcenters[i_edges, j]
            y = grid.ycellcenters[i_edges, j]

            with rasterio.open(bathymetry_raster) as src:
                bathy = np.squeeze(list(src.sample(zip(x, y))))
                bathy[(bathy == src.nodata) | (bathy == 0)] = np.nan
                bathy = datum - bathy

            x_edges = grid.ycellcenters[i_edges, j]
            z_edges = np.vstack([top[i_edges, j]] + [b for b in botm[:, i_edges, j]])

            plt.figure()
            plt.pcolormesh(x_edges, z_edges, idomain[:, i_values, j], cmap='Blues_r', shading='auto', edgecolors='k',
                           lw=0.1)
            plt.plot(x_edges, bathy, color='r', label=bathymetry_raster)
            plt.title(f'Column {j}')
            plt.legend()
            pdf.savefig()
            plt.close()
