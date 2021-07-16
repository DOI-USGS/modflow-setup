"""Functions for working with zone budget.
"""
import os
from pathlib import Path

import numpy as np
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from flopy.utils import binaryfile as bf


def write_zonebudget6_input(zones, budgetfile, binary_grid_file,
                            zone_arrays_subfolder='external',
                            outname=None):
    """Given a numpy array of zones, a MODFLOW 6 budget file and binary grid file,
    write input files for zone budget. Zone definitions are written as external text
    array files, at the location specifeid by `zone_arrays_subfolder`. Zone budget
    input files are located and named based on `outname`.

    TODO: adapt this function to work with 1-D array of zones for advanced stress packages
    (with no binary grid file), or unstructured models.

    Parameters
    ----------
    zones : ndarray
        Array of zone numbers, of same shape as the model.
        A 2D zones array will be broadcast to each layer of a 3D model.
    budgetfile : str or pathlike
        MODFLOW 6 cell budget output file
    binary_grid_file : str or pathlike
        MODFLOW 6 binary grid file
    zone_arrays_subfolder : str or pathlike (optional)
        Relative location for writing external files,
        by default 'external'
    outname : str or pathlike (optional)
        Location and filename stem for Zone budget input files.
        by default None, in which case the Zone budget input files are named after
        the input `budgetfile`.
    """
    # make sure zones is the right shape
    bgf = MfGrdFile(binary_grid_file)
    nlay, nrow, ncol = bgf.modelgrid.nlay, bgf.modelgrid.nrow, bgf.modelgrid.ncol
    layered = False
    if len(zones.shape) == 3:
        layered = True
        assert zones.shape == (nlay, nrow, ncol)
    elif len(zones.shape) == 2:
        assert zones.shape == (nrow, ncol)
        # if zones is 2d and multiple layers
        # broadcast zones to all layers
        if nlay > 1:
            layered = True
            zones = np.broadcast_to(zones, (nlay, nrow, ncol))
    else:
        assert len(zones) == nlay * nrow * ncol
    assert zones.shape[-2:] == (nrow, ncol)

    budgetfile = Path(budgetfile)
    if outname is None:
        outpath = budgetfile.parent
        outname = budgetfile.stem
    else:
        outpath = Path(outname).parent
        outname = Path(outname).stem

    output_namefile = outpath / (outname + '.zbud.nam')
    output_zonefile = outpath / (outname + '.zbud.zon')
    output_zone_array_format = outpath

    # relative path to budget file from zone budget input
    #budgetfile = budgetfile.relative_to(outpath.absolute())
    # use os.path.relpath because pathlib relative_to()
    # only works with strings and therefore doesn't handle two paths that branch
    budgetfile = os.path.relpath(budgetfile, outpath.absolute())
    # relative path to grb file
    if binary_grid_file is not None:
        binary_grid_file = os.path.relpath(binary_grid_file, outpath.absolute())

    # write the name file
    with open(output_namefile, 'w') as dest:
        dest.write('begin zonebudget\n')
        dest.write(f'  bud {budgetfile}\n')
        dest.write(f'  zon {output_zonefile.name}\n')
        dest.write(f'  grb {binary_grid_file}\n')
        dest.write('end zonebudget\n')
    print(f'wrote {output_namefile}')

    # write the zone arrays
    zone_arrays_path = outpath / zone_arrays_subfolder
    zone_arrays_path.mkdir(exist_ok=True)
    zone_array_format = 'budget-zones_{:03d}.dat'
    open_close_text = ''
    if not layered:
        array_fname = zone_arrays_path / zone_array_format.format(0)
        np.savetxt(array_fname, zones, fmt='%d')
        array_fname = array_fname.relative_to(outpath)
        print(f'wrote {array_fname}')
        open_close_text += f'    open/close {array_fname}\n'
    else:
        for k, layer_zones in enumerate(zones):
            array_fname = zone_arrays_path / zone_array_format.format(k)
            np.savetxt(array_fname, layer_zones, fmt='%d')
            array_fname = array_fname.relative_to(outpath)
            open_close_text += f'    open/close {array_fname}\n'
            print(f'wrote {array_fname}')

    # write the zone file
    with open(output_zonefile, 'w') as dest:
        dest.write('begin dimensions\n')
        dest.write(f'  ncells {zones.size}\n')
        dest.write('end dimensions\n')
        dest.write('\n')
        dest.write('begin griddata\n')
        izone_text = 'izone'
        if layered:
            izone_text += ' layered'
        dest.write(f'  {izone_text}\n')
        dest.write(open_close_text)
        dest.write('end griddata\n')
    print(f'wrote {output_zonefile}')
