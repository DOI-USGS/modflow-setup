"""
Tests for zbud.py module
"""
import shutil
from pathlib import Path
from subprocess import PIPE, Popen

import numpy as np
import pytest
from mfexport.zbud import write_zonebudget6_input


@pytest.mark.parametrize('outname', (None, 'shellmound'))
def test_write_zonebudget6_input(shellmound_model, outname, tmpdir, mf6_exe):

    nlay, nrow, ncol = shellmound_model.dis.botm.array.shape
    zones2d = np.zeros((nrow, ncol), dtype=int)
    zones2d[10:20, 10:20] = 1
    model_ws = Path(shellmound_model.model_ws)
    budgetfile = model_ws / (shellmound_model.name + '.cbc')
    binary_grid_file = model_ws / (shellmound_model.name + '.dis.grb')
    dest_budgetfile = tmpdir / budgetfile.name
    dest_binary_grid_file = tmpdir / binary_grid_file.name
    shutil.copy(budgetfile, dest_budgetfile)
    shutil.copy(binary_grid_file, dest_binary_grid_file)
    if outname is not None:
        outname = tmpdir / outname
    # delete output files
    (tmpdir / 'shellmound.zbud.nam').unlink(missing_ok=True)
    (tmpdir / 'shellmound.zbud.nam').unlink(missing_ok=True)
    (tmpdir / 'external/budget-zones_000.dat').unlink(missing_ok=True)

    # write zonebudget input
    write_zonebudget6_input(zones2d, budgetfile=dest_budgetfile,
                             binary_grid_file=dest_binary_grid_file,
                             outname=outname)
    assert (tmpdir / 'shellmound.zbud.nam').exists()
    assert (tmpdir / 'shellmound.zbud.zon').exists()
    assert (tmpdir / 'external/budget-zones_000.dat').exists()

    # run zonebudget
    process = Popen([str(mf6_exe), 'shellmound.zbud.nam'], cwd=tmpdir,
                 stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    assert process.returncode == 0
