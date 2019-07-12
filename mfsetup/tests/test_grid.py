import copy
import os
import numpy as np
import pytest
from ..grid import MFsetupGrid

# TODO: add tests for grid.py

@pytest.fixture(scope='module')
def modelgrid():
    return MFsetupGrid(xoff=100., yoff=200., angrot=20.,
                       proj4='+init=epsg:3070',
                       delr=np.ones(10), delc=np.ones(2))


def test_grid_eq(modelgrid):
    grid2 = copy.deepcopy(modelgrid)
    assert modelgrid == grid2
