"""
Tests for utils.py module
"""
import sys
sys.path.append('..')
import time
import shutil
import os
import numpy as np
import pytest
from ..utils import fill_layers


def test_fill_layers():

    nlay, nrow, ncol = 10, 10, 10
    all_layers = np.zeros((nlay, nrow, ncol), dtype=float) * np.nan
    ni = 6
    nj= 3
    all_layers[0, 2:2+ni, 2:2+ni] = 10
    all_layers[2] = 8
    all_layers[5, 2:2+ni, 2:2+nj] = 5
    all_layers[9] = 1
    filled = fill_layers(all_layers)
    a = np.array([ni*ni, ni*ni, nrow*ncol,
                  ni*nj, ni*nj, ni*nj, ni*nj, ni*nj, ni*nj,
                  nrow*ncol])
    b = np.arange(1, 11, dtype=float)[::-1]
    assert np.array_equal(np.nansum(filled, axis=(1, 2)),
                          a*b)
    make_plot = False
    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(nlay):
            lw = 0.5
            if i in [0, 2, 5, 9]:
                lw = 2
            ax.plot(all_layers[i, 5, :], lw=lw)
