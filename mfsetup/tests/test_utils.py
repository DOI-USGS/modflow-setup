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
from utils import fill_layers


def test_fill_layers():

    nlay, nrow, ncol = 10, 10, 10
    all_layers = np.zeros((nlay, nrow, ncol), dtype=float) * np.nan
    all_layers[0, 2:8, 2:8] = 10
    all_layers[2] = 8
    all_layers[5, 2:8, 2:5] = 5
    all_layers[9] = 1
    filled = fill_layers(all_layers)
    assert np.array_equal(np.nansum(filled, axis=(1, 2)),
                          np.array([360., 324., 800., 252., 216., 500., 72., 54., 36., 100.]))
