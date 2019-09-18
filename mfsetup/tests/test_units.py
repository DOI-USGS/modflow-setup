"""
Tests for units.py module
"""
import numpy as np

from ..units import convert_flux_units, convert_length_units, convert_time_units


def test_convert_flux():

    result = convert_flux_units('inches', 'years',
                                'feet', 'days')
    assert np.allclose(result, 1/12 * 1/365.25)
    result = convert_flux_units('inches', 'years',
                                'feet', 'days')
    assert np.allclose(result, 1 / 12 * 1 / 365.25)


def test_convert_length_units():
    assert np.allclose(convert_length_units(2, 1), 1/.3048)
    assert np.allclose(convert_length_units(1, 2), .3048)
    assert np.allclose(convert_length_units('meters', 'feet'), 1/.3048)
    assert np.allclose(convert_length_units('feet', 'meters'), .3048)
    assert np.allclose(convert_length_units('m', 'ft'), 1/.3048)
    assert np.allclose(convert_length_units('ft', 'm'), .3048)
    assert np.allclose(convert_length_units(None, 'm'), 1.)


def test_convert_time_units():
    assert np.allclose(convert_time_units(4, 1), 86400)
    assert np.allclose(convert_time_units('days', 'seconds'), 86400)
    assert np.allclose(convert_time_units('d', 's'), 86400)
    assert np.allclose(convert_time_units(1, 4), 1/86400)
    assert np.allclose(convert_time_units(None, 'd'), 1.)

