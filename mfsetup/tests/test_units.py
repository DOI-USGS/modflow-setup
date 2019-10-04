"""
Tests for units.py module
"""
import numpy as np
import pytest
from ..units import (convert_flux_units, convert_length_units,
                     convert_volume_units,
                     convert_time_units, parse_length_units,
                     lenuni_values)


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


def test_convert_volume_units():
    assert np.allclose(convert_volume_units('cubic meters', 'cubic feet'), 35.3147)
    assert np.allclose(convert_volume_units('cubic feet', 'cubic meters'), 0.0283168)
    assert np.allclose(convert_volume_units('meters', 'feet'), 35.3147)
    assert np.allclose(convert_volume_units('feet', 'meters'), 0.0283168)
    assert np.allclose(convert_volume_units('feet3', 'm3'), 0.0283168)
    assert np.allclose(convert_volume_units('feet3', 'meters3'), 0.0283168)
    assert np.allclose(convert_volume_units('gallons', 'ft3'), 1/7.48052)
    assert np.allclose(convert_volume_units('gallons', 'm3'), (.3048**3)/7.48052)
    assert np.allclose(convert_volume_units('gallons', 'acre foot'), 1/7.48052/43560)
    assert np.allclose(convert_volume_units('gallons', 'af'), 1/7.48052/43560)
    assert np.allclose(convert_volume_units('gallons', 'acre-ft'), 1/7.48052/43560)
    assert np.allclose(convert_volume_units('mgal', 'acre-ft'), 1e6 / 7.48052 / 43560)
    assert np.allclose(convert_volume_units('liters', 'gallon'), 1/3.78541)
    assert np.allclose(convert_volume_units(None, 'cubic feet'), 1.)
    assert np.allclose(convert_volume_units('cubic feet', None), 1.)
    assert np.allclose(convert_volume_units('junk', 'junk'), 1.)


@pytest.mark.parametrize('prefix', ['cubic', 'square', ''])
@pytest.mark.parametrize('length_units', [' meters', ' m',
                                          ' feet', ' ft'])
@pytest.mark.parametrize('time_units', [' per day',
                                        '/day',
                                        ' /day',
                                        '/d',
                                         ])
def test_parse_length_units_prefix(prefix, length_units, time_units):
    text = prefix + length_units + time_units
    length_units = parse_length_units(text)
    assert length_units in lenuni_values


@pytest.mark.parametrize('length_units', ['meters', 'm',
                                          'feet', 'ft'])
@pytest.mark.parametrize('exp', ['3', '2', '^3', '^2'])
@pytest.mark.parametrize('time_units', [' per day',
                                        '/day',
                                        ' /day',
                                        '/d',
                                         ])
def test_parse_length_units_exponent(length_units, exp, time_units):
    text = length_units + exp + time_units
    length_units = parse_length_units(text)
    assert length_units in lenuni_values
