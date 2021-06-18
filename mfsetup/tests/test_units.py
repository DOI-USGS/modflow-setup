"""
Tests for units.py module
"""
import numpy as np
import pytest

from mfsetup.units import (
    convert_flux_units,
    convert_length_units,
    convert_temperature_units,
    convert_time_units,
    convert_volume_units,
    lenuni_values,
    parse_length_units,
)


def test_convert_flux():
    result = convert_flux_units('inches', 'years',
                                'feet', 'days')
    assert np.allclose(result, 1/12 * 1/365.25)
    result = convert_flux_units('feet', 'second',
                                'meters', 'days')
    assert np.allclose(result, 0.3048 * 86400)


def test_convert_length_units():
    assert np.allclose(convert_length_units('centimeters', 'inches'), 1 / 2.54)
    assert np.allclose(convert_length_units('in', 'cm'), 2.54)
    assert np.allclose(convert_length_units(2, 1), 1/.3048)
    assert np.allclose(convert_length_units(1, 2), .3048)
    assert np.allclose(convert_length_units('meters', 'feet'), 1/.3048)
    assert np.allclose(convert_length_units('feet', 'meters'), .3048)
    assert np.allclose(convert_length_units('m', 'ft'), 1/.3048)
    assert np.allclose(convert_length_units('ft', 'm'), .3048)
    assert np.allclose(convert_length_units(None, 'm'), 1.)
    assert np.allclose(convert_length_units('millimeters', 'meters'), 1/1000)
    assert np.allclose(convert_length_units('meters', 'millimeters'), 1000)
    assert np.allclose(convert_length_units('meters', 'km'), 0.001)
    assert np.allclose(convert_length_units('kilometers', 'meters'), 1000)
    assert np.allclose(convert_length_units('kilometers', 'cm'), 1000*100)


def test_convert_time_units():
    assert np.allclose(convert_time_units('s', 'day'), 1/86400)
    assert np.allclose(convert_time_units(4, 1), 86400)
    assert np.allclose(convert_time_units('days', 'seconds'), 86400)
    assert np.allclose(convert_time_units('d', 's'), 86400)
    assert np.allclose(convert_time_units(1, 4), 1/86400)
    assert np.allclose(convert_time_units(None, 'd'), 1.)
    assert np.allclose(convert_time_units(5, 4), 365.25)
    assert np.allclose(convert_time_units(4, 5), 1/365.25)
    assert np.allclose(convert_time_units('years', 'days'), 365.25)


def test_convert_volume_units():
    assert np.allclose(convert_volume_units('cubic meters', 'mgal'), 264.172/1e6)
    assert np.allclose(convert_volume_units('$m^3$', '$ft^3$'), 35.3147)
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


@pytest.mark.parametrize('in_out_units_value_expected', [('Celsius', 'fahrenheit', 0, 32),
                                                         ('celsius', 'F', -40, -40),
                                                         ('Fahrenheit', 'celsius', -40, -40),
                                                         ('Fahrenheit', 'C', 32, 0),
                                                         ('C', 'F', 100, 212)
                                         ])
def test_convert_temp_units(in_out_units_value_expected):
    inunits, outunits, value, expected = in_out_units_value_expected
    fn = convert_temperature_units(inunits, outunits)
    result = fn(value)
    assert result == expected
