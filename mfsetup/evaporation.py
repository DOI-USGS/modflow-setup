"""
Implements the Hamon Method for estimating open water evaporation.
See Hamon (1961) and Harwell (2012)

Hamon, W.R., 1961, Estimating potential evapotranspiration:
Journal of Hydraulics Division, Proceedings of the
American Society of Civil Engineers, v. 87, p. 107–120.

Harwell, G.R., 2012, Estimation of evaporation from open water—A review of selected studies, summary of U.S.
Army Corps of Engineers data collection and methods, and evaluation of two methods for estimation of evaporation
from five reservoirs in Texas: U.S. Geological Survey Scientific Investigations Report 2012–5202, 96 p.
"""
import numpy as np

from mfsetup.units import convert_length_units


def solar_declination(julian_day):
    """

    Parameters
    ----------
    julian_day : int
        Julian day of the year
    Returns
    -------
    delta : float
        solar_declination, in radians
    """
    return 0.4093 * np.sin((2 * np.pi/365) * julian_day - 1.405)


def sunset_hour_angle(latitude_dd, delta):
    """

    Parameters
    ----------
    latitude_dd : float
        Latitude, decimal degrees
    delta : float
        solar_declination, in radians

    Returns
    -------
    omega : float
        sunset_hour_angle, in radians
    """
    return np.arccos(-np.tan(np.radians(latitude_dd)) *
                     np.tan(delta))


def max_daylight_hours(omega):
    """

    Parameters
    ----------
    omega : float
        sunset_hour_angle, in radians

    Returns
    -------
    D : float
        maximum possible daylight hours
    """
    return (24/np.pi) * omega


def saturation_vapor_pressure(avg_daily_air_temp):
    """

    Parameters
    ----------
    avg_daily_air_temp : float
        Average daily air temperature, in Celsius

    Returns
    -------
    svp : float
        saturation vapor pressure, in kilopascals
    """
    return 0.6108 * np.exp((17.27 * avg_daily_air_temp)/
                           (237.3 + avg_daily_air_temp))


def saturation_vapor_density(svp,
                             avg_daily_air_temp):
    """

    Parameters
    ----------
    svp : float
        saturation vapor pressure, in kilopascals
    avg_daily_air_temp : float
        Average daily air temperature, in Celsius

    Returns
    -------
    svd : float
        is the saturation vapor density, in grams per
        cubic meter
    """
    avg_daily_air_temp_kelvin = avg_daily_air_temp + 273.15
    return 2166.74 * (svp/
                      avg_daily_air_temp_kelvin)


def hamon_evaporation(day_of_year, tmean_c, latitude_dd,
                      dest_length_units='inches'):
    """

    Parameters
    ----------
    day_of_year : int
        (Julian) day of the year
    tmean_c : float
        Average daily air temperature, in Celsius
    latitude_dd : float
        Latitude, decimal degrees
    dest_length_units : str
        Length units of output (e.g. ft., feet, meters, etc.)

    Returns
    -------
    E : float
        Open water evaporation, in inches per day
    """
    delta = solar_declination(day_of_year)
    omega = sunset_hour_angle(latitude_dd, delta)
    D = max_daylight_hours(omega)
    svp = saturation_vapor_pressure(tmean_c)
    svd = saturation_vapor_density(svp,
                                   tmean_c)
    E_inches = 0.55 * (D/12)**2 * (svd/100)
    mult = convert_length_units('inches', dest_length_units)
    return E_inches * mult
