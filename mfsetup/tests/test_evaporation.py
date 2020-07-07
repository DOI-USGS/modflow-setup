import numpy as np
import pytest

from mfsetup.evaporation import (
    hamon_evaporation,
    max_daylight_hours,
    solar_declination,
    sunset_hour_angle,
)


@pytest.mark.parametrize('jday_lat_expected', [(105, 30, 12.7)  # Jan 2003
                                           ]
                         )
def test_make_daylight_hours(jday_lat_expected):
    jday, lat, expected_hours = jday_lat_expected
    delta = solar_declination(jday)
    omega = sunset_hour_angle(lat, delta)
    D = max_daylight_hours(omega)
    assert np.allclose(D, expected_hours, rtol=0.01)


@pytest.mark.parametrize('jday_tavg_lat_expected', [(1, 11, 29.8752, 1.21/31),  # Jan 2004
                                                    (196, 27.5, 29.8752, 5.93/31) # July 2004
                                           ]
                         )
def test_hamon_evaporation(jday_tavg_lat_expected):
    """
    Test Hamon calculation for Canyon Lake, TX
    See p 16 in Harwell (2012)
    """
    day_of_year, tmean_c, lat, expected_evap_inches = jday_tavg_lat_expected
    evap_inches_daily = hamon_evaporation(day_of_year, tmean_c, lat)
    assert np.allclose(evap_inches_daily, expected_evap_inches,
                       rtol=0.01)
