"""
Stuff for handling units
"""
import numpy as np

lenuni_values = {'unknown': 0,
                 'undefined': 0,
                 'feet': 1,
                 'meters': 2,
                 'centimeters': 3,
                 'millimeters': 4,
                 'kilometers': 9,
                 'inches': 10,
                 'miles': 11,
                 'ft': 1,
                 'm': 2,
                 'cm': 3,
                 'mm': 4,
                 'in': 10,
                 'mi': 11,
                 'km': 9,
                 'foot': 1,
                 'meter': 2,
                 'centimeter': 3,
                 'millimeter': 4,
                 'kilometer': 9,
                 'inch': 10,
                 'mile': 11,
                 }

fullnames = {'unknown', 'undefined', 'feet', 'meters', 'centimeters',
             'millimeters', 'inches', 'miles', 'kilometers',
             'seconds', 'minutes', 'hours', 'days', 'years'}

lenuni_text = {v: k for k, v in lenuni_values.items() if k in fullnames}

volumetric_units = {'liters': 13,
                    'L': 13,
                    'gallons': 14,
                    'gallon': 14,
                    'gal': 14,
                    'mgal': 15,
                    'million gallons': 15,
                    'acre feet': 16,
                    'acre-feet': 16,
                    'af': 16,
                    'acre-ft': 16,
                    'acre foot': 16,
                    'acre-foot': 16
                    }

itmuni_values = {"unknown": 0,
                 "seconds": 1,
                 "minutes": 2,
                 "hours": 3,
                 "days": 4,
                 "years": 5,
                 "second": 1,
                 "minute": 2,
                 "hour": 3,
                 "day": 4,
                 "year": 5,
                 "s": 1,
                 "m": 2,
                 "h": 3,
                 "d": 4,
                 "y": 5
                 }

# convert from model length units to the unit abbreviations that pandas uses
pandas_units = {"seconds": "s",
                "minutes": "m",
                "hours": "h",
                "days": "D"
                }

itmuni_text = {v: k for k, v in itmuni_values.items() if k in fullnames}


def convert_length_units(lenuni1, lenuni2):
    """Convert length units, takes MODFLOW-2005 style lenuni numbers
    or MF-6 style text.

    Parameters
    ----------
    lenuni1 : int or str
        Convert from.
    lenuni2 : int or str
        Convert to.

    Returns
    -------
    mult : float
        Multiplier to convert from lenuni1 to lenuni2.
    """
    if lenuni1 is None or lenuni2 is None:
        return 1.
    if isinstance(lenuni1, str):
        lenuni1 = lenuni_values.get(lenuni1.lower(), 0)
    if isinstance(lenuni2, str):
        lenuni2 = lenuni_values.get(lenuni2.lower(), 0)

    length_conversions = get_length_conversions()
    mult = length_conversions[lenuni1, lenuni2]
    return mult


def convert_time_units(itmuni1, itmuni2):
    """Convert time units, takes MODFLOW-2005 style itmuni numbers
    or MF-6 style text.

    Parameters
    ----------
    itmuni1 : int or str
        Convert from.
    itmuni2 : int or str
        Convert to.

    Returns
    -------
    mult : float
        Multiplier to convert from itmuni1 to itmuni2.
    """
    if itmuni1 is None or itmuni2 is None:
        return 1.
    if isinstance(itmuni1, str):
        itmuni1 = itmuni_values.get(itmuni1.lower(), 0)
    if isinstance(itmuni2, str):
        itmuni2 = itmuni_values.get(itmuni2.lower(), 0)

    yearlen = 365.25
    # "seconds": 1,
    # "minutes": 2,
    # "hours": 3,
    # "days": 4,
    # "years": 5,
    mults = {(1, 2): 1/60,   # seconds to minutes
             (1, 3): 1/3600,
             (1, 4): 1/86400,
             (1, 5): 1/(86400 * yearlen),
             (2, 3): 1/60,
             (2, 4): 1/1440,
             (2, 5): 1/(1440 * yearlen),
             (3, 4): 1/24,
             (3, 5): 1/(24 * yearlen),
             (4, 5): 1/yearlen}  # days to years
    convert_time_units = np.ones((6, 6), dtype=float)
    for (u0, u1), mult in mults.items():
        convert_time_units[u0, u1] = mult
        convert_time_units[u1, u0] = 1/mult
    mult = convert_time_units[itmuni1, itmuni2]
    return mult


def get_length_conversions():
    mults = {(1, 2): 1 * 0.3048,  # feet to m
             (1, 3): 100 * 0.3048,
             (1, 4): 1000 * 0.3048,
             (1, 9): 1 * 0.3048 / 5280,  # feet to km
             (1, 10): 1 * 12,
             (1, 11): 1 / 5280,  # feet to miles
             (2, 3): 100,  # meters to cm
             (2, 4): 1000,
             (2, 9): 1 / 1000,
             (2, 10): 1 * 12 / .3048,
             (2, 11): 1 / (.3048 * 5280),
             (3, 4): 10,  # cm to mm
             (3, 9): 1 / (100 * 1000),
             (3, 10): 1 * 12 / (100 * .3048),
             (3, 11): 1 / (.3048 * 5280 * 100),
             (4, 9): 1 / 1e6,  # mm to km
             (4, 10): 1 * 12 / (1000 * .3048),
             (4, 11): 1 / (.3048 * 5280 * 1000),
             }
    length_conversions = np.ones((12, 12), dtype=float)
    for (u0, u1), mult in mults.items():
        length_conversions[u0, u1] = mult
        length_conversions[u1, u0] = 1 / mult
    return length_conversions


def get_volume_conversions():
    length_conversions = get_length_conversions()
    m, n = length_conversions.shape
    size = np.max(list(volumetric_units.values())) + 1
    volume_conversions = np.ones((size, size), dtype=float)
    volume_conversions[:m, :n] = length_conversions **3
    mults = {(13, 1): (1/.3048**3)/1000,  # liters to ft3
             (13, 2): 1/1000,
             (13, 3): 1000,
             (13, 4): 1e6,
             (13, 10): (1/.3048**3)/1000/(12**3), # liters to cubic inches
             (13, 14): 1/3.78541, # liters to gallons
             (13, 15): 1/(3.78541 * 1e6), # liters to million gallons
             (13, 16): (1/.3048**3)/1000/43560, # liters to acre feet
             (14, 1): 1 / 7.48052,  # gallons to ft3
             (14, 2): (.3048**3) / 7.48052, # gallons to m3
             (14, 3): 1e6 * (.3048**3) / 7.48052,  # gallons to cm3
             (14, 4): 1e9 * (.3048**3) / 7.48052,  # gallons to mm3
             (14, 10): 1/231, # gallons to cubic inches
             (14, 15): 1/1e6, # gallons to million gallons
             (14, 16): 1 / 7.48052 / 43560, # gallons to acre feet
             (15, 1): 1e6 / 7.48052,  # million gallons to ft3
             (15, 2): 1e6 * (.3048 ** 3) / 7.48052,  # million gallons to m3
             (15, 10): 1e6 / 231,
             (15, 16): 1e6 / 7.48052 / 43560, # million gallons to acre feet
             (16, 1): 1/43560,  # acre feet to ft3
             (16, 2): 1/43560 * (.3048 ** 3), # acre feet to m3
             }
    for (u0, u1), mult in mults.items():
        volume_conversions[u0, u1] = mult
        volume_conversions[u1, u0] = 1 / mult
    return volume_conversions


def get_unit_text(length_unit, time_unit, length_unit_exp):
    """Get text abbreviation for common units.
    Needs to be filled out more."""
    if isinstance(length_unit, str):
        length_unit = lenuni_values.get(length_unit.lower(), 0)
    if isinstance(time_unit, str):
        time_unit = itmuni_values.get(time_unit.lower(), 0)
    text = {(1, 1, 3): 'cfs',
            (1, 4, 3): 'cfd',
            (2, 1, 3): 'cms',
            (2, 4, 3): 'cmd'
            }
    return text.get((length_unit, time_unit, length_unit_exp), 'units')


def convert_volume_units(input_volume_units, output_volume_units):
    if input_volume_units is None or output_volume_units is None:
        return 1.

    # if both units are expressed as lengths cubed
    in_units = parse_length_units(input_volume_units, text_output=False)
    if in_units is not None:
        if isinstance(in_units, str):
            in_units = lenuni_values.get(in_units.lower(), 0)
    else:
        in_units = volumetric_units.get(input_volume_units.lower(), 0)
    out_units = parse_length_units(output_volume_units, text_output=False)
    if out_units is not None:
        if isinstance(out_units, str):
            out_units = lenuni_values.get(out_units.lower(), 0)
    else:
        out_units = volumetric_units.get(output_volume_units.lower(), 0)

    # get the volume conversions matrix
    vol_conversions = get_volume_conversions()

    # look up the multiplier
    mult = vol_conversions[in_units, out_units]
    return mult


def convert_flux_units(input_length_units, input_time_units,
                       output_length_units, output_time_units):
    # TODO: add support for areas and volumes

    lmult = convert_length_units(input_length_units, output_length_units)
    tmult = convert_time_units(input_time_units, output_time_units)
    return lmult / tmult


def parse_length_units(text, text_output=True):
    for k in volumetric_units.keys():
        if k in text.lower():
            return
    for k, v in lenuni_values.items():
        if k in text:
            if text_output:
                return k
            else:
                return v


def convert_temperature_units(input_temp_units, output_temp_units):
    temp_units = {'celsius': 1,
                  'c': 1,
                  'fahrenheit': 2,
                  'f': 2
                  }

    input_temp_units = temp_units.get(input_temp_units.lower(), 0)
    output_temp_units = temp_units.get(output_temp_units.lower(), 0)

    def unknown(temp):
        return temp

    def c_to_f(temp):
        return temp * (9/5) + 32

    def f_to_c(temp):
        return (5/9) * (temp - 32)

    conversions = {(1, 2): c_to_f,
                   (2, 1): f_to_c}
    conversion = conversions.get((input_temp_units, output_temp_units), unknown)
    return conversion
