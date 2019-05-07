"""
Stuff for handling units
"""
import numpy as np

lenuni_values = {'unknown': 0,
                 'feet': 1,
                 'meters': 2,
                 'centimeters': 3,
                 'ft': 1,
                 'm': 2,
                 'cm': 3
                 }

lenuni_text = {v: k for k, v in lenuni_values.items()}

itmuni_values = {"unknown": 0,
                 "seconds": 1,
                 "minutes": 2,
                 "hours": 3,
                 "days": 4,
                 "years": 5,
                 "s": 1,
                 "m": 2,
                 "h": 3,
                 "d": 4,
                 "y": 5
                 }

itmuni_text = {v: k for k, v in itmuni_values.items()}


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
    if isinstance(lenuni1, str):
        lenuni1 = lenuni_values.get(lenuni1.lower(), 0)
    if isinstance(lenuni2, str):
        lenuni2 = lenuni_values.get(lenuni2.lower(), 0)

    mults = {(1, 2): 0.3048,
             (1, 3): 100 * 0.3048,
             (2, 3): 100
             }
    convert_length_units = np.ones((4, 4), dtype=float)
    for (u0, u1), mult in mults.items():
        convert_length_units[u0, u1] = mult
        convert_length_units[u1, u0] = 1 / mult
    mult = convert_length_units[lenuni1, lenuni2]

    #convert_length_units = np.identity((4, 4), dtype=float)
    #convert_length_units[1, 2] = 0.3048
    #convert_length_units[2, 1] = 1/convert_length_units[1, 2]
    #convert_length_units[1, 3] = 100 * convert_length_units[1, 2]
    #convert_length_units[3, 1] = 1/convert_length_units[1, 3]
    #convert_length_units[2, 3] = 100
    #convert_length_units[3, 2] = 1/convert_length_units[2, 3]
    #mult = convert_length_units[lenuni1, lenuni2]
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
    if isinstance(itmuni1, str):
        itmuni1 = itmuni_values.get(itmuni1.lower(), 0)
    if isinstance(itmuni2, str):
        itmuni2 = itmuni_values.get(itmuni2.lower(), 0)

    yearlen = 365.25
    mults = {(1, 2): 1/60,
             (1, 3): 1/3600,
             (1, 4): 1/86400,
             (1, 5): 1/(86400 * yearlen),
             (2, 3): 1/60,
             (2, 4): 1/1440,
             (2, 5): 1/(1440 * yearlen),
             (3, 4): 1/24,
             (3, 5): 1/(24 * yearlen),
             (4, 5): yearlen}
    convert_time_units = np.ones((6, 6), dtype=float)
    for (u0, u1), mult in mults.items():
        convert_time_units[u0, u1] = mult
        convert_time_units[u1, u0] = 1/mult
    mult = convert_time_units[itmuni1, itmuni2]
    return mult


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