"""
Utilities for swapping data between MODFLOW-6 and MODFLOW-2005 models
"""
from mfsetup.units import itmuni_text, itmuni_values, lenuni_text, lenuni_values

# mapping of variables to packages
variable_packages = {'mf6': {'sy': 'sto',
                             'ss': 'sto',
                             'k': 'npf',
                             'k33': 'npf',
                             'idomain': 'dis',
                             'strt': 'ic',
                             },
                     'mfnwt': {'sy': 'upw',
                               'ss': 'upw',
                               'hk': 'upw',
                               'vka': 'upw',
                               'ibound': 'bas6',
                               'strt': 'bas6',
                               },
                     'mf2005': {'sy': 'lpf',
                                'ss': 'lpf',
                                'hk': 'lpf',
                                'vka': 'lpf',
                                'ibound': 'bas6',
                                'strt': 'bas6',
                                },
                     'mf2k': {'sy': 'lpf',
                                'ss': 'lpf',
                                'hk': 'lpf',
                                'vka': 'lpf',
                                'ibound': 'bas6',
                                'strt': 'bas6',
                                }
                     }

# mapping between packages
packages = {'mf6': {'upw': {'npf', 'sto'},
                    'lpf': {'npf', 'sto'},
                    'bas6': {'ic', 'dis'},
                    'dis': {'dis', 'tdis'},
                    },
            'mfnwt': {'npf': {'upw'},
                      'sto': {'upw'},
                      'ic': {'bas6'},
                      'dis': {'dis', 'bas6'},
                      'tdis': {'dis'},
                      'ims': 'nwt'
                      },
            'mf2005': {'npf': {'lpf'},
                       'sto': {'lpf'},
                       'ic': {'bas6'},
                       'dis': {'dis', 'bas6'},
                       'tdis': {'dis'},
                       'ims': 'pcg2'
                       },
            'mf2k': {'npf': {'lpf'},
                       'sto': {'lpf'},
                       'ic': {'bas6'},
                       'dis': {'dis', 'bas6'},
                       'tdis': {'dis'},
                       'ims': 'pcg'
                       },
            }


# mapping of variables between modflow versions
mf6_variables = {'hk': 'k',
                 'vka': 'k33',
                 'ibound': 'idomain',
                 'rech': 'recharge'
                 }
mf2005_variables = {v:k for k, v in mf6_variables.items()}


def get_variable_name(variable, model_version):
    """Get the name for a variable in another version of MODFLOW.
    For example, given the variable=idomain, get the equivalent
    variable in model_version='mfnwt' (ibound)
    """
    if model_version == 'mf6':
        return mf6_variables.get(variable, variable)
    elif model_version in {'mfnwt', 'mf2005', 'mf2k'}:
        return mf2005_variables.get(variable, variable)
    else:
        msg = ('Could not get variable {}; '
               'unrecognized MODFLOW version: {}'.format(variable, model_version))
        raise ValueError(msg)


def get_variable_package_name(variable, model_version, source_package=None):
    """Get the package for a variable in another version of MODFLOW.
    For example, given the variable=idomain, which is in package=dis
    in MODFLOW-6, get the package for the equivalent variable (ibound)
    in model_version='mfnwt'. If the package names are consistent between
    MODFLOW versions, the source_package name will be returned.
    """
    if model_version in variable_packages:
        equiv_variable = get_variable_name(variable, model_version)
        return variable_packages[model_version].get(equiv_variable, source_package)
    else:
        msg = ('Could not get package for variable {}; '
               'unrecognized MODFLOW version: {}'.format(variable, model_version))
        raise ValueError(msg)


def get_package_name(package, model_version):
    """Get the name of the package(s) in another version of MODFLOW
    (model_version) that have the information in package.
    For example, package='upw' and model_version='mf6' would return
    both the npf and sto packages, which have the equivalent
    variables in upw (hk, vka, sy, ss).
    """
    if model_version in packages:
        return packages[model_version].get(package, {package})
    else:
        msg = ('Could not get equivalent package for {}; '
               'unrecognized MODFLOW version: {}'.format(package, model_version))
        raise ValueError(msg)


def get_model_length_units(model, lenuni_format=False):
    if model.version == 'mf6':
        unit_text = model.dis.length_units.array
    else:
        unit_text =  lenuni_text[model.dis.lenuni]
    if lenuni_format:
        return lenuni_values[unit_text]
    return unit_text


def get_model_time_units(model, itmuni_format=False):
    if model.version == 'mf6':
        unit_text = model.simulation.tdis.time_units.array
    else:
        unit_text = itmuni_text[model.dis.itmuni]
    if itmuni_format:
        return itmuni_values[unit_text]
    return unit_text
