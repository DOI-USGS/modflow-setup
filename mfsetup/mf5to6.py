"""
Utilities for swapping data between MODFLOW-6 and MODFLOW-2005 models
"""

# mapping of mf2005 variables to mf6 counterparts
mf6_variables = {}

# mapping of mf6 packages to mf2005 counterparts
mf2005_packages = {'tdis': 'dis',
                   'sto': 'upw',
                   'npf': 'upw'
                   }

mf6_packages = {'upw': {'npf', 'sto'},
                'dis': {'dis', 'tdis'}
                }


def set_cfg(model, variable):
    j=2