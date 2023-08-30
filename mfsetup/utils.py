import inspect
import pprint
from collections.abc import Mapping

import numpy as np
import pandas as pd


def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out], thresh)
    return out


def flatten(d):
    """Recursively flatten a dictionary of varying depth,
    putting all keys at a single level.
    """
    flatd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flatd.update(flatten(v))
        else:
            flatd[k] = v
    return flatd


def update(d, u):
    """Recursively update a dictionary of varying depth
    d with items from u.
    from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(d, Mapping):
            if isinstance(v, Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        else:
            d = {k: v}
    return d


def get_input_arguments(kwargs, function, verbose=False, warn=False, errors='coerce',
                        exclude=None):
    """Return subset of keyword arguments in kwargs dict
    that are valid parameters to a function or method.

    Parameters
    ----------
    kwargs : dict (parameter names, values)
    function : function of class method
    warn : bool;
        If true, print supplied argument that are not in the function's signature
    exclude : sequence


    Returns
    -------
    input_kwargs : dict
    """
    np.set_printoptions(threshold=20, edgeitems=1)

    # translate the names of some variables
    # to valid flopy arguments
    # (not sure if this is the best place for this)
    translations = {'continue': 'continue_'
                    }

    if verbose:
        print('\narguments to {}:'.format(function.__qualname__))
    params = inspect.signature(function)
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)
    input_kwargs = {}
    not_arguments = {}
    for k, v in kwargs.items():
        k_original = k
        k = translations.get(k, k)
        if k in params.parameters and not {k, k_original}.intersection(exclude):
            input_kwargs[k] = v
            if verbose:
                print_item(k, v)
        else:
            not_arguments[k] = v
    if verbose and warn:
        print('\nother arguments:')
        for k, v in not_arguments.items():
            #print('{}: {}'.format(k, v))
            print_item(k, v)
    if errors == 'raise' and len(not_arguments) > 0:
        raise ValueError(
            f'Invalid input arguments to {function.__name__}(): '
            f"{', '.join(not_arguments.keys())}\n"
            f"Valid arguments: {', '.join(params.parameters.keys())}")
    if verbose:
        print('\n')
    return input_kwargs


def print_item(k, v):
    print('{}: '.format(k), end='')
    if isinstance(v, dict):
        if len(v) > 1:
            print('{{{}: {}\n ...\n}}'.format(*next(iter(v.items()))))
        else:
            print(v)
    elif isinstance(v, list):
        if len(v) > 3:
            print('[{} ... {}]'.format(v[0], v[-1]))
        else:
            pprint.pprint(v, compact=True)
    elif isinstance(v, pd.DataFrame):
        print(v.head())
    elif isinstance(v, np.ndarray):
        txt = 'array: {}, {}'.format(v.shape, v.dtype)
        try:
            txt += ', min: {:g}, mean: {:g}, max: {:g}'.format(v.min(), v.mean(), v.max())
        except:
            pass
        print(txt)
    else:
        print(v)


def get_packages(namefile):
    packages = []
    with open(namefile) as src:
        read = True
        for line in src:
            if line.startswith('#') or \
                    len(line.strip()) < 1 or \
                    line.lower().startswith('data') or \
                    line.lower().strip().startswith('list'):
                continue
            elif 'begin options' in line.lower():
                read = False
            elif 'begin packages' in line.lower():
                read = True
            elif 'end packages' in line.lower():
                read = False
            elif read:
                package_name = line.strip().lower().split()
                if len(package_name) > 0:
                    package_name = package_name[0]
                else:
                    continue
                if package_name not in {'bas6'}:
                    package_name = package_name.replace('6', '')
                packages.append(package_name)
    return packages
