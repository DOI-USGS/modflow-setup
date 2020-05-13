import collections
import inspect
import pprint
import numpy as np


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
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        else:
            d = {k: v}
    return d


def get_input_arguments(kwargs, function, verbose=False, warn=False, exclude=None):
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
    np.set_printoptions(threshold=20)
    if verbose:
        print('\narguments to {}:'.format(function.__qualname__))
    params = inspect.signature(function)
    input_kwargs = {}
    not_arguments = {}
    if exclude is None:
        exclude = []
    for k, v in kwargs.items():
        if k in params.parameters and k not in exclude:
            input_kwargs[k] = v
            print_item(k, v)
        else:
            not_arguments[k] = v
    if verbose and warn:
        print('\nother arguments:')
        for k, v in not_arguments.items():
            #print('{}: {}'.format(k, v))
            print_item(k, v)
    if verbose:
        print('\n')
    return input_kwargs


def print_item(k, v):
    print('{}: '.format(k), end='')
    if isinstance(v, dict):
        #print(json.dumps(v, indent=4))
        pprint.pprint(v)
    elif isinstance(v, list):
        pprint.pprint(v)
    else:
        print(v)


def get_packages(namefile):
    packages = []
    with open(namefile) as src:
        for line in src:
            if line.startswith('#') or \
                    line.lower().startswith('data') or \
                    line.lower().startswith('list'):
                continue
            else:
                packages.append(line.lower().split()[0])
    return packages

