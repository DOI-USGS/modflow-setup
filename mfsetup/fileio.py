import os
import json
import yaml
import numpy as np
from flopy.utils import SpatialReference

def check_source_files(fileslist):
    """Check that the files in fileslist exist.
    """
    if isinstance(fileslist, str):
        fileslist = [fileslist]
    for f in fileslist:
        if not os.path.exists(f):
            raise IOError('Cannot find {}'.format(f))

def load(filename):
    """Load a configuration file."""
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        return load_yml(filename)
    elif filename.endswith('.json'):
        return load_json(filename)


def dump(filename, data):
    """Write a dictionary to a configuration file."""
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        return dump_yml(filename, data)
    elif filename.endswith('.json'):
        return dump_json(filename, data)


def load_json(jsonfile):
    """Convenience function to load a json file; replacing
    some escaped characters."""
    with open(jsonfile) as f:
        return json.load(f)


def dump_json(jsonfile, data):
    """Write a dictionary to a json file."""
    with open(jsonfile, 'w') as output:
        json.dump(data, output, indent=4, sort_keys=True)


def load_sr(filename):
    """Create a SpatialReference instance from model config json file."""
    cfg = load(filename)
    return SpatialReference(delr=np.ones(cfg['ncol'])* cfg['delr'],
                            delc=np.ones(cfg['nrow']) * cfg['delc'],
                            xul=cfg['xul'], yul=cfg['yul'],
                            epsg=cfg['epsg']
                              )


def load_yml(yml_file):
    """Load yaml file into a dictionary."""
    with open(yml_file) as src:
        cfg = yaml.load(src)
    return cfg


def dump_yml(yml_file, data):
    """Write a dictionary to a yaml file."""
    with open(yml_file, 'w') as output:
        yaml.dump(data, output)


def load_array(filename, shape=None):
    """Load an array, ensuring the correct shape."""
    arr = np.loadtxt(filename)
    if shape is not None:
        if arr.shape != shape:
            if arr.size == np.prod(shape):
                arr = np.reshape(arr, shape)
            else:
                raise ValueError("Data in {} have size {}; should be {}"
                                 .format(filename, arr.shape, shape))
    return arr


def save_array(filename, arr, **kwargs):
    """Save and array and print that it was written."""
    np.savetxt(filename, arr, **kwargs)
    print('wrote {}'.format(filename))