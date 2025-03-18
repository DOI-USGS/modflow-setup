"""Functions for reading and writing stuff to disk, and working with file paths.
"""
import datetime as dt
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import yaml
from flopy.mf6.data import mfstructure
from flopy.mf6.mfbase import (
    ExtFileAction,
    FlopyException,
    MFDataException,
    MFFileMgmt,
    PackageContainer,
    PackageContainerType,
    VerbosityLevel,
)
from flopy.mf6.modflow import mfims, mftdis
from flopy.modflow.mf import ModflowGlobal
from flopy.utils import mfreadnam

import mfsetup
from mfsetup.grid import MFsetupGrid
from mfsetup.utils import get_input_arguments, update


def check_source_files(fileslist):
    """Check that the files in fileslist exist.
    """
    if isinstance(fileslist, str):
        fileslist = [fileslist]
    for f in fileslist:
        f = Path(f)
        if not f.exists():
            raise IOError(f'Cannot find {f.absolute()}')


def load(filename):
    """Load a configuration file."""
    filename = Path(filename)
    if set(filename.suffixes).intersection({'.yml', '.yaml'}):
        return load_yml(filename)
    elif filename.suffix == '.json':
        return load_json(filename)


def dump(filename, data):
    """Write a dictionary to a configuration file."""
    if str(filename).endswith('.yml') or str(filename).endswith('.yaml'):
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
    print('wrote {}'.format(jsonfile))


def load_modelgrid(filename):
    """Create a MFsetupGrid instance from model config json file."""
    cfg = load(filename)
    rename = {'xll': 'xoff',
              'yll': 'yoff',
              }
    for k, v in rename.items():
        if k in cfg:
            cfg[v] = cfg.pop(k)
    if np.isscalar(cfg['delr']):
        cfg['delr'] = np.ones(cfg['ncol'])* cfg['delr']
    if np.isscalar(cfg['delc']):
        cfg['delc'] = np.ones(cfg['nrow']) * cfg['delc']
    kwargs = get_input_arguments(cfg, MFsetupGrid)
    return MFsetupGrid(**kwargs)


def load_yml(yml_file):
    """Load yaml file into a dictionary."""
    with open(yml_file) as src:
        cfg = yaml.load(src, Loader=yaml.Loader)
    return cfg


def dump_yml(yml_file, data):
    """Write a dictionary to a yaml file."""
    with open(yml_file, 'w') as output:
        yaml.dump(data, output)#, Dumper=yaml.Dumper)
    print('wrote {}'.format(yml_file))


def load_array(filename, shape=None, nodata=-9999):
    """Load an array, ensuring the correct shape."""
    t0 = time.time()
    if not isinstance(filename, list):
        filename = [filename]
    shape2d = shape
    if shape is not None and len(shape) == 3:
        shape2d = shape[1:]

    arraylist = []
    for f in filename:
        if isinstance(f, dict):
            f = f['filename']
        txt = 'loading {}'.format(f)
        if shape2d is not None:
            txt += ', shape={}'.format(shape2d)
        print(txt, end=', ')
        # arr = np.loadtxt
        # pd.read_csv is >3x faster than np.load_txt
        arr = pd.read_csv(f, delim_whitespace=True, header=None).values
        if shape2d is not None:
            if arr.shape != shape2d:
                if arr.size == np.prod(shape2d):
                    arr = np.reshape(arr, shape2d)
                else:
                    raise ValueError("Data in {} have size {}; should be {}"
                                     .format(f, arr.shape, shape2d))
        arraylist.append(arr)
    array = np.squeeze(arraylist)
    if issubclass(array.dtype.type, np.floating):
        array[array == nodata] = np.nan
    print("took {:.2f}s".format(time.time() - t0))
    return array


def save_array(filename, arr, nodata=-9999,
               **kwargs):
    """Save and array and print that it was written."""
    if isinstance(filename, dict) and 'filename' in filename.keys():
        filename = filename.copy().pop('filename')
    t0 = time.time()
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        arr = arr.copy()
        arr = arr.astype(int)
    arr[np.isnan(arr)] = nodata
    np.savetxt(filename, arr, **kwargs)
    print('wrote {}'.format(filename), end=', ')
    print("took {:.2f}s".format(time.time() - t0))


def append_csv(filename, df, **kwargs):
    """Read data from filename,
    append to dataframe, and write appended dataframe
    back to filename."""
    if os.path.exists(filename):
        written = pd.read_csv(filename)
        df = pd.concat([df, written], axis=0)
    df.to_csv(filename, **kwargs)


def load_cfg(cfgfile, verbose=False, default_file=None):
    """This method loads a YAML or JSON configuration file,
    applies configuration defaults from a default_file if specified,
    adds the absolute file path of the configuration file
    to the configuration dictionary, and converts any
    relative paths in the configuration dictionary to
    absolute paths, assuming the paths are relative to
    the configuration file location.

    Parameters
    ----------
    cfgfile : str
        Path to MFsetup configuration file (json or yaml)

    Returns
    -------
    cfg : dict
        Dictionary of configuration data

    Notes
    -----
    This function is used by the model instance load and setup_from_yaml
    classmethods, so that configuration defaults can be applied to the
    simulation and model blocks before they are passed to the flopy simulation
    constructor and the model constructor.
    """
    print('loading configuration file {}...'.format(cfgfile))
    source_path = Path(__file__).parent
    default_file = Path(default_file)
    check_source_files([cfgfile, source_path / default_file])

    # default configuration
    default_cfg = {}
    if default_file is not None:
        default_cfg = load(source_path / default_file)
        default_cfg['filename'] = source_path / default_file

        # for now, only apply defaults for the model and simulation blocks
        # which are needed for the model instance constructor
        # other defaults are applied in _set_cfg,
        # which is called by model.__init__
        # intermediate_data is needed by some tests
        apply_defaults = {'simulation', 'model', 'intermediate_data'}
        default_cfg = {k: v for k, v in default_cfg.items()
                       if k in apply_defaults}

    # recursively update defaults with information from yamlfile
    cfg = default_cfg.copy()
    user_specified_cfg = load(cfgfile)

    update(cfg, user_specified_cfg)
    cfg['model'].update({'verbose': verbose})
    cfg['filename'] = os.path.abspath(cfgfile)

    # convert relative paths in the configuration dictionary
    # to absolute paths, based on the location of the config file
    config_file_location = os.path.split(os.path.abspath(cfgfile))[0]
    cfg = set_cfg_paths_to_absolute(cfg, config_file_location)
    return cfg


def set_cfg_paths_to_absolute(cfg, config_file_location):
    version = None
    if 'simulation' in cfg:
        version = 'mf6'
    else:
        version = cfg['model'].get('version')
    if version == 'mf6':
        file_path_keys_relative_to_config = [
            'simulation.sim_ws',
            'parent.model_ws',
            'parent.simulation.sim_ws',
            'parent.headfile',
            #'setup_grid.lgr.config_file'
        ]
        model_ws = os.path.normpath(os.path.join(config_file_location,
                                                 cfg['simulation']['sim_ws']))
    else:
        file_path_keys_relative_to_config = [
            'model.model_ws',
            'parent.model_ws',
            'parent.simulation.sim_ws',
            'parent.headfile',
            'nwt.use_existing_file'
        ]
        model_ws = os.path.normpath(os.path.join(config_file_location,
                                                 cfg['model']['model_ws']))
    file_path_keys_relative_to_model_ws = [
        'setup_grid.grid_file'
    ]
    # add additional paths by looking for source_data
    # within these input blocks, convert file paths to absolute
    look_for_files_in = ['source_data',
                         'perimeter_boundary',
                         'lgr',
                         'sfrmaker_options'
                         ]
    for pckgname, pckg in cfg.items():
        if isinstance(pckg, dict):
            for input_block in look_for_files_in:
                if input_block in pckg.keys():
                    # handle LGR sub-blocks separately
                    # if LGR configuration is specified within the yaml file
                    # (or as a dictionary), we don't want to touch it at this point
                    # (just convert filepaths to configuration files for sub-models)
                    if input_block == 'lgr':
                        for model_name, config in pckg[input_block].items():
                            if 'filename' in config:
                                file_keys = _parse_file_path_keys_from_source_data(
                                    {model_name: config})
                    else:
                        file_keys = _parse_file_path_keys_from_source_data(pckg[input_block])
                    for key in file_keys:
                        file_path_keys_relative_to_config. \
                            append('.'.join([pckgname, input_block, key]))
                for loc in ['output_files',
                            'output_folders',
                            'output_folder',
                            'output_path']:
                    if loc in pckg.keys():
                        file_keys = _parse_file_path_keys_from_source_data(pckg[loc], paths=True)
                        for key in file_keys:
                            file_path_keys_relative_to_model_ws. \
                                append('.'.join([pckgname, loc, key]).strip('.'))

    # set locations that are relative to configuration file
    cfg = _set_absolute_paths_to_location(file_path_keys_relative_to_config,
                                          config_file_location, cfg)

    # set locations that are relative to model_ws
    cfg = _set_absolute_paths_to_location(file_path_keys_relative_to_model_ws,
                                         model_ws,
                                         cfg)
    return cfg


def _set_path(keys, abspath, cfg):
    """From a sequence of keys that point to a file
    path in a nested dictionary, convert the file
    path at that location from relative to absolute,
    based on a provided absolute path.

    Parameters
    ----------
    keys : sequence or str of dict keys separated by '.'
        that point to a relative path
        Example: 'parent.model_ws' for cfg['parent']['model_ws']
    abspath : absolute path
    cfg : dictionary

    Returns
    -------
    updates cfg with an absolute path based on abspath,
    at the location in the dictionary specified by keys.
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    d = cfg.get(keys[0])
    if d is not None:
        for level in range(1, len(keys)):
            if level == len(keys) - 1:
                k = keys[level]
                if k in d:
                    if d[k] is not None:
                        d[k] = os.path.normpath(os.path.join(abspath, d[k]))
                elif k.isdigit():
                    k = int(k)
                    if d[k] is not None:
                        d[k] = os.path.join(abspath, d[k])
            else:
                key = keys[level]
                if key in d:
                    d = d[keys[level]]
    return cfg


def _set_absolute_paths_to_location(paths, location, cfg):
    """Set relative file paths in a configuration dictionary
    to a specified location.

    Parameters
    ----------
    paths : sequence
        Sequence of dictionary keys read by set_path.
        e.g. ['parent.model_ws', 'parent.headfile']
    location : str (path to folder)
    cfg : configuration dictionary  (as read in by load_cfg)

    """
    for keys in paths:
        cfg = _set_path(keys, location, cfg)
    return cfg


def _parse_file_path_keys_from_source_data(source_data, prefix=None, paths=False):
    """Parse a source data entry in the configuration file.

    pseudo code:
    For each key or item in source_data,
        If it is a string that ends with a valid extension,
            a file is expected.
        If it is a dict or list,
            it is expected to be a file or set of files with metadata.
        For each item in the dict or list,
            If it is a string that ends with a valid extension,
                a file is expected.
            If it is a dict or list,
                A set of files corresponding to
                model layers or stress periods is expected.

    valid source data file extensions: csv, shp, tif, asc

    Parameters
    ----------
    source_data : dict
    prefix : str
        text to prepend to results, e.g.
        keys = prefix.keys
    paths = Bool
        if True, overrides check for valid extension

    Returns
    -------
    keys
    """
    valid_extensions = ['csv', 'shp', 'tif',
                        'ref', 'dat',
                        'nc',
                        'yml', 'json',
                        'hds', 'cbb', 'cbc',
                        'grb']
    file_keys = ['filename',
                 'filenames',
                 'binaryfile',
                 'nhdplus_paths']
    keys = []
    if source_data is None:
        return []
    if isinstance(source_data, str):
        return ['']
    if isinstance(source_data, list):
        items = enumerate(source_data)
    elif isinstance(source_data, dict):
        items = source_data.items()
    for k0, v in items:
        if isinstance(v, str):
            if k0 in file_keys:
                keys.append(k0)
            elif v[-3:] in valid_extensions or paths:
                keys.append(k0)
            elif 'output' in source_data:
                keys.append(k0)
        elif isinstance(v, list):
            for i, v1 in enumerate(v):
                if k0 in file_keys:
                    keys.append('.'.join([str(k0), str(i)]))
                elif paths or isinstance(v1, str) and v1[-3:] in valid_extensions:
                    keys.append('.'.join([str(k0), str(i)]))
        elif isinstance(v, dict):
            keys += _parse_file_path_keys_from_source_data(v, prefix=k0, paths=paths)
    if prefix is not None:
        keys = ['{}.{}'.format(prefix, k) for k in keys]
    return keys


def setup_external_filepaths(model, package, variable_name,
                             filename_format, file_numbers=None,
                             relative_external_paths=True):
    """Set up external file paths for a MODFLOW package variable. Sets paths
    for intermediate files, which are written from the (processed) source data.
    Intermediate files are supplied to Flopy as external files for a given package
    variable. Flopy writes external files to a specified location when the MODFLOW
    package file is written. This method gets the external file paths that
    will be written by FloPy, and puts them in the configuration dictionary
    under their respective variables.

    Parameters
    ----------
    model : mfsetup.MF6model or mfsetup.MFnwtModel instance
        Model with cfg attribute to update.
    package : str
        Three-letter package abreviation (e.g. 'DIS' for discretization)
    variable_name : str
        FloPy name of variable represented by external files (e.g. 'top' or 'botm')
    filename_format : str
        File path to the external file(s). Can be a string representing a single file
        (e.g. 'top.dat'), or for variables where a file is written for each layer or
        stress period, a format string that will be formated with the zero-based layer
        number (e.g. 'botm{}.dat') for files botm0.dat, botm1.dat, ...
    file_numbers : list of ints
        List of numbers for the external files. Usually these represent zero-based
        layers or stress periods.

    Returns
    -------
    filepaths : list
        List of external file paths

    Adds intermediated file paths to model.cfg[<package>]['intermediate_data']
    For MODFLOW-6 models, Adds external file paths to model.cfg[<package>][<variable_name>]
    """
    package = package.lower()
    if file_numbers is None:
        file_numbers = [0]

    # in lieu of a way to get these from Flopy somehow
    griddata_variables = ['top', 'botm', 'idomain', 'strt',
                          'k', 'k33', 'sy', 'ss']
    transient2D_variables = {'rech', 'recharge',
                             'finf', 'pet', 'extdp', 'extwc',
                             }
    transient3D_variables = {'lakarr', 'bdlknc'}
    tabular_variables = {'connectiondata'}
    transient_tabular_variables = {'stress_period_data'}
    transient_variables = transient2D_variables | transient3D_variables | transient_tabular_variables

    model.get_package(package)
    # intermediate data
    filename_format = os.path.split(filename_format)[-1]
    if not relative_external_paths:
        intermediate_files = [os.path.normpath(os.path.join(model.tmpdir,
                              filename_format).format(i)) for i in file_numbers]
    else:
        intermediate_files = [os.path.join(model.tmpdir,
                              filename_format).format(i) for i in file_numbers]

    if variable_name in transient2D_variables or variable_name in transient_tabular_variables:
        model.cfg['intermediate_data'][variable_name] = {per: f for per, f in
                                                         zip(file_numbers, intermediate_files)}
    elif variable_name in transient3D_variables:
        model.cfg['intermediate_data'][variable_name] = {0: intermediate_files}
    elif variable_name in tabular_variables:
        model.cfg['intermediate_data']['{}_{}'.format(package, variable_name)] = intermediate_files
    else:
        model.cfg['intermediate_data'][variable_name] = intermediate_files

    # external array(s) read by MODFLOW
    # (set to reflect expected locations where flopy will save them)
    if not relative_external_paths:
        external_files = [os.path.normpath(os.path.join(model.model_ws,
                                       model.external_path,
                                       filename_format.format(i))) for i in file_numbers]
    else:
        external_files = [os.path.join(model.model_ws,
                                       model.external_path,
                                       filename_format.format(i)) for i in file_numbers]

    if variable_name in transient2D_variables or variable_name in transient_tabular_variables:
        model.cfg['external_files'][variable_name] = {per: f for per, f in
                                                         zip(file_numbers, external_files)}
    elif variable_name in transient3D_variables:
        model.cfg['external_files'][variable_name] = {0: external_files}
    else:
        model.cfg['external_files'][variable_name] = external_files

    if model.version == 'mf6':
        # skip these for now (not implemented yet for MF6)
        if variable_name in transient3D_variables:
            return
        ext_files_key = 'external_files'
        if variable_name not in transient_variables:
            filepaths = [{'filename': f} for f in model.cfg[ext_files_key][variable_name]]
        else:
            filepaths = {per: {'filename': f}
                         for per, f in model.cfg[ext_files_key][variable_name].items()}
        # set package variable input (to Flopy)
        if variable_name in griddata_variables:
            model.cfg[package]['griddata'][variable_name] = filepaths
        elif variable_name in tabular_variables:
            model.cfg[package][variable_name] = filepaths[0]
            model.cfg[ext_files_key]['{}_{}'.format(package, variable_name)] = model.cfg[ext_files_key].pop(variable_name)
            #elif variable_name in transient_variables:
            #    filepaths = {per: {'filename': f} for per, f in
            #                 zip(file_numbers, model.cfg[ext_files_key][variable_name])}
            #    model.cfg[package][variable_name] = filepaths
        elif variable_name in transient_tabular_variables:
            model.cfg[package][variable_name] = filepaths
            model.cfg[ext_files_key]['{}_{}'.format(package, variable_name)] = model.cfg[ext_files_key].pop(variable_name)
        else:
            model.cfg[package][variable_name] = filepaths # {per: d for per, d in zip(file_numbers, filepaths)}
    else:
        filepaths = model.cfg['intermediate_data'][variable_name]
        model.cfg[package][variable_name] = filepaths

    return filepaths


def flopy_mf2005_load(m, load_only=None, forgive=False, check=False):
    """Execute the code in flopy.modflow.Modflow.load on an existing
    flopy.modflow.Modflow instance."""
    version = m.version
    verbose = m.verbose
    model_ws = m.model_ws

    # similar to modflow command: if file does not exist , try file.nam
    namefile_path = os.path.join(model_ws, m.namefile)
    if (not os.path.isfile(namefile_path) and
            os.path.isfile(namefile_path + '.nam')):
        namefile_path += '.nam'
    if not os.path.isfile(namefile_path):
        raise IOError('cannot find name file: ' + str(namefile_path))

    files_successfully_loaded = []
    files_not_loaded = []

    # set the reference information
    attribs = mfreadnam.attribs_from_namfile_header(namefile_path)

    #ref_attributes = SpatialReference.load(namefile_path)

    # read name file
    ext_unit_dict = mfreadnam.parsenamefile(
        namefile_path, m.mfnam_packages, verbose=verbose)
    if m.verbose:
        print('\n{}\nExternal unit dictionary:\n{}\n{}\n'
              .format(50 * '-', ext_unit_dict, 50 * '-'))

    # create a dict where key is the package name, value is unitnumber
    ext_pkg_d = {v.filetype: k for (k, v) in ext_unit_dict.items()}

    # reset version based on packages in the name file
    if "NWT" in ext_pkg_d or "UPW" in ext_pkg_d:
        version = "mfnwt"
    if "GLOBAL" in ext_pkg_d:
        if version != "mf2k":
            m.glo = ModflowGlobal(m)
        version = "mf2k"
    if "SMS" in ext_pkg_d:
        version = "mfusg"
    if "DISU" in ext_pkg_d:
        version = "mfusg"
        m.structured = False
    # update the modflow version
    m.set_version(version)

    # reset unit number for glo file
    if version == "mf2k":
        if "GLOBAL" in ext_pkg_d:
            unitnumber = ext_pkg_d["GLOBAL"]
            filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
            m.glo.unit_number = [unitnumber]
            m.glo.file_name = [filepth]
        else:
            # TODO: is this necessary? it's not done for LIST.
            m.glo.unit_number = [0]
            m.glo.file_name = [""]

    # reset unit number for list file
    if 'LIST' in ext_pkg_d:
        unitnumber = ext_pkg_d['LIST']
        filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
        m.lst.unit_number = [unitnumber]
        m.lst.file_name = [filepth]

    # look for the free format flag in bas6
    bas_key = ext_pkg_d.get('BAS6')
    if bas_key is not None:
        bas = ext_unit_dict[bas_key]
        start = bas.filehandle.tell()
        line = bas.filehandle.readline()
        while line.startswith("#"):
            line = bas.filehandle.readline()
        if "FREE" in line.upper():
            m.free_format_input = True
        bas.filehandle.seek(start)
    if verbose:
        print("ModflowBas6 free format:{0}\n".format(m.free_format_input))

    # load dis
    dis_key = ext_pkg_d.get('DIS') or ext_pkg_d.get('DISU')
    if dis_key is None:
        raise KeyError('discretization entry not found in nam file')
    disnamdata = ext_unit_dict[dis_key]
    dis = disnamdata.package.load(
        disnamdata.filename, m,
        ext_unit_dict=ext_unit_dict, check=False)
    files_successfully_loaded.append(disnamdata.filename)
    if m.verbose:
        print('   {:4s} package load...success'.format(dis.name[0]))
    m.setup_grid()  # reset model grid now that DIS package is loaded
    assert m.pop_key_list.pop() == dis_key
    ext_unit_dict.pop(dis_key)  #.filehandle.close()

    if load_only is None:
        # load all packages/files
        load_only = ext_pkg_d.keys()
    else:  # check items in list
        if not isinstance(load_only, list):
            load_only = [load_only]
        not_found = []
        for i, filetype in enumerate(load_only):
            load_only[i] = filetype = filetype.upper()
            if filetype not in ext_pkg_d:
                not_found.append(filetype)
        if not_found:
            raise KeyError(
                "the following load_only entries were not found "
                "in the ext_unit_dict: " + str(not_found))

    # zone, mult, pval
    if "PVAL" in ext_pkg_d:
        m.mfpar.set_pval(m, ext_unit_dict)
        assert m.pop_key_list.pop() == ext_pkg_d.get("PVAL")
    if "ZONE" in ext_pkg_d:
        m.mfpar.set_zone(m, ext_unit_dict)
        assert m.pop_key_list.pop() == ext_pkg_d.get("ZONE")
    if "MULT" in ext_pkg_d:
        m.mfpar.set_mult(m, ext_unit_dict)
        assert m.pop_key_list.pop() == ext_pkg_d.get("MULT")

    # try loading packages in ext_unit_dict
    for key, item in ext_unit_dict.items():
        if item.package is not None:
            if item.filetype in load_only:
                if forgive:
                    try:
                        package_load_args = \
                            list(inspect.getfullargspec(item.package.load))[0]
                        if "check" in package_load_args:
                            item.package.load(
                                item.filename, m,
                                ext_unit_dict=ext_unit_dict, check=False)
                        else:
                            item.package.load(
                                item.filename, m,
                                ext_unit_dict=ext_unit_dict)
                        files_successfully_loaded.append(item.filename)
                        if m.verbose:
                            print('   {:4s} package load...success'
                                  .format(item.filetype))
                    except Exception as e:
                        m.load_fail = True
                        if m.verbose:
                            print('   {:4s} package load...failed\n   {!s}'
                                  .format(item.filetype, e))
                        files_not_loaded.append(item.filename)
                else:
                    package_load_args = \
                        list(inspect.getfullargspec(item.package.load))[0]
                    if "check" in package_load_args:
                        item.package.load(
                            item.filename, m,
                            ext_unit_dict=ext_unit_dict, check=False)
                    else:
                        item.package.load(
                            item.filename, m,
                            ext_unit_dict=ext_unit_dict)
                    files_successfully_loaded.append(item.filename)
                    if m.verbose:
                        print('   {:4s} package load...success'
                              .format(item.filetype))
            else:
                if m.verbose:
                    print('   {:4s} package load...skipped'
                          .format(item.filetype))
                files_not_loaded.append(item.filename)
        elif "data" not in item.filetype.lower():
            files_not_loaded.append(item.filename)
            if m.verbose:
                print('   {:4s} package load...skipped'
                      .format(item.filetype))
        elif "data" in item.filetype.lower():
            if m.verbose:
                print('   {} file load...skipped\n      {}'
                      .format(item.filetype,
                              os.path.basename(item.filename)))
            if key not in m.pop_key_list:
                # do not add unit number (key) if it already exists
                if key not in m.external_units:
                    m.external_fnames.append(item.filename)
                    m.external_units.append(key)
                    m.external_binflag.append("binary"
                                              in item.filetype.lower())
                    m.external_output.append(False)
        else:
            raise KeyError('unhandled case: {}, {}'.format(key, item))

    # pop binary output keys and any external file units that are now
    # internal
    for key in m.pop_key_list:
        try:
            m.remove_external(unit=key)
            ext_unit_dict.pop(key)
        except KeyError:
            if m.verbose:
                print('Warning: external file unit {} does not exist in '
                      'ext_unit_dict.'.format(key))

    # write message indicating packages that were successfully loaded
    if m.verbose:
        print('')
        print('   The following {0} packages were successfully loaded.'
              .format(len(files_successfully_loaded)))
        for fname in files_successfully_loaded:
            print('      ' + os.path.basename(fname))
        if len(files_not_loaded) > 0:
            print('   The following {0} packages were not loaded.'
                  .format(len(files_not_loaded)))
            for fname in files_not_loaded:
                print('      ' + os.path.basename(fname))
    if check:
        m.check(f='{}.chk'.format(m.name), verbose=m.verbose, level=0)

    # return model object
    return m


def flopy_mfsimulation_load(sim, model, strict=True, load_only=None,
             verify_data=False):
    """Execute the code in flopy.mf6.MFSimulation.load on
    existing instances of flopy.mf6.MFSimulation and flopy.mf6.MF6model"""

    instance = sim
    if not isinstance(model, list):
        model_instances = [model]
    else:
        model_instances = model
    version = sim.version
    exe_name = sim.exe_name
    verbosity_level = instance.simulation_data.verbosity_level

    if verbosity_level.value >= VerbosityLevel.normal.value:
        print('loading simulation...')

    # build case consistent load_only dictionary for quick lookups
    load_only = PackageContainer._load_only_dict(load_only)

    # load simulation name file
    if verbosity_level.value >= VerbosityLevel.normal.value:
        print('  loading simulation name file...')
    instance.name_file.load(strict)

    # load TDIS file
    tdis_pkg = 'tdis{}'.format(mfstructure.MFStructure().
                               get_version_string())
    tdis_attr = getattr(instance.name_file, tdis_pkg)
    instance._tdis_file = mftdis.ModflowTdis(instance,
                                             filename=tdis_attr.get_data())

    instance._tdis_file._filename = instance.simulation_data.mfdata[
        ('nam', 'timing', tdis_pkg)].get_data()
    if verbosity_level.value >= VerbosityLevel.normal.value:
        print('  loading tdis package...')
    instance._tdis_file.load(strict)

    # load models
    try:
        model_recarray = instance.simulation_data.mfdata[('nam', 'models',
                                                          'models')]
        models = model_recarray.get_data()
    except MFDataException as mfde:
        message = 'Error occurred while loading model names from the ' \
                  'simulation name file.'
        raise MFDataException(mfdata_except=mfde,
                              model=instance.name,
                              package='nam',
                              message=message)
    for item in models:
        # resolve model working folder and name file
        path, name_file = os.path.split(item[1])

        # get the existing model instance
        # corresponding to its entry in the simulation name file
        # (in flopy the model instance is obtained from PackageContainer.model_factory below)
        model_obj = [m for m in model_instances if m.namefile == name_file]
        if len(model_obj) == 0:
            print('model {} attached to {} not found in {}'.format(item, instance, model_instances))
            return
        model_obj = model_obj[0]
        #model_obj = PackageContainer.model_factory(item[0][:-1].lower())

        # load model
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print('  loading model {}...'.format(item[0].lower()))

        instance._models[item[2]] = flopy_mf6model_load(instance, model_obj,
                                                        strict=strict,
                                                        model_rel_path=path,
                                                        load_only=load_only)

        # original flopy code to load model
        #instance._models[item[2]] = model_obj.load(
        #    instance,
        #    instance.structure.model_struct_objs[item[0].lower()], item[2],
        #    name_file, version, exe_name, strict, path, load_only)

    # load exchange packages and dependent packages
    try:
        exchange_recarray = instance.name_file.exchanges
        has_exch_data = exchange_recarray.has_data()
    except MFDataException as mfde:
        message = 'Error occurred while loading exchange names from the ' \
                  'simulation name file.'
        raise MFDataException(mfdata_except=mfde,
                              model=instance.name,
                              package='nam',
                              message=message)
    if has_exch_data:
        try:
            exch_data = exchange_recarray.get_data()
        except MFDataException as mfde:
            message = 'Error occurred while loading exchange names from ' \
                      'the simulation name file.'
            raise MFDataException(mfdata_except=mfde,
                                  model=instance.name,
                                  package='nam',
                                  message=message)
        for exgfile in exch_data:
            if load_only is not None and not \
                    PackageContainer._in_pkg_list(load_only, exgfile[0],
                                          exgfile[2]):
                if instance.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('    skipping package {}..'
                          '.'.format(exgfile[0].lower()))
                continue
            # get exchange type by removing numbers from exgtype
            exchange_type = ''.join([char for char in exgfile[0] if
                                     not char.isdigit()]).upper()
            # get exchange number for this type
            if exchange_type not in instance._exg_file_num:
                exchange_file_num = 0
                instance._exg_file_num[exchange_type] = 1
            else:
                exchange_file_num = instance._exg_file_num[exchange_type]
                instance._exg_file_num[exchange_type] += 1

            exchange_name = '{}_EXG_{}'.format(exchange_type,
                                               exchange_file_num)
            # find package class the corresponds to this exchange type
            package_obj = PackageContainer.package_factory(
                exchange_type.replace('-', '').lower(), '')
            if not package_obj:
                message = 'An error occurred while loading the ' \
                          'simulation name file.  Invalid exchange type ' \
                          '"{}" specified.'.format(exchange_type)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(instance.name,
                                      'nam',
                                      'nam',
                                      'loading simulation name file',
                                      exchange_recarray.structure.name,
                                      inspect.stack()[0][3],
                                      type_, value_, traceback_, message,
                                      instance._simulation_data.debug)

            # build and load exchange package object
            exchange_file = package_obj(instance, exgtype=exgfile[0],
                                        exgmnamea=exgfile[2],
                                        exgmnameb=exgfile[3],
                                        filename=exgfile[1],
                                        pname=exchange_name,
                                        loading_package=True)
            if verbosity_level.value >= VerbosityLevel.normal.value:
                print('  loading exchange package {}..'
                      '.'.format(exchange_file._get_pname()))
            exchange_file.load(strict)
            # Flopy>=3.9
            if hasattr(instance, '_package_container'):
                instance._package_container.add_package(exchange_file)
            instance._exchange_files[exgfile[1]] = exchange_file

    # load simulation packages
    solution_recarray = instance.simulation_data.mfdata[('nam',
                                                         'solutiongroup',
                                                         'solutiongroup'
                                                         )]

    try:
        solution_group_dict = solution_recarray.get_data()
    except MFDataException as mfde:
        message = 'Error occurred while loading solution groups from ' \
                  'the simulation name file.'
        raise MFDataException(mfdata_except=mfde,
                              model=instance.name,
                              package='nam',
                              message=message)
    for solution_group in solution_group_dict.values():
        for solution_info in solution_group:
            if load_only is not None and not PackageContainer._in_pkg_list(
                    load_only, solution_info[0], solution_info[2]):
                if instance.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('    skipping package {}..'
                          '.'.format(solution_info[0].lower()))
                continue
            ims_file = mfims.ModflowIms(instance, filename=solution_info[1],
                                        pname=solution_info[2])
            if verbosity_level.value >= VerbosityLevel.normal.value:
                print('  loading ims package {}..'
                      '.'.format(ims_file._get_pname()))
            ims_file.load(strict)

    instance.simulation_data.mfpath.set_last_accessed_path()
    if verify_data:
        instance.check()
    return instance


def flopy_mf6model_load(simulation, model, strict=True, model_rel_path='.',
                        load_only=None):
    """Execute the code in flopy.mf6.MFmodel.load_base on an
        existing instance of MF6model."""

    instance = model
    modelname = model.name
    structure = model.structure

    # build case consistent load_only dictionary for quick lookups
    load_only = PackageContainer._load_only_dict(load_only)

    # load name file
    instance.name_file.load(strict)

    # order packages
    vnum = mfstructure.MFStructure().get_version_string()
    # FIX: Transport - Priority packages maybe should not be hard coded
    priority_packages = {'dis{}'.format(vnum): 1, 'disv{}'.format(vnum): 1,
                         'disu{}'.format(vnum): 1}
    packages_ordered = []
    package_recarray = instance.simulation_data.mfdata[(modelname, 'nam',
                                                        'packages',
                                                        'packages')]
    for item in package_recarray.get_data():
        if item[0] in priority_packages:
            packages_ordered.insert(0, (item[0], item[1], item[2]))
        else:
            packages_ordered.append((item[0], item[1], item[2]))

    # load packages
    sim_struct = mfstructure.MFStructure().sim_struct
    instance._ftype_num_dict = {}
    for ftype, fname, pname in packages_ordered:
        ftype_orig = ftype
        ftype = ftype[0:-1].lower()
        if ftype in structure.package_struct_objs or ftype in \
                sim_struct.utl_struct_objs:
            if (
                load_only is not None
                and not PackageContainer._in_pkg_list(
                    priority_packages, ftype_orig, pname
                )
                and not PackageContainer._in_pkg_list(load_only, ftype_orig, pname)
            ):
                if (
                    simulation.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(f"    skipping package {ftype}...")
                continue
            if model_rel_path and model_rel_path != '.':
                # strip off model relative path from the file path
                filemgr = simulation.simulation_data.mfpath
                fname = filemgr.strip_model_relative_path(modelname,
                                                          fname)
            if simulation.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('    loading package {}...'.format(ftype))
            # load package
            instance.load_package(ftype, fname, pname, strict, None)

    # load referenced packages
    if modelname in instance.simulation_data.referenced_files:
        for ref_file in \
                instance.simulation_data.referenced_files[modelname].values():
            if (ref_file.file_type in structure.package_struct_objs or
                ref_file.file_type in sim_struct.utl_struct_objs) and \
                    not ref_file.loaded:
                instance.load_package(ref_file.file_type,
                                      ref_file.file_name, None, strict,
                                      ref_file.reference_path)
                ref_file.loaded = True

    # TODO: fix jagged lists where appropriate

    return instance


def which(program):
    """Check for existance of executable.
    https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file


def exe_exists(exe_name):
    exe_path = which(exe_name)
    if exe_path is not None:
        return os.path.exists(exe_path) and \
               os.access(which(exe_path), os.X_OK)


def read_mf6_block(filename, blockname):
    blockname = blockname.lower()
    data = {}
    read = False
    per = None
    with open(filename) as src:
        for line in src:
            line = line.lower()
            if 'begin' in line and blockname in line:
                if blockname == 'period':
                    per = int(line.strip().split()[-1])
                    data[per] = []
                elif blockname == 'continuous':
                    fname = line.strip().split()[-1]
                    data[fname] = []
                elif blockname == 'packagedata':
                    data['packagedata'] = []
                else:
                    blockname = line.strip().split()[-1]
                    data[blockname] = []
                read = blockname
                continue
            if 'end' in line and blockname in line:
                per = None
                read = False
                #break
            if read == 'options':
                line = line.strip().split()
                data[line[0]] = line[1:]
            elif read == 'packages':
                pckg, fname, ext = line.strip().split()
                data[pckg] = fname
            elif read == 'period':
                data[per].append(' '.join(line.strip().split()))
            elif read == 'continuous':
                data[fname].append(' '.join(line.strip().split()))
            elif read == 'packagedata':
                data['packagedata'].append(' '.join(line.strip().split()))
            elif read == blockname:
                data[blockname].append(' '.join(line.strip().split()))
    return data


def read_lak_ggo(f, model,
                 start_datetime='1970-01-01',
                 keep_only_last_timestep=True):
    lake, hydroid = os.path.splitext(os.path.split(f)[1])[0].split('_')
    lak_number = int(lake.strip('lak'))
    df = read_ggofile(f, model=model,
                      start_datetime=start_datetime,
                      keep_only_last_timestep=keep_only_last_timestep)
    df['lake'] = lak_number
    df['hydroid'] = hydroid
    return df


def read_ggofile(gagefile, model,
                start_datetime='1970-01-01',
                keep_only_last_timestep=True):
    with open(gagefile) as src:
        next(src)
        namesstr = next(src)
        names = namesstr.replace('DATA:', '').replace('.', '')\
            .replace('-', '_').replace('(', '').replace(')', '')\
            .replace('"','').strip().split()
        names = [n.lower() for n in names]
        df = pd.read_csv(src, skiprows=0,
                            header=None,
                            delim_whitespace=True,
                            names=names
                            )
    kstp = []
    kper = []
    for i, nstp in enumerate(model.dis.nstp.array):
        for j in range(nstp):
            kstp.append(j)
            kper.append(i)
    if len(df) == len(kstp) + 1:
        df = df.iloc[1:].copy()
    if df.time.iloc[0] == 1:
        df['time'] -= 1
    df['kstp'] = kstp
    df['kper'] = kper
    if keep_only_last_timestep:
        df = df.groupby('kper').last()

    start_ts = pd.Timestamp(start_datetime)
    df['datetime'] = pd.to_timedelta(df.time, unit='D') + start_ts
    df.index = df.datetime
    return df


def add_version_to_fileheader(filename, model_info=None):
    """Add modflow-setup, flopy and optionally model
    version info to an existing file header denoted by
    the comment characters ``#``, ``!``, or ``//``.
    """
    tempfile = str(filename) + '.temp'
    shutil.copy(filename, tempfile)
    with open(tempfile) as src:
        with open(filename, 'w') as dest:
            if model_info is None:
                header = ''
            else:
                header = f'# {model_info}\n'
            read_header = True
            for line in src:
                if read_header and len(line.strip()) > 0 and \
                        line.strip()[0] in {'#', '!', '//'}:
                    if model_info is None or model_info not in line:
                        header += line
                elif read_header:
                    if 'modflow-setup' not in header:
                        headerlist = header.strip().split('\n')
                        if 'flopy' in header.lower():
                            pos, flopy_info = [(i, s) for i, s in enumerate(headerlist)
                                               if 'flopy' in s.lower()][0]
                            #flopy_info = header.strip().split('\n')[-1]
                            if 'version' not in flopy_info.lower():
                                flopy_version = f'flopy version {flopy.__version__}'
                                flopy_info = flopy_info.lower().replace('flopy',
                                                                        flopy_version)
                                headerlist[pos] = flopy_info

                                #header = '\n'.join(header.split('\n')[:-2] +
                                #                   [flopy_info + '\n'])
                            mfsetup_text = '# via '
                            pos += 1  # insert mfsetup header after flopy
                        else:
                            mfsetup_text = '# File created by '
                            pos = -1  # insert mfsetup header at end
                        mfsetup_text += 'modflow-setup version {}'.format(mfsetup.__version__)
                        mfsetup_text += ' at {:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now())
                        headerlist.insert(pos, mfsetup_text)
                        header = '\n'.join(headerlist) + '\n'
                    dest.write(header)
                    read_header = False
                    dest.write(line)
                else:
                    dest.write(line)
    os.remove(tempfile)


def remove_file_header(filename):
    """Remove the header of a MODFLOW input file,
    to allow comparison betwee files that have different
    headers but are otherwise the same, for example."""
    backup_file = str(filename) + '.backup'
    shutil.copy(filename, backup_file)
    with open(backup_file) as src:
        with open(filename, 'w') as dest:
            for line in src:
                if not line.strip().startswith('#'):
                    dest.write(line)
    os.remove(backup_file)
