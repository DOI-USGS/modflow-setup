import inspect
import sys
import os
import json
import yaml
import time
import numpy as np
import pandas as pd
from flopy.utils import SpatialReference, mfreadnam, TemporalReference
from flopy.mf6.mfbase import PackageContainer, MFFileMgmt, ExtFileAction, \
    PackageContainerType, MFDataException, FlopyException, \
    VerbosityLevel
from flopy.mf6.data import mfstructure
from flopy.mf6.modflow import mfnam, mfims, mftdis, mfgwfgnc, mfgwfmvr
from .grid import MFsetupGrid
from .utils import get_input_arguments, update


def check_source_files(fileslist):
    """Check that the files in fileslist exist.
    """
    if isinstance(fileslist, str):
        fileslist = [fileslist]
    for f in fileslist:
        if not os.path.exists(f):
            raise IOError('Cannot find {}'.format(f))


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
    print('wrote {}'.format(jsonfile))


def load_sr(filename):
    """Create a SpatialReference instance from model config json file."""
    cfg = load(filename)
    return SpatialReference(delr=np.ones(cfg['ncol'])* cfg['delr'],
                            delc=np.ones(cfg['nrow']) * cfg['delc'],
                            xul=cfg['xul'], yul=cfg['yul'],
                            epsg=cfg['epsg']
                              )


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
    arr[np.isnan(arr)] = nodata
    np.savetxt(filename, arr, **kwargs)
    print('wrote {}'.format(filename), end=', ')
    print("took {:.2f}s".format(time.time() - t0))


def load_cfg(cfgfile, default_file='/mfnwt_defaults.yml'):
    """

    Parameters
    ----------
    cfgfile : str
        Path to MFsetup configuration file (json or yaml)

    Returns
    -------
    cfg : dict
        Dictionary of configuration data
    """
    print('loading configuration file {}...'.format(cfgfile))
    source_path = os.path.split(__file__)[0]
    # default configuration
    default_cfg = load(source_path + default_file)
    default_cfg['filename'] = source_path + default_file

    # recursively update defaults with information from yamlfile
    cfg = default_cfg.copy()
    update(cfg, load(cfgfile))
    cfg['model'].update({'verbose': cfgfile})
    cfg['filename'] = os.path.abspath(cfgfile)

    # convert relative paths in the configuration dictionary
    # to absolute paths, based on the location of the config file
    config_file_location = os.path.split(os.path.abspath(cfgfile))[0]
    cfg = set_cfg_paths_to_absolute(cfg, config_file_location)
    return cfg


def set_cfg_paths_to_absolute(cfg, config_file_location):
    if cfg['model']['version'] == 'mf6':
        file_path_keys_relative_to_config = [
            'simulation.sim_ws',
            'parent.simulation.sim_ws',
            'parent.headfile',
        ]
        model_ws = os.path.normpath(os.path.join(config_file_location,
                                                 cfg['simulation']['sim_ws']))
    else:
        file_path_keys_relative_to_config = [
            'model.model_ws',
            'parent.model_ws',
            'parent.headfile',
            'nwt.use_existing_file'
        ]
        model_ws = os.path.normpath(os.path.join(config_file_location,
                                                 cfg['model']['model_ws']))
    file_path_keys_relative_to_model_ws = [
        'setup_grid.grid_file'
    ]
    # add additional paths by looking for source_data
    for pckgname, pckg in cfg.items():
        if isinstance(pckg, dict):
            if 'source_data' in pckg.keys():
                file_keys = _parse_file_path_keys_from_source_data(pckg['source_data'])
                for key in file_keys:
                    file_path_keys_relative_to_config. \
                        append('.'.join([pckgname, 'source_data', key]))
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
    d = cfg[keys[0]]
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
                        'hds', 'cbb', 'cbc']
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
                             filename_format, nfiles=1):
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
    nfiles : int
        Number of external files for the variable (e.g. nlay or nper)

    Returns
    -------
    filepaths : list
        List of external file paths

    Adds intermediated file paths to model.cfg[<package>]['intermediate_data']
    For MODFLOW-6 models, Adds external file paths to model.cfg[<package>][<variable_name>]
    """
    package = package.lower()

    # in lieu of a way to get these from Flopy somehow
    griddata_variables = ['top', 'botm', 'idomain', 'strt',
                          'k', 'k33', 'sy', 'ss']
    transient2D_variables = {'rech', 'recharge',
                             'finf', 'pet', 'extdp', 'extwc',
                             }
    transient3D_variables = {'lakarr', 'bdlknc'}

    model.get_package(package)
    # intermediate data
    filename_format = os.path.split(filename_format)[-1]
    intermediate_files = [os.path.normpath(os.path.join(model.tmpdir,
                          filename_format).format(i)) for i in range(nfiles)]

    if variable_name in transient2D_variables:
        model.cfg['intermediate_data'][variable_name] = {i: f for i, f in
                                                         enumerate(intermediate_files)}
    elif variable_name in transient3D_variables:
        model.cfg['intermediate_data'][variable_name] = {0: intermediate_files}
    else:
        model.cfg['intermediate_data'][variable_name] = intermediate_files

    # external array(s) read by MODFLOW
    # (set to reflect expected locations where flopy will save them)
    external_files = [os.path.normpath(os.path.join(model.model_ws,
                                   model.external_path,
                                   filename_format.format(i))) for i in range(nfiles)]

    if variable_name in transient2D_variables:
        model.cfg['external_files'][variable_name] = {i: f for i, f in
                                                         enumerate(external_files)}
    elif variable_name in transient3D_variables:
        model.cfg['external_files'][variable_name] = {0: external_files}
    else:
        model.cfg['external_files'][variable_name] = external_files

    if model.version == 'mf6':
        # skip these for now (not implemented yet for MF6)
        if variable_name in transient3D_variables:
            return
        filepaths = [{'filename': model.cfg['external_files'][variable_name][i]}
                     for i in range(len(external_files))]
        # set package variable input (to Flopy)
        if variable_name in griddata_variables:
            model.cfg[package]['griddata'][variable_name] = filepaths
        else:
            model.cfg[package][variable_name] = {per: d for per, d in enumerate(filepaths)}
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
    ref_attributes = SpatialReference.load(namefile_path)

    # read name file
    ext_unit_dict = mfreadnam.parsenamefile(
        namefile_path, m.mfnam_packages, verbose=verbose)
    if m.verbose:
        print('\n{}\nExternal unit dictionary:\n{}\n{}\n'
              .format(50 * '-', ext_unit_dict, 50 * '-'))

    # create a dict where key is the package name, value is unitnumber
    ext_pkg_d = {v.filetype: k for (k, v) in ext_unit_dict.items()}

    # version is assumed to be mfnwt
    m.set_version(version)

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
    assert m.pop_key_list.pop() == dis_key
    ext_unit_dict.pop(dis_key)
    start_datetime = ref_attributes.pop("start_datetime", "01-01-1970")
    itmuni = ref_attributes.pop("itmuni", 4)
    ref_source = ref_attributes.pop("source", "defaults")
    # if m.structured:
    #    # get model units from usgs.model.reference, if provided
    #    if ref_source == 'usgs.model.reference':
    #        pass
    #    # otherwise get them from the DIS file
    #    else:
    #        itmuni = dis.itmuni
    #        ref_attributes['lenuni'] = dis.lenuni
    #    sr = SpatialReference(delr=m.dis.delr.array, delc=ml.dis.delc.array,
    #                          **ref_attributes)
    # else:
    #    sr = None
    #
    dis.sr = m.sr
    dis.tr = TemporalReference(itmuni=itmuni, start_datetime=start_datetime)
    dis.start_datetime = start_datetime

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

    # try loading packages in ext_unit_dict
    for key, item in ext_unit_dict.items():
        if item.package is not None:
            if item.filetype in load_only:
                if forgive:
                    try:
                        package_load_args = \
                            list(inspect.getargspec(item.package.load))[0]
                        if "check" in package_load_args:
                            pck = item.package.load(
                                item.filename, m,
                                ext_unit_dict=ext_unit_dict, check=False)
                        else:
                            pck = item.package.load(
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
                        list(inspect.getargspec(item.package.load))[0]
                    if "check" in package_load_args:
                        pck = item.package.load(
                            item.filename, m,
                            ext_unit_dict=ext_unit_dict, check=False)
                    else:
                        pck = item.package.load(
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


def flopy_mfsimulation_load(sim, model, strict=True):
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
        model_obj = [m for m in model_instances if m.namefile == name_file]
        if len(model_obj) == 0:
            print('model {} attached to {} not found in {}'.format(item, instance, model_instances))
            return
        model_obj = model_obj[0]
        #model_obj = PackageContainer.model_factory(item[0][:-1].lower())
        # load model
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print('  loading model {}...'.format(item[0].lower()))
        instance._models[item[2]] = flopy_mf6model_load(instance, model_obj, strict=strict, model_rel_path=path)
        #instance._models[item[2]] = model_obj.load(
        #    instance,
        #    instance.structure.model_struct_objs[item[0].lower()], item[2],
        #    name_file, version, exe_name, strict, path)
#
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
            message = 'Error occurred while loading exchange names from the ' \
                      'simulation name file.'
            raise MFDataException(mfdata_except=mfde,
                                  model=instance.name,
                                  package='nam',
                                  message=message)
        for exgfile in exch_data:
            # get exchange type by removing numbers from exgtype
            exchange_type = ''.join([char for char in exgfile[0] if
                                     not char.isdigit()]).upper()
            # get exchange number for this type
            if not exchange_type in instance._exg_file_num:
                exchange_file_num = 0
                instance._exg_file_num[exchange_type] = 1
            else:
                exchange_file_num = instance._exg_file_num[exchange_type]
                instance._exg_file_num[exchange_type] += 1

            exchange_name = '{}_EXG_{}'.format(exchange_type,
                                               exchange_file_num)
            # find package class the corresponds to this exchange type
            package_obj = instance.package_factory(
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
            ims_file = mfims.ModflowIms(instance, filename=solution_info[1],
                                        pname=solution_info[2])
            if verbosity_level.value >= VerbosityLevel.normal.value:
                print('  loading ims package {}..'
                      '.'.format(ims_file._get_pname()))
            ims_file.load(strict)

    instance.simulation_data.mfpath.set_last_accessed_path()
    return instance


def flopy_mf6model_load(simulation, model, strict=True, model_rel_path='.'):
    """Execute the code in flopy.mf6.MFmodel.load_base on an
        existing instance of MF6model."""

    instance = model
    modelname = model.name
    structure = model.structure

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
        ftype = ftype[0:-1].lower()
        if ftype in structure.package_struct_objs or ftype in \
                sim_struct.utl_struct_objs:
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
    with open(filename) as src:
        for line in src:
            line = line.lower()
            if 'begin' in line and blockname in line:
                read = True
                continue
            if blockname == 'options' and read:
                line = line.strip().split()
                data[line[0]] = line[1:]
            if 'end' in line and blockname in line:
                read = False
                break
    return data
