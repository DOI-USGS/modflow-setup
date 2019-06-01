import inspect
#try:
#    from ruamel import yaml
#except:
import yaml
import os
import json
try:
    import yaml
except:
    from ruamel import yaml
import time
import numpy as np
import pandas as pd
from flopy.utils import SpatialReference, mfreadnam, TemporalReference
from .grid import MFsetupGrid
from .utils import get_input_arguments


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

def load_array(filename, shape=None):
    """Load an array, ensuring the correct shape."""
    t0 = time.time()
    txt = 'loading {}'.format(filename)
    if shape is not None:
        txt += ', shape={}'.format(shape)
    print(txt, end=', ')
    # arr = np.loadtxt
    # pd.read_csv is >3x faster than np.load_txt
    arr = pd.read_csv(filename, delim_whitespace=True, header=None).values
    if shape is not None:
        if arr.shape != shape:
            if arr.size == np.prod(shape):
                arr = np.reshape(arr, shape)
            else:
                raise ValueError("Data in {} have size {}; should be {}"
                                 .format(filename, arr.shape, shape))
    print("took {:.2f}s".format(time.time() - t0))
    return arr


def save_array(filename, arr, **kwargs):
    """Save and array and print that it was written."""
    t0 = time.time()
    np.savetxt(filename, arr, **kwargs)
    print('wrote {}'.format(filename), end=', ')
    print("took {:.2f}s".format(time.time() - t0))

def flopy_mf2005_load(m, load_only=None, forgive=False, check=False):
    """Execute the code in flopy.modflow.Modflow.load"""
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