"""
Functions for setting up starting heads
"""
import shutil

from mfsetup.fileio import save_array, setup_external_filepaths
from mfsetup.mf5to6 import get_variable_name, get_variable_package_name
from mfsetup.sourcedata import (
    ArraySourceData,
    MFArrayData,
    MFBinaryArraySourceData,
    get_source_data_file_ext,
)
from mfsetup.utils import get_input_arguments


def setup_strt(model, package, strt=None, source_data_config=None,
               filename_fmt='strt_{:03d}.dat', write_fmt='%.2f',
               write_nodata=None,
               **kwargs):

    # default arguments to ArraySourceData
    default_kwargs = {
        'resample_method': 'linear'
    }
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    var = 'strt'
    datatype = 'array3d'
    # model that strt values could come from
    source_model = model.parent
    # for LGR models, use the ultimate parent model if there is one
    if model._is_lgr:
        source_data_config = model.parent.cfg['ic']['source_data']

    default_parent_source_data = model._parent_default_source_data
    strt_config = source_data_config.get('strt')

    # check for parent model and a binary file
    binary_file = False
    if strt_config is not None:
        if 'from_parent' in strt_config:
            if model._is_lgr:
                source_model = source_model.parent
            if source_model is None:
                raise ValueError(("'from_parent' in configuration by no parent model."
                                f"{package} package, {model.name} model.\n"
                                f"source_data config:\n{source_data_config}"))
            if strt_config == 'from_parent':
                default_parent_source_data = True
            else:
                from_parent_cfg = source_data_config['strt'].get('from_parent', {})
                binary_file = from_parent_cfg.get('binaryfile', False)
                kwargs.update(from_parent_cfg)
        elif 'from_model_top' in strt_config:
            default_parent_source_data = False

    sd = None  # source data instance
    # data specified directly
    if strt is not None:
        sd = MFArrayData(variable=var,
                         values=strt,
                         datatype=datatype,
                         dest_model=model, **kwargs)
    # data read from binary file with parent model head solution
    elif binary_file:
        kwargs = get_input_arguments(kwargs, MFBinaryArraySourceData)
        sd = MFBinaryArraySourceData(variable='strt', filename=binary_file,
                                     datatype=datatype,
                                     dest_model=model,
                                     source_modelgrid=source_model.modelgrid,
                                     from_source_model_layers=None,
                                     length_units=model.length_units,
                                     time_units=model.time_units,
                                     **kwargs)
    # data read from Flopy instance of parent model
    elif default_parent_source_data:
        source_variable = get_variable_name(var, source_model.version)
        source_package = get_variable_package_name(var, source_model.version, package)
        source_array = getattr(source_model, source_package).__dict__[source_variable].array
        kwargs = get_input_arguments(kwargs, ArraySourceData)
        sd = ArraySourceData(variable=source_variable, filenames=None,
                             datatype=datatype,
                             dest_model=model,
                             source_modelgrid=source_model.modelgrid,
                             source_array=source_array,
                             from_source_model_layers=model.parent_layers,
                             length_units=model.length_units,
                             time_units=model.time_units,
                             **kwargs)
    # default to setting strt from model top
    elif strt_config is None or 'from_model_top' in strt_config:
        sd = MFArrayData(variable=var,
                         values=[model.dis.top.array] * model.nlay,
                         datatype=datatype,
                         dest_model=model, **kwargs)
    # data from files
    elif isinstance(source_data_config, dict) and source_data_config.get(var) is not None:
            #ext = get_source_data_file_ext(source_data_config, package, var)
            source_data_files = source_data_config.get(var)
            kwargs = get_input_arguments(kwargs, ArraySourceData)
            sd = ArraySourceData.from_config(source_data_files,
                                            datatype=datatype,
                                            variable=var,
                                            dest_model=model,
                                            **kwargs)

    if sd is None:
        raise ValueError((f'Unrecognized input for variable {var}, '
                         f'{package} package, {model.name} model.\n'
                          f'{var} values: {strt}\n'
                          f'source_data config:\n{source_data_config}'))
    else:
        data = sd.get_data()

    filepaths = model.setup_external_filepaths(package, var, filename_fmt,
                                               file_numbers=list(data.keys()))

    # write out array data to intermediate files
    if write_nodata is None:
        write_nodata = model._nodata_value
    for i, arr in data.items():
        save_array(filepaths[i], arr,
                   nodata=write_nodata,
                   fmt=write_fmt)
        # still write intermediate files for MODFLOW-6
        # even though input and output filepaths are same
        if model.version == 'mf6':
            src = filepaths[i]['filename']
            dst = model.cfg['intermediate_data'][var][i]
            shutil.copy(src, dst)
    return filepaths
