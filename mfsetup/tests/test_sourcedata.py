import os
import copy
import pytest
import numpy as np
from ..fileio import _parse_file_path_keys_from_source_data
from ..sourcedata import (SourceData, ArraySourceData, TabularSourceData,
                          MFArrayData, MFBinaryArraySourceData,
                          weighted_average_between_layers)
from ..units import convert_length_units, convert_time_units
from mfsetup import MFnwtModel


@pytest.fixture(scope="session")
def cfg(mfnwt_inset_test_cfg_path):
    cfg = MFnwtModel.load_cfg(mfnwt_inset_test_cfg_path)
    # add some stuff just for the tests
    cfg['gisdir'] = os.path.join(cfg['model']['model_ws'], 'gis')
    return cfg


@pytest.fixture(scope="function")
def inset(cfg):
    print('pytest fixture inset')
    cfg = cfg.copy()
    m = MFnwtModel(cfg=cfg, **cfg['model'])
    return m


@pytest.fixture(scope="function")
def inset_with_grid(inset):
    print('pytest fixture inset_with_grid')
    m = inset  #deepcopy(inset)
    cfg = inset.cfg.copy()
    cfg['setup_grid']['grid_file'] = inset.cfg['setup_grid'].pop('output_files').pop('grid_file')
    sd = cfg['setup_grid'].pop('source_data').pop('features_shapefile')
    sd['features_shapefile'] = sd.pop('filename')
    cfg['setup_grid'].update(sd)
    m.setup_grid(**cfg['setup_grid'])
    return inset


@pytest.fixture
def source_data_cases(tmpdir, inset_with_grid):
    inf_array = 'mfsetup/tests/data/plainfieldlakes/source_data/net_infiltration__2012-01-01_to_2017-12-31__1066_by_1145__SUM__INCHES_PER_YEAR.tif'
    botm0 = os.path.abspath('{}/junk0.dat'.format(tmpdir))
    botm1 = os.path.abspath('{}/junk1.dat'.format(tmpdir))
    cases = ['junk_str.csv', # key that is a string
             ['junk_list.tif', 'junk_list2.tif'], # key that is a list
             {'filenames': ['junk_dict_list.tif', # key that is a dict
                            'junk_dict_list.tif']}, # key that is a list within a dict
             {'features_shapefile': # key that is a dict within a dict
                  {'filename': 'plainfieldlakes/source_data/all_lakes.shp',
                   'id_column': 'HYDROID',
                   'include_ids': [600054357, 600054319]
                   }},
             {'infiltration_arrays':
                  {'filenames':
                       {0: inf_array,
                        2: inf_array
                        },
                   'mult': 0.805,
                   'length_units': 'inches',
                   'time_units': 'years',
                   }},
             {'botm':
                  {0: 'junk.tif',
                   1: 0.5,
                   2: 'junk2.tif'
                   },
              'elevation_units': 'feet',
              'top': 'junk.tif'
              },
             {'hk':
                  {0: botm0,
                   1: 0.7,
                   2: botm1,
                   },
              'elevation_units': 'feet',
              },
             {'grid_file': 'grid.yml'},
             {'pdfs': 'postproc/pdfs',
              'rasters': 'postproc/rasters',
              'shapefiles': 'postproc/shps'
              }
             ]

    np.savetxt(botm0, np.ones(inset_with_grid.modelgrid.shape[1:]))
    np.savetxt(botm1, np.zeros(inset_with_grid.modelgrid.shape[1:]))
    return cases


@pytest.fixture
def source_data_from_model_cases(inset):
    nlay, nrow, ncol = inset.parent.dis.botm.array.shape
    alllayers = np.zeros((nlay+1, nrow, ncol))
    alllayers[0] = inset.parent.dis.top.array
    alllayers[1:] = inset.parent.dis.botm.array
    cases = [{'from_parent': {
                0: 0.5, # bottom of layer zero in inset is positioned at half the thickness of parent layer 1
                1: 1, # bottom of layer 1 in inset corresponds to bottom of layer 0 in parent
                2: 2,
                3: 3,
                4: 4},
              'source_modelgrid': inset.parent.modelgrid,
              'source_array': alllayers # source array of different shape than model grid
    },
        {'from_parent': {
            0: 0,  # bottom of layer zero in inset is positioned at half the thickness of parent layer 1
            1: 0.3,  # bottom of layer 1 in inset corresponds to bottom of layer 0 in parent
            2: 0.6,
            3: 1,
            4: 1.5,
            5: 1.9,
            6: 2
        },
            'source_modelgrid': inset.parent.modelgrid,
            'source_array': inset.parent.dis.botm.array
        },
        {'from_parent': {
            'binaryfile': os.path.normpath(os.path.join(inset._config_path, 'plainfieldlakes/pfl.hds'))
        }
        }
    ]
    return cases


def test_parse_source_data(source_data_cases,
                           source_data_from_model_cases,
                           inset_with_grid):
    model = inset_with_grid
    cases = source_data_cases + source_data_from_model_cases
    results = []

    sd = SourceData.from_config(cases[0], type='tabular')
    assert isinstance(sd.filenames, dict)
    assert sd.length_unit_conversion == 1.
    assert sd.time_unit_conversion == 1.
    assert sd.unit_conversion == 1.

    sd = SourceData.from_config(cases[1], type='tabular')
    assert isinstance(sd.filenames, dict)

    sd = SourceData.from_config(cases[2], type='tabular')
    assert isinstance(sd.filenames, dict)

    sd = TabularSourceData.from_config(cases[3]['features_shapefile'])
    assert isinstance(sd.filenames, dict)

    var = 'rech'
    sd = SourceData.from_config(cases[4]['infiltration_arrays'],
                                variable=var,
                                type='array')
    assert isinstance(sd.filenames, dict)
    assert sd.unit_conversion == 1. # no dest model

    # test conversion to model units
    sd = ArraySourceData.from_config(cases[4]['infiltration_arrays'],
                                     variable=var,
                                     dest_model=model)
    assert isinstance(sd.filenames, dict)
    assert sd.unit_conversion == convert_length_units('inches', 'meters') *\
        convert_time_units('years', 'days')
    data = sd.get_data()
    assert isinstance(data, dict)
    assert len(data) == len(cases[4]['infiltration_arrays']['filenames'])
    assert data[0].shape == model.modelgrid.shape[1:]
    assert sd.unit_conversion == 1/12 * .3048 * 1/365.25

    # test averaging of layer between two files
    sd = ArraySourceData.from_config(cases[6]['hk'],
                                     variable='hk',
                                     dest_model=model)
    data = sd.get_data()
    assert isinstance(sd.filenames, dict)
    assert np.allclose(data[1].mean(axis=(0, 1)), cases[6]['hk'][1])

    # test averaging of layers provided in source array
    sd = ArraySourceData.from_config(source_data_from_model_cases[0],
                                     variable='botm',
                                     dest_model=model)
    data = sd.get_data()
    mask = sd._source_grid_mask
    arr0 = sd.regrid_from_source_model(sd.source_array[0],
                                        mask=mask,
                                        method='linear')
    arr1 = sd.regrid_from_source_model(sd.source_array[1],
                                        mask=mask,
                                        method='linear')
    assert np.allclose(np.mean([arr0, arr1], axis=(0)), data[0])

    # TODO: write test for multiplier intermediate layers

    # test mapping of layers from binary file;
    # based on layer bottom mapping
    filename = source_data_from_model_cases[2]['from_parent']['binaryfile']
    source_model = inset_with_grid.parent
    modelname = 'parent'
    inset_with_grid._parent_layers = {0: -0.5, 1: 0, 2: 1, 3: 2, 4: 3}
    sd = MFBinaryArraySourceData(variable='strt', filename=filename,
                                 dest_model=model,
                                 source_modelgrid=source_model.modelgrid,
                                 from_source_model_layers={},
                                 length_units=model.cfg[modelname]['length_units'],
                                 time_units=model.cfg[modelname]['time_units'])
    data = sd.get_data()
    # first two layers in dest model should both be from parent layer 0
    mask = sd._source_grid_mask
    arr0 = sd.regrid_from_source_model(sd.source_array[0],
                                       mask=mask,
                                       method='linear')
    assert np.array_equal(data[0], data[1])
    assert np.array_equal(arr0, data[0])
    inset_with_grid._parent_layers = None # reset


@pytest.mark.parametrize('values, length_units, time_units, mult, expected',
                         [(0.001, 'meters', 'days', 1, 0.001),
                          (0.001, 'feet', 'unknown', 2, 0.002 * .3048),
                          (10, 'inches', 'years', 1, 10/12 * .3048 * 1/365.25)
                          ])
def test_mfarraydata(values, length_units, time_units, mult, expected,
                     inset_with_grid, tmpdir):
    variable = 'rech'
    mfad = MFArrayData(variable=variable,
                       values=values,
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=inset_with_grid)
    data = mfad.get_data()
    assert isinstance(data, dict)
    assert np.allclose(data[0].mean(axis=(0, 1)), expected)

    values = [values, values*2]
    expected = expected * 2
    mfad = MFArrayData(variable=variable,
                       values=values,
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=inset_with_grid)
    data = mfad.get_data()
    assert np.allclose(data[1].mean(axis=(0, 1)), expected)

    values = {0: values[0], 2: values[1]*2}
    expected = expected * 2
    mfad = MFArrayData(variable=variable,
                       values=values,
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=inset_with_grid)
    data = mfad.get_data()
    assert len(data) == 2
    assert np.allclose(data[2].mean(axis=(0, 1)), expected)

    arrayfile0 = os.path.abspath('{}/junk0.txt'.format(tmpdir))
    arrayfile2 = os.path.abspath('{}/junk2.txt'.format(tmpdir))
    origdata = data.copy()
    np.savetxt(arrayfile0, origdata[0])
    np.savetxt(arrayfile2, origdata[2])
    mfad = MFArrayData(variable=variable,
                       values={0: arrayfile0,
                               2: arrayfile2},
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=inset_with_grid)
    data = mfad.get_data()
    assert np.allclose(data[2].mean(axis=(0, 1)),
                       origdata[2].mean(axis=(0, 1)) * mult * mfad.unit_conversion)
    assert np.allclose(data[2].mean(axis=(0, 1)), expected * mult * mfad.unit_conversion)
    assert np.allclose(data[0].mean(axis=(0, 1)),
                       origdata[0].mean(axis=(0, 1)) * mult * mfad.unit_conversion)
    assert np.allclose(data[0].mean(axis=(0, 1)), expected * mult * mfad.unit_conversion/4)


def test_parse_source_data_file_keys(source_data_cases):

    cases = source_data_cases
    expected = [[''],
                [0, 1],
                ['filenames.0', 'filenames.1'],
                ['features_shapefile.filename'],
                ['infiltration_arrays.filenames.0',
                 'infiltration_arrays.filenames.2'
                 ],
                ['botm.0', 'botm.2', 'top'],
                ['hk.0', 'hk.2'],
                ['grid_file'],
                ['pdfs', 'rasters', 'shapefiles']
                ]
    for i, case in enumerate(cases[:-1]):
        keys = _parse_file_path_keys_from_source_data(case)
        assert keys == expected[i]
    keys = _parse_file_path_keys_from_source_data(cases[-1], paths=True)
    assert keys == expected[-1]


def test_weighted_average_between_layers():
    arr0 = np.ones((2, 2))
    arr1 = np.zeros((2, 2))
    weight0 = 0.7
    result = weighted_average_between_layers(arr0, arr1, weight0=0.7)
    assert np.allclose(result.mean(axis=(0, 1)), weight0)
