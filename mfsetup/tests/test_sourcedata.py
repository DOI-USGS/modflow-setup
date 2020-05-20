import os
import copy
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from ..fileio import _parse_file_path_keys_from_source_data
from ..sourcedata import (ArraySourceData, TransientArraySourceData,
                          TabularSourceData, TransientTabularSourceData,
                          MFArrayData, MFBinaryArraySourceData, transient2d_to_xarray)
from mfsetup.discretization import weighted_average_between_layers
from ..units import convert_length_units, convert_time_units


@pytest.fixture
def source_data_cases(tmpdir, pfl_nwt_with_grid):
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
              },
             {'flowlines':  # case 9
                  {'nhdplus_paths': ['junk.shp', 'junk2.shp']
                   }
              }
             ]

    np.savetxt(botm0, np.ones(pfl_nwt_with_grid.modelgrid.shape[1:]))
    np.savetxt(botm1, np.zeros(pfl_nwt_with_grid.modelgrid.shape[1:]))
    return cases


@pytest.fixture
def source_data_from_model_cases(pfl_nwt):
    nlay, nrow, ncol = pfl_nwt.parent.dis.botm.array.shape
    alllayers = np.zeros((nlay+1, nrow, ncol))
    alllayers[0] = pfl_nwt.parent.dis.top.array
    alllayers[1:] = pfl_nwt.parent.dis.botm.array
    cases = [{'from_parent': {
                0: 0.5, # bottom of layer zero in pfl_nwt is positioned at half the thickness of parent layer 1
                1: 1, # bottom of layer 1 in pfl_nwt corresponds to bottom of layer 0 in parent
                2: 2,
                3: 3,
                4: 4},
              'source_modelgrid': pfl_nwt.parent.modelgrid,
              'source_array': alllayers # source array of different shape than model grid
    },
        {'from_parent': {
            0: 0,  # bottom of layer zero in pfl_nwt is positioned at half the thickness of parent layer 1
            1: 0.3,  # bottom of layer 1 in pfl_nwt corresponds to bottom of layer 0 in parent
            2: 0.6,
            3: 1,
            4: 1.5,
            5: 1.9,
            6: 2
        },
            'source_modelgrid': pfl_nwt.parent.modelgrid,
            'source_array': pfl_nwt.parent.dis.botm.array
        },
        {'from_parent': {
            'binaryfile': os.path.normpath(os.path.join(pfl_nwt._config_path, 'plainfieldlakes/pfl.hds'))
        }
        }
    ]
    return cases


def test_parse_source_data(source_data_cases,
                           source_data_from_model_cases,
                           pfl_nwt_with_grid, project_root_path):
    model = pfl_nwt_with_grid
    cases = source_data_cases + source_data_from_model_cases
    results = []

    sd = TabularSourceData.from_config(cases[0], type='tabular')
    assert isinstance(sd.filenames, dict)
    assert sd.length_unit_conversion == 1.
    assert sd.time_unit_conversion == 1.
    assert sd.unit_conversion == 1.

    sd = TabularSourceData.from_config(cases[1], type='tabular')
    assert isinstance(sd.filenames, dict)

    sd = TabularSourceData.from_config(cases[2], type='tabular')
    assert isinstance(sd.filenames, dict)

    sd = TabularSourceData.from_config(cases[3]['features_shapefile'])
    assert isinstance(sd.filenames, dict)

    var = 'rech'
    sd = ArraySourceData.from_config(cases[4]['infiltration_arrays'],
                                variable=var,
                                type='array')
    assert isinstance(sd.filenames, dict)
    assert sd.unit_conversion == 1. # no dest model

    sd = TabularSourceData.from_config(cases[9]['flowlines']['nhdplus_paths'])
    assert isinstance(sd.filenames, dict)

    # test conversion to model units
    for i, f in cases[4]['infiltration_arrays']['filenames'].items():
        cases[4]['infiltration_arrays']['filenames'][i] = os.path.join(project_root_path, f)
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
    source_model = pfl_nwt_with_grid.parent
    modelname = 'parent'
    pfl_nwt_with_grid._parent_layers = {0: -0.5, 1: 0, 2: 1, 3: 2, 4: 3}
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
    pfl_nwt_with_grid._parent_layers = None # reset


@pytest.mark.parametrize('values, length_units, time_units, mult, expected',
                         [(0.001, 'meters', 'days', 1, 0.001),
                          (0.001, 'feet', 'unknown', 2, 0.002 * .3048),
                          (10, 'inches', 'years', 1, 10/12 * .3048 * 1/365.25)
                          ])
def test_mfarraydata(values, length_units, time_units, mult, expected,
                     pfl_nwt_with_grid, tmpdir):
    variable = 'rech'
    mfad = MFArrayData(variable=variable,
                       values=values,
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=pfl_nwt_with_grid)
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
                       dest_model=pfl_nwt_with_grid)
    data = mfad.get_data()
    assert np.allclose(data[1].mean(axis=(0, 1)), expected)

    values = {0: values[0], 2: values[1]*2}
    expected = expected * 2
    mfad = MFArrayData(variable=variable,
                       values=values,
                       multiplier=mult,
                       length_units=length_units,
                       time_units=time_units,
                       dest_model=pfl_nwt_with_grid)
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
                       dest_model=pfl_nwt_with_grid)
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
                [], #['pdfs', 'rasters', 'shapefiles']
                ['flowlines.nhdplus_paths.0',
                 'flowlines.nhdplus_paths.1']
                ]
    for i, case in enumerate(cases):
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


def test_transient2d_to_DataArray():
    data = np.random.randn(2, 2, 2)
    times = ['2008-01-01', '2008-02-01']
    result = transient2d_to_xarray(data, time=times)
    assert isinstance(result, xr.DataArray)
    assert result.shape == data.shape
    assert np.all(result['time'] == times)
    assert np.array_equal(result['x'], np.arange(2))
    assert np.array_equal(result['y'], np.arange(2)[::-1])
    assert np.array_equal(result, data)


def test_tabular_source_data(tmpdir, project_root_path, shellmound_model_with_dis):
    
    m = shellmound_model_with_dis
    # capitalize the column names so that they're mixed case
    csvfile = os.path.join(project_root_path, 'mfsetup/tests/data/shellmound/tables/head_obs_well_info.csv')
    input_csv = os.path.join(tmpdir, 'csv_cap_cols.csv')
    df = pd.read_csv(csvfile)
    df.columns = [c.capitalize() for c in df.columns]
    df.to_csv(input_csv, index=False)

    sd = TabularSourceData(filenames=input_csv, data_column='Head_m', id_column='Site_no',
                                    length_units = 'unknown', volume_units=None,
                                    column_mappings=None,
                                    dest_model=m)
    sd.get_data()


def test_transient_tabular_source_data(tmpdir, project_root_path, shellmound_model_with_dis):
    
    m = shellmound_model_with_dis
    # capitalize the column names so that they're mixed case
    csvfile = os.path.join(project_root_path, 'mfsetup/tests/data/shellmound/tables/iwum_m3_1M.csv')
    input_csv = os.path.join(tmpdir, 'csv_cap_cols.csv')
    df = pd.read_csv(csvfile)
    df.columns = [c.capitalize() for c in df.columns]
    # create a geometry column of wkt strings, as would result if the csv were written by geopandas.to_csv
    df['geometry'] = [str(Point(x, y)) for x, y in zip(df.X, df.Y)]
    df.to_csv(input_csv, index=False)

    sd = TransientTabularSourceData(filenames=input_csv, data_column='Flux_m3', 
                                    datetime_column='Start_datetime', id_column='Node',
                                    x_col='X', y_col='Y', 
                                    period_stats={0: 'mean', 1: None, 2: 'none', 3: 'mean'},
                                    length_units='unknown', time_units='unknown', volume_units=None,
                                    column_mappings=None,
                                    dest_model=m)
    # expected period stats
    sd.period_stats
    assert sd.period_stats[0] == {'period_stat': 'mean'}
    assert sd.period_stats[1] == None
    assert sd.period_stats[2] == None
    for per in 3, m.nper-1:
        assert sd.period_stats[per] == {'period_stat': 'mean', 
                                        'start_datetime': m.perioddata.start_datetime[per], 
                                        'end_datetime': m.perioddata.end_datetime[per] - pd.Timedelta(1, unit='days')
                                        }
    data = sd.get_data()
    
    # verify that first (steady-state) period has data
    # if period is steady and no end_datetime_column is given, default should be to take mean for whole file
    assert not any(np.in1d([1, 2], data.per.unique()))

@pytest.mark.skip(reason='still working on tests for other SourceData classes')
def test_transient_array_source_data(pfl_nwt_with_dis):
    filenames = None
    sd = TransientArraySourceData(filenames, 'rech', period_stats=None,
                 length_units='unknown', time_units='days',
                 dest_model=None, source_modelgrid=None, source_array=None,
                 from_source_model_layers=None, datatype=None,
                 resample_method='nearest', vmin=-1e30, vmax=1e30
                 )
