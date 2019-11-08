import os
import copy
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from ..fileio import _parse_file_path_keys_from_source_data
from ..sourcedata import (ArraySourceData, TabularSourceData,
                          MFArrayData, MFBinaryArraySourceData, transient2d_to_xarray)
from mfsetup.tdis import aggregate_dataframe_to_stress_period
from mfsetup.discretization import weighted_average_between_layers
from ..units import convert_length_units, convert_time_units
from mfsetup import MFnwtModel
from mfsetup.utils import get_input_arguments


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

@pytest.mark.parametrize('dates', [('2007-04-01', '2007-03-31'),
                                   ('2008-04-01', '2008-09-30'),
                                   ('2008-10-01', '2009-03-31')])
@pytest.mark.parametrize('sourcefile', ['tables/sp69_pumping_from_meras21_m3.csv',
                                        'tables/iwum_m3_1M.csv',
                                        'tables/iwum_m3_6M.csv'])
def test_aggregate_dataframe_to_stress_period(shellmound_datapath, sourcefile, dates):
    """
    dates
    1) similar to initial steady-state period that doesn't represent a real time period
    2) period that spans one or more periods in source data
    3) period that doesn't span any periods in source data

    sourcefiles

    'tables/sp69_pumping_from_meras21_m3.csv'
        case where dest. period is completely within the start and end dates for the source data
        (source start date is before dest start date; source end date is after dest end date)

    'tables/iwum_m3_6M.csv'
        case where 1) start date coincides with start date in source data; end date spans
        one period in source data. 2) start and end date do not span any periods
        in source data (should return a result of length 0)

    'tables/iwum_m3_1M.csv'
        case where 1) start date coincides with start date in source data; end date spans
        multiple periods in source data. 2) start and end date do not span any periods
        in source data (should return a result of length 0)
    Returns
    -------

    """
    start, end = dates
    welldata = pd.read_csv(os.path.join(shellmound_datapath, sourcefile
                                        ))

    welldata['start_datetime'] = pd.to_datetime(welldata.start_datetime)
    welldata['end_datetime'] = pd.to_datetime(welldata.end_datetime)
    duplicate_well = welldata.groupby('node').get_group(welldata.node.values[0])
    welldata = welldata.append(duplicate_well)
    start_datetime = pd.Timestamp(start)
    end_datetime = pd.Timestamp(end)  # pandas convention of including last day
    result = aggregate_dataframe_to_stress_period(welldata,
                                                  start_datetime=start_datetime,
                                                  end_datetime=end_datetime,
                                                  period_stat='mean',
                                                  id_column='node',
                                                  data_column='flux_m3')
    overlap = (welldata.start_datetime < end_datetime) & \
                               (welldata.end_datetime > start_datetime)
    #period_inside_welldata = (welldata.start_datetime < start_datetime) & \
    #                         (welldata.end_datetime > end_datetime)
    #overlap = welldata_overlaps_period #| period_inside_welldata

    # for each location (id), take the mean across source data time periods
    agg = welldata.loc[overlap].copy().groupby(['start_datetime', 'node']).sum().reset_index()
    agg = agg.groupby('node').mean().reset_index()
    if end_datetime < start_datetime:
        assert result['flux_m3'].sum() == 0
    if overlap.sum() == 0:
        assert len(result) == 0
    expected_sum = agg['flux_m3'].sum()
    if duplicate_well.node.values[0] in agg.index:
        dw_overlaps = (duplicate_well.start_datetime < end_datetime) & \
                (duplicate_well.end_datetime > start_datetime)
        expected_sum += duplicate_well.loc[dw_overlaps, 'flux_m3'].mean()
    assert np.allclose(result['flux_m3'].sum(), expected_sum)


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