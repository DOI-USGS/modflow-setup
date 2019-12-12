"""
Test lake package functionality
"""
import numpy as np
import pandas as pd
import pytest
from mfsetup.lakes import PrismSourceData, setup_lake_info


@pytest.fixture
def source_data_from_prism_cases():
    cases = [{'climate':
                  {'filenames':
                       {600059060: 'pleasant/source_data/PRISM_ppt_tmean_stable_4km_189501_201901_43.9850_-89.5522.csv'
                        },
                   'format': 'prism',
                   'period_stats':
                       {0: ['mean', '2012-01-01', '2018-12-31'],  # average daily rate for model period for initial steady state
                        1: 'mean'}
                   }
              }
    ]
    return cases


def get_prism_data(prism_datafile):
    """Read data from prism file; subset to model time period
    and convert units."""
    df = pd.read_csv(prism_datafile, header=None,
                     skiprows=11, names=['datetime', 'ppt_inches', 'tmean'])
    df.index = pd.to_datetime(df.datetime)

    # subset data to the model timeperiod
    model_start, model_end = '2012-01-01', '2018-12-31'
    df_model = df.loc[model_start:model_end]

    # compute mean precip in meters/day
    df_model['ppt_m'] = df_model['ppt_inches']*.3048/12
    df_model['ppt_md'] = df_model['ppt_m']/df_model.index.days_in_month
    df_model['tmean_c'] = (5/9) * (df_model['tmean'] - 32)
    return df_model


def test_parse_prism_source_data(source_data_from_prism_cases, pleasant_nwt_with_grid):
    m = pleasant_nwt_with_grid
    cases = source_data_from_prism_cases
    sd = PrismSourceData.from_config(cases[0]['climate'], dest_model=m)
    data = sd.get_data()
    assert np.array_equal(data.per, m.perioddata.per)
    prism = get_prism_data(sd.filenames[600059060])
    assert np.allclose(data.loc[data.per == 0, 'temp'], prism['tmean_c'].mean())
    assert np.allclose(data.loc[data.per == 0, 'precipitation'], prism['ppt_md'].mean())
    assert np.allclose(data.loc[1:, 'temp'], prism['tmean_c'])
    assert np.allclose(data.loc[1:, 'precipitation'], prism['ppt_md'])


def test_setup_lake_info(pleasant_nwt):

    # test setting up lake info without any lake package
    m = pleasant_nwt
    del m.cfg['lak']
    result = setup_lake_info(m)
    assert result is None
    assert m.lake_recharge is None
