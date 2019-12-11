"""
Test lake package functionality
"""
import numpy as np
import pandas as pd
import pytest
from mfsetup.lakes import PrismSourceData


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


def test_parse_prism_source_data(source_data_from_prism_cases, pleasant_nwt_with_grid):
    m = pleasant_nwt_with_grid
    cases = source_data_from_prism_cases
    sd = PrismSourceData.from_config(cases[0]['climate'], dest_model=m)
    data = sd.get_data()
    assert np.array_equal(data.per, m.perioddata.per)
    df = pd.read_csv(sd.filenames[600059060], header=None,
                     skiprows=11, names=['datetime', 'ppt_inches', 'tmean'])
    df.index = pd.to_datetime(df.datetime)

    # subset data to the model timeperiod
    model_start, model_end = '2012-01-01', '2018-12-31'
    df_model = df.loc[model_start:model_end]

    # compute mean precip in meters/day
    df_model['ppt_m'] = df_model['ppt_inches']*.3048/12
    ndays = (df_model.index[-1] - df_model.index[0]).days
    df_model['ppt_md'] = df_model['ppt_m']/df_model.index.days_in_month
    tmean = df_model['tmean'].mean()
    tmean_c = (5/9) * (tmean - 32)
    assert np.allclose(data.loc[data.per == 0, 'temp_c'], tmean_c)
    assert np.allclose(data.loc[data.per == 0, 'ppt_md'], df_model['ppt_md'].mean())
