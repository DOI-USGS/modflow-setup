import numpy as np
import pandas as pd
import pytest

from mfsetup.obs import (
    make_obsname,
    parse_observation_layers,
    read_observation_data,
)


def test_make_obsname():
    maxlen = 13
    names = ['345612091522401',
             '345612091522401',
             '345612091522401',
             '445612091522401',
             '345612091522401',
]
    expected = [names[0][-maxlen:],
                names[1][-maxlen-1:-1],
                names[2][-maxlen-2:-2],
                names[3][-maxlen-2:-2],
                names[0][-maxlen:]
                ]
    unique_names = set()
    for i, n in enumerate(names):
        result = make_obsname(n, unique_names, maxlen=maxlen)
        unique_names.add(result)
        assert len(result) <= maxlen
        assert result == expected[i]


@pytest.mark.parametrize('value,expected', [
    # single layer specifications
    (2, [2]),
    (3.0, [3]),
    ('2', [2]),
    # lists/tuples/arrays of layers
    ([0, 2], [0, 2]),
    ((1, 3), [1, 3]),
    (np.array([0, 4]), [0, 4]),
    # strings with multiple layers (comma- and/or space-separated)
    ('0, 2', [0, 2]),
    ('1 3', [1, 3]),
    ('0,2,4', [0, 2, 4]),
    # missing/empty values fall back to all layers (None)
    (None, None),
    (np.nan, None),
    ('', None),
    ('   ', None),
    ('nan', None),
    ('None', None),
])
def test_parse_observation_layers(value, expected):
    result = parse_observation_layers(value, nlay=5)
    assert result == expected
    # parsed layers should always be a list of python ints (or None)
    if result is not None:
        assert all(isinstance(k, int) for k in result)


@pytest.mark.parametrize('value', [9, -1, [0, 9], '2, 7'])
def test_parse_observation_layers_out_of_range(value):
    with pytest.raises(ValueError, match='outside of the model'):
        parse_observation_layers(value, nlay=5)


def test_parse_observation_layers_no_nlay_skips_validation():
    # with no nlay provided, out-of-range values are not validated
    assert parse_observation_layers(99) == [99]


def test_read_observation_data(test_data_path):

    csvfile = test_data_path / 'shellmound/tables/observations.csv'

    results = read_observation_data(csvfile, column_info={},#'x_location_col', 'x',
                                                          #'y_location_col', 'y'},
                          column_mappings={'obsname': 'comid'})
    assert pd.api.types.is_string_dtype(results['obsname'].dtype)
    assert isinstance(results['obsname'].values[0], str)
