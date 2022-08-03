from mfsetup.obs import make_obsname, read_observation_data


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


def test_read_observation_data(test_data_path):

    csvfile = test_data_path / 'shellmound/tables/observations.csv'

    results = read_observation_data(csvfile, column_info={},#'x_location_col', 'x',
                                                          #'y_location_col', 'y'},
                          column_mappings={'obsname': 'comid'})
    assert results['obsname'].dtype == object
    assert isinstance(results['obsname'].values[0], str)
