"""Tests for the oc.py module
"""
import pytest

from mfsetup.oc import fill_oc_stress_period_data, parse_oc_period_input


@pytest.mark.parametrize('input,expected,output_fmt', [
    # dictionary-based flopy-like input to (mf6) flopy-style input
    ({'saverecord': {0: {'head': 'last', 'budget': 'last'}}},
     {'saverecord': {0: [('head', 'last'), ('budget', 'last')]}},
     'mf6'),
    # mf6-style input to (mf6) flopy-style input
    ({'period_options': {0: ['save head last', 'save budget last']}},
     {'saverecord': {0: [('head', 'last'), ('budget', 'last')]}},
     'mf6'),
    # mf6-style input to flopy-style input
    ({'period_options': {0: ['save head first', 'save budget first']}},
     {'stress_period_data': {(0, 0): ['save head', 'save budget']}},
     'mfnwt'),
    ({'period_options': {0: ['save head last', 'save budget last']}},
     {'stress_period_data': {(0, 9): ['save head', 'save budget']}},
     'mfnwt'),
    ({'period_options': {0: ['save head frequency 5', 'save budget frequency 5']}},
     {'stress_period_data': {(0, 0): ['save head', 'save budget'],
                             (0, 5): ['save head', 'save budget']}},
     'mfnwt'),
    ({'period_options': {0: ['save head steps 2 3', 'save budget steps 2 3']}},
     {'stress_period_data': {(0, 2): ['save head', 'save budget'],
                             (0, 3): ['save head', 'save budget']}},
     'mfnwt'),
    ({'period_options': {0: ['save head all', 'save budget all']}},
     None, 'mfnwt'
    ),
    # input already in flopy format
    ({'stress_period_data': {(0, 2): ['save head', 'save budget'],
                             (0, 3): ['save head', 'save budget']}},
     {},
    'mfnwt')
                                        ],
                         )
def test_parse_oc_period_input(input, expected, output_fmt):
    results = parse_oc_period_input(input, nstp=[10], output_fmt=output_fmt)
    # kludge for testing 'all'
    if expected is None:
        expected = {'stress_period_data': {(0, i): ['save head', 'save budget']
                                           for i in range(10)}}
    assert results == expected


@pytest.mark.parametrize('stress_period_data,nper',
                         [({(0, 0): ['save head', 'save budget'],
                            (3, 0): ['save head']
                            },
                           5)]
                         )
def test_fill_oc_stress_period_data(stress_period_data, nper):
    results = fill_oc_stress_period_data(stress_period_data, nper)
    expected = {(0, 0): ['save head', 'save budget'],
                (1, 0): ['save head', 'save budget'],
                (2, 0): ['save head', 'save budget'],
                (3, 0): ['save head'],
                (4, 0): ['save head']}
    assert results == expected
