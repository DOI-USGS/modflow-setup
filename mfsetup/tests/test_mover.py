"""Test functions in the mover.py module. See test_lgr.py for
a test of the test_mover_get_sfr_package_connections function.

"""
from copy import deepcopy

import pytest
from shapely.geometry import Point

from mfsetup import MF6model
from mfsetup.mover import get_connections


@pytest.mark.parametrize('from_features,to_features,expected_n_connections',
                         (([Point(1, 1)], [Point(1, 1)], 1),
                          ([Point(1, 1)], [Point(2, 2)], 0)
                          )
                         )
def test_get_connections(from_features, to_features, expected_n_connections):
    """Simple test for the get_connections function,
    which is not currently in the code but may be useful
    for implementing a proximity-based routing feature in
    SFRmaker (for flowlines that only include routing information).
    """
    results = get_connections(from_features, to_features,
                              distance_threshold=1)
    assert len(results) == expected_n_connections


def test_sfr_mover(pleasant_mf6_cfg, project_root_path, tmpdir):
    """Test that 'mover' gets added to the SFR Package input file options block
    when it is specified in the configuration file options block."""
    cfg = deepcopy(pleasant_mf6_cfg)
    cfg['sfr']['options']['mover'] = True
    m = MF6model(cfg=cfg)
    m.setup_dis()
    m.setup_tdis()
    m.setup_sfr()
    m.write_input()
    with open(m.sfr.filename) as src:
        text = src.read()
        assert 'mover' in text
