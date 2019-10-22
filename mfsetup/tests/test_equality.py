import copy
from ..equality import model_eq, package_eq, list_eq


def test_model_equality(shellmound_model_with_dis):
    m1 = copy.deepcopy(shellmound_model_with_dis)
    m2 = copy.deepcopy(shellmound_model_with_dis)
    assert model_eq(m1, m2)
