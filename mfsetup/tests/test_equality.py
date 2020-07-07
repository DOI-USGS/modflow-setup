import copy

from flopy.datbase import DataInterface
from flopy.mf6.data.mfdatalist import MFList

from mfsetup.equality import list_eq, model_eq, package_eq


def test_model_equality(shellmound_model_with_dis):
    m1 = copy.deepcopy(shellmound_model_with_dis)
    m2 = copy.deepcopy(shellmound_model_with_dis)
    assert model_eq(m1, m2)


def test_package_equality(shellmound_model_with_dis):
    m1 = copy.deepcopy(shellmound_model_with_dis)
    m2 = copy.deepcopy(shellmound_model_with_dis)
    for package in m1.get_package_list():
        pck1 = getattr(m1, package.lower())
        pck2 = getattr(m2, package.lower())
        assert package_eq(pck1, pck2)


def test_list_equality(pleasant_model):
    m1 = copy.deepcopy(pleasant_model)
    m2 = copy.deepcopy(pleasant_model)
    for package in m1.get_package_list():
        pck1 = getattr(m1, package.lower())
        pck2 = getattr(m2, package.lower())
        for v in pck1.data_list:
            if isinstance(v, DataInterface):
                try:
                    arr = v.array
                except:
                    arr = None
                if arr is not None:
                    if isinstance(v, MFList):
                        assert list_eq(pck1, pck2)
