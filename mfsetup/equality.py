"""
Functions for checking equality between Flopy objects
"""
import flopy
import numpy as np

fm = flopy.modflow
mf6 = flopy.mf6
from flopy.datbase import DataInterface, DataType
from flopy.mbase import ModelInterface


def get_package_list(model):
    if model.version == 'mf6':
        packages = [p.name[0].upper() for p in model.packagelist]
    else:
        packages = model.get_package_list()
    return packages


def package_eq(pk1, pk2):
    """Test for equality between two package objects."""
    for k, v in pk1.__dict__.items():
        v2 = pk2.__dict__[k]
        if k in ['_child_package_groups',
                 '_data_list',
                 '_packagelist',
                 '_simulation_data',
                 'blocks',
                 'dimensions',
                 'package_key_dict',
                 'package_name_dict',
                 'package_type_dict',
                 'post_block_comments',
                 'simulation_data',
                 'structure'
                 ]:
            continue
        # skip packages within packages to avoid RecursionError
        elif isinstance(v, mf6.mfpackage.MFPackage) or\
            isinstance(v, mf6.mfbase.PackageContainer):
            continue
        elif isinstance(v, mf6.mfpackage.MFChildPackages):
            if not package_eq(v, v2):
                return False
        elif k not in pk2.__dict__:
            return False
        elif type(v) == bool:
            if not v == v2:
                return False
        elif type(v) == dict:
            for v_k, v_v in v.items():
                if v_k not in v2:
                    return False
                # skip packages within packages to avoid RecursionError
                if isinstance(v_v, mf6.mfpackage.MFPackage):
                    continue
                elif v[v_k] != v2[v_k]:
                    return False
        elif type(v) in [str, int, float, list]:
            if v != v2:
                return False
        elif isinstance(v, ModelInterface):
            # weak, but calling model_eq would result in recursion
            if v.__repr__() != v2.__repr__():
                return False
        elif isinstance(v, DataInterface):
            if v != v2:
                if v.data_type == DataType.transientlist or \
                        v.data_type == DataType.list:
                    if not list_eq(v, v2):
                        return False
                else:
                    a1, a2 = v.array, v2.array
                    if a1 is None and a2 is None:
                        continue
                    if not isinstance(a1, np.ndarray):
                        if a1 != a2:
                            return False
                    # TODO: this may return False if there are nans
                    elif not np.allclose(v.array, v2.array):
                        return False
        elif v != v2:
            return False
    return True


def list_eq(mflist1, mflist2):
    """Compare two transientlists.
    """
    if isinstance(mflist1, mf6.data.mfdatalist.MFTransientList):
        data1 = {per: ra for per, ra in enumerate(mflist1.array)}
        data2 = {per: ra for per, ra in enumerate(mflist2.array)}
    elif isinstance(mflist1, mf6.data.mfdatalist.MFList):
        data1 = {0: mflist1.array}
        data2 = {0: mflist2.array}
    elif hasattr(mflist1, 'data'):
        data1 = mflist1.data
        data2 = mflist2.data
    else:  # pass on lists that don't have a data attribute for now;
        # this affects the ModflowGwfoc package
        return True
    for k, v in data1.items():
        if k not in data2:
            return False
        v2 = data2[k]
        if v is None and v2 is None:
            continue
        elif not isinstance(v, np.recarray):
            if v != v2:
                return False
        else:
            # compare the two np.recarrays
            # not sure if this will work for all relevant cases
            for c, dtype in v.dtype.fields.items():
                c1 = v[c].copy()
                c2 = v2[c].copy()
                if np.issubdtype(dtype[0].type, np.floating):
                    c1[np.isnan(c1)] = 0
                    c2[np.isnan(c2)] = 0
                if not np.array_equal(c1, c2):
                    return False
    return True


def model_eq(m1, m2):
    """Test for equality between two model objects."""
    if not isinstance(m2, m1.__class__):
        return False
    m1packages = get_package_list(m1)
    m2packages = get_package_list(m2)
    if m1packages != m2packages:
        return False
    if m2.modelgrid != m1.modelgrid:
        if m1.modelgrid.__repr__() != m2.modelgrid.__repr__():
            return False
        for attr in ['xcellcenters', 'ycellcenters']:
            if not np.allclose(getattr(m1.modelgrid, attr),
                               getattr(m2.modelgrid, attr)
                               ):
                return False
    for k, v in m1.__dict__.items():
        if k in [
                 '_packagelist',
                 '_package_paths',
                 'package_key_dict',
                 'package_type_dict',
                 'package_name_dict',
                 '_ftype_num_dict']:
            continue
        elif k not in m2.__dict__:
            return False
        elif type(v) == bool:
            if not v == m2.__dict__[k]:
                return False
        elif type(v) in [str, int, float, dict, list]:
            if v != m2.__dict__[k]:
                pass
            continue
    for pk in m1packages:
        if not package_eq(getattr(m1, pk), getattr(m2, pk)):
            return False
    return True
