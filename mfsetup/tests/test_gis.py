import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
import pyproj
from ..gis import get_proj4, df2shp, shp2df, shp_properties


def test_get_proj4(tmpdir):
    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-90 +k=0.9996 +x_0=520000 +y_0=-4480000 +datum=NAD83 +units=m +no_defs '
    p1 = pyproj.Proj(proj4)
    f = os.path.join(tmpdir, 'junk.shp')
    df2shp(pd.DataFrame({'id': [0],
                         'geometry': [Point(0, 0)]
                         }),
           f, proj4=proj4)
    proj4_2 = get_proj4(f.replace('shp', 'prj'))
    p2 = pyproj.Proj(proj4_2)
    assert p1 == p2


def test_shp_properties():
    df = pd.DataFrame({'reach': [1], 'value': [1.0], 'name': ['stuff']}, index=[0])
    df = df[['name', 'reach', 'value']].copy()
    assert [d.name for d in df.dtypes] == ['object', 'int64', 'float64']
    assert shp_properties(df) == {'name': 'str', 'reach': 'int', 'value': 'float'}


def test_integer_dtypes(tmpdir):

    # verify that pandas is recasting numpy ints as python ints when converting to dict
    # (numpy ints invalid for shapefiles)
    d = pd.DataFrame(np.ones((3, 3)), dtype=int).astype(object).to_dict(orient='records')
    for i in range(3):
        assert isinstance(d[i][0], int)

    df = pd.DataFrame({'r': np.arange(100), 'c': np.arange(100)})
    f = '{}/ints.dbf'.format(tmpdir)
    df2shp(df, f)
    df2 = shp2df(f)
    assert np.all(df == df2)


def test_boolean_dtypes(tmpdir):

    df = pd.DataFrame([False, True]).transpose()
    df.columns = ['true', 'false']
    f = '{}/bool.dbf'.format(tmpdir)
    df2shp(df, f)
    df2 = shp2df(f, true_values='True', false_values='False')
    assert np.all(df == df2)
