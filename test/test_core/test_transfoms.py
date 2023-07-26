from gempy import GeoModel
from test.test_api.test_gempy3_api import _create_data


def test_transform_1():
    geo_data: GeoModel = _create_data()
    print(geo_data.transform)
    transformed_xyz = geo_data.transform.apply(geo_data.surface_points.xyz)
    print(transformed_xyz)
    return 