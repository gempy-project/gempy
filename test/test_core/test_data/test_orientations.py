from gempy.core.data import GeoModel
from test.test_api.test_initialization_and_compute_api import _create_data

def test_surface_points_copy_df():
    geo_data: GeoModel = _create_data()

    assert list(geo_data.structural_frame.surface_points_copy.df.columns) == ['X', 'Y', 'Z', 'id', 'nugget', 'formation']