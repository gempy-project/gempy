from gempy.core.data import GeoModel
from test.test_api.test_initialization_and_compute_api import _create_data

def test_id_to_name():
    geo_data: GeoModel = _create_data()

    id_number1 = geo_data.structural_frame.surface_points_copy.df['id'].unique()[0]
    id_number2 = geo_data.structural_frame.surface_points_copy.df['id'].unique()[1]

    assert geo_data.structural_frame.surface_points_copy.id_to_name(id_number1) == 'rock1'
    assert geo_data.structural_frame.surface_points_copy.id_to_name(id_number2) == 'rock2'


def test_surface_points_copy_df():
    geo_data: GeoModel = _create_data()

    assert list(geo_data.structural_frame.surface_points_copy.df.columns) == ['X', 'Y', 'Z', 'id', 'nugget', 'formation']