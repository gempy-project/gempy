from gempy import GeoModel
from test.test_gempy_v3.test_model_construction import test_create_geomodel


def test_SurfacePointsTable_to_df():
    geo_model: GeoModel = test_create_geomodel()
    df = geo_model.structural_frame.surface_points.df
    return 
