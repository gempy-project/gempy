from gempy import GeoModel
from .test_gempy3_api import _create_data
import gempy as gp
import gempy_viewer


def test_geo_model_to_legacy():

    geo_data: GeoModel = _create_data()

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )
    
    gempy_viewer.plot_2d(geo_data, direction=['y'])
    
    # Convert to legacy data class and plot with legacy function

