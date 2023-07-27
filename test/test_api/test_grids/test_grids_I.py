import pytest
import gempy as gp
from gempy.core.data.enumerators import ExampleModel


def  test_octree():
    geo_model: gp.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.HORIZONTAL_STRAT,
        compute_model=False
    ) 
    
    
    