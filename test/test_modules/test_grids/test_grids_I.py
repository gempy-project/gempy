import pytest

import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data.output.blocks_value_type import ValueType
from gempy_engine.plugins.plotting.helper_functions import plot_block_and_input_2d
from test.conftest import TEST_SPEED, TestSpeed


pytestmark = pytest.mark.skipif(TEST_SPEED.value < TestSpeed.MINUTES.value, reason="Global test speed below this test value.")


def  test_octree():
    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ANTICLINE,
        compute_model=False
    ) 
    
    geo_model.interpolation_options.number_octree_levels = 4
    gp.compute_model(geo_model)

    plot_block_and_input_2d(
        stack_number=0,
        interpolation_input=geo_model.interpolation_input_copy,
        outputs=geo_model.solutions.octrees_output,
        structure=geo_model.structural_frame.input_data_descriptor.stack_structure,
        value_type=ValueType.ids
    )
    
    gpv.plot_3d(geo_model, show_data=True, show_boundaries=True, show_lith=False, image=True)
    
    geo_model.interpolation_options.number_octree_levels = 2
    gp.compute_model(geo_model)

    plot_block_and_input_2d(
        stack_number=0,
        interpolation_input=geo_model.interpolation_input_copy,
        outputs=geo_model.solutions.octrees_output,
        structure=geo_model.structural_frame.input_data_descriptor.stack_structure,
        value_type=ValueType.ids
    )

    gpv.plot_3d(geo_model, show_data=True, show_boundaries=True, show_lith=False, image=True)
