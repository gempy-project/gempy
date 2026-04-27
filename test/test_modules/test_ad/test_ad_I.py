import pytest
from gempy_viewer import GemPyToVista

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer
from gempy_engine.core.data.interp_output import InterpOutput
from test.conftest import TEST_SPEED, TestSpeed
from test.verify_helper import gempy_verify_array

PLOT = True


def test_generate_fold_model():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # Compute the geological model
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64',
            compute_grads=True
        )
    )

    foo = geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field[0]
    foo.backward(retain_graph=True, create_graph=True)
    geo_data.taped_interpolation_input.surface_points
                 

    if PLOT or False:
        gpv = require_gempy_viewer()
        gpv.plot_3d(geo_data, image=True)
