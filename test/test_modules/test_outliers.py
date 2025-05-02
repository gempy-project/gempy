import gempy as gp
from gempy.optional_dependencies import require_gempy_viewer
from gempy.core.data.enumerators import ExampleModel
from gempy.core.data.grid_modules import RegularGrid
import numpy as np

PLOT = True


def test_outliers_model_1():
    # Path to input data
    data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"
    path_to_data = data_path + "/data/input_data/video_tutorials_v3/"

    # Create instance of geomodel
    model = gp.create_geomodel(
        project_name='tutorial_model_onlap_1',
        extent=[0, 2000, 0, 1000, 0, 1000],
        resolution=[100, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "tutorial_model_onlap_1_orientations.csv?cache=",
            path_to_surface_points=path_to_data + "tutorial_model_onlap_1_surface_points.csv?cache="
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=model,
        mapping_object={
                "Young_Series": ("basin_fill_2", "basin_fill_1"),
                "Old_Series"  : ("basin_top", "basin_bottom")
        }
    )

    # Set the relation of the youngest group to Onlap
    from gempy_engine.core.data.stack_relation_type import StackRelationType
    model.structural_frame.structural_groups[0].structural_relation = StackRelationType.ONLAP
    
    model.interpolation_options.sigmoid_slope = 5_000_000

    # Compute a solution for the model
    gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy
        )
    )

    # Assert
    arrays = model.solutions.raw_arrays  # * arrays is equivalent to gempy v2 solutions
    assert arrays.scalar_field_matrix.shape == (2, 250_000)  # * 2 groups, 250000 points

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_2d(
            model=model,
            show_data=True,
            show_boundaries=False,
            show=True
        )

        gpv.plot_2d(
            model=model,
            show_data=False,
            show_boundaries=False,
            show_scalar=True,
            show=True
        )


        gpv.plot_2d(
            model=model,
            show_data=False,
            show_boundaries=False,
            show_scalar=True,
            series_n=1,
            show=True
        )
