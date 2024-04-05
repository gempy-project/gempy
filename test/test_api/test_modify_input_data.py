from gempy import generate_example_model
from gempy.core.data.enumerators import ExampleModel
import gempy_viewer as gp_viewer
import gempy as gp


def test_modify_surface_point_by_name_and_index():
    model = generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    print(model.structural_frame)
    gp_viewer.plot_2d(
        model,
        direction=['y'],
        show_boundaries=False,  # TODO: Fix boundaries
    )
    
    gp.modify_surface_points(
        geo_model=model,
        elements_names=["fault"],
        Z=800 # This can be an array of the same length as the number of points
    )

    gp_viewer.plot_2d(
        model,
        direction=['y'],
        show_boundaries=False,  # TODO: Fix boundaries
    )



def test_modify_surface_point_by_global_index():
    model = generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    print(model.surface_points_copy.df)
    gp_viewer.plot_2d(
        model,
        direction=['y'],
        show_boundaries=False,  # TODO: Fix boundaries
    )

    gp.modify_surface_points(
        geo_model=model,
        slice=0,
        Z=800
    )

    print(model.surface_points_copy.df)
    gp_viewer.plot_2d(
        model,
        direction=['y'],
        show_boundaries=False,  # TODO: Fix boundaries
    )

