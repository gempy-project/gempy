import numpy as np

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer

PLOT = True


def test_plot_transformed_data_only_transform_input():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    print(model.structural_frame)

    if PLOT:
        gpv = require_gempy_viewer()

        gpv.plot_3d(
            model,
            image=True,
            transformed_data=False,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': 10
            }
        )
        
        gpv.plot_3d(
            model,
            image=False,
            transformed_data=True,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )
        
        
def test_plot_transformed_data_including_grid_transform():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=False)

    # Calculate point_y_axis
    regular_grid = gp.data.grid.RegularGrid.from_corners_box(
        pivot=(200, 200),
        point_x_axis=(800,800),
        distance_point3=1000,
        zmin=model.extent[4],
        zmax=model.extent[5],
        resolution=np.array([20, 20, 20]),
        plot=True
    )

    model.grid = gp.data.grid.Grid()
    model.grid.dense_grid = regular_grid
    
    gp.compute_model(model)
    

    if PLOT:
        gpv = require_gempy_viewer()
    
        
        gpv.plot_3d(
            model,
            image=False,
            transformed_data=True,
            show_boundaries=True,
            show_lith=False,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )


        gpv.plot_3d(
            model,
            image=False,
            transformed_data=False,
            show_boundaries=True,
            show_lith=False,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )


def test_transformed_data():
    if transfromed_data := True:  # TODO: Expose this to user
        xyz2 = surface_points.model_transform.apply_with_pivot(
            points=xyz,
            # pivot=np.array([5_478_256.5, 5_698_528.946534388,0]),
            pivot=np.array([5.47825650e+06, 5.69852895e+06, -1.48920000e+03])

        )

        xyz = np.vstack([xyz, xyz2])

    agggg = np.concatenate([mapped_array, mapped_array])

    if transfromed_data := True:
        orientations_xyz2 = orientations.model_transform.apply_with_pivot(
            points=orientations_xyz,
            pivot=np.array([5.47825650e+06, 5.69852895e+06, -1.48920000e+03])
        )
        orientations_grads2 = orientations.model_transform.transform_gradient(orientations_grads)
        arrows_factor /= orientations.model_transform.isometric_scale

        orientations_xyz = np.vstack([orientations_xyz, orientations_xyz2])

        input_transform = regular_grid.input_transform
        transformed = input_transform.apply(regular_grid.bounding_box)  # ! grid already has the grid transform applied
        new_extents = np.array([transformed[:, 0].min(), transformed[:, 0].max(),
                                transformed[:, 1].min(), transformed[:, 1].max(),
                                transformed[:, 2].min(), transformed[:, 2].max()])
