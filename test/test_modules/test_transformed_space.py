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
            image=True,
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
        point_x_axis=(800, 800),
        distance_point3=1000,
        zmin=model.grid.extent[4],
        zmax=model.grid.extent[5],
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
            image=True,
            transformed_data=True,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )

        gpv.plot_3d(
            model,
            image=True,
            transformed_data=False,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )


def test_plot_transformed_data_including_grid_transform_octree():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=False)

    i = 4
    # Calculate point_y_axis
    regular_grid = gp.data.grid.RegularGrid.from_corners_box(
        pivot=(200, 200),
        point_x_axis=(800, 800),
        distance_point3=1000,
        zmin=model.grid.extent[4],
        zmax=model.grid.extent[5],
        resolution=np.array([2 ** i] * 3),
        plot=True
    )

    # model.grid.active_grids ^= gp.data.grid.Grid.GridTypes.DENSE

    options = model.interpolation_options
    options.number_octree_levels = 4

    model.grid.set_octree_grid(regular_grid, model.interpolation_options.evaluation_options)
    
    gp.compute_model(model)

    if PLOT:
        gpv = require_gempy_viewer()

        gpv.plot_3d(
            model,
            image=True,
            transformed_data=True,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )

        gpv.plot_3d(
            model,
            image=True,
            transformed_data=False,
            show_boundaries=True,
            show_lith=True,
            kwargs_plot_data={
                    'arrow_size': .01
            }
        )
