import numpy as np
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.core.data.grid_modules import RegularGrid
from gempy.modules.mesh_extranction import marching_cubes
from gempy.optional_dependencies import require_gempy_viewer

PLOT = True


def test_marching_cubes_implementation():
    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)

    # Change the grid to only be the dense grid
    dense_grid: RegularGrid = RegularGrid(
        extent=model.grid.extent,
        resolution=np.array([40, 20, 20])
    )

    model.grid.dense_grid = dense_grid
    gp.set_active_grid(
        grid=model.grid,
        grid_type=[model.grid.GridTypes.DENSE],
        reset=True
    )

    model.interpolation_options.evaluation_options.number_octree_levels = 1
    model.interpolation_options.evaluation_options.mesh_extraction = False  # * Not extracting the mesh with dual contouring
    gp.compute_model(model)

    # Assert
    assert model.solutions.block_solution_type == RawArraysSolution.BlockSolutionType.DENSE_GRID
    assert model.solutions.dc_meshes is None
    arrays = model.solutions.raw_arrays  # * arrays is equivalent to gempy v2 solutions

    # assert arrays.scalar_field_matrix.shape == (3, 8_000)  # * 3 surfaces, 8000 points

    marching_cubes.set_meshes_with_marching_cubes(model)

    if PLOT:
        gpv = require_gempy_viewer()
        gtv: gpv.GemPyToVista = gpv.plot_3d(
            model=model,
            show_data=True,
            image=False,
            show=True
        )
