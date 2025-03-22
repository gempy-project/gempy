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
        resolution=np.array([20, 20, 20])
    )

    model.grid.dense_grid = dense_grid
    gp.set_active_grid(
        grid=model.grid,
        grid_type=[model.grid.GridTypes.DENSE],
        reset=True
    )

    model.interpolation_options.evaluation_options.mesh_extraction = False  # * Not extracting the mesh with dual contouring
    gp.compute_model(model)

    # Assert
    assert model.solutions.block_solution_type == RawArraysSolution.BlockSolutionType.DENSE_GRID
    assert model.solutions.dc_meshes is None
    arrays = model.solutions.raw_arrays  # * arrays is equivalent to gempy v2 solutions

    assert arrays.scalar_field_matrix.shape == (3, 8_000)  # * 3 surfaces, 8000 points

    mc_edges, mc_vertices = marching_cubes.compute_marching_cubes(model)

    if PLOT:
        gpv = require_gempy_viewer()
        gtv: gpv.GemPyToVista = gpv.plot_3d(
            model=model,
            show_data=True,
            image=False,
            show=False
        )
        import pyvista as pv

        # TODO: This opens interactive window as of now
        pyvista_plotter: pv.Pltter = gtv.p

        # Add the meshes to the plot
        for i in range(len(mc_vertices)):
            pyvista_plotter.add_mesh(
                pv.PolyData(mc_vertices[i],
                            np.insert(mc_edges[i], 0, 3, axis=1).ravel()),
                color='blue')

        pyvista_plotter.show()


