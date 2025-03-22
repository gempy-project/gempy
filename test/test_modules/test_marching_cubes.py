import numpy as np
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.core.data.grid_modules import RegularGrid
from gempy.optional_dependencies import require_gempy_viewer

from skimage import measure

PLOT = True

def marching_cubes(block, elements, spacing, extent):
    """
        Extract the surface meshes using marching cubes
        Args:
            block (np.array): The block to extract the surface meshes from.
            elements (list): IDs of unique structural elements in model
            spacing (tuple): The spacing between grid points in the block.

        Returns:
            mc_vertices (list): Vertices of the surface meshes.
            mc_edges (list): Edges of the surface meshes.
        """

    # Extract the surface meshes using marching cubes
    mc_vertices = []
    mc_edges = []
    for i in range(0, len(elements)):
        verts, faces, _, _ = measure.marching_cubes(block, i,
                                                    spacing=spacing)
        mc_vertices.append(verts + [extent[0], extent[2], extent[4]])
        mc_edges.append(faces)
    return mc_vertices, mc_edges


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

    # TODO: Maybe to complicated because it includes accounting for faults, multiple elements in groups
    #  and transformation to real coordinates

    # Empty lists to store vertices and edges
    mc_vertices = []
    mc_edges = []

    # Boolean list of fault groups
    faults = model.structural_frame.group_is_fault

    # MC for faults, directly on fault block not on scalar field
    if faults is not None:
        # TODO: This should also use the scalar fields probably
        for i in np.unique(model.solutions.raw_arrays.fault_block)[:-1]:
            fault_block = model.solutions.raw_arrays.fault_block.reshape(model.grid.regular_grid.resolution)
            verts, faces, _, _ = measure.marching_cubes(fault_block,
                                                        i,
                                                        spacing=(model.grid.regular_grid.dx,
                                                                 model.grid.regular_grid.dy,
                                                                 model.grid.regular_grid.dz))
            mc_vertices.append(verts + [model.grid.regular_grid.extent[0],
                                        model.grid.regular_grid.extent[2],
                                        model.grid.regular_grid.extent[4]])
            mc_edges.append(faces)
    else:
        pass

    # Extract scalar field values for elements
    scalar_values = model.solutions.raw_arrays.scalar_field_at_surface_points

    # Get indices of non fault elements
    if faults is not None:
        false_indices = [i for i, fault in enumerate(faults) if not fault]
    else:
        false_indices = np.arange(len(model.structural_frame.structural_groups))

    # Extract marching cubes for non fault elements
    for idx in false_indices:

        # Get correct scalar field for structural group
        scalar_field = model.solutions.raw_arrays.scalar_field_matrix[idx].reshape(model.grid.regular_grid.resolution)

        # Extract marching cubes for each scalar value for all elements of a group
        for i in range(len(scalar_values[idx])):
            verts, faces, _, _ = measure.marching_cubes(scalar_field, scalar_values[idx][i],
                                                        spacing=(model.grid.regular_grid.dx,
                                                                 model.grid.regular_grid.dy,
                                                                 model.grid.regular_grid.dz))

            mc_vertices.append(verts + [model.grid.regular_grid.extent[0],
                                        model.grid.regular_grid.extent[2],
                                        model.grid.regular_grid.extent[4]])
            mc_edges.append(faces)

    # Reorder everything correctly if faults exist
    # TODO: All of the following is just complicated code to reorder the elements to match the order of the elements
    #  in the structural frame, probably unnecessary in gempy strucuture
    #
    # if faults is not None:
    #
    #     # TODO: This is a very convoluted way to get a boolean list of faults per element
    #     bool_list = np.zeros(4, dtype=bool)
    #     for i in range(len(model.structural_frame.structural_groups)):
    #         print(i)
    #         if model.structural_frame.group_is_fault[i]:
    #             for j in range(len(model.structural_frame.structural_groups[i].elements)):
    #                 bool_list[i + j] = True
    #         if not model.structural_frame.group_is_fault[i]:
    #             for k in range(len(model.structural_frame.structural_groups[i].elements)):
    #                 bool_list[i + k] = False
    #
    #     true_count = sum(bool_list)
    #
    #     # Split arr_list into two parts
    #     true_elements_vertices = mc_vertices[:true_count]
    #     false_elements_vertices = mc_vertices[true_count:]
    #     true_elements_edges = mc_edges[:true_count]
    #     false_elements_edges = mc_edges[true_count:]
    #
    #     # Create a new list to store reordered elements
    #     mc_vertices = []
    #     mc_edges = []
    #
    #     # Iterator for both true and false elements
    #     true_idx, false_idx = 0, 0
    #
    #     # Populate reordered_list based on bool_list
    #     for is_true in bool_list:
    #         if is_true:
    #             mc_vertices.append(true_elements_vertices[true_idx] + [model.grid.regular_grid.extent[0],
    #                                                                    model.grid.regular_grid.extent[2],
    #                                                                    model.grid.regular_grid.extent[4]])
    #             mc_edges.append(true_elements_edges[true_idx])
    #             true_idx += 1
    #         else:
    #             mc_vertices.append(false_elements_vertices[false_idx] + [model.grid.regular_grid.extent[0],
    #                                                                      model.grid.regular_grid.extent[2],
    #                                                                      model.grid.regular_grid.extent[4]])
    #             mc_edges.append(false_elements_edges[false_idx])
    #             false_idx += 1

    if PLOT:
        gpv = require_gempy_viewer()
        # gtv: gpv.GemPyToVista = gpv.plot_3d(model, show_data=True, image=True)
        import pyvista as pv
        # pyvista_plotter: pv.Plotter = gtv.p

        # TODO: This opens interactive window as of now
        pyvista_plotter = pv.Plotter()

        # Add the meshes to the plot
        for i in range(len(mc_vertices)):
            pyvista_plotter.add_mesh(
                pv.PolyData(mc_vertices[i],
                            np.insert(mc_edges[i], 0, 3, axis=1).ravel()),
                color='blue')

        pyvista_plotter.show()
