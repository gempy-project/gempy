import numpy as np
from typing import Optional
from skimage import measure
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution

from gempy.core.data import GeoModel, StructuralElement, StructuralGroup
from gempy.core.data.grid_modules import RegularGrid


def set_meshes_with_marching_cubes(model: GeoModel) -> None:
    """Extract meshes for all structural elements using the marching cubes algorithm.
    
    Parameters
    ----------
    model : GeoModel
        The geological model containing solutions and structural elements.
    
    Raises
    ------
    ValueError
        If the model solutions do not contain dense grid data.
    """
    # Verify that solutions contain dense grid data
    solution_not_having_dense: bool = model.solutions.block_solution_type != RawArraysSolution.BlockSolutionType.DENSE_GRID
    if model.solutions is None or solution_not_having_dense:
        raise ValueError("Model solutions must contain dense grid data for mesh extraction.")

    regular_grid: RegularGrid = model.grid.regular_grid
    structural_groups: list[StructuralGroup] = model.structural_frame.structural_groups

    if not model.solutions.octrees_output or not model.solutions.octrees_output[0].outputs_centers:
        raise ValueError("No interpolation outputs available for mesh extraction.")

    output_lvl0: list[InterpOutput] = model.solutions.octrees_output[0].outputs_centers

    # TODO: How to get this properly in gempy
    # get a list of indices of the lithological groups
    lith_group_indices = []
    fault_group_indices = []
    index = 0
    for i in model.structural_frame.structural_groups:
        if i.is_fault:
            fault_group_indices.append(index)
        else:
            lith_group_indices.append(index)
        index += 1

    # extract scalar field values at surface points
    scalar_values = model.solutions.raw_arrays.scalar_field_at_surface_points

    # TODO: Here I just get my own masks, cause the gempy masks dont work as expected
    masks = _get_masking_arrays(lith_group_indices, model, scalar_values)

    # TODO: Attribute of element.scalar_field was None, changed it to scalar field value of that element
    #  This should probably be done somewhere else and maybe renamed to scalar_field_value?
    #  This is just the most basic solution to be clear what I did
    # _set_scalar_field_to_element(model, output_lvl0, structural_groups)

    # Trying to use the exiting gempy masks
    # masks = []
    # masks.append(
    #     np.ones_like(model.solutions.raw_arrays.scalar_field_matrix[0].reshape(model.grid.regular_grid.resolution),
    #                  dtype=bool))
    # for idx in lith_group_indices:
    #     output_group: InterpOutput = output_lvl0[idx]
    #     masks.append(output_group.mask_components[8:].reshape(model.grid.regular_grid.resolution))

    non_fault_counter = 0
    for e, structural_group in enumerate(structural_groups):
        if e >= len(output_lvl0):
            continue

        # Outdated?
        # output_group: InterpOutput = output_lvl0[e]
        # scalar_field_matrix = output_group.exported_fields_dense_grid.scalar_field

        # Specify the correct scalar field, can be removed in the future
        scalar_field = model.solutions.raw_arrays.scalar_field_matrix[e].reshape(model.grid.regular_grid.resolution)

        # pick mask depending on whether the structural group is a fault or not
        if structural_group.is_fault:
            mask = np.ones_like(scalar_field, dtype=bool)
        else:
            mask = masks[non_fault_counter]  # TODO: I need the entry without faults here
            non_fault_counter += 1

        for element in structural_group.elements:
            extract_mesh_for_element(
                structural_element=element,
                regular_grid=regular_grid,
                scalar_field=scalar_field,
                mask=mask
            )


# TODO: This should be set somewhere else
def _set_scalar_field_to_element(model, output_lvl0, structural_groups):
    element: StructuralElement
    counter = 0
    for e, structural_group in enumerate(structural_groups):
        if e >= len(output_lvl0):
            continue

        for element in structural_group.elements:
            element.scalar_field_at_interface = model.solutions.scalar_field_at_surface_points[counter]
            counter += 1


# TODO: This should be set somewhere else
def _get_masking_arrays(lith_group_indices, model, scalar_values):
    masks = []
    masks.append(np.ones_like(model.solutions.raw_arrays.scalar_field_matrix[0].reshape(model.grid.regular_grid.resolution),
                              dtype=bool))
    for idx in lith_group_indices:
        mask = model.solutions.raw_arrays.scalar_field_matrix[idx].reshape(model.grid.regular_grid.resolution) <= \
               scalar_values[idx][-1]

        masks.append(mask)
    return masks


def extract_mesh_for_element(structural_element: StructuralElement,
                             regular_grid: RegularGrid,
                             scalar_field: np.ndarray,
                             mask: Optional[np.ndarray] = None) -> None:
    """Extract a mesh for a single structural element using marching cubes.
    
    Parameters
    ----------
    structural_element : StructuralElement
        The structural element for which to extract a mesh.
    regular_grid : RegularGrid
        The regular grid defining the spatial discretization.
    scalar_field : np.ndarray
        The scalar field used for isosurface extraction.
    mask : np.ndarray, optional
        Optional mask to restrict the mesh extraction to specific regions.
    """
    # Extract mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        volume=scalar_field,
        level=structural_element.scalar_field_at_interface,
        spacing=(regular_grid.dx, regular_grid.dy, regular_grid.dz),
        mask=mask
    )

    # Adjust vertices to correct coordinates in the model's extent
    verts = (verts + [regular_grid.extent[0],
                      regular_grid.extent[2],
                      regular_grid.extent[4]])

    # Store mesh in the structural element
    structural_element.vertices = verts
    structural_element.edges = faces
