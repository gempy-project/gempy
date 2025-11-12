import os
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

    for e, structural_group in enumerate(structural_groups):
        if e >= len(output_lvl0):
            continue

        output_group: InterpOutput = output_lvl0[e]
        scalar_field_matrix = output_group.exported_fields_dense_grid.scalar_field
        if structural_group.is_fault is False:
            slice_: slice = output_group.grid.dense_grid_slice
            mask = output_group.combined_scalar_field.squeezed_mask_array[slice_]
        else:
            mask = np.ones_like(scalar_field_matrix, dtype=bool)

        for element in structural_group.elements:
            extract_mesh_for_element(
                structural_element=element,
                regular_grid=regular_grid,
                scalar_field=scalar_field_matrix,
                mask=mask
            )


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
    if type(scalar_field).__module__ == 'torch':
        import torch
        scalar_field = scalar_field.detach().numpy()
    if type(mask).__module__ == "torch":
        import torch
        mask = torch.to_numpy(mask)


    # Extract mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        volume=scalar_field.reshape(regular_grid.resolution),
        level=structural_element.scalar_field_at_interface,
        spacing=(regular_grid.dx, regular_grid.dy, regular_grid.dz),
        mask=mask.reshape(regular_grid.resolution) if mask is not None else None,
        allow_degenerate=False,
        method="lewiner"
    )

    # Adjust vertices to correct coordinates in the model's extent
    verts = (verts + [regular_grid.extent[0],
                      regular_grid.extent[2],
                      regular_grid.extent[4]])

    # Store mesh in the structural element
    structural_element.vertices = verts
    structural_element.edges = faces
