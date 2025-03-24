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
    if (model.solutions is None or 
            model.solutions.block_solution_type != RawArraysSolution.BlockSolutionType.DENSE_GRID):
        raise ValueError("Model solutions must contain dense grid data for mesh extraction.")
    
    regular_grid: RegularGrid = model.grid.regular_grid
    structural_groups: list[StructuralGroup] = model.structural_frame.structural_groups

    print(model.solutions.scalar_field_at_surface_points)

    if not model.solutions.octrees_output or not model.solutions.octrees_output[0].outputs_centers:
        raise ValueError("No interpolation outputs available for mesh extraction.")
    
    output_lvl0: list[InterpOutput] = model.solutions.octrees_output[0].outputs_centers

    # TODO: Attribute of element.scalar_field was None, changed it to scalar field value of that element
    #  This should probably be done somewhere else and maybe renamed to scalar_field_value?
    #  This is just the most basic solution to be clear what I did
    counter = 0
    for e, structural_group in enumerate(structural_groups):
        if e >= len(output_lvl0):
            continue

        for element in structural_group.elements:
            element.scalar_field = model.solutions.scalar_field_at_surface_points[counter]
            counter += 1

    for e, structural_group in enumerate(structural_groups):
        if e >= len(output_lvl0):
            continue
            
        output_group: InterpOutput = output_lvl0[e]
        scalar_field_matrix = output_group.exported_fields_dense_grid.scalar_field

        for element in structural_group.elements:
            extract_mesh_for_element(
                structural_element=element,
                regular_grid=regular_grid,
                scalar_field=scalar_field_matrix
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
    # Apply mask if provided
    volume = scalar_field.reshape(regular_grid.resolution)
    if mask is not None:
        volume = volume * mask
    
    # TODO: We need to pass the mask arrays to the marching cubes to account for discontinuities. The mask array are
    #  in InterpOutput too if I remember correctly.

    # Extract mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        volume=volume,
        level=structural_element.scalar_field,
        spacing=(regular_grid.dx, regular_grid.dy, regular_grid.dz),
        mask=None
    )
    
    # Adjust vertices to correct coordinates in the model's extent
    verts = (verts + [regular_grid.extent[0],
                      regular_grid.extent[2],
                      regular_grid.extent[4]])

    # Store mesh in the structural element
    structural_element.vertices = verts
    structural_element.edges = faces