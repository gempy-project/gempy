import numpy as np
from skimage import measure


def compute_marching_cubes(model):
    # Empty lists to store vertices and edges
    mc_vertices = []
    mc_edges = []
    # Boolean list of fault groups
    faults = model.structural_frame.group_is_fault
    # MC for faults, directly on fault block not on scalar field
    if faults is not None:
        _extract_fault_mesh(mc_edges, mc_vertices, model)
    else:
        pass
    
    # Extract scalar field values for elements
    scalar_values = model.solutions.raw_arrays.scalar_field_at_surface_points
    
    # Get indices of non fault elements
    false_indices = _get_lithology_idx(faults, model)
    # Extract marching cubes for non fault elements
    for idx in false_indices:
        _extract_meshes_for_lithologies(idx, mc_edges, mc_vertices, model, scalar_values)
    return mc_edges, mc_vertices


def _extract_meshes_for_lithologies(idx, mc_edges, mc_vertices, model, scalar_values):
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


def _get_lithology_idx(faults, model):
    if faults is not None:
        false_indices = [i for i, fault in enumerate(faults) if not fault]
    else:
        false_indices = np.arange(len(model.structural_frame.structural_groups))
    return false_indices


def _extract_fault_mesh(mc_edges, mc_vertices, model):
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
