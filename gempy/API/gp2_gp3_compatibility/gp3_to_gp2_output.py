import numpy as np

from gempy import Project, Solution
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level


def set_gp3_solutions_to_gp2_solution(gp3_solution: Solutions, geo_model: Project) -> Solution:
    octree_lvl = -1

    octree_output: OctreeLevel = gp3_solution.octrees_output[octree_lvl]
    
    regular_grid_scalar = get_regular_grid_value_for_level(gp3_solution.octrees_output).astype("int8")

    _set_block_matrix(geo_model, octree_output)
    _set_lith_block(geo_model, octree_output)
    _set_scalar_field(geo_model, octree_output)

    _set_scalar_field_at_surface_points(geo_model, octree_output)
    
    meshes: list[DualContouringMesh] = gp3_solution.dc_meshes

    geo_model.set_surface_order_from_solution()
    
    if meshes: _set_surfaces_meshes(geo_model, meshes)
    
    return geo_model.solutions


def _set_surfaces_meshes(geo_model: Project, meshes: list[DualContouringMesh]) -> Project:
    
    geo_model.solutions.vertices = [mesh.vertices for mesh in meshes]
    geo_model.solutions.edges = [mesh.edges for mesh in meshes]
    
    rescaling_factor: float = geo_model._additional_data.rescaling_data.df.loc['values', 'rescaling factor']
    shift: np.array = geo_model._additional_data.rescaling_data.df.loc['values', 'centers']
        
    surfaces_df = geo_model.solutions.surfaces.df
    idx_of_vertices = surfaces_df.columns.get_loc('vertices')
    idx_of_edges = surfaces_df.columns.get_loc('edges')

    # flip mesh 4 and 5
    # meshed_flipped = [ meshes[0], meshes[1], meshes[2], meshes[4], meshes[3]]  bug: meshes are not in the right order yet
    in_ = meshes
    
    for i in range(0, len(meshes)):  
        surfaces_df.iloc[i, idx_of_vertices] = [(in_[i].vertices - 0.5001) * rescaling_factor + shift] # ! remember the 0.5001
        surfaces_df.iloc[i, idx_of_edges] = [in_[i].edges]
    
    return geo_model


def _set_block_matrix(geo_model: Project, octree_output: OctreeLevel) -> Project:
    temp_list = []
    for i in range(octree_output.number_of_outputs):
        temp_list.append(octree_output.outputs_centers[i].values_block)

    block_matrix_stacked = np.vstack(temp_list)
    geo_model.solutions.block_matrix = block_matrix_stacked
    return geo_model


def _set_scalar_field(geo_model: Project, octree_output: OctreeLevel) -> Project:
    temp_list = []
    for i in range(octree_output.number_of_outputs):
        temp_list.append(octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field)

    scalar_field_stacked = np.vstack(temp_list)
    geo_model.solutions.scalar_field_matrix = scalar_field_stacked
    return geo_model


def _set_scalar_field_at_surface_points(geo_model: Project, octree_output: OctreeLevel) -> Project:
    temp_list = []
    for i in range(octree_output.number_of_outputs):
        temp_list.append(octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field_at_surface_points)
    
    geo_model.solutions.scalar_field_at_surface_points = temp_list
    return geo_model


def _set_lith_block(geo_model: Project, octree_output: OctreeLevel) -> Project:
    block = octree_output.last_output_center.ids_block
    block[block == 0] = block.max() + 1
    geo_model.solutions.lith_block = block
    return geo_model
