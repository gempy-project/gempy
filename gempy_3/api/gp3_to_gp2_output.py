import numpy as np

from gempy import Project, Solution
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.solutions import Solutions


def set_gp3_solutions_to_gp2_solution(gp3_solution: Solutions, geo_model: Project) -> Solution:
    octree_lvl = -1

    octree_output: OctreeLevel = gp3_solution.octrees_output[octree_lvl]
    
    _set_block_matrix(geo_model, octree_output)
    _set_lith_block(geo_model, octree_output)
    _set_scalar_field(geo_model, octree_output)
    
    meshes: list[DualContouringMesh] = gp3_solution.dc_meshes

    _set_surfaces_meshes(geo_model, meshes)

    geo_model.set_surface_order_from_solution()
    
    return geo_model.solutions


def _set_surfaces_meshes(geo_model: Project, meshes: list[DualContouringMesh]) -> Project:
    geo_model.solutions.vertices = [mesh.vertices for mesh in meshes]
    geo_model.solutions.edges = [mesh.edges for mesh in meshes]
    
    rescaling_factor: float = geo_model._additional_data.rescaling_data.df.loc['values', 'rescaling factor']
    shift: np.array = geo_model._additional_data.rescaling_data.df.loc['values', 'centers']
    
    # TODO: Here we need to look exactly in which surface we add the mesh
    geo_model.solutions.surfaces.df.loc[4, 'vertices'] = [meshes[0].vertices * rescaling_factor - shift]
    geo_model.solutions.surfaces.df.loc[4, 'edges'] = [meshes[0].edges]
    geo_model.solutions.surfaces.df.loc[1, 'vertices'] = [meshes[1].vertices * rescaling_factor - shift]
    geo_model.solutions.surfaces.df.loc[1, 'edges'] = [meshes[1].edges]
    
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


def _set_lith_block(geo_model: Project, octree_output: OctreeLevel) -> Project:
    block = octree_output.last_output_center.ids_block
    block[block == 0] = block.max() + 1
    geo_model.solutions.lith_block = block
    return geo_model
