from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.stack_relation_type import StackRelationType
from typing import Optional, Union

import numpy as np

from gempy.core.data import StructuralGroup, GeoModel, FaultsRelationSpecialCase, StructuralElement, StructuralFrame
from gempy.core.data.mesh_scalar_field import MeshScalarField
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable


def add_structural_group(
        model: GeoModel, group_index: int, structural_group_name: str, elements: list[StructuralElement],
        structural_relation: StackRelationType,
        fault_relations: FaultsRelationSpecialCase = FaultsRelationSpecialCase.OFFSET_ALL,
        custom_interpolation: CustomInterpolationFunctions = None,
        null_space: bool = False,
        mesh_source: Optional[MeshScalarField] = None) -> StructuralFrame:
    """Insert a group at ``group_index``.

    ``null_space`` makes the region the group claims void (id -1); ``mesh_source`` takes its scalar field
    from a mesh instead of from its elements' input data (see ``add_mesh_group``).
    """

    # Check elements are a Sequence
    if not isinstance(elements, list):
        raise TypeError("elements must be a list of StructuralElement objects.")

    new_group = StructuralGroup(
        name=structural_group_name,
        elements=elements,
        structural_relation=structural_relation,
        fault_relations=fault_relations,
        custom_interpolation=custom_interpolation,
        null_space=null_space,
        mesh_source=mesh_source,
    )

    # Insert the fault group into the structural frame:
    model.structural_frame.insert_group(group_index, new_group)
    return model.structural_frame


def add_mesh_group(
        model: GeoModel, group_name: str, mesh_source: Union[MeshScalarField, str],
        null_space: bool = False,
        structural_relation: StackRelationType = StackRelationType.ERODE,
        group_index: int = 0,
        element_name: Optional[str] = None,
        color: Optional[str] = None) -> StructuralGroup:
    """Insert a group whose surface is a mesh: the mesh's signed distance is its scalar field.

    The group gets one element (``element_name``, default the group name) and no input data. With
    ``null_space`` everything the group claims is void (e.g. above topography); otherwise its element is
    the unit on the positive side of the mesh. Load the mesh with ``mesh_source.load``; computing needs a
    backend registered with ``gempy.modules.mesh_fields``.
    """
    if isinstance(mesh_source, str):
        mesh_source = MeshScalarField(uri=mesh_source)
    element = StructuralElement(
        name=element_name or group_name,
        surface_points=SurfacePointsTable(np.zeros(0, dtype=SurfacePointsTable.dt)),
        orientations=OrientationsTable(np.zeros(0, dtype=OrientationsTable.dt)),
        color=color or next(model.structural_frame.color_generator),
    )
    add_structural_group(model, group_index, group_name, [element], structural_relation,
                         fault_relations=FaultsRelationSpecialCase.OFFSET_NONE,
                         null_space=null_space, mesh_source=mesh_source)
    return model.structural_frame.structural_groups[group_index]


def remove_structural_group_by_index(model: GeoModel, group_index: int) -> StructuralFrame:
    model.structural_frame.structural_groups.pop(group_index)
    return model.structural_frame


def remove_structural_group_by_name(model: GeoModel, group_name: str) -> StructuralFrame:
    group = model.structural_frame.get_group_by_name(group_name)
    group_index = model.structural_frame.structural_groups.index(group)
    model.structural_frame.structural_groups.pop(group_index)
    return model.structural_frame
    

def remove_element_by_name(model: GeoModel, element_name: str) -> StructuralFrame:
    element = model.structural_frame.get_element_by_name(element_name)
    element_group: StructuralGroup = model.structural_frame.get_group_by_element(element)
    element_group.remove_element(element)
    return model.structural_frame   
