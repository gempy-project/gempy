from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.core.data import StructuralGroup, GeoModel, FaultsRelationSpecialCase, StructuralElement, StructuralFrame


def add_structural_group(
        model: GeoModel, group_index: int, structural_group_name: str, elements: list[StructuralElement],
        structural_relation: StackRelationType, fault_relations: FaultsRelationSpecialCase = FaultsRelationSpecialCase.OFFSET_ALL) -> StructuralFrame:
    
    # Check elements are a Sequence
    if not isinstance(elements, list):
        raise TypeError("elements must be a list of StructuralElement objects.")
    
    new_group = StructuralGroup(
        name=structural_group_name,
        elements=elements,
        structural_relation=structural_relation,
        fault_relations=fault_relations
    )

    # Insert the fault group into the structural frame:
    model.structural_frame.insert_group(group_index, new_group)
    return model.structural_frame


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
