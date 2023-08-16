from core.data.stack_relation_type import StackRelationType
from gempy.core.data import StructuralGroup, GeoModel, FaultsRelationSpecialCase, StructuralElement, StructuralFrame


def add_structural_group(
        model: GeoModel, group_index: int, structural_group_name: str, elements: list[StructuralElement],
        structural_relation: StackRelationType, fault_relations: FaultsRelationSpecialCase = FaultsRelationSpecialCase.OFFSET_ALL) -> StructuralFrame:
    new_group = StructuralGroup(
        name=structural_group_name,
        elements=elements,
        structural_relation=structural_relation,
        fault_relations=fault_relations
    )

    # Insert the fault group into the structural frame:
    model.structural_frame.insert_group(group_index, new_group)
    return model.structural_frame
