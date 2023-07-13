from typing import Union

from gempy import GeoModel
from gempy.core.data.structural_group import StructuralGroup


def map_stack_to_surfaces(gempy_model: GeoModel, mapping_object: Union[dict[str, list[str]] | dict[str, tuple]]):
    for group_name, elements in mapping_object.items():
        for element_name in elements:
            # Here we need to find out the current group of the element.
            # This can be done by looking up the element in each group.
            from_group_name = None
            for group in gempy_model.structural_frame.structural_groups:
                if any(element.name == element_name for element in group.elements):
                    from_group_name = group.name
                    break

            # If we've found the group, we can proceed with moving the element.
            if from_group_name:
                _move_element(
                    structural_groups=gempy_model.structural_frame.structural_groups,
                    element_name=element_name,
                    from_group_name=from_group_name,
                    to_group_name=group_name
                )
            else:
                print(f"Could not find element '{element_name}' in any group.")


def _move_element(structural_groups: list[StructuralGroup], element_name: str, from_group_name: str, to_group_name: str):
    # Find the source and destination groups
    from_group = next((group for group in structural_groups if group.name == from_group_name), None)
    to_group = next((group for group in structural_groups if group.name == to_group_name), None)

    # Check if both groups exist
    if from_group is None or to_group is None:
        raise ValueError("One or both group names are not found.")

    # Find the element in the source group
    element = next((element for element in from_group.elements if element.name == element_name), None)

    # Check if element exists
    if element is None:
        raise ValueError(f"Element '{element_name}' not found in group '{from_group_name}'.")

    # Remove the element from the source group and add it to the destination group
    from_group.elements.remove(element)
    to_group.elements.append(element)