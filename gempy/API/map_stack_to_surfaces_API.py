from typing import Union

from gempy_engine.core.data.stack_relation_type import StackRelationType
from ..core.data.geo_model import GeoModel
from ..core.data.structural_frame import StructuralFrame
from ..core.data.structural_group import StructuralGroup


def map_stack_to_surfaces(gempy_model: GeoModel, mapping_object: Union[dict[str, list[str]] | dict[str, tuple]],
                          set_series: bool = True, remove_unused_series=True, series_data: list = None) -> StructuralFrame:
    """
    Map stack (series) to surfaces by reorganizing elements between groups in a GeoModel's structural frame.

    This function reorganizes structural elements (surfaces) based on a mapping object 
    and updates the structural frame of the GeoModel. It can also create new series 
    and remove unused ones.

    Args:
        gempy_model (GeoModel): The GeoModel object whose structural frame is to be modified.
        mapping_object (Union[dict[str, list[str]] | dict[str, tuple]]): Dictionary mapping group names to element names.
        set_series (bool, optional): If True, creates new series for groups not present in the GeoModel. Defaults to True.
        remove_unused_series (bool, optional): If True, removes groups without any elements. Defaults to True.
        series_data (list, optional): List of series data from JSON containing structural relations. Defaults to None.

    Returns:
        StructuralFrame: The updated StructuralFrame object.
    """
    structural_groups: list[StructuralGroup] = gempy_model.structural_frame.structural_groups

    for index, (group_name, elements) in enumerate(mapping_object.items()):
        # region Create new series if needed
        group_already_exists = any(group.name == group_name for group in structural_groups)
        if set_series and not group_already_exists:
            # Get structural relation from series_data if available
            structural_relation = StackRelationType.ERODE  # Default value
            if series_data:
                for series in series_data:
                    if series['name'] == group_name:
                        structural_relation = StackRelationType[series['structural_relation']]
                        break

            new_group = StructuralGroup(
                name=group_name,
                elements=[],
                structural_relation=structural_relation
            )
            structural_groups.insert(index, new_group)
        # endregion

        # Make sure elements is a list or tuple
        if isinstance(elements, str):
            elements = [elements]
            
        for element_name in elements:
            # Here we need to find out the current group of the element.
            # This can be done by looking up the element in each group.
            from_group_name = None
            for group in structural_groups:
                if any(element.name == element_name for element in group.elements):
                    from_group_name = group.name
                    break

            # If we've found the group, we can proceed with moving the element.
            if from_group_name:
                _move_element(
                    structural_groups=structural_groups,
                    element_name=element_name,
                    from_group_name=from_group_name,
                    to_group_name=group_name
                )
            else:
                print(f"Could not find element '{element_name}' in any group.")

    # Remove empty groups
    if remove_unused_series:
        structural_groups = [group for group in structural_groups if group.elements]

    gempy_model.structural_frame.structural_groups = structural_groups
    return gempy_model.structural_frame


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
