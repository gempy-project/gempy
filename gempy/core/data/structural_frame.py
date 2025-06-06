import hashlib

import numpy as np
import warnings
from dataclasses import dataclass
from pydantic import model_validator, computed_field, ValidationError, Field
from pydantic.functional_validators import ModelWrapValidatorHandler
from typing import Generator, Union

from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.stack_relation_type import StackRelationType

from .encoders.binary_encoder import deserialize_input_data_tables
from .encoders.converters import loading_model_context
from .orientations import OrientationsTable
from .structural_element import StructuralElement
from .structural_group import StructuralGroup, FaultsRelationSpecialCase
from .surface_points import SurfacePointsTable
from ..color_generator import ColorsGenerator


@dataclass
class StructuralFrame:
    """
    Represents a structural frame, which is a collection of structural groups that constitute a geological model.

    Attributes:
        structural_groups (list[StructuralGroup]): List of structural groups that constitute the geological model.
        color_generator (ColorsGenerator): Instance of ColorsGenerator used for assigning distinct colors to different structural elements.
        is_dirty (bool): Boolean flag indicating if the structural frame has been modified.
    """

    structural_groups: list[StructuralGroup]
    color_generator: ColorsGenerator =  Field(default_factory=ColorsGenerator)
    basement_color: str = None
    # ? Should I create some sort of structural options class? For example, the masking descriptor and faults relations pointer
    is_dirty: bool = True

    # region Constructor
    # 
    def __init__(self, structural_groups: list[StructuralGroup], color_gen: ColorsGenerator):
        self.structural_groups = structural_groups  # ? This maybe could be optional
        self.color_generator = color_gen
        
    def __post_init__(self):
        pass

    @classmethod
    def from_data_tables(cls, surface_points: SurfacePointsTable, orientations: OrientationsTable):
        surface_points_groups: list[SurfacePointsTable] = surface_points.get_surface_points_by_id_groups()
        colors_generator = ColorsGenerator()

        structural_elements = []
        for i in range(len(surface_points_groups)):
            id_ = surface_points_groups[i].id
            orientation_i = orientations.get_orientations_by_id(id_)
            if len(orientation_i) == 0:
                orientation_i = OrientationsTable.empty_orientation(id_)

            structural_element: StructuralElement = StructuralElement(
                name=surface_points.id_to_name(i),
                id=id_,
                surface_points=surface_points_groups[i],
                orientations=orientation_i,
                color=next(colors_generator)
            )

            structural_elements.append(structural_element)
        # * Structural groups definitions
        default_formation: StructuralGroup = StructuralGroup(
            name="default_formation",
            elements=structural_elements,
            structural_relation=StackRelationType.ERODE
        )
        # ? Should I move this to the constructor?
        structural_frame: StructuralFrame = cls(
            structural_groups=[default_formation],
            color_gen=colors_generator
        )

        return structural_frame

    @classmethod
    def initialize_default_structure(cls) -> 'StructuralFrame':
        """
        Initialize the default structure.

        This method is used to initialize the default structure for a `StructuralFrame` object.

        Args:
            None

        Returns:
            'StructuralFrame': A `StructuralFrame` object representing the default structure.

        Example:
            structural_frame = initialize_default_structure()
        """
        color_gen = ColorsGenerator()

        structural_group = StructuralGroup(
            name="default_formations",
            elements=[
                    StructuralElement(
                        name="surface1",
                        surface_points=SurfacePointsTable.initialize_empty(),
                        orientations=OrientationsTable.initialize_empty(),
                        color=next(color_gen)
                    )
            ],
            structural_relation=StackRelationType.ERODE
        )

        structural_frame = cls(
            structural_groups=[structural_group],
            color_gen=color_gen
        )

        return structural_frame

    # endregion

    # region Methods
    def get_element_by_name(self, element_name: str) -> StructuralElement:
        elements: Generator = (group.get_element_by_name(element_name) for group in self.structural_groups)
        valid_elements: Generator = (element for element in elements if element is not None)
        element = next(valid_elements, None)
        if element is None:
            raise ValueError(f"Element with name {element_name} not found in the structural frame.")
        return element

    def get_group_by_name(self, group_name: str) -> StructuralGroup:
        groups: Generator = (group for group in self.structural_groups if group.name == group_name)
        group = next(groups, None)
        if group is None:
            raise ValueError(f"Group with name {group_name} not found in the structural frame.")
        return group

    def get_group_by_element(self, element: StructuralElement) -> StructuralGroup:
        groups: Generator = (group for group in self.structural_groups if element in group.elements)
        group = next(groups, None)
        if group is None:
            raise ValueError(f"Element {element.name} not found in any group in the structural frame.")
        return group

    def append_group(self, group: StructuralGroup):
        self.structural_groups.append(group)

    def insert_group(self, index: int, group: StructuralGroup):
        self.structural_groups.insert(index, group)

    def __repr__(self):
        structural_groups_repr = ',\n'.join([repr(g) for g in self.structural_groups])
        fault_relations_str = np.array2string(self.fault_relations, precision=2, separator=', ', suppress_small=True) if self.fault_relations is not None else 'None'
        return (f"StructuralFrame(\n"
                f"\tstructural_groups=[\n{structural_groups_repr}\n],\n"
                f"\tfault_relations=\n{fault_relations_str},\n"
                )

    def _repr_html_(self):
        structural_groups_html = '<br>'.join([g._repr_html_() for g in self.structural_groups])
        if self.fault_relations is not None:
            # Define the colors for True and False values
            true_color = '#527682'
            false_color = '#FFB6C1'

            table_headers = '<th></th>' + ''.join('<th style="transform: rotate(-35deg); height:150px; vertical-align: bottom; text-align: center;">{}</th>'.format((g.name[:10] + '...') if len(g.name) > 10 else g.name) for g in self.structural_groups)
            table_rows = ''.join('<tr><th>{}</th>{}</tr>'.format(self.structural_groups[i].name, ''.join('<td style="background-color: {}; width: 20px; height: 20px; border: 1px solid black;"></td>'.format(true_color if cell else false_color) for cell in row)) for i, row in enumerate(self.fault_relations))
            fault_relations_str = '<table style="border-collapse: collapse; table-layout: fixed;">{}{}</table>'.format(table_headers, table_rows)
        else:
            fault_relations_str = 'None'

        # Define the legend
        legend = f"""
        <table>
          <tr>
            <td><div style="display: inline-block; background-color: {true_color}; width: 20px; height: 20px; border: 1px solid black;"></div> True</td>
            <td><div style="display: inline-block; background-color: {false_color}; width: 20px; height: 20px; border: 1px solid black;"></div> False</td>
          </tr>
        </table>
        """

        html = f"""
        <table>
          <tr><td>Structural Groups:</td><td>{structural_groups_html}</td></tr>
          <tr><td>Fault Relations:</td><td>{fault_relations_str}</td></tr>
          <tr><td></td><td>{legend}</td></tr>
        </table>
        """
        return html

    # endregion

    # region Properties
    @property
    def structural_elements(self) -> list[StructuralElement]:
        """Returns a list of all structural elements across the structural groups."""
        elements = []
        for group in self.structural_groups:
            elements.extend(group.elements)
        elements.append(self._basement_element)
        return elements

    @property
    def n_elements(self) -> int:
        """Returns the total number of elements in the structural frame."""
        return len(self.structural_elements)


    @property
    def _basement_element(self) -> StructuralElement:
        """Returns the basement structural element with a unique color."""

        def _get_unique_basement_color(color_generator: ColorsGenerator, used_colors: list[str]) -> str:
            color = next(color_generator)
            if color in used_colors:
                return _get_unique_basement_color(color_generator, used_colors)
            return color

        elements = []
        for group in self.structural_groups:
            elements.extend(group.elements)

        used_colors = [element.color for element in elements]

        if self.basement_color is None or self.basement_color in used_colors:
            self.basement_color = _get_unique_basement_color(
                color_generator=self.color_generator, 
                used_colors=used_colors
            )

        basement = StructuralElement(
            name="basement",
            surface_points=SurfacePointsTable(data=np.zeros(0, dtype=SurfacePointsTable.dt)),
            orientations=OrientationsTable(data=np.zeros(0, dtype=OrientationsTable.dt)),
            color=self.basement_color
        )

        return basement

    # ? Should I move this property to StructuralGroup?
    @property
    def fault_relations(self) -> np.ndarray:
        """Returns a  array describing the fault relations between the structural groups."""
        # Initialize an empty boolean array with dimensions len(structural_groups) x len(structural_groups)

        fault_relations = np.zeros((len(self.structural_groups), len(self.structural_groups)), dtype=bool)

        # We assume that the list is ordered from older to younger
        # Iterate over the list of structural_groups
        for i, group in enumerate(self.structural_groups):
            match (group.structural_relation, group.fault_relations):
                case (StackRelationType.FAULT, FaultsRelationSpecialCase.OFFSET_ALL):  # It affects all younger groups
                    fault_relations[i, i + 1:] = True
                case (StackRelationType.FAULT, FaultsRelationSpecialCase.OFFSET_NONE):  # It affects no groups
                    pass
                case (StackRelationType.FAULT, FaultsRelationSpecialCase.OFFSET_FORMATIONS):  # It affects all younger groups that are formations
                    do_offset = []
                    for group_internal in self.structural_groups[i + 1:]:
                        do_offset.append(group_internal.structural_relation != StackRelationType.FAULT)
                    fault_relations[i, i + 1:] = do_offset
                case (StackRelationType.FAULT, list(fault_groups)) if fault_groups:  # It affects only the specified groups
                    for fault_group in fault_groups:
                        j = self.structural_groups.index(fault_group)
                        if j <= i:  # Only consider groups that are 
                            raise ValueError(f"Fault {group.name} cannot affect older fault {fault_group.name}")
                case (StackRelationType.FAULT, _):
                    raise ValueError(f"Fault {group.name} has an invalid fault relation")
                case _:
                    pass  # If not a fault or fault relation is not specified, do nothing
        return fault_relations

    @fault_relations.setter
    def fault_relations(self, matrix: np.ndarray):
        """Sets the fault relations between structural groups using the provided matrix."""
        assert matrix.shape == (len(self.structural_groups), len(self.structural_groups))

        # Iterate over each StructuralGroup
        for i, group in enumerate(self.structural_groups):

            affected_groups = matrix[i, :]  # * If the group is a fault
            # If all younger groups are affected
            all_younger_groups_affected = np.all(affected_groups[i + 1:])
            any_younger_groups_affected = np.any(affected_groups[i + 1:])

            if all_younger_groups_affected:
                group.fault_relations = FaultsRelationSpecialCase.OFFSET_ALL
            elif not any_younger_groups_affected:
                group.fault_relations = FaultsRelationSpecialCase.OFFSET_NONE
            else:  # * A specific set of groups are affected
                group.fault_relations = [g for j, g in enumerate(self.structural_groups) if affected_groups[j]]

    @property
    def group_is_fault(self) -> list[bool]:
        """Returns a list of booleans indicating if each structural element is a fault."""
        return [group.is_fault for group in self.structural_groups]

    @property
    def group_is_lithology(self) -> list[bool]:
        """Returns a list of booleans indicating if each structural element is a lithology."""
        return [group.is_lithology for group in self.structural_groups]

    @property
    def input_data_descriptor(self):
        """Returns a descriptor for the input data, detailing the relations and faults between groups."""
        # TODO: This should have the exact same dirty logic as interpolation_input

        self._validate_faults_relations()
        return InputDataDescriptor.from_structural_frame(
            structural_frame=self,
            making_descriptor=self.groups_structural_relation,
            faults_relations=self.fault_relations,
            faults_input_data=self.faults_input_data

        )

    @property
    def faults_input_data(self):
        """Returns a descriptor for the input data, detailing the relations and faults between groups."""
        faults_input_data: list[FaultsData] = [group.faults_input_data for group in self.structural_groups]
        return faults_input_data

    @property
    def groups_structural_relation(self) -> list[StackRelationType]:
        """Returns a list of the structural relations for each group."""
        groups_ = [group.structural_relation for group in self.structural_groups]
        groups_[-1] = StackRelationType.BASEMENT
        return groups_

    @property
    def number_of_points_per_element(self) -> np.ndarray:
        """Returns an array with the number of points for each structural element."""
        return np.array([element.number_of_points for element in self.structural_elements])

    @property
    def number_of_points_per_group(self) -> np.ndarray:
        """Returns an array with the number of points for each structural group."""
        return np.array([group.number_of_points for group in self.structural_groups])

    @property
    def number_of_orientations_per_group(self) -> np.ndarray:
        """Returns an array with the number of orientations for each structural group."""
        return np.array([group.number_of_orientations for group in self.structural_groups])

    @property
    def number_of_elements_per_group(self) -> np.ndarray:
        """Returns an array with the number of elements for each structural group."""
        return np.array([group.number_of_elements for group in self.structural_groups])

    @property
    def surfaces(self) -> list[StructuralElement]:
        """Returns a list of all surfaces in the structural elements."""
        return self.structural_elements

    @property
    def number_of_elements(self) -> int:
        """Returns the total number of elements in the structural frame."""
        return len(self.structural_elements)

    @property
    def elements_names(self) -> list[str]:
        """Returns a list of names of all structural elements."""
        return [element.name for element in self.structural_elements]

    @property
    def elements_ids(self) -> np.ndarray:
        """Returns an array of IDs for all structural elements."""
        return np.arange(len(self.structural_elements)) + 1

    @property
    def surface_points_copy(self) -> SurfacePointsTable:
        """Returns a SurfacePointsTable for all surface points across the structural elements. This is a copy!"""
        all_data: np.ndarray = np.concatenate([element.surface_points.data for element in self.structural_elements])
        return SurfacePointsTable(data=all_data, name_id_map=self.element_name_id_map)

    @property
    def surface_points(self):
        raise AttributeError("This property can only be set, not read. You can access the copy with `surface_points_copy` or"
                             "the original on the individual structural elements.")

    @surface_points.setter
    def surface_points(self, modified_surface_points: SurfacePointsTable) -> None:
        """Distributes the modified surface points back to the structural elements."""
        for element in self.structural_elements:
            element.surface_points.data = modified_surface_points.get_surface_points_by_id(element.id).data

    @property
    def orientations_copy(self) -> OrientationsTable:
        """Returns an OrientationsTable for all orientations across the structural elements."""
        all_data: np.ndarray = np.concatenate([element.orientations.data for element in self.structural_elements])
        return OrientationsTable(data=all_data)

    @property
    def orientations(self) -> OrientationsTable:
        raise AttributeError("This property can only be set, not read. You can access the copy with `orientations_copy` or"
                             "the original on the individual structural elements.")

    @orientations.setter
    def orientations(self, modified_orientations: OrientationsTable) -> None:
        """Distributes the modified orientations back to the structural elements."""
        for element in self.structural_elements:
            element.orientations.data = modified_orientations.get_orientations_by_id(element.id).data
            
    @property
    def input_tables_binary(self):
        return self.surface_points_copy.data.tobytes() + self.orientations_copy.data.tobytes()

    @property
    def element_id_name_map(self) -> dict[int, str]:
        """Returns a dictionary mapping element IDs to names."""
        return {element.id: element.name for i, element in enumerate(self.structural_elements)}

    @property
    def element_name_id_map(self) -> dict[str, int]:
        """Returns a dictionary mapping element names to IDs."""
        return {element.name: element.id for i, element in enumerate(self.structural_elements)}

    @property
    def elements_colors(self) -> list[str]:
        """Returns a list of colors assigned to each structural element. Used in matplotlib"""
        # reversed
        return [element.color for element in self.structural_elements][::-1]

    @property
    def elements_colors_volumes(self) -> list[str]:
        """Returns a list of colors assigned to each structural element for volume representation. Used in pyvista"""
        return self.elements_colors

    @property
    def elements_colors_contacts(self) -> list[str]:
        """Returns a list of colors assigned to each structural element for contact representation. Used in many places"""
        points_ = [element.color for element in self.structural_elements if len(element.surface_points) > 0]
        return points_

    @property
    def elements_colors_orientations(self) -> list[str]:
        """Returns a list of colors assigned to each structural element for orientation representation. Used to paint
        orientations in pyvista
        """
        orientations_ = [element.color for element in self.structural_elements if len(element.orientations) > 0]
        return orientations_

    @property
    def surface_points_colors_per_item(self) -> list[str]:
        """Returns a list of colors assigned to each surface point across structural elements. Used in matplotlib"""
        surface_points_colors = [element.color for element in self.structural_elements for _ in range(element.number_of_points)]
        return surface_points_colors

    @property
    def orientations_colors_per_item(self) -> list[str]:
        """Returns a list of colors assigned to each orientation across structural elements. Used in matplotlib"""
        orientations_colors = [element.color for element in self.structural_elements for _ in range(element.number_of_orientations)]
        return orientations_colors

    @property
    def groups_to_mapper(self) -> dict[str, list[str]]:
        """Returns a dictionary mapping each structural group to its corresponding elements."""
        result_dict = {}
        for group in self.structural_groups:
            element_names = [element.name for element in group.elements]
            result_dict[group.name] = element_names
        return result_dict

    # region Depends on Pandas
    @property
    def surfaces_df(self) -> 'pd.DataFrame':
        """Returns a DataFrame representation of all surfaces across structural elements."""
        # TODO: Loop every structural element. Each element should be a row in the dataframe
        # TODO: The columns have to be ['element, 'group', 'color']

        raise NotImplementedError

    # endregion

    # endregion
    # region Pydantic

    @model_validator(mode="wrap")
    @classmethod
    def deserialize_binary(cls, data: Union["StructuralFrame", dict], constructor: ModelWrapValidatorHandler["StructuralFrame"]) -> "StructuralFrame":
        match data:
            case StructuralFrame():
                return data
            case dict():
                instance: StructuralFrame = constructor(data)
                metadata = data.get('binary_meta_data', {})
                context = loading_model_context.get()

                if 'input_binary' not in context:
                    return instance

                instance.orientations, instance.surface_points = deserialize_input_data_tables(
                    binary_array=context['input_binary'],
                    name_id_map=instance.surface_points_copy.name_id_map,
                    sp_binary_length_=metadata["sp_binary_length"],
                    ori_binary_length_=metadata["ori_binary_length"]
                )

                return instance
            case _:
                raise ValidationError(f"Invalid data type for StructuralFrame: {type(data)}")

        # Access the context variable to get injected data

    @computed_field
    def binary_meta_data(self) -> dict:
        return {
                'sp_binary_length': len(self.surface_points_copy.data.tobytes()),
                'ori_binary_length': len(self.orientations_copy.data.tobytes()) ,
        }

    # endregion

    def _validate_faults_relations(self):
        """Check that if there are any StackRelationType.FAULT in the structural groups the fault relation matrix is
        given and shape is the right one, i.e. a square matrix of size equals to len(groups)"""

        if any([group.structural_relation == StackRelationType.FAULT for group in self.structural_groups]):
            if self.fault_relations is None:
                raise ValueError("The fault relations matrix is not given")
            if self.fault_relations.shape != (len(self.structural_groups), len(self.structural_groups)):
                raise ValueError("The fault relations matrix is not the right shape")
