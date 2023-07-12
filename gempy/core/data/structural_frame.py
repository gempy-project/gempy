from dataclasses import dataclass

import numpy as np

from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.stacks_structure import StacksStructure

from .orientations import OrientationsTable
from .structural_element import StructuralElement
from .structural_group import StructuralGroup
from .surface_points import SurfacePointsTable
from ..color_generator import ColorsGenerator


@dataclass
class StructuralFrame:
    structural_groups: list[StructuralGroup]  # ? should this be lazy?
    structural_elements: list[StructuralElement]

    # ? Should I create some sort of structural options class? For example, the masking descriptor and faults relations pointer

    color_gen: ColorsGenerator = ColorsGenerator()  # ? Do I need a method to regenerate this?
    is_dirty: bool = True  # This changes when the structural frame is modified

    def __init__(self, structural_groups: list[StructuralGroup], structural_elements: list[StructuralElement]):
        self.structural_groups = structural_groups  # ? This maybe could be optional
        self.structural_elements = structural_elements

        # Append basement element
        self.structural_elements.append(
            StructuralElement(
                name="basement",
                surface_points=SurfacePointsTable(data=np.zeros(0, dtype=SurfacePointsTable.dt)),
                orientations=OrientationsTable(data=np.zeros(0, dtype=OrientationsTable.dt)),
                color=next(StructuralFrame.color_gen),
            )
        )

    @property
    def input_data_descriptor(self):
        # TODO: This should have the exact same dirty logic as interpolation_input
        return InputDataDescriptor.from_structural_frame(
            structural_frame=self,
            making_descriptor=[StackRelationType.ERODE],
            faults_relations=None
        )

    @property
    def number_of_points_per_element(self) -> np.ndarray:
        return np.array([element.number_of_points for element in self.structural_elements])

    @property
    def number_of_points_per_group(self) -> np.ndarray:
        return np.array([group.number_of_points for group in self.structural_groups])

    @property
    def number_of_orientations_per_group(self) -> np.ndarray:
        return np.array([group.number_of_orientations for group in self.structural_groups])

    @property
    def number_of_elements_per_group(self) -> np.ndarray:
        return np.array([group.number_of_elements for group in self.structural_groups])

    @property
    def surfaces(self) -> list[StructuralElement]:
        return self.structural_elements

    @property
    def number_of_elements(self) -> int:
        return len(self.structural_elements)

    @property
    def elements_names(self) -> list[str]:
        return [element.name for element in self.structural_elements]

    @property
    def elements_colors(self) -> list[str]:
        # reversed
        return [element.color for element in self.structural_elements]

    @property
    def elements_ids(self) -> np.ndarray:
        """Return id given by the order of the structural elements"""
        return np.arange(len(self.structural_elements)) + 1

    @property
    def surface_points(self) -> SurfacePointsTable:
        all_data: np.ndarray = np.concatenate([element.surface_points.data for element in self.structural_elements])
        return SurfacePointsTable(data=all_data)

    @property
    def surface_points_colors(self) -> list[str]:
        """Using the id record of surface_points map the elements colors to each point"""
        elements_colors = self.elements_colors[1::-1]  # remove first element (basement)
        surface_points = self.surface_points
        surface_points_id = surface_points.data['id']

        return [elements_colors[surface_points_id[i]] for i in range(len(surface_points))]

    @property
    def orientations_colors(self) -> list[str]:
        """Using the id record of orientations map the elements colors to each point"""
        elements_colors = self.elements_colors[1::-1]  # remove first element (basement)
        orientations = self.orientations
        orientations_id = orientations.data['id']

        return [elements_colors[orientations_id[i]] for i in range(len(orientations))]

    @property
    def orientations(self) -> OrientationsTable:
        all_data: np.ndarray = np.concatenate([element.orientations.data for element in self.structural_elements])
        return OrientationsTable(data=all_data)

    # region Depends on Pandas
    @property
    def surfaces_df(self) -> 'pd.DataFrame':
        # TODO: Loop every structural element. Each element should be a row in the dataframe
        # TODO: The columns have to be ['element, 'group', 'color']

        raise NotImplementedError
    # endregion
