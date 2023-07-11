from dataclasses import dataclass

import numpy as np

from .orientations import OrientationsTable
from .structural_element import StructuralElement
from .structural_group import StructuralGroup
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, TensorsStructure, StacksStructure, StackRelationType
from .surface_points import SurfacePointsTable


@dataclass
class StructuralFrame:
    structural_groups: list[StructuralGroup]  # ? should this be lazy?
    structural_elements: list[StructuralElement]

    input_data_descriptor: InputDataDescriptor  # ? This maybe is just a property

    def __init__(self, structural_groups: list[StructuralGroup], structural_elements: list[StructuralElement]):
        self.structural_groups = structural_groups  # ? This maybe could be optional
        self.structural_elements = structural_elements

    @property
    def input_data_descriptor(self) -> InputDataDescriptor:
        tensor_struct = TensorsStructure(
            number_of_points_per_surface=np.array([9, 12, 12, 13, 12, 12])
        )

        stack_structure = StacksStructure(
            number_of_points_per_stack=np.array([9, 24, 37]),
            number_of_orientations_per_stack=np.array([1, 4, 6]),
            number_of_surfaces_per_stack=np.array([1, 2, 3]),
            masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE],
            faults_relations=None
        )

        return InputDataDescriptor(
            tensors_structure=tensor_struct,
            stack_structure=stack_structure
        )

    @property
    def surfaces(self) -> list[StructuralElement]:
        return self.structural_elements
    
    @property
    def elements_names(self) -> list[str]:
        return [element.name for element in self.structural_elements]
    
    @property
    def elements_colors(self) -> list[str]:
        return [element.color for element in self.structural_elements]
    
    @property
    def surface_points(self) -> SurfacePointsTable:
        all_data: np.ndarray = np.concatenate([element.surface_points.data for element in self.structural_elements])
        return SurfacePointsTable(data=all_data)
        
    @property
    def surface_points_colors(self) -> list[str]:
        """Using the id record of surface_points map the elements colors to each point"""
        elements_colors = self.elements_colors
        surface_points = self.surface_points
        surface_points_id = surface_points.data['id']
        
        return [elements_colors[surface_points_id[i]] for i in range(len(surface_points))]
        
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
    
    @property
    def surfaces_points_df(self) -> 'pd.DataFrame':
        raise NotImplementedError
    
    @property
    def orientations_df(self) -> 'pd.DataFrame':
        raise NotImplementedError
    
    # endregion
        