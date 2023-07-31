from typing import Union
import numpy as np

from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy import GeoModel, StructuralFrame
from gempy.core.data.structural_group import StructuralGroup, FaultsRelationSpecialCase


def set_is_fault(frame: Union[GeoModel, StructuralFrame], feature_fault: Union[list[str], list[StructuralGroup]] = None,
                 toggle: bool = False, change_color: bool = True) -> StructuralFrame:
    # * Find the groups passed and set structural relation to fault
    if isinstance(frame, GeoModel):
        frame = frame.structural_frame
    
    for index, group in enumerate(feature_fault):
        if isinstance(group, str):
            group = next((g for g in frame.structural_groups if g.name == group), None)
        if isinstance(group, StructuralGroup):
            group.structural_relation = StackRelationType.FAULT
            group.fault_relations = FaultsRelationSpecialCase.OFFSET_ALL  # * Set the default fault relations
        else:
            raise ValueError(f"Could not find group '{group}' in structural frame.")
        
    
    # * TODO: Set the fault colors
    
    return frame
    
    
def set_fault_relation(frame: Union[GeoModel, StructuralFrame], rel_matrix: np.ndarray) -> StructuralFrame:
    """"""
    if isinstance(frame, GeoModel):
        frame = frame.structural_frame
    
    frame.fault_relations = rel_matrix 
    return frame


def set_is_finite_fault(self, series_fault=None, toggle: bool = True):
    """"""
    raise NotImplementedError
    s = self._faults.set_is_finite_fault(series_fault,
                                         toggle)  # change df in Fault obj
    # change shared aesara variable for infinite factor
    self._interpolator.set_aesara_shared_is_finite()
    return s


