from typing import Union
import numpy as np

from core.data.stack_relation_type import StackRelationType
from gempy import GeoModel
from gempy.core.data.structural_group import StructuralGroup, FaultsRelationSpecialCase


def set_is_fault(gempy_model: GeoModel, feature_fault: Union[list[str], list[StructuralGroup]] = None,
                 toggle: bool = False, change_color: bool = True):
    # * Find the groups passed and set structural relation to fault

    for index, group in enumerate(feature_fault):
        if isinstance(group, str):
            group = next((g for g in gempy_model.structural_frame.structural_groups if g.name == group), None)
        if isinstance(group, StructuralGroup):
            group.structural_relation = StackRelationType.FAULT
            group.fault_relations = FaultsRelationSpecialCase.OFFSET_ALL  # * Set the default fault relations
        else:
            raise ValueError(f"Could not find group '{group}' in structural frame.")
        
    
    # * TODO: Set the fault colors
    
    
def set_is_finite_fault(self, series_fault=None, toggle: bool = True):
    """"""
    s = self._faults.set_is_finite_fault(series_fault,
                                         toggle)  # change df in Fault obj
    # change shared aesara variable for infinite factor
    self._interpolator.set_aesara_shared_is_finite()
    return s


def set_fault_relation(self, rel_matrix):
    """"""
    self._faults.set_fault_relation(rel_matrix)

    # Updating
    self._interpolator.set_aesara_shared_fault_relation()
    self._interpolator.set_aesara_shared_weights()
    return self._faults.faults_relations_df
