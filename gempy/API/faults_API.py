from typing import Union
import numpy as np

from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy import GeoModel, StructuralFrame
from gempy.core.data.structural_group import StructuralGroup, FaultsRelationSpecialCase


def set_is_fault(frame: Union[GeoModel, StructuralFrame], feature_fault: Union[list[str], list[StructuralGroup]],
                 change_color: bool = True) -> StructuralFrame:
    if isinstance(frame, GeoModel):
        frame = frame.structural_frame

    frame = _find_and_set_fields(
        frame=frame,
        feature_fault=feature_fault,
        faults_relation_type=FaultsRelationSpecialCase.OFFSET_ALL,
        stack_relation_type=StackRelationType.FAULT,
        change_color=change_color
    )

    # * TODO: Set the fault colors

    return frame


def unset_is_fault(frame: Union[GeoModel, StructuralFrame], feature_fault: Union[list[str], list[StructuralGroup]]) -> StructuralFrame:
    if isinstance(frame, GeoModel):
        frame = frame.structural_frame

    frame = _find_and_set_fields(
        frame=frame,
        feature_fault=feature_fault,
        faults_relation_type=FaultsRelationSpecialCase.OFFSET_NONE,
        stack_relation_type=StackRelationType.ERODE,
        change_color=False
    )

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


def _find_and_set_fields(frame: StructuralFrame, feature_fault: list[StructuralGroup],
                         faults_relation_type: FaultsRelationSpecialCase, stack_relation_type: StackRelationType,
                         change_color: bool) -> StructuralFrame:
    for index, group in enumerate(feature_fault):
        if isinstance(group, str):
            group = next((g for g in frame.structural_groups if g.name == group), None)
        if isinstance(group, StructuralGroup):
            group.structural_relation = stack_relation_type
            group.fault_relations = faults_relation_type  # * Set the default fault relations
            if change_color:
                for element in group.elements:
                    element.color = '#527682'
        else:
            raise ValueError(f"Could not find group '{group}' in structural frame.")
    return frame
