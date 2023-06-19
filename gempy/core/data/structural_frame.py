from dataclasses import dataclass

from gempy.core.data.geo_model import GeoModel
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_group import StructuralGroup


@dataclass
class StructuralFrame:
    geo_model: GeoModel
    structural_groups: list[StructuralGroup]
    structural_elements: list[StructuralElement]