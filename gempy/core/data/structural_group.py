from dataclasses import dataclass

from gempy.core.data.geo_model import GeoModel
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_frame import StructuralFrame


@dataclass
class StructuralGroup:
    name: str
    elements: list[StructuralElement]
    linked_geomodel: GeoModel

    @property
    def id(self):
        raise NotImplementedError

    @property
    def linked_structural_frame(self) -> StructuralFrame:
        raise self.linked_geomodel.structural_frame


@dataclass
class Stack(StructuralGroup): 
    pass


@dataclass
class Fault(StructuralGroup): 
    pass
