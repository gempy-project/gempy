from dataclasses import dataclass
from gempy.core.data.structural_element import StructuralElement


@dataclass
class StructuralGroup:
    name: str
    elements: list[StructuralElement]

    @property
    def id(self):
        raise NotImplementedError


@dataclass
class Stack(StructuralGroup): 
    pass


@dataclass
class Fault(StructuralGroup): 
    pass
