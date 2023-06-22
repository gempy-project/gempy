from abc import ABC
from dataclasses import dataclass
from gempy.core.data.structural_element import StructuralElement


@dataclass
class StructuralGroup(ABC):
    name: str
    elements: list[StructuralElement]

    @property
    def id(self):
        raise NotImplementedError


@dataclass
class Stack(StructuralGroup): 
    def __int__(self, name: str, elements: list[StructuralElement]):
        super().__init__(name, elements)


@dataclass
class Fault(StructuralGroup): 
    pass
