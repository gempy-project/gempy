import pprint
from abc import ABC
from dataclasses import dataclass

from core.data.stack_relation_type import StackRelationType
from gempy.core.data.structural_element import StructuralElement


@dataclass
class StructuralGroup(ABC):
    name: str
    elements: list[StructuralElement]
    structural_relation: StackRelationType
    
    def __repr__(self):
        return pprint.pformat(self.__dict__)
    
    @property
    def id(self):
        raise NotImplementedError
    
    @property
    def number_of_points(self) -> int:
        return sum([element.number_of_points for element in self.elements])
    
    @property
    def number_of_orientations(self) -> int:
        return sum([element.number_of_orientations for element in self.elements])
    
    @property
    def number_of_elements(self) -> int:
        return len(self.elements)


@dataclass
class Stack(StructuralGroup): 
    def __int__(self, name: str, elements: list[StructuralElement]):
        super().__init__(name, elements)


@dataclass
class Fault(StructuralGroup): 
    pass
