import pprint
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from gempy_engine.core.data.legacy_solutions import LegacySolution
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.core.data.structural_element import StructuralElement


@dataclass
class StructuralGroup(ABC):
    name: str
    elements: list[StructuralElement] = field(repr=False)
    structural_relation: StackRelationType

    solution: Optional[LegacySolution] = field(init=False, default=None, repr=False)
    
    def __repr__(self):
        elements_repr = ',\n'.join([repr(e) for e in self.elements])
        return f"StructuralGroup(\n" \
               f"\tname={self.name},\n" \
               f"\tstructural_relation={self.structural_relation},\n" \
               f"\telements=[\n{elements_repr}\n]\n)"
    
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
        
    def __repr__(self):
        return pprint.pformat(self.__dict__)


@dataclass
class Fault(StructuralGroup): 
    pass
