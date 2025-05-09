import pprint
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union, Generator

from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.core.data.structural_element import StructuralElement


class FaultsRelationSpecialCase(Enum):
    OFFSET_FORMATIONS = auto()
    OFFSET_NONE = auto()
    OFFSET_ALL = auto()
    
    
@dataclass
class StructuralGroup(ABC):
    """
    An abstract base class that represents a structural group within a geological model.
    
    """
    name: str  #: The name of the structural group.
    
    elements: list[StructuralElement] = field(repr=False)  #: A list of structural elements within the group.
    structural_relation: StackRelationType  #: The type of relation between the structural elements in the group.

    #: Relations with other groups in terms of faults.
    fault_relations: Optional[Union[list["StructuralGroup"], FaultsRelationSpecialCase]] = field(default=None, repr=False)
    faults_input_data: Optional[FaultsData] = field(default=None, repr=False)
    
    solution: Optional[RawArraysSolution] = field(init=False, default=None, repr=False)  #: Solution related to this group from geological computations.
    
    
    def __post_init__(self):
        if not isinstance(self.elements, list):
            raise TypeError("elements must be a list of StructuralElement objects.")
        for e in self.elements:
            if not isinstance(e, StructuralElement):
                raise TypeError("elements must be a list of StructuralElement objects.")
        
    def __repr__(self):
        elements_repr = ',\n'.join([repr(e) for e in self.elements])
        return f"StructuralGroup(\n" \
               f"\tname={self.name},\n" \
               f"\tstructural_relation={self.structural_relation},\n" \
               f"\telements=[\n{elements_repr}\n]\n)"

    def _repr_html_(self):
        elements_html = '<br>'.join([e._repr_html_() for e in self.elements])
        html = f"""
    <table style="border-left:1.2px solid black;>
      <tr><th colspan="2"><b>StructuralGroup:</b></th></tr>
      <tr><td>Name:</td><td>{self.name}</td></tr>
      <tr><td>Structural Relation:</td><td>{self.structural_relation}</td></tr>
      <tr><td>Elements:</td><td>{elements_html}</td></tr>
    </table>
        """
        return html
    
    def append_element(self, element: StructuralElement):
        self.elements.append(element)
    
    def remove_element(self, element: StructuralElement):
        self.elements.remove(element)

    @property
    def id(self):
        raise NotImplementedError
    
    @property
    def is_fault(self)-> bool:
        return self.structural_relation == StackRelationType.FAULT
    
    @property
    def is_lithology(self)-> bool:
        return self.structural_relation == StackRelationType.ERODE or self.structural_relation == StackRelationType.ONLAP
    
    @property
    def number_of_points(self) -> int:
        return sum([element.number_of_points for element in self.elements])
    
    @property
    def number_of_orientations(self) -> int:
        return sum([element.number_of_orientations for element in self.elements])
    
    @property
    def number_of_elements(self) -> int:
        return len(self.elements)
    
    def get_element_by_name(self, element_name: str) -> StructuralElement | None:
        matched_elements: Generator = (element for element in self.elements if element.name == element_name)
        return next(matched_elements, None)


# ? I think these two subclasses are not necessary
@dataclass
class Stack(StructuralGroup): 
    def __int__(self, name: str, elements: list[StructuralElement]):
        super().__init__(name, elements)
        
    def __repr__(self):
        return pprint.pformat(self.__dict__)


@dataclass
class Fault(StructuralGroup): 
    pass
