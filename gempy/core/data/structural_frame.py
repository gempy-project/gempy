from dataclasses import dataclass

from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_group import StructuralGroup


@dataclass
class StructuralFrame:
    structural_groups: list[StructuralGroup]
    structural_elements: list[StructuralElement]
    
    def __init__(self):
        self.structural_groups = []
        self.structural_elements = []