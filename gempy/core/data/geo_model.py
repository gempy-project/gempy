from dataclasses import dataclass

from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from .structural_frame import StructuralFrame
from ..grid import Grid

"""
TODO:
    - [ ] StructuralFrame will all input points chunked on Elements. Here I will need a property to put all
    together to feed to InterpolationInput

"""


@dataclass(init=False)
class GeoModel:
    name: str
    structural_frame: StructuralFrame
    grid: Grid
    
    # GemPy engine data types?
    input_data_descriptor: InputDataDescriptor  # ? This maybe is just a property
    interpolation_options: InterpolationOptions
    
    def __init__(self, name):
        self.name = name
        self.structural_frame = StructuralFrame()
