from dataclasses import dataclass

from gempy_engine.core.data import InterpolationOptions
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
    interpolation_options: InterpolationOptions

    def __init__(self, name: str, structural_frame: StructuralFrame, grid: Grid,
                 interpolation_options: InterpolationOptions):
        self.name = name
        self.structural_frame = structural_frame  # ? This could be Optional

        self.grid = grid
        self.interpolation_options = interpolation_options

    