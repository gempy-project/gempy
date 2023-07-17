import pprint
from dataclasses import dataclass

import gempy_engine.core.data.grid
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from .structural_frame import StructuralFrame
from .transforms import Transform
from ..grid import Grid

"""
TODO:
    - [ ] StructuralFrame will all input points chunked on Elements. Here I will need a property to put all
    together to feed to InterpolationInput

"""


@dataclass
class GeoModelMeta:
    name: str
    creation_date: str
    last_modification_date: str
    owner: str


@dataclass(init=False)
class GeoModel:
    meta: GeoModelMeta
    structural_frame: StructuralFrame
    grid: Grid  # * This is the general gempy grid
    transform: Transform
    
    # region GemPy engine data types
    interpolation_options: InterpolationOptions  # * This has to be fed by USER

    # ? Are this more caching fields than actual fields?
    interpolation_grid: gempy_engine.core.data.grid.Grid = None
    _interpolationInput: InterpolationInput = None  # * This has to be fed by structural_frame
    _input_data_descriptor: InputDataDescriptor = None # * This has to be fed by structural_frame

    # endregion

    solutions: gempy_engine.core.data.solutions.Solutions = None
    
    legacy_model: "gpl.Project" = None
    
    def __init__(self, name: str, structural_frame: StructuralFrame, grid: Grid, interpolation_options: InterpolationOptions):
        # TODO: Fill the arguments properly
        self.meta = GeoModelMeta(
            name=name,
            creation_date=None,
            last_modification_date=None,
            owner=None
        )
        
        self.structural_frame = structural_frame  # ? This could be Optional

        self.grid = grid
        self.interpolation_options = interpolation_options
        self.transform = Transform.from_input_points(
            surface_points=self.surface_points,
            orientations=self.orientations
        )


    def __repr__(self):
        return pprint.pformat(self.__dict__)
    
    @property
    def surface_points(self):
        return self.structural_frame.surface_points
    
    @property
    def orientations(self):
        return self.structural_frame.orientations
    
    @property
    def interpolation_input(self):
        if self.structural_frame.is_dirty:
            self._interpolationInput = InterpolationInput.from_structural_frame(
                structural_frame=self.structural_frame,
                grid=self.grid,
                transform=self.transform
            )
        return self._interpolationInput
    
    
    @property
    def input_data_descriptor(self):
        # TODO: This should have the exact same dirty logic as interpolation_input
        return self.structural_frame.input_data_descriptor
        