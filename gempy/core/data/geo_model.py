from dataclasses import dataclass

import gempy_engine.core.data.grid
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType
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
    interpolation_grid: gempy_engine.core.data.grid.Grid
    
    _interpolationInput: InterpolationInput  # * This has to be fed by structural_frame
    _input_data_descriptor: InputDataDescriptor  # * This has to be fed by structural_frame
    
    # endregion

    def __init__(self, name: str, structural_frame: StructuralFrame, grid: Grid,
                 interpolation_options: InterpolationOptions):
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
        self.transform = Transform()
    
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
                grid=self.grid
            )
        return self._interpolationInput
    
    
    @property
    def input_data_descriptor(self):
        # TODO: This should have the exact same dirty logic as interpolation_input
        return InputDataDescriptor.from_structural_frame(
            structural_frame=self.structural_frame,
            making_descriptor=[StackRelationType.ERODE],
            faults_relations=None
        )
        
        
