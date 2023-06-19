import re
from dataclasses import dataclass

import numpy as np

from .orientations import Orientations
from .surface_points import SurfacePoints

"""
TODO:
 - [ ] Add repr and _repr_html_ methods. Legacy representations depended on pandas, which is optional now

"""




@dataclass
class StructuralElement:
    name: str
    is_active: bool
    _color: str
    surface_points: SurfacePoints
    orientations: Orientations

    # Output
    vertices: np.ndarray
    edges: np.ndarray
    scalar_field: float

    def __init__(self, name, is_active, color):
        self.name = name
        self.is_active = is_active
        self.color = color

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, str) or not re.match("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", value):
            raise ValueError(f"Invalid color: {value}")
        self._color = value

    @property
    def is_basement(self):
        # ? Not sure if this will be necessary
        raise NotImplementedError

    @property
    def has_data(self):
        raise NotImplementedError

    @property
    def index(self):
        raise NotImplementedError

    @property
    def id(self):
        raise NotImplementedError

    @property
    def structural_group(self):
        raise NotImplementedError
