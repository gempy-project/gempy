import pprint
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .orientations import OrientationsTable
from .surface_points import SurfacePointsTable

"""
TODO:
 - [ ] Add repr and _repr_html_ methods. Legacy representations depended on pandas, which is optional now

"""


@dataclass
class StructuralElement:
    name: str
    is_active: bool
    _color: str
    surface_points: SurfacePointsTable
    orientations: OrientationsTable

    # Output
    # ? Should we extract this to a separate class?
    vertices: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    scalar_field: Optional[float] = None
    
    
    def __init__(self, name: str, surface_points: SurfacePointsTable, orientations: OrientationsTable,
                 is_active: Optional[bool] = True, color: Optional[str] = None):
        self.name = name
        
        self.surface_points = surface_points
        self.orientations = orientations
        
        self.is_active = is_active
        self.color = color


    def __repr__(self):
        r, g, b = int(self._color[1:3], 16), int(self._color[3:5], 16), int(self._color[5:7], 16)
        colored_color = f'\033[38;2;{r};{g};{b}m' + self._color + '\033[0m'
        return f"Element(\n\tname={self.name},\n\tcolor={colored_color},\n\tis_active={self.is_active}\n)"

    @property
    def number_of_points(self) -> int:
        return len(self.surface_points)
    
    @property
    def number_of_orientations(self) -> int:
        return len(self.orientations)

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
