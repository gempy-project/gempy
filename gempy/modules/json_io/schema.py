"""
Schema definitions for JSON I/O operations in GemPy.
This module defines the expected structure of JSON files for loading and saving GemPy models.
"""

from typing import TypedDict, List, Dict, Any, Optional

class SurfacePoint(TypedDict):
    x: float
    y: float
    z: float
    id: int
    nugget: float

class Orientation(TypedDict):
    x: float
    y: float
    z: float
    G_x: float
    G_y: float
    G_z: float
    id: int
    polarity: int

class Fault(TypedDict):
    name: str
    id: int
    is_active: bool

class Series(TypedDict):
    name: str
    id: int
    is_active: bool
    is_fault: bool
    order_series: int

class GridSettings(TypedDict):
    regular_grid_resolution: List[int]
    regular_grid_extent: List[float]
    octree_levels: Optional[int]

class ModelMetadata(TypedDict):
    name: str
    creation_date: str
    last_modification_date: str
    owner: str

class GemPyModelJson(TypedDict):
    metadata: ModelMetadata
    surface_points: List[SurfacePoint]
    orientations: List[Orientation]
    faults: List[Fault]
    series: List[Series]
    grid_settings: GridSettings
    interpolation_options: Dict[str, Any] 