"""
Schema definitions for JSON I/O operations in GemPy.
This module defines the expected structure of JSON files for loading and saving GemPy models.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Sequence

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
    G_x: float  # X component of the gradient
    G_y: float  # Y component of the gradient
    G_z: float  # Z component of the gradient
    id: int
    nugget: float
    polarity: int  # 1 for normal, -1 for reverse

class Surface(TypedDict):
    name: str
    id: int
    color: Optional[str]  # Hex color code
    vertices: Optional[List[List[float]]]  # List of [x, y, z] coordinates

class Fault(TypedDict):
    name: str
    id: int
    is_active: bool
    surface: Surface

class Series(TypedDict):
    name: str
    id: int
    is_active: bool
    is_fault: bool
    order_series: int
    surfaces: List[Surface]
    faults: List[Fault]

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
    series: List[Series]
    grid_settings: GridSettings
    interpolation_options: Dict[str, Any] 