"""
Schema definitions for JSON I/O operations in GemPy.
This module defines the expected structure of JSON files for loading and saving GemPy models.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Sequence

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired # Fallback for older Python versions


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

class Surface(TypedDict, total=False):
    name: str  # Required
    id: NotRequired[int]  # Optional, will be auto-generated if not provided
    color: NotRequired[Optional[str]]  # Optional hex color code
    vertices: NotRequired[Optional[List[List[float]]]]  # Optional list of [x, y, z] coordinates

class Fault(TypedDict, total=False):
    name: str  # Required
    id: NotRequired[int]  # Optional, will be auto-generated
    is_active: NotRequired[bool]  # Optional, defaults to True
    surface: Surface

class Series(TypedDict, total=False):
    name: str  # Required
    surfaces: Union[List[str], List[Surface]]  # Required, can be list of names or Surface objects
    id: NotRequired[int]  # Optional, will be auto-generated
    is_active: NotRequired[bool]  # Optional, defaults to True
    is_fault: NotRequired[bool]  # Optional, defaults to False
    order_series: NotRequired[int]  # Optional, will be auto-generated
    faults: NotRequired[List[Fault]]  # Optional, defaults to empty list
    structural_relation: NotRequired[str]  # Optional, defaults to "ONLAP"
    colors: NotRequired[Optional[List[str]]]  # Optional

class GridSettings(TypedDict, total=False):
    regular_grid_resolution: NotRequired[List[int]]  # Optional, defaults to [10, 10, 10]
    regular_grid_extent: NotRequired[List[float]]  # Optional, auto-calculated from data
    octree_levels: NotRequired[Optional[int]]  # Optional

class ModelMetadata(TypedDict, total=False):
    name: NotRequired[str]  # Optional, defaults to "GemPy Model"
    creation_date: NotRequired[str]  # Optional, defaults to current date
    last_modification_date: NotRequired[str]  # Optional, defaults to current date
    owner: NotRequired[Optional[str]]  # Optional, defaults to "GemPy Team"

class IdNameMapping(TypedDict, total=False):
    name_to_id: Dict[str, int]  # Required if id_name_mapping is provided

class GemPyModelJson(TypedDict):
    surface_points: List[SurfacePoint]  # Required
    orientations: List[Orientation]  # Required
    series: List[Series]  # Required but with minimal required fields
    metadata: NotRequired[ModelMetadata]  # Optional
    grid_settings: NotRequired[Optional[GridSettings]]  # Optional
    interpolation_options: NotRequired[Optional[Dict[str, Any]]]  # Optional
    id_name_mapping: NotRequired[IdNameMapping]  # Optional 