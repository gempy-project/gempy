from dataclasses import dataclass
from typing import Optional


@dataclass
class ImporterHelper:
    path_to_surface_points: Optional[str] 
    path_to_orientations: Optional[str]
    
    hash_surface_points: Optional[str] = None
    hash_orientations: Optional[str] = None
    
    # ? Add here pandas reader options