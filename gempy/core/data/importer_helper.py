from dataclasses import dataclass
from typing import Optional


@dataclass
class ImporterHelper:
    path_to_surface_points: Optional[str] 
    path_to_orientations: Optional[str]
    
    hash_surface_points: Optional[str] = None
    hash_orientations: Optional[str] = None
    
    coord_x_name: Optional[str] = "X"
    coord_y_name: Optional[str] = "Y"
    coord_z_name: Optional[str] = "Z"
    surface_name: Optional[str] = "formation"
    
    gx_name: Optional[str] = "G_x"
    gy_name: Optional[str] = "G_y"
    gz_name: Optional[str] = "G_z"
    
    pandas_reader_kwargs: Optional[dict] = None
    
    
    # ? Add here pandas reader options