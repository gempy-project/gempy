from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_NUGGET = 0.01

# ? Maybe we should merge this with the SurfacePoints class from gempy_engine

@dataclass
class Orientations:
    data: np.ndarray  # * (x, y, z, G_x, G_y, G_z, id, nugget) # 

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray, 
                    id: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'Orientations':
        
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_NUGGET
            
        return cls(np.array([x, y, z, G_x, G_y, G_z, id, nugget]).T)
    
    def get_orientations_by_id(self, id: int) -> 'Orientations':
        return Orientations(self.data[self.data[:, 6] == id])
    
    
    def get_orientations_by_id_groups(self) -> list['Orientations']:
        ids = np.unique(self.data[:, 6])
        return [self.get_orientations_by_id(id) for id in ids]
