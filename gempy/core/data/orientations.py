from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_NUGGET = 0.01


@dataclass
class Orientations:
    data: np.ndarray  # * (x, y, z, G_x, G_y, G_z, id, nugget) # 

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray, 
                    id: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'Orientations':
        return cls(np.array([x, y, z, G_x, G_y, G_z, id, nugget]).T)
