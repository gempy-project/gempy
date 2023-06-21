from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_NUGGET = 0.00001


@dataclass
class SurfacePoints:
    data: np.ndarray  # * (x, y, z, id, nugget) # 

    # TODO: Do we need a id to formation mapper here?

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    id: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'SurfacePoints':
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_NUGGET
        return cls(np.array([x, y, z, id, nugget]).T)


    def get_surface_points_by_id(self, id: int) -> 'SurfacePoints':
        return SurfacePoints(self.data[self.data[:, 3] == id])
    
    
    def get_surface_points_by_id_groups(self) -> list['SurfacePoints']:
        ids = np.unique(self.data[:, 3])
        return [self.get_surface_points_by_id(id) for id in ids]