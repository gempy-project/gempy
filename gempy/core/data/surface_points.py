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
