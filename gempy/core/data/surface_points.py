from dataclasses import dataclass

import numpy as np


@dataclass
class SurfacePoints:
    data: np.ndarray  # * (x, y, z, id, nugget) # 
