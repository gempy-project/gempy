from dataclasses import dataclass

import numpy as np


@dataclass
class Orientations:
    data: np.ndarray  # * (x, y, z, G_x, G_y, G_z, id, nugget) # 
    