from dataclasses import dataclass

import numpy as np


@dataclass
class Transform:
    position: np.ndarray = np.zeros(3)
    rotation: np.ndarray = np.zeros(3)
    scale: np.ndarray | float = 1.0
    
    @property
    def isometric_scale(self) -> float:
        return np.mean(self.scale)