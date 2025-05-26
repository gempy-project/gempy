"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""
from typing import Optional

import numpy as np

from ...core.data.grid_modules.topography import _LoadDEMArtificial


def create_random_topography(extent: np.array, resolution: np.array, dz: Optional[np.array] = None,
                             fractal_dimension: Optional[float] = 2.0) -> np.array:
    dem = _LoadDEMArtificial(
        extent=extent,
        resolution=resolution,
        d_z=dz,
        fd=fractal_dimension
    )

    return dem.get_values()



