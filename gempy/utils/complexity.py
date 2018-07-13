"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


@author: Alexander Schaaf
"""

import numpy as np
from gempy.topology import get_unique_regions


def nearest_neighbors(lb):
    """
    Simplest 3D nearest neighbor comparison to (6-stamp) for lithology values.

    Args:
        lb (np.ndarray): lb[0].reshape(*geo_data.resolution).astype(int)

    Returns:
        (np.ndarray)
    """
    shp = lb.shape
    nn = np.zeros((shp[0] - 1, shp[0] - 1, shp[0] - 1))
    # x
    nn += np.abs(lb[1:, :-1, :-1] - lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] - lb[1:, :-1, :-1])
    # y
    nn += np.abs(lb[:-1, 1:, :-1] - lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] - lb[:-1, 1:, :-1])
    # z
    nn += np.abs(lb[:-1, :-1, 1:] - lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] - lb[:-1, :-1, 1:])

    return nn
