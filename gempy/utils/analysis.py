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
from warnings import warn
import numpy as np
try:
    from skimage.measure import regionprops, label
except ImportError:
    warn("skimage package is not installed, which is required for geomodel complexity analysis.")


def get_nearestneighbor_block(lb):
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
    nn += np.abs(lb[1:, :-1, :-1] ^ lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] ^ lb[1:, :-1, :-1])
    # y
    nn += np.abs(lb[:-1, 1:, :-1] ^ lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] ^ lb[:-1, 1:, :-1])
    # z
    nn += np.abs(lb[:-1, :-1, 1:] ^ lb[:-1, :-1, :-1])
    nn += np.abs(lb[:-1, :-1, :-1] ^ lb[:-1, :-1, 1:])

    return nn


def get_geobody_volume(rprops, geo_data=None):
    """Compute voxel counts per unique integer body in given block model

    Args:
        rprops (list): List of regionprops object for each unique region of the model.
        (skimage.measure._regionprops._RegionProperties object)

    Returns:
        (dict): Dict with node labels as keys and geobody volume values.
    """
    if geo_data is None:
        return {rprop.label:rprop.area for rprop in rprops}
    else:
        return {rprop.label:rprop.area * np.product(geo_data.extent[1::2] / geo_data.resolution) for rprop in rprops}


def get_geobody_tops(rprops, geo_data=None):
    """Get the top vertical limit coordinate of geobodies (via bbox).

    Args:
        rprops (list): List of regionprops object for each unique region of the model.
        (skimage.measure._regionprops._RegionProperties object)
        geo_data (gempy.data_management.InputData):

    Returns:
        (dict): Dict with node labels as keys and geobody top coordinates as values.
    """

    if geo_data is None:
        return {rprop.label: rprop.bbox[5] for rprop in rprops}
    else:
        return {rprop.label: rprop.bbox[5] * geo_data.extent[5] / geo_data.resolution[2] for rprop in rprops}


def get_geobody_bots(rprops, geo_data=None):
    """Get the bottom vertical limit coordinate of geobodies (via bbox).

    Args:
        rprops (list): List of regionprops object for each unique region of the model.
        (skimage.measure._regionprops._RegionProperties object)
        geo_data (gempy.data_management.InputData):

    Returns:
        (dict): Dict with node labels as keys and geobody bottom coordinates as values.
    """
    if geo_data is None:
        return {rprop.label: rprop.bbox[2] for rprop in rprops}
    else:
        return {rprop.label: rprop.bbox[2] * geo_data.extent[5] / geo_data.resolution[2] for rprop in rprops}


def get_centroids(rprops):
    """Get node centroids in 2d and 3d as {node id (int): tuple(x,y,z)}."""
    centroids = {}
    for rp in rprops:
            centroids[rp.label] = rp.centroid
    return centroids


def get_unique_regions(lith_block, fault_block, n_faults, neighbors=8, noddy=False):
    """

    Args:
        lith_block (np.ndarray): Lithology block model
        fault_block (np.ndarray): Fault block model
        n_faults (int): Number of faults.
        neighbors (int, optional): Specifies the neighbor voxel connectivity taken into account for the topology
            analysis. Must be either 4 or 8 (default: 8)
        noddy (bool): If a noddy block is handed to the function, equalizes the results to be comparable with GemPy

    Returns:
        (np.ndarray): Model block with uniquely labeled regions.
    """
    lith_block = np.round(lith_block).astype(int)
    fault_block = np.round(fault_block).astype(int)

    # label the fault block for normalization (comparability of e.g. pynoddy and gempy models)
    fault_block = label(fault_block, neighbors=neighbors, background=9999)

    if noddy:
        # then this is a gempy model, numpy starts with 1
        lith_block[lith_block == 0] = int(np.max(lith_block) + 1)  # set the 0 to highest value + 1
        lith_block -= n_faults  # lower by n_faults to equal with pynoddy models
        # so the block starts at 1 and goes continuously to max

    ublock = (lith_block.max() + 1) * fault_block + lith_block
    labels_block, labels_n = label(ublock, neighbors=neighbors, return_num=True, background=9999)
    if 0 in np.unique(labels_block):
        labels_block += 1

    return labels_block
