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

from itertools import combinations
from logging import debug
import numpy as np
from nptyping import Array
from typing import Iterable, List, Set, Tuple, Dict, Union
import matplotlib.pyplot as plt


def compute_topology(
        geo_model, 
        cell_number:int=None,
        direction:str=None,
        n_shift:int=1,
        voxel_threshold:int=1
    ):
    res = geo_model.grid.regular_grid.resolution
    n_unconf = np.count_nonzero(geo_model.series.df.BottomRelation == "Erosion") - 2  # TODO -2 n other lith series
    n_faults = np.count_nonzero(geo_model.faults.df.isFault)

    fb = np.round(
        geo_model.solutions.block_matrix[n_unconf:n_faults + n_unconf, 0, :]
    ).astype(int).sum(axis=0).reshape(*res)

    lb = np.round(
        geo_model.solutions.lith_block
    ).astype(int).reshape(*res)

    n_lith = len(np.unique(lb))  # ? quicker looking it up in geomodel?

    if cell_number is None or direction is None:
        direction = "None"
    elif direction.capitalize() == "X":
        lb = lb[cell_number, :, :]
        fb = fb[cell_number, :, :]
        res = (1, res[1], res[2])
    elif direction.capitalize() == "Y":
        lb = lb[:, cell_number, :]
        fb = fb[:, cell_number, :]
        res = (res[0], 1, res[2])
    elif direction.capitalize() == "Z":
        lb = lb[:, :, cell_number]
        fb = fb[:, :, cell_number]
        res = (res[0], res[1], 1)

    edges, centroids = _analyze_topology(
        fb.ravel(), 
        lb.ravel(), 
        n_lith, 
        res, 
        n_shift, 
        voxel_threshold, 
        direction
    )
    
    edges = set((n1, n2) for n1, n2 in edges)
    edges = _filter_reverse_edges(edges)
    return edges, centroids


def _filter_reverse_edges(edges:Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Filter reversed topology edge tuples to fix doubling of topology edges
    like (1,5) (5,1).
    
    Source:
        https://stackoverflow.com/a/9922322/8040299

    Args:
        edges (Set[Tuple[int, int]]): Set of topologyedge tuples.
    
    Returns:
        Set[Tuple[int, int]]: Filtered set of topology edge tuples
    """
    
    edges_unique=set()
    for e in edges:
        if not (e in edges_unique or (e[1], e[0]) in edges_unique):
            edges_unique.add(e)
    return edges_unique


def _analyze_topology(
        fault_matrix_sum, 
        lith_matrix, 
        n_lith, 
        res, 
        n_shift, 
        voxel_threshold,
        direction
    ):
    fault_shift = fault_matrix_sum.min()
    fault_matrix_sum_shift = fault_matrix_sum - fault_shift

    where = np.tile(lith_matrix, (n_lith, 1)) == np.unique(lith_matrix).reshape(-1, 1)
    lith_matrix_shift = np.sum(where * np.arange(n_lith).reshape(-1, 1), axis=0) + 1

    topo_matrix = lith_matrix_shift + n_lith * fault_matrix_sum_shift
    topo_matrix_3D = topo_matrix.reshape(*res)

    # x direction
    if direction.capitalize() != "X":
        x_l = topo_matrix_3D[n_shift:,:,:]
        x_r = topo_matrix_3D[:-n_shift,:,:]
        x_edges, x_count = _get_edges(x_l, x_r)
        x_edges = x_edges[:, x_count > voxel_threshold]
    else:
        x_edges = np.array([[],[]])
    
    # y direction
    if direction.capitalize() != "Y":
        y_l = topo_matrix_3D[:, n_shift:,:]
        y_r = topo_matrix_3D[:, :-n_shift,:]
        y_edges, y_count = _get_edges(y_l, y_r)
        y_edges = y_edges[:, y_count > voxel_threshold]
    else:
        y_edges = np.array([[],[]])

    # z direction
    if direction.capitalize() != "Z":
        z_l = topo_matrix_3D[:, :, n_shift:]
        z_r = topo_matrix_3D[:, :, :-n_shift]
        z_edges, z_count = _get_edges(z_l, z_r)
        z_edges = z_edges[:, z_count > voxel_threshold]
    else:
        z_edges = np.array([[],[]])

    edges = np.unique(
        np.concatenate((x_edges.T, y_edges.T, z_edges.T), axis=0), axis=0
    )

    centroids = _get_centroids(topo_matrix_3D)

    return edges, centroids


def get_lot_node_to_lith_id(
        geo_model, 
        centroids:Dict[int, Array[float, 3]]
    ) -> Dict[int, int]:
    """Get look-up table to translate topology node id's back into GemPy lith
    id's.
    
    Args:
        geo_model: GemPy geomodel instance with solutions.
        centroids (Dict[int, Array[float, 3]]): Topology node centroids.
    
    Returns:
        Dict[int, int]: Look-up table translating node id -> lith id.
    """
    lb = geo_model.solutions.lith_block.reshape(
        geo_model.grid.regular_grid.resolution
    ).astype(int)

    lot = {}
    for node, pos in centroids.items():
        p = np.round(pos).astype(int)
        lith_id = lb[p[0], p[1], p[2]]
        lot[node] = lith_id
    return lot


def get_lot_lith_to_node_id(
        lot:Dict[int, Array[float, 3]]
    ) -> Dict[int, List[int]]:
    """Get look-up table to translate lith id's back into topology node
    id's.
    
    Args:
        lot (Dict[int, Array[float, 3]]): Node to lith id look-up table. Can be
        computed using the function 'get_lot_node_to_lith_id'.
    
    Returns:
        Dict[int, List[int]]: Look-up table.
    """
    lot2 = {}
    for k, v in lot.items():
        if v not in lot2.keys():
            lot2[v] = [k]
        else:
            lot2[v].append(k)
    return lot2


def _get_edges(
        l:Array[int, ..., ..., ...], 
        r:Array[int, ..., ..., ...]
    ) -> Array[int, ..., 2]:
    """Get edges from given shifted arrays.

    Args:
        l (Array): Topology labels array shifted to one direction.
        r (Array): Topology labels array shifted to the other direction.

    Returns:
        Array: Topology edges.
    """
    shift = np.stack([l.ravel(),  r.ravel()])
    i1, i2 = np.nonzero(np.diff(shift, axis=0))
    return np.unique(shift[:, i2], axis=1, return_counts=True)


def clean_unconformity_topology(
        geo_model, 
        unconf_lith_id:int, 
        edges:Array[int, ..., 2], 
        centroids:Dict[int, Array[int, ..., 3]]
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, Array[int, ..., 3]]]:
    """Clean unconformity topology edges and centroids. Needs to be run for
    each unconformity separately.
    
    Args:
        geo_model ([type]): [description]
        unconf_lith_id (int): [description]
        edges (Array[int, ..., 2]): [description]
        centroids (Dict[int, Array[int, ..., 3]]): [description]
    
    Returns:
        Tuple[Set[Tuple[int, int]], Dict[int, Array[int, ..., 3]]]: [description]
    """
    lot = get_lot_node_to_lith_id(geo_model, centroids)
    lot2 = get_lot_lith_to_node_id(lot)
    edges_clean = []
    
    unconf_centroid = np.mean(
        [centroids[i] for i in lot2[unconf_lith_id]], axis=0
    )
    for node_id in lot2[unconf_lith_id]:
        centroids.pop(node_id)
    centroids[lot2[unconf_lith_id][0]] = unconf_centroid
    
    for n1, n2 in edges:
        if n1 in lot2[unconf_lith_id]:
            n1 = lot2[unconf_lith_id][0]
        if n2 in lot2[unconf_lith_id]:
            n2 = lot2[unconf_lith_id][0]
        if n1 == n2:
            continue
        edges_clean.append((n1, n2))
    
    return set(edges_clean), centroids


def jaccard_index(
        edges1:Set[Tuple[int, int]], 
        edges2:Set[Tuple[int, int]]
    ) -> float:
    """Jaccard index.
    
    Args:
        edges1 (Set[Tuple[int, int]]): Set of topology edges.
        edges2 (Set[Tuple[int, int]]): Set of topology edges.
    
    Returns:
        float: Jaccard index.
    """
    intersection_cardinality = len(edges1.intersection(edges2))
    union_cardinality = len(edges1.union(edges2))
    return intersection_cardinality / union_cardinality



# def compute_topology(
#         geo_model, 
#         cell_number:int=None,
#         direction:str=None,
#         n_shift:int=1,
#         voxel_threshold:int=0
#     ) -> Tuple[Set, Dict]:
#     """Compute geomodel topology.

#     Wrapper function
    
#     Args:
#         geo_model ([type]): Computed GemPy geomodel instance. 
#         n_shift (int, optional): Number of voxels to shift. Defaults to 1.
    
#     Returns:
#         tuple: set of edges, centroid dictionary
#     """
#     res = geo_model.grid.regular_grid.resolution
#     n_unconf = np.count_nonzero(geo_model.series.df.BottomRelation == "Erosion") - 2
#     n_faults = np.count_nonzero(geo_model.faults.df.isFault)
#     # n_liths = len(geo_model.surfaces.df) - n_faults

#     lb = np.round(geo_model.solutions.lith_block).astype(int).reshape(*res)
#     n_liths = len(np.unique(lb))

#     fb = np.round(
#         geo_model.solutions.block_matrix[n_unconf:-1, :]
#     ).astype(int).reshape(n_faults, *res)

#     if cell_number is None or direction is None:
#         pass
#     elif direction.capitalize() == "X":
#         lb = lb[cell_number, :, :]
#         fb = fb[:, cell_number, :, :]
#         res = (1, res[1], res[2])
#     elif direction.capitalize() == "Y":
#         lb = lb[:, cell_number, :]
#         fb = fb[:, :, cell_number, :]
#         res = (res[0], 1, res[2])
#     elif direction.capitalize() == "Z":
#         lb = lb[:, :, cell_number]
#         fb = fb[:, :, :, cell_number]
#         res = (res[0], res[1], 1)

#     lith_ids = get_lith_ids(geo_model)

#     debug(f"lb shape: {lb.flatten().shape}")
#     debug(f"fb shape: {fb.reshape(n_faults, -1).shape}")

#     return _analyze_topology(
#         lb.flatten(), 
#         fb.reshape(n_faults, -1), 
#         n_faults, 
#         n_liths, 
#         res,
#         n_shift,
#         voxel_threshold,
#         lith_ids,
#         n_unconf
#     )
    

# def _analyze_topology(
#         lb:Array[int, ...], 
#         fb:Array[int, ..., ...], 
#         n_faults:int, 
#         n_liths:int, 
#         res:Tuple[int, int, int], 
#         n_shift:int, 
#         voxel_threshold:int,
#         lith_ids:Array[int, ...],
#         n_unconf:int
#     ) -> Tuple:
#     topology_labels = _get_topology_labels(lb, fb, n_liths, lith_ids, n_unconf)
#     shift_xyz_block = _topology_shift(topology_labels, res, n_shift=n_shift)
#     labels = _bitstack_topology_labels(topology_labels).reshape(*res)
#     edges = _get_edges(
#         shift_xyz_block, 
#         labels, 
#         res, 
#         n_shift, 
#         voxel_threshold=voxel_threshold
#     )
#     centroids = _get_centroids(labels)

#     return edges, centroids


# def get_lith_ids(geo_model) -> Array[int, ...]:
#     lith_ids = []
#     for series, i in geo_model.surfaces.df.groupby("series").groups.items():
#         if geo_model.faults.df.loc[series].isFault:
#             continue

#         lith_id = geo_model.surfaces.df.loc[i, "id"].values
#         for li in lith_id:
#             lith_ids.append(li)

#     return np.array(lith_ids)


# def _get_topology_labels(
#         lb:Array[int, ...], 
#         fb:Array[int, ..., ...],
#         n_lith:int,
#         lith_ids:Array[int, ...],
#         n_unconf:int
#     ) -> Array[bool, ..., ...]:
#     """Get unique topology labels block.
    
#     Args:
#         lb (Array[int, ...]): Flattened lithology block from GemPy model.
#         fb (Array[int, ..., ...]): Flattened fault block stack from GemPy 
#             model.
#         n_lith (int): Number of lithologies.
    
#     Returns:
#         Array[bool, ..., ...]: Boolean topology label array, with first
#             dimension representing the topology id's and the second axis
#             representing the flattened voxel array.
#     """
#     debug(f"lb shape: {lb.shape}")

#     # fb -= np.arange(1, fb.shape[0] + 1)[None, :].T
#     fb -= np.arange(1 + n_unconf, fb.shape[0] + 1 + n_unconf)[None, :].T
#     fb = np.repeat(fb, 2, axis=0).astype(bool)  # 2 digits for each fb
#     fb[::2] = ~fb[::2]  # invert bool for every duplicate fb
#     # lb = lb - lb.min()
    
#     lb_labels = np.tile(lb, (n_lith, 1)) == lith_ids.reshape(-1,1)
#     debug(f"lb labels shp: {lb_labels.shape}")
#     return np.concatenate((lb_labels, fb), axis=0).astype(bool)


# def _topology_shift(
#         topology_labels:Array[bool, ..., ...],
#         res:Iterable[int],
#         n_shift:int=1
#     ) -> Tuple[Array[int, ...]]:
#     n_digits = topology_labels.shape[0]
#     topology_block = topology_labels.reshape(n_digits, *res)

#     x = np.logical_or(
#         topology_block[:, n_shift:, :, :],
#         topology_block[:, :-n_shift, :, :]
#     )
#     debug(f"x shift shp: {x.shape}")

#     y = np.logical_or(
#         topology_block[:, :, n_shift:, :],
#         topology_block[:, :, :-n_shift, :]
#     )
#     debug(f"y shift shp: {y.shape}")

#     z = np.logical_or(
#         topology_block[:, :, :, n_shift:],
#         topology_block[:, :, :, :-n_shift]
#     )
#     debug(f"z shift shp: {z.shape}")

#     x_flat = x.reshape(
#         n_digits, (res[0] - n_shift) * res[1] * res[2]
#     ).astype(int)
#     x_flat_bin = np.sum(x_flat * 2**np.arange(n_digits)[None, ::-1].T, axis=0)

#     y_flat = y.reshape(
#         n_digits, (res[1] - n_shift) * res[0] * res[2]
#     ).astype(int)
#     y_flat_bin = np.sum(y_flat * 2**np.arange(n_digits)[None, ::-1].T, axis=0)

#     z_flat = z.reshape(
#         n_digits, (res[2] - n_shift) * res[1] * res[0]
#     ).astype(int)
#     z_flat_bin = np.sum(z_flat * 2**np.arange(n_digits)[None, ::-1].T, axis=0)
    
#     debug(f"x_flat_bin unique: {np.unique(x_flat_bin)}")

#     return x_flat_bin, y_flat_bin, z_flat_bin


# def _bitstack_topology_labels(
#         topology_labels:Array[bool, ..., ...]
#     ) -> Array[int, ...]:
#     n = topology_labels.shape[0]
#     return np.sum(topology_labels * 2**np.arange(n)[None, ::-1].T, axis=0)


# def _get_edges(
#         shift_xyz_block:Tuple[Array[int, ...]], 
#         labels:Array[int, ..., ..., ...],
#         res:tuple, 
#         n_shift:int,
#         voxel_threshold:int=10
#     ) -> Set[Tuple]:
#     """Extract binary topology edges from given labels and shift blocks.
    
#     Args:
#         topology_labels (Array[bool, ..., ...]): [description]
#         shift_xyz_block (Array[int, 3, ...]): [description]
#         res (tuple): [description]
#         n_shift (int): [description]
    
#     Returns:
#         set: Set of topology edges as node tuples (n1, n2).
#     """
#     edges = set()
#     reshaper = _get_reshapers(res, n_shift)
    
#     for i, shift_block, shape in zip(range(3), shift_xyz_block, reshaper):
#         debug(f"i: {i}")
#         shift_block = shift_block.reshape(*shape)
#         debug(shift_block.shape)
#         edge_sum_bins, counts = np.unique(shift_block, return_counts=True)
#         debug(f"edge sum bins: {edge_sum_bins}")

#         for edge_sum_bin, count in zip(edge_sum_bins, counts):
#             if count <= voxel_threshold:
#                 continue

#             x, y, z = np.argwhere(shift_block == edge_sum_bin)[0]

#             e1 = labels[x, y, z]

#             if i == 0:
#                 e2 = labels[x + 1, y, z]
#             elif i == 1:
#                 e2 = labels[x, y + 1, z]
#             elif i == 2:
#                 e2 = labels[x, y, z + 1]
            
#             if e1 == e2:
#                 continue
            
#             edges.add((e1, e2))

#     return edges


# def _get_reshapers(res:tuple, n_shift:int) -> tuple:
#     """Get reshaping tuples based on given geomodel
#     voxel resolution and voxel shift.
    
#     Args:
#         res (tuple): Geomodel resolution (nx, ny, nz)
#         n_shift (int): Shift in number of voxels
    
#     Returns:
#         tuple: Contains three reshaper tuples for x, y and z 
#             direction considering the given shift. 
#     """
#     reshaper = (
#         (res[0] - n_shift, res[1], res[2]),
#         (res[0], res[1] - n_shift, res[2]),
#         (res[0], res[1], res[2] - n_shift)
#     )
#     return reshaper


def _get_centroids(labels:Array[int, ..., ..., ...]) -> dict:
    """Get geobody node centroids in array coordinates.
    
    Args:
        labels (Array[int, ..., ..., ...]): Uniquely labeled block.
    
    Returns:
        dict: Geobody node keys yield centroid coordinate tuples in array
            coordinates.
    """
    node_locs = []
    ulabels = np.unique(labels)
    for node in ulabels:
        node_pos = np.argwhere(labels==node)
        node_locs.append(node_pos.mean(axis=0))
    centroids = {n:loc for n, loc in zip(ulabels, node_locs)}
    # for k, v in centroids.items():
        # debug(f"{k}: {v}")
    return centroids


# *****************************************************************************
# *****************************************************************************
# * ADJACENCY MATRIX
# *****************************************************************************


def adj_matrix(
        edges:Set[tuple], 
        labels:Array[int, ..., ..., ...],  
        n_faults:int, 
        n_liths:int
    ) -> (Array[bool, ..., ...], List[str]):
    f_labels = _get_fault_labels(n_faults)
    fault_labels_bin = _get_fault_label_comb_bin(f_labels)
    lith_labels_bin = _get_lith_labels_bin(n_liths)
    adj_matrix_labels = _get_adj_matrix_labels(lith_labels_bin, fault_labels_bin)
    return _get_adj_matrix(edges, adj_matrix_labels, labels), adj_matrix_labels


def plot_adj_matrix(adj_matrix:Array[bool, ..., ...], adj_labels:List[str]):
    fig, ax = plt.subplots(figsize=(10, 10))

    n_labels = len(adj_labels)

    ax.imshow(adj_matrix, cmap="YlOrRd")

    fontdict = dict(
        fontsize=10
    )

    ax.set_xticklabels(adj_labels, fontdict=fontdict, rotation=90)
    ax.set_yticklabels(adj_labels, fontdict=fontdict)
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))

    ax.set_ylim(-.5, n_labels - .5)
    ax.set_xlim(-.5, n_labels - .5)

    ax.set_title("Adjacency Matrix")
    plt.show()


def _get_fault_labels(n_faults:int) -> Array[int, ..., 2]:
    """Get unique fault label id pairs for each fault block. For two faults
    this looks like: [[0 1]
                      [2 3]]
    
    Args:
        n_faults (int): Number of faults.
    
    Returns:
        Array[int, ..., 2]: Unique consecutive fault label id pairs.
    """
    flabels = np.stack(
        (
            np.arange(n_faults), 
            np.arange(1, n_faults + 1)
        )
    ).T + np.arange(n_faults)[None, :].T
    return flabels


def _get_fault_label_comb_bin(fault_labels:Array[int, ..., 2]) -> List[str]:
    """Get unique binary fault label combinations. E.g. for two faults the 
    output looks like: ['0101', '1001', '0110', '1010'].
    
    Args:
        fault_labels (Array[int, ..., 2]): Unique base-10 fault label array.
    
    Returns:
        List[str]: List of binary fault label combinations.
    """
    n_faults = fault_labels.shape[0]
    fault_labels_bin = []
    for comb in combinations(fault_labels.flatten(), n_faults):
        if sum(comb) in np.sum(fault_labels, axis=1):
            continue  # skip combinations within the same fault block
        fault_labels_bin.append(
            np.binary_repr(sum(2**np.array(comb))).zfill(n_faults * 2)
        )

    return fault_labels_bin


def _get_lith_labels_bin(n_layers:int) -> List[str]:
    """Get unique binary lith labels list. For five layers this looks like:
    ['00001', '00010', '00100', '01000', '10000'].

    Args:
        n_layers (int): Number of layers.
    
    Returns:
        List[str]: Unique binary lith labels.
    """
    return [np.binary_repr(2**i).zfill(n_layers) for i in range(n_layers)]


def _get_adj_matrix_labels(
        lith_labels_bin:List[str], 
        fault_labels_bin:List[str]
    ) -> List[str]:
    """Get all possible valid combinations between lithology id's and fault
    blocks in binary.
    
    Args:
        lith_labels_bin (List[str]): Unique binary lithology labels.
        fault_labels_bin (List[str]): Unique binary fault combination labels. 
    
    Returns:
        List[str]: ['000010101', '000011001', '000010110', ...]
    """
    return [l+f for l in lith_labels_bin for f in fault_labels_bin]


def _get_adj_matrix(
        edges:Iterable, 
        adj_matrix_labels:Iterable, 
        labels:Array[int, ..., ..., ...]
    ) -> Array[bool, ..., ...]:
    """Generate adjacency matrix from given list of edges, all possible unique
    geo- model nodes and actual unique geobody labels.
    
    Args:
        edges (Iterable): [(n, m), ...]
        adj_matrix_labels (Iterable): ["000010101", ...]
        labels (Array[int, ..., ..., ...]): Uniquely labeled block matrix.
    
    Returns:
        Array[bool, ..., ...]: Boolean adjacency matrix encoding the geomodel
            topology.
    """
    n = len(adj_matrix_labels)
    adj_matrix = np.zeros((n, n)).astype(bool)

    n_entities = len(adj_matrix_labels[0])

    for n1, n2 in edges:
        i = adj_matrix_labels.index(np.binary_repr(n1).zfill(n_entities))
        j = adj_matrix_labels.index(np.binary_repr(n2).zfill(n_entities))
        adj_matrix[i, j] = True
        adj_matrix[j, i] = True

    for bin_label in [np.binary_repr(l).zfill(9) for l in np.unique(labels)]:
        i = adj_matrix_labels.index(bin_label)
        adj_matrix[i, i] = True  

    return adj_matrix