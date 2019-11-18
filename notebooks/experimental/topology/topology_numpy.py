"""
*******************************************************************************
Prototyping functionality for new improved implementation of GemPy geomodel 
topology analysis that is mainly numpy-based for taking advantage of vectorized 
computation and serves as a prototype for the future theano/tensorflow 
implementation for GemPy. 

*******************************************************************************
@author: Alexander Schaaf

*******************************************************************************
GemPy is licensed under the GNU Lesser General Public License v3.0

*******************************************************************************
"""

from itertools import combinations
from logging import debug
import numpy as np
from nptyping import Array
from typing import Iterable, List


def lithblock_to_lb_fb(geo_model) -> tuple:
    """Serve flattened lithology id and fault blocks from solutions
    stored in given geomodel instance for topological analysis (i.e.
    rounded integer arrays).
    
    Args:
        geo_model (gempy.core.model.Model): GemPy Model instance with solutions.
    
    Returns:
        (tuple) of np.ndarray's containing the lithilogy id block
            and fault block stack.
    """
    lb = np.round(geo_model.solutions.block_matrix[-1, 0, :]).astype(int)
    fb = np.round(geo_model.solutions.block_matrix[:-1, 0, :]).astype(int)
    return lb, fb


def get_fault_ids(geo_model) -> np.array:
    """Get surface id's for all faults in given geomodel.
    
    Args:
        geo_model (gempy.core.model.Model): GemPy Model instance with solutions.
        
    Returns:
        (np.array) of int surface id's.
    """
    isfault = np.isin(
        geo_model.surfaces.df.series, 
        geo_model.faults.df.index[geo_model.faults.df.isFault]
    )
    return geo_model.surfaces.df.id[isfault].values


def get_labels_block(
        lb:Array[int, ...], 
        fb:Array[int, ..., ...]) -> Array[int, ...]:
    """Uniquely binary labeled geobodies in geomodel for topology analysis.
    
    Args:
        lb (Array[int, ...]): Lithology id matrix.
        fb (Array[int, n_faults, ...]): Fault id matrix.
    
    Returns:
        Array[int, ...]: Uniquely labeled matrix.
    """
    # faults
    n_faults = fb.shape[0]
    fb -= 1  # shift fb's to start at 0
    fb += np.arange(n_faults)[None, :].T  # add fault numbers vector to 
    # consecutively label fb's uniquely
    
    debug(f"fb shp: {fb.shape}; fb unique: {np.unique(fb)}")
    for i, block in enumerate(fb):
        debug(f"fb {i}: {np.unique(block)}")
    
    # lithologies
    debug(f"lb shp: {lb.shape}; lb unique: {np.unique(lb)}")
    # shift lb id's to 0, then shift to number of faults + 2 to create a
    # consecutive labeling of lithologies starting after the highest fault 
    # block id
    lb = lb - lb.min() + n_faults + 2
    debug(f"lb shift unique: {np.unique(lb)}")
    # concatenate lb and fb's, then raise labels to the power of 2 for 
    # binary labeling
    labels = 2**np.concatenate((lb[None, :], fb), axis=0)

    debug(f"labels shp: {labels.shape}")
    debug(f"\nunique labels:")
    for label in np.unique(labels):
        debug(np.binary_repr(label).zfill(9) + " <-> " + str(label))
    # sum along concatenation axis to create uniquely labeled geobodies
    # with unique id's within each fault block
    labels = labels.sum(axis=0)

    debug(f"\nsummed labels:\nsum unique: {np.unique(labels)}")
    for label in np.unique(labels):
        debug(np.binary_repr(label).zfill(9) + " <-> " + str(label))

    return labels


def get_topology_labels(
        lb:Array[int, ...], 
        fb:Array[int, ..., ...],
        n_lith:int
    ) -> Array[bool, ..., ...]:
    """Get unique topology labels block.
    
    Args:
        lb (Array[int, ...]): Flattened lithology block from GemPy model.
        fb (Array[int, ..., ...]): Flattened fault block stack from GemPy 
            model.
        n_lith (int): Number of lithologies.
    
    Returns:
        Array[bool, ..., ...]: Boolean topology label array, with first
            dimension representing the topology id's and the second axis
            representing the flattened voxel array.
    """
    fb -= np.arange(1, fb.shape[0] + 1)[None, :].T
    fb = np.repeat(fb, 2, axis=0).astype(bool)  # 2 digits for each fb
    fb[::2] = ~fb[::2]  # invert bool for every duplicate fb
    lb = lb - lb.min()
    lb_labels = np.tile(lb, (n_lith,1)) == np.arange(n_lith).reshape(-1,1)
    return np.concatenate((lb_labels, fb), axis=0).astype(bool)


def bitstack_topology_labels(
        topology_labels:Array[bool, ..., ...]
    ) -> Array[int, ...]:
    n = topology_labels.shape[0]
    return np.sum(topology_labels * 2**np.arange(n)[None, ::-1].T, axis=0)


def topology_shift(
        topology_labels:Array[bool, ..., ...],
        res:Iterable[int],
        n_shift:int=1
    ) -> Array[int, 3, ...]:
    n_digits = topology_labels.shape[0]
    topology_block = topology_labels.reshape(n_digits, *res)

    x = np.logical_or(
        topology_block[:, n_shift:, :, :],
        topology_block[:, :-n_shift, :, :]
    )
    debug(f"x shift shp: {x.shape}")

    y = np.logical_or(
        topology_block[:, :, n_shift:, :],
        topology_block[:, :, :-n_shift, :]
    )
    debug(f"y shift shp: {y.shape}")

    z = np.logical_or(
        topology_block[:, :, :, n_shift:],
        topology_block[:, :, :, :-n_shift]
    )
    debug(f"z shift shp: {z.shape}")

    x_flat = x.reshape(
        n_digits, (res[0] - n_shift) * res[1] * res[2]
    ).astype(int)

    y_flat = y.reshape(
        n_digits, (res[1] - n_shift) * res[0] * res[2]
    ).astype(int)

    z_flat = z.reshape(
        n_digits, (res[2] - n_shift) * res[1] * res[0]
    ).astype(int)

    xyz_flat = np.stack((x_flat, y_flat, z_flat))
    debug(f"xys shape: {xyz_flat.shape}")
    xyz_bin = np.sum(xyz_flat * 2**np.arange(n_digits)[None, ::-1].T, axis=1)
    debug(f"xyz bin shape: {xyz_bin.shape}")
    return xyz_bin#, np.stack((x_flat[None, :], y_flat[None, :], z_flat[None, :]))[:, 0, :, :]


def get_topo_block(
        labels:Array[int, ..., ..., ...], 
        n_shift:int=1) -> Array[int, 3, ..., ..., ...]:
    """Create topology block by shifting along x, y, z axes and
    summing up.
    
    Args:
        labels (Array[int, ..., ..., ...]): Labels block shaped (nx, ny, nz).
        n_shift (int, optional): Number of voxels to shift. Defaults to 1.
    
    Returns:
        Array[int, 3, ..., ..., ...]: Shifted and summed labels block used to
            analyze the geobody topology of the geomodel.
    """
    sum_x = np.abs(labels[n_shift:, :, :] + labels[:-n_shift, :, :])
    debug(f"sum_x shp: {sum_x.shape}")
    sum_y = np.abs(labels[:, n_shift:, :] + labels[:, :-n_shift, :])
    debug(f"sum_y shp: {sum_y.shape}")
    sum_z = np.abs(labels[:, :, n_shift:] + labels[:, :, :-n_shift])
    debug(f"sum_z shp: {sum_z.shape}")
    
    slx, sly, slz = (slice(n_shift // 2, -n_shift//2) for i in range(3))
    debug(f"slx {slx}; sly {sly}; slz {slz}")
    
    topo_block = np.concatenate(
        (
         sum_x[None, :, sly, slz], 
         sum_y[None, slx, :, slz], 
         sum_z[None, slx, sly, :]
        ), axis=0
    )
    debug(f"{topo_block.shape}")
    return topo_block


def get_node_label_sum_lot(ulabels:np.array) -> dict:
    """Get look-up table from sum of nodes (key) to
    constituent nodes (value) in the form of a tuple
    of geobody nodes.
    
    Args:
        ulabels (np.array):
        
    Returns:
        (dict)
    """
    possible_edges = list(combinations(ulabels, 2))
    debug(f"possible node combinations: {possible_edges}")
    ulabel_LOT = {sum(comb):comb for comb in possible_edges}
    for k, v in ulabel_LOT.items():
        debug(f"{k} = {v[0]} + {v[1]}")
    return ulabel_LOT


def get_edges(
        shift_xyz_block:Array[int, 3, ...], 
        labels:Array[int, ..., ..., ...],
        res:tuple, 
        n_shift:int,
    ) -> set:
    """Extract binary topology edges from given labels and shift blocks.
    
    Args:
        topology_labels (Array[bool, ..., ...]): [description]
        shift_xyz_block (Array[int, 3, ...]): [description]
        res (tuple): [description]
        n_shift (int): [description]
    
    Returns:
        set: Set of topology edges as node tuples (n1, n2).
    """
    edges = set()
    slicers = get_slicers(n_shift)
    reshaper = get_reshapers(res, n_shift)
    
    # labels = bitstack_topology_labels(topology_labels).reshape(*res)

    for i, shift_block, shape in zip(range(3), shift_xyz_block, reshaper):
        # for every binary shift block with the correct
        # reshaper for depending on n_shift
        shift_block = shift_block.reshape(*shape)

        b1 = labels[slicers[i][0]]  # get correctly shifted labels block 1
        b2 = labels[slicers[i][1]]  # get correctly shifted labels block 2

        
        for edge_sum_bin in np.unique(shift_block, return_counts=False):
            # for every unique binary edge sum in the
            # xyz shift block

            # select only voxels where they equal
            # the unique binary edge sum in the 
            # shift block
            filter_ = shift_block == edge_sum_bin
            # select the real binary pre-comparison
            # labels for each edge
            e1 = b1[filter_]
            e2 = b2[filter_]

            # some edges do not exist in the labels
            # block, skip those
            if len(e1) == 0 or len(e2) == 0:
                continue

            # skip self topology
            if e1[0] == e2[0]:
                continue

            debug(f"({e1[0]}, {e2[0]})")

            edges.add(
                (e1[0],
                 e2[0])
            )
    
    return edges


def get_centroids(labels:Array[int, ..., ..., ...]) -> dict:
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
    for k, v in centroids.items():
        debug(f"{k}: {v}")
    return centroids


def get_lith_lot(
        labels:Array[int, ..., ..., ...], 
        n_faults:int, 
        n_layers:int
    ) -> dict:
    """Create look-up table to go from combined geobody node id to 
    lithology id.
    
    Args:
        labels (Array[int, ..., ..., ...]): Uniquely labeled block matrix.
        n_faults (int): Number of faults in the model.
    
    Returns:
        dict: Mapping node id's to lithology id's 
    """
    ulabels = np.unique(labels)
    layer_ids = {
        np.binary_repr(2**i).zfill(n_faults * 2 + n_layers):i 
        for i in range(n_faults * 2, n_faults * 2 + n_layers)
    }  

    node_to_layer_LOT = {}
    for node in ulabels:
        node_bin = np.binary_repr(node).zfill(n_faults * 2 + n_layers)

        node_bin_nofault = node_bin[:-n_faults * 2]
        node_bin_nofault += "0" * n_faults * 2

        for k, v in layer_ids.items():
            if node_bin_nofault in k:
                node_to_layer_LOT[node] = v
                
    return node_to_layer_LOT


def get_adj_matrix(
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


def get_fault_labels(n_faults:int) -> Array[int, ..., 2]:
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


def get_fault_label_comb_bin(fault_labels:Array[int, ..., 2]) -> List[str]:
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


def get_lith_labels_bin(n_layers:int) -> List[str]:
    """Get unique binary lith labels list. For five layers this looks like:
    ['00001', '00010', '00100', '01000', '10000'].

    Args:
        n_layers (int): Number of layers.
    
    Returns:
        List[str]: Unique binary lith labels.
    """
    return [np.binary_repr(2**i).zfill(n_layers) for i in range(n_layers)]


def get_adj_matrix_labels(
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


def get_slicers(n_shift:int) -> tuple:
    """Get slice instances for given shift.
    
    Args:
        n_shift (int): Shift in number of voxels.
    
    Returns:
        tuple: Containing tuple of two slice instances for x, y and z
            direction.
    """
    slice_fit = slice(None, None)
    
    slicers = (
        (
            (slice(n_shift, None), slice_fit, slice_fit), 
            (slice(-n_shift), slice_fit, slice_fit)
        ),
        (
            (slice_fit, slice(n_shift, None), slice_fit), 
            (slice_fit, slice(-n_shift), slice_fit)
        ),
        (
            (slice_fit, slice_fit, slice(n_shift, None)), 
            (slice_fit, slice_fit, slice(-n_shift))
        )
    )
    return slicers


def get_reshapers(res:tuple, n_shift:int) -> tuple:
    """Get reshaping tuples based on given geomodel
    voxel resolution and voxel shift.
    
    Args:
        res (tuple): Geomodel resolution (nx, ny, nz)
        n_shift (int): Shift in number of voxels
    
    Returns:
        tuple: Contains three reshaper tuples for x, y and z 
            direction considering the given shift. 
    """
    reshaper = (
        (res[0] - 1, res[1], res[2]),
        (res[0], res[1] - 1, res[2]),
        (res[0], res[1], res[2] - 1)
    )
    return reshaper


def compute_topology(geo_model, n_shift:int=1) -> tuple:
    """Compute geomodel topology.
    
    Args:
        geo_model ([type]): Computed GemPy geomodel instance. 
        n_shift (int, optional): Number of voxels to shift. Defaults to 1.
    
    Returns:
        tuple: set of edges, centroid dictionary
    """
    lb, fb = lithblock_to_lb_fb(geo_model)
    n_faults = np.count_nonzero(geo_model.faults.df.isFault)
    n_liths = len(geo_model.surfaces.df) - n_faults
    res = geo_model.grid.regular_grid.resolution

    topology_labels = get_topology_labels(lb, fb, n_liths)
    shift_xyz_block = topology_shift(topology_labels, res, n_shift=n_shift)
    labels = bitstack_topology_labels(topology_labels).reshape(*res)
    edges = get_edges(shift_xyz_block, labels, res, n_shift)
    centroids = get_centroids(labels)

    return edges, centroids