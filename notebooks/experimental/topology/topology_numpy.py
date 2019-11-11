from itertools import combinations
from logging import debug
import numpy as np
from nptyping import Array


def lithblock_to_lb_fb(geo_model) -> tuple:
    """Serve flattened lithology id and fault blocks from solutions
    stored in given geomodel instance for topological analysis (i.e.
    rounded integer arrays).
    
    Args:
        geo_model (): GemPy Model instance with solutions.
    
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
        geo_model ():
        
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
        topo_block_f:np.ndarray, 
        labels_block:np.ndarray, 
        n_shift:int) -> list:
    """Evaluate the actual edge nodes from the labels block.
    
    Args:
        topo_block_f (np.ndarray): [description]
        labels_block (np.ndarray): [description]
        n_shift (n): [description]
    
    Returns:
        list: List of edge tuples (node_a, node_b)
    """
    edges = set()
    slice_fit = slice(n_shift - 1, -(n_shift))
    
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

    for i, topo_block_dir in enumerate(topo_block_f):
        for edge_sum in np.unique(topo_block_dir):
            if edge_sum == 0:
                continue

            shift_1 = labels_block[slicers[i][0]]
            shift_2 = labels_block[slicers[i][1]]
            filter_ = topo_block_dir == edge_sum
            edges.add(
                (np.unique(shift_1[filter_])[0], 
                 np.unique(shift_2[filter_])[0])
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