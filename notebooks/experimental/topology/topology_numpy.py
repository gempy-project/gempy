from itertools import combinations
import logging
import numpy as np


def lithblock_to_lb_fb(geo_model) -> tuple:
    """Serve flattened lithology id and fault blocks from solutions
    stored in given geomodel instance for topological analysis (i.e.
    rounded integer arrays).
    
    Args:
        geo_model (gp.core.model.Model): GemPy Model instance with 
            solutions.
    
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
    
    
def get_labels_block(geo_model) -> tuple:
    """Uniquely label geobodies in geomodel for topology analysis.
    
    Args:
        geo_model ():
    
    Returns:
        (tuple) [0] uniquely labeled geobody block (nx, ny, nz)
                [1] olabels: list of original labels before summation
                [2] ulabels: list of labels after summation
    """
    # get data
    lb, fb = lithblock_to_lb_fb(geo_model)
    fault_ids = get_fault_ids(geo_model)
    logging.debug(f"fault id's: {fault_ids}")
    
    # label lithologies
    lb_labels = lb # - lb.min()
    logging.debug(f"lb labels: {np.unique(lb_labels)}")
    
    # label faults
    fb_labels = fb - fb.min(axis=1)[:, None]
    logging.debug(f"fb_labels shp: {fb_labels.shape}")
    fb_labels = fb_labels * fault_ids[:, None]
    # fb_labels += len(np.unique(lb_labels)) - 1
    
    logging.debug(f"fb labels: {np.unique(fb_labels)}")
    for i, fblock in enumerate(fb_labels):
        logging.debug(f"fb {i}: {np.unique(fblock)}")
        
    # combine labels
    # concatenate lith labels with fault labels array into (n, ...)
    labels = 2**np.concatenate((lb_labels[None, :], fb_labels), axis=0)
    
    # store original labels for lith and all faults for late ruse
    
    olabels = [np.unique(label) for label in labels]

    for block in labels:
        logging.debug(f"olabels (pre-sum): {np.unique(block)}")
        
    # sum along axis 0 to combine into unique geobody labels
    # for each fault block
    labels = labels.sum(axis=0)
    
    logging.debug(f"ulabels: {np.unique(labels)}")
#     logging.debug(f"labels binary: {[np.binary_repr(label) for label in np.unique(labels)]}")
    logging.debug(f"n labels: {len(np.unique(labels))}")
    
    return labels.reshape(*geo_model.grid.regular_grid.resolution), olabels


def get_topo_block(labels:np.ndarray, n_shift:int=1) -> np.ndarray:
    """Create topology block by shifting along x, y, z axes and
    summing up.
    
    Args:
        labels (np.ndarray):
        n_shift (int):
    
    Returns:
        (np.ndarray) (3, nx, ny, nz)
    """
    sum_x = np.abs(labels[n_shift:, :, :] + labels[:-n_shift, :, :])
    logging.debug(f"sum_x shp: {sum_x.shape}")
    sum_y = np.abs(labels[:, n_shift:, :] + labels[:, :-n_shift, :])
    logging.debug(f"sum_y shp: {sum_y.shape}")
    sum_z = np.abs(labels[:, :, n_shift:] + labels[:, :, :-n_shift])
    logging.debug(f"sum_z shp: {sum_z.shape}")
    
    slx, sly, slz = (slice(n_shift // 2, -n_shift//2) for i in range(3))
    logging.debug(f"slx {slx}; sly {sly}; slz {slz}")
    
    sums_xyz = np.concatenate(
        (
         sum_x[None, :, sly, slz], 
         sum_y[None, slx, :, slz], 
         sum_z[None, slx, sly, :]
        ), axis=0
    )
    logging.debug(f"{sums_xyz.shape}")
    return sums_xyz


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
    logging.debug(f"possible node combinations: {possible_edges}")
    ulabel_LOT = {sum(comb):comb for comb in possible_edges}
    for k, v in ulabel_LOT.items():
        logging.debug(f"{k} = {v[0]} + {v[1]}")
    return ulabel_LOT


# def get_edges(topo_block:np.ndarray, labels_LOT:dict) -> list:
#     """Look up edges from topology block (shifted block) in the
#     labels-sum LOT and return list of edges.
    
#     Args:
#         topo_block (np.ndarray):
#         labels_LOT (dict):
    
#     Return:
#         (list):
#     """
#     edges = []
#     for blob in np.unique(topo_block):
#         edge = labels_LOT.get(blob)
#         if edge:
#             logging.debug(f"Valid edge: {edge}")
#             edges.append(edge)
#         else:
#             logging.debug(f"Invalid node: {blob}")
#     return edges

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


def get_centroids(labels:np.ndarray, ulabels:np.array) -> dict:
    """Get geobody node centroids in array coordinates.
    
    Args:
        labels (np.ndarray): [description]
        ulabels (np.array): [description]
    
    Returns:
        dict: [description]
    """
    node_locs = []
    for node in ulabels:
        node_pos = np.argwhere(labels==node)
        node_locs.append(node_pos.mean(axis=0))
    centroids = {n:loc for n, loc in zip(ulabels, node_locs)}
    for k, v in centroids.items():
        logging.debug(f"{k}: {v}")
    return centroids