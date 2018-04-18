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

import warnings
import numpy as np
try:
    from skimage.future import graph
    from skimage.measure import label
    from skimage.measure import regionprops
except ImportError:
    warnings.warn("skimage package is not installed, which is required for geomodel topology analysis.")


def topology_analyze(lith_block, fault_block, n_faults,
                     areas_bool=False,
                     return_block=False):
    """
    Analyses the block models adjacency topology. Every lithological entity is described by a uniquely labeled node
    (centroid) and its connections to other entities by edges.

    Args:
        lith_block (np.ndarray): Lithology block model
        fault_block (np.ndarray): Fault block model
        n_faults (int): Number of faults.


    Keyword Args:
        areas_bool (bool): If True computes adjacency areas for connected nodes in voxel number. Default False.
        return_block (bool): If True additionally returns the uniquely labeled block model as np.ndarray.

    Return:
        tuple:
            G: Region adjacency graph object (skimage.future.graph.rag.RAG) containing the adjacency topology graph
                (G.adj).
            centroids (dict): Centroid node coordinates as a dictionary with node id's (int) as keys and (x,y,z) coordinates
                as values.
            labels_unique (np.array): List of all labels used.
            lith_to_labels_lot (dict): Dictionary look-up-table to go from lithology id to node id.
            labels_to_lith_lot (dict): Dictionary look-up-table to go from node id to lithology id.
    """

    lith_block = lith_block.astype(int)
    lithologies = np.unique(lith_block.astype(int))
    # store a safe copy of the lith block for reference
    block_original = lith_block.astype(int)
    fault_block = fault_block.astype(int)
    # label the fault block for normalization (comparability of e.g. pynoddy and gempy models)
    fault_block = label(fault_block, neighbors=8, background=9999)

    if 0 in lith_block:
        # then this is a gempy model, numpy starts with 1
        lith_block[lith_block == 0] = int(np.max(lith_block) + 1)  # set the 0 to highest value + 1
        lith_block -= n_faults  # lower by n_faults to equal with pynoddy models
        # so the block starts at 1 and goes continuously to max

    # make sure that faults seperate lithologies in labeling, YUGE clever algorithm of the narcisist
    ublock = (lith_block.max() + 1) * fault_block + lith_block

    # label the block for unique regions
    labels_block, labels_n = label(ublock, neighbors=8, return_num=True, background=9999)
    if 0 in np.unique(labels_block):
        labels_block += 1

    labels_unique = np.unique(labels_block)
    # create adjacency graph from labeled block
    G = graph.RAG(labels_block)
    # get the centroids from the labeled block
    centroids = get_centroids(labels_block)
    # create look-up-tables in both directions
    lith_to_labels_lot = lithology_labels_lot(lithologies, labels_block, block_original, labels_unique)
    labels_to_lith_lot = labels_lithology_lot(labels_unique, labels_block, block_original)
    # classify the edges (stratigraphic, across-fault)
    classify_edges(G, centroids, block_original, fault_block)
    # compute the adjacency areas for each edge
    if areas_bool:
        # TODO: 2d option (if slice only), right now it only works for 3d
        compute_areas(G, labels_block)

    if not return_block:
        return G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot
    else:
        return G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot, labels_block


def compute_areas(G, labels_block):
    """
    Computes adjacency areas and stores them in G.adj[n1][n2]["area"].

    Args:
        G (skimage.future.graph.rag.RAG): Topology graph object.
        labels_block (np.ndarray): Uniquely labeled block model.
    """
    # TODO: AS: make area computation function more modular to support additional functionality (e.g. fault throw) FABIAN?
    # get all bool arrays for each label, for filtering
    labels_bools = np.array([(labels_block == l).astype("bool") for l in np.unique(labels_block)])
    for n1, n2 in G.edges_iter():  # iterate over every edge in the graph
        # modify labels block to avoid non-unique values when doing later comparison
        b = np.square(labels_block * (labels_bools[n1 - 1] + labels_bools[n2 - 1]))
        # translate block by 1 voxel in each dimension and substract, take absolute of results; this gets you
        # the boundary voxels of the regions of n1 and n2, including the shared one
        d = np.absolute(b[0:-1, 0:-1, 0:-1] - b[1:, 1:, 1:])
        # filter out the shared boundary
        d = (d == np.absolute(n1 ** 2 - n2 ** 2))
        # count the shared boundary, which is the shared area voxel count of n1 and n2
        area = np.count_nonzero(d)
        # store in adjacency dict of graph for access
        G.adj[n1][n2]["area"] = area
        G.adj[n2][n1]["area"] = area


def classify_edges(G, centroids, lith_block, fault_block):
    """
    Classifies edges by type into stratigraphic or fault in "G.adj". Accessible via G.adj[node1][node2]["edge_type"]

    Args:
        G (skimage.future.graph.rag.RAG): Topology graph object.
        centroids (dict): Centroid dictionary {node id (int): tuple(x,y,z)}
        lith_block (np.ndarray): Shaped lithology block model.
        fault_block (np.ndarray): Shaped fault block model.

    Returns:

    """
    # loop over every node in adjacency dictionary
    for n1, n2 in G.edges_iter():
        # get centroid coordinates
        if n2 == 0 or n1 == 0:
            continue
        n1_c = centroids[n1]
        n2_c = centroids[n2]
        # get fault block values at node positions
        if len(np.shape(lith_block)) == 3:
            n1_fb_val = fault_block[int(n1_c[0]), int(n1_c[1]), int(n1_c[2])]
            n2_fb_val = fault_block[int(n2_c[0]), int(n2_c[1]), int(n2_c[2])]
        else:
            n1_fb_val = fault_block[int(n1_c[0]), int(n1_c[1])]
            n2_fb_val = fault_block[int(n2_c[0]), int(n2_c[1])]

        if n1_fb_val == n2_fb_val:
            # both are in the same fault entity
            G.adj[n1][n2]["edge_type"] = "stratigraphic"
        else:
            G.adj[n1][n2]["edge_type"] = "fault"


def get_centroids(label_block):
    """Get node centroids in 2d and 3d as {node id (int): tuple(x,y,z)}."""
    _rprops = regionprops(label_block)
    centroids = {}
    for rp in _rprops:
            centroids[rp.label] = rp.centroid
    return centroids


def lithology_labels_lot(lithologies, labels, block_original, labels_unique, verbose=0):
    """Create LOT from lithology id to label."""
    lot = {}
    for lith in lithologies:
        lot[str(lith)] = {}
    for l in labels_unique:
        if len(np.where(labels == l)) == 3:
            _x, _y, _z = np.where(labels == l)
            lith_id = np.unique(block_original[_x, _y, _z])[0]
        else:
            _x, _z = np.where(labels == l)
            lith_id = np.unique(block_original[_x, _z])[0]

        if verbose:
            print("label:", l)
            print("lith:", lith_id)
        lot[str(lith_id)][str(l)] = {}
    return lot


def labels_lithology_lot(labels_unique, labels, block_original, verbose=0):
    """Create LOT from label to lithology id."""
    lot = {}
    for l in labels_unique:
        if len(np.where(labels == l)) == 3:
            _x, _y, _z = np.where(labels == l)
            lith_id = np.unique(block_original[_x, _y, _z])[0]
        else:
            _x, _z = np.where(labels == l)
            lith_id = np.unique(block_original[_x, _z])[0]
        if verbose:
            print(l)
        lot[l] = str(lith_id)
    if verbose:
        print(lot)
    return lot


def check_adjacency(G, n1, n2):
    """Check if n2 is adjacent/shares edge with n1."""
    if n2 in G.adj[n1]:
        return True
    else:
        return False


def compute_adj_shape(n1, n2, labels_block):
    """Compute shape of adjacency area: number of voxels in X, Y and Z (column height). (Fabian)"""
    labels_bools = np.array([(labels_block == l).astype("bool") for l in np.unique(labels_block)])
    b = np.square(labels_block * (labels_bools[n1 - 1] + labels_bools[n2 - 1]))
    d = np.absolute(b[0:-1, 0:-1, 0:-1] - b[1:, 1:, 1:])
    d = (d == np.absolute(n1 ** 2 - n2 ** 2))
    if np.count_nonzero(d) != 0:
        nz_xyz = np.ndarray.nonzero(d)
        x_len = max(nz_xyz[0])-min(nz_xyz[0])+1
        y_len = max(nz_xyz[1])-min(nz_xyz[1])+1
        z_len = max(nz_xyz[2])-min(nz_xyz[2])+1
    else:
        print('Unable to calculate an adjacency area.')  # Implementing this due to problem, where empty lists returned
        nz_xyz = 0
        x_len = 0
        y_len = 0
        z_len = 0
    # z_len (maximal column height) to be used to calculate fault throw. (Fabian)
    return x_len, y_len, z_len, nz_xyz, d


def compare_graphs(G1, G2):
    """
    Compares two graph objects using the Jaccard-Index.

    Args:
        G1 (skimage.future.graph.rag.RAG): Topology graph object.
        G2 (skimage.future.graph.rag.RAG): Another topology graph object.

    Returns:
        (float): Jaccard-Index
    """
    intersection = 0
    union = G1.number_of_edges()

    for edge in G1.edges_iter():
        if G2.has_edge(edge[0], edge[1]):
            intersection += 1
        else:
            union += 1

    return intersection / union