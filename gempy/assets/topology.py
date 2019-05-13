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
    warnings.warn("skimage package is not installed, which is required for geomodel topology \ "
                  "analysis.")
from gempy.utils.analysis import get_centroids, get_unique_regions
from networkx import convert_node_labels_to_integers, relabel_nodes

def compute_topology(geo_model,
                     cell_number=None,
                     direction=None,
                     compute_areas=False,
                     return_label_block=False,
                     return_rprops=False,
                     filter_rogue=False,
                     noddy=False,
                     filter_threshold_area=10,
                     neighbors=8,
                     enhanced_labels=True):
    """
    Computes model topology and returns graph, centroids and look-up-tables.

    Args:
        geo_model (gempy.model.Model): Container class of all objects that constitute a GemPy model.
        cell_number (int): Cell number for 2-D slice topology analysis. Default None.
        direction (str): "x", "y" or "z" specifying the slice direction for 2-D topology analysis.
            Default None.
        compute_areas (bool): If True computes adjacency areas for connected nodes in voxel number.
            Default False.
        return_label_block (bool): If True additionally returns the uniquely labeled block model as
            np.ndarray. Default False.
        return_rprops (bool, optional): If True additionally returns region properties of the unique regions
            (see skimage.measure.regionprops).
        filter_rogue (bool, optional): If True filters nodes with region areas below threshold (default: 1) from
            topology graph.
        filter_threshold_area (int, optional): Specifies the threshold area value (number of pixels) for filtering
            small regions that may thow off topology analysis.
        neighbors (int, optional): Specifies the neighbor voxel connectivity taken into account for the topology
            analysis. Must be either 4 or 8 (default: 8).
        enhanced_labels (bool, optional): If True enhances the topology graph node labeling with fb_id, lb_id and instance
            id (e.g. 1_6_b), if False reverses to just numeric labeling (default: True).

    Returns:
        tuple:
            G: Region adjacency graph object (skimage.future.graph.rag.RAG) containing the adjacency
                topology graph (G.adj).
            centroids (dict): Centroid node coordinates as a dictionary with node id's (int) as keys
                and (x,y,z) coordinates as values. {node id (int): tuple(x,y,z)}
            labels_unique (np.array): List of all labels used.
            lith_to_labels_lot (dict): Dictionary look-up-table to go from lithology id to node id.
            labels_to_lith_lot (dict): Dictionary look-up-table to go from node id to lithology id.
    """
    assert hasattr(geo_model, "solutions"), "geo_model object must contain .solutions attribute."
    isfault = np.where(geo_model.faults.df.isFault == True)[0]
    fault_arrays = np.round(np.atleast_2d(geo_model.solutions.block_matrix[isfault])[:, 0, :])
    iszero = np.where(fault_arrays==0)
    fault_arrays[iszero] += 1
    fault_arrays = fault_arrays * ~(fault_arrays == np.min(np.round(fault_arrays), axis=1).reshape(-1, 1))

    fault_block = fault_arrays.sum(axis=0)
    res = geo_model.grid.regular_grid.resolution

    if cell_number is None or direction is None:  # topology of entire block
        lb = geo_model.solutions.lith_block.reshape(*res)
        fb = fault_block.reshape(*res)
    elif direction == "x":
        lb = geo_model.solutions.lith_block.reshape(*res)[cell_number, :, :]
        fb = fault_block.reshape(*res)[cell_number, :, :]
    elif direction == "y":
        lb = geo_model.solutions.lith_block.reshape(*res)[:, cell_number, :]
        fb = fault_block.reshape(*res)[:, cell_number, :]
    elif direction == "z":
        lb = geo_model.solutions.lith_block.reshape(*res)[:, :, cell_number]
        fb = fault_block.reshape(*res)[:, :, cell_number]

    return _topology_analyze(lb, fb, geo_model.faults.n_faults,
                             areas_bool=compute_areas,
                             return_block=return_label_block,
                             return_rprops=return_rprops,
                             filter_rogue=filter_rogue,
                             noddy=noddy,
                             filter_threshold_area=filter_threshold_area,
                             neighbors=neighbors,
                             enhanced_labels=enhanced_labels)

def _topology_analyze(lith_block,
                      fault_block,
                      n_faults,
                      areas_bool=False,
                      return_block=False,
                      return_rprops=False,
                      filter_rogue=False,
                      noddy=False,
                      filter_threshold_area=10,
                      neighbors=8,
                      enhanced_labels=True):
    """
    Analyses the block models adjacency topology. Every lithological entity is described by a uniquely labeled node
    (centroid) and its connections to other entities by edges.

    Args:
        lith_block (np.ndarray): Lithology block model
        fault_block (np.ndarray): Fault block model
        n_faults (int): Number of df.


    Keyword Args:
        areas_bool (bool): If True computes adjacency areas for connected nodes in voxel number. Default False.
        return_block (bool): If True additionally returns the uniquely labeled block model as np.ndarray.
        n_faults (int): Number of faults.
        areas_bool (bool, optional): If True computes adjacency areas for connected nodes in voxel number.
            Default False.
        return_block (bool, optional): If True additionally returns the uniquely labeled block model as np.ndarray.
        return_rprops (bool, optional): If True additionally returns region properties of the unique regions
            (see skimage.measure.regionprops).
        filter_rogue (bool, optional): If True filters nodes with region areas below threshold (default: 1) from
            topology graph.
        filter_threshold_area (int, optional): Specifies the threshold area value (number of pixels) for filtering
            small regions that may thow off topology analysis.
        neighbors (int, optional): Specifies the neighbor voxel connectivity taken into account for the topology
            analysis. Must be either 4 or 8 (default: 8).
        enhanced_labels (bool, optional): If True enhances the topology graph node labeling with fb_id, lb_id and instance
            id (e.g. 1_6_b), if False reverses to just numeric labeling (default: True).

    Return:
        tuple:
            G: Region adjacency graph object (skimage.future.graph.rag.RAG) containing the adjacency topology graph
                (G.adj).
            centroids (dict): Centroid node coordinates as a dictionary with node id's (int) as keys and (x,y,z)
                coordinates as values.
            labels_unique (np.array): List of all labels used.
            lith_to_labels_lot (dict): Dictionary look-up-table to go from lithology id to node id.
            labels_to_lith_lot (dict): Dictionary look-up-table to go from node id to lithology id.
    """
    block_original = lith_block.astype(int)  # do we really still need this?

    # generate unique labels block by combining lith and fault blocks
    labels_block = get_unique_regions(lith_block, fault_block, n_faults, neighbors=neighbors, noddy=noddy)

    # create adjacency graph from labeled block
    G = graph.RAG(labels_block)
    rprops = regionprops(labels_block)  # get properties of uniquely labeles regions
    centroids = get_centroids(rprops)  # unique region centroids coordinates

    # classify edges (stratigraphic, fault)
    classify_edges(G, centroids, fault_block)

    # filter rogue pixel nodes from graph if wanted
    if filter_rogue:
        G, centroids = filter_region_areas(G, centroids, rprops, area_threshold=filter_threshold_area)
        G = convert_node_labels_to_integers(G, first_label=1)
        centroids = {i+1: coords for i, coords in enumerate(centroids.values())}

    # enhanced node labeling containing fault block and lith id
    if enhanced_labels:
        labels = enhanced_labeling(G, rprops, lith_block, fault_block)
        G = relabel_nodes(G, labels)  # relabel graph
        centroids = get_centroids(rprops)  # redo centroids for updated labeling

    # compute the adjacency areas for each edge
    if areas_bool:
       compute_areas(G, labels_block)   # TODO: 2d option (if slice only), right now it only works for 3d

    # prep returned objects
    topo = [G, centroids]

    # keep option for old labeling for legacy support
    if not enhanced_labels:  # create look-up-tables in both directions
        topo.append(lithology_labels_lot(labels_block, block_original))
        topo.append(labels_lithology_lot(labels_block, block_original))
    if return_block:  # append the labeled block to return
        topo.append(labels_block)
    if return_rprops:  # append rprops to return
        topo.append(rprops)

    return tuple(topo)


def enhanced_labeling(G, rprops, lith_block, fault_block):
    """Relabel the given graph's nodes with scheme "{fault_id}_{lith_id}_{instance}"

    Args:
        G (skimage.future.graph.rag.RAG): Region adjacency graph object containing the adjacency topology graph.
        rprops (list): List of regionprops object for each unique region of the model.
        lith_block (np.ndarray): Lithology block model (3d shape).
        fault_block (np.ndarray): Fault block model (3d shape).

    Returns:
        (dict): Mapping of old to new labels to be used with networkx.relabel_nodes(G, Labels_dict)
    """
    labels = []

    for n, rp in zip(G.nodes(), rprops):
        _c = tuple(np.array(rp.centroid).astype(int))  # centroid location
        lid = lith_block[_c].astype(int)  # inquire lb
        fid = fault_block[_c].astype(int)  # and fb id at centroid loc
        label = str(fid) + "_" + str(lid)  # fuse label
        labels.append(label)

    # post-process for multiple lith region instances in single fault block
    for label in labels:
        ids = np.where(np.array(labels) == label)[0]
        if len(ids) > 1:
            for i, id_ in enumerate(ids):
                if i > 0:
                    labels[id_] += ("_" + chr(ord("a") + i))
    # fuse into dict mapping new labels to old
    labels = {n: l for n, l in zip(G.nodes(), labels)}

    # fix rprop labels
    for rprop, l in zip(rprops, labels.values()):
        rprop.label = l

    return labels


def filter_region_areas(graph, centroids, rprops, area_threshold=10):
    """
    Filters nodes with region areas with an area below the given threshold (default: 10) from given graph.
    Useful for filtering rogue pixels that throw off topology graph comparisons.

    Args:
        graph (skimage.future.graph.rag.RAG): Topology graph object to be filtered.
        rprops (list): Region property objects (skimage.measure._regionprops._RegionProperties) for all nodes within given topology graph.
    Keyword Args:
        area_threshold (int): Region areas with number of pixels below or equal of this value will be removed. Default 10

    Returns:
        None (in-place removal)
    """
    from copy import deepcopy

    if len(graph.nodes()) != len(rprops):   # failsafe if the function is run with mismatching rprops
        return graph, centroids

    ngraph = deepcopy(graph)
    ncentroids = deepcopy(centroids)

    for node, rprop in zip(graph.nodes(), rprops):
        if rprop.area <= area_threshold:  # if region area is below given threshold area
            ngraph.remove_node(node)  # then pop the node and all its edges
            ncentroids.pop(node)  # pop centroid
    return ngraph, ncentroids


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
    for n1, n2 in G.edges():  # iterate over every edge in the graph
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


def classify_edges(G, centroids, fault_block):
    """
    Classifies edges by type into stratigraphic or fault in "G.adj". Accessible via G.adj[node1][node2]["edge_type"]

    Args:
        G (skimage.future.graph.rag.RAG): Topology graph object.
        centroids (dict): Centroid dictionary {node id (int): tuple(x,y,z)}
        fault_block (np.ndarray): Shaped fault block model.

    Returns:

    """
    # loop over every node in adjacency dictionary
    for n1, n2 in G.edges():
        # get centroid coordinates
        if n2 == 0 or n1 == 0:
            continue
        n1_c = centroids[n1]
        n2_c = centroids[n2]
        # get fault block values at node positions
        if len(n1_c) == 3:
            n1_fb_val = fault_block[int(n1_c[0]), int(n1_c[1]), int(n1_c[2])]
            n2_fb_val = fault_block[int(n2_c[0]), int(n2_c[1]), int(n2_c[2])]
        else:
            # if 2d slice, but in 3d array (e.g. shape (50,1,50))
            if len(fault_block.shape) == 3:
                axis = np.argmin(fault_block.shape)
                n1_fb_val = np.squeeze(fault_block, axis)[int(n1_c[0]), int(n1_c[1])]
                n2_fb_val = np.squeeze(fault_block, axis)[int(n2_c[0]), int(n2_c[1])]
            # if actual 2d slice of shape e.g. (50,50)
            else:
                n1_fb_val = fault_block[int(n1_c[0]), int(n1_c[1])]
                n2_fb_val = fault_block[int(n2_c[0]), int(n2_c[1])]

        if n1_fb_val == n2_fb_val:
            # both are in the same fault entity
            G.adj[n1][n2]["edge_type"] = "stratigraphic"
        else:
            G.adj[n1][n2]["edge_type"] = "fault"


def lithology_labels_lot(labels, block_original, verbose=0):
    """Create LOT from lithology id to label."""
    lot = {}
    for lith in np.unique(block_original):
        lot[str(lith)] = {}
    for l in np.unique(labels):
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


def labels_lithology_lot(labels, lb, verbose=0):
    """Create LOT from label to lithology id."""
    lot = {}
    for l in np.unique(labels):
        if len(np.where(labels == l)) == 3:
            _x, _y, _z = np.where(labels == l)
            lith_id = np.unique(lb[_x, _y, _z])[0]
        else:
            _x, _z = np.where(labels == l)
            lith_id = np.unique(lb[_x, _z])[0]
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
    """Jaccard index for numeric topology graph comparisons.

    Args:
        G1 (skimage.future.graph.rag.RAG): Topology graph object.
        G2 (skimage.future.graph.rag.RAG): Another topology graph object.

    Returns:
        (float): Jaccard-Index (1 if identical graph, 0 if entirely dissimilar)

    Source:
        http://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
    """
    intersection_cardinality = len(set.intersection(*[set(G1.edges()), set(G2.edges())]))
    union_cardinality = len(set.union(*[set(G1.edges()), set(G2.edges())]))
    return intersection_cardinality / float(union_cardinality)


def convert_centroids_2d(centroids, direction="y"):
    """
    Args:
        centroids (dict): Dictionary containing graph node numbers as keys and centroid coordinates (x,z,y) as values
        (as given by skimage.measure.regionprops function).
        direction (str, optional): Specifies for which direction to flatten the coordinates from 3D to 2D (default: "y").

    Returns:
        (dict): Dictionary containing graph node numbers as keys an centroid coordinates in 2D depending on direction
    """
    centroids_2d = {}
    for key, val in centroids.items():
        if direction == "y":
            centroids_2d[key] = (val[0], val[2])
        elif direction == "x":
            centroids_2d[key] = (val[1], val[2])
        elif direction == "z":
            centroids_2d[key] = (val[0], val[1])
    return centroids_2d
