# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:topology]
#     language: python
#     name: conda-env-topology-py
# ---

import sys, os
# os.environ["THEANO_FLAGS"] = "device=cuda"  # use gpu
sys.path.append("../../../")
import gempy as gp
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import logging
from importlib import reload
from copy import copy

""
geo_model = gp.load_model(
    'Tutorial_ch1-9b_Fault_relations', 
    path= '../../data/gempy_models', 
    recompile=True
)
gp.compute_model(geo_model)
gp.plot.plot_section(geo_model, show_data=True)

""
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

"""
# NUMPY TOPOLOGY
"""

import topology_numpy as tpn

""
# get the lb, fb from block matrix
lb, fb = tpn.lithblock_to_lb_fb(geo_model)
# generate unique labels block from combination of lb and fb's
labels = tpn.get_labels_block(lb, fb).reshape(50, 50, 50)
# shift the blocks to get the "topology block"
shift_blocks = tpn.get_topo_block(labels, n_shift=1)
# remove non-edge sums
shift_blocks[np.isin(shift_blocks, 2 * np.unique(labels))] = 0
# extract edges from the shift block
edges = tpn.get_edges(shift_blocks, labels, 1)
# calculate centroids of geobodies
centroids = tpn.get_centroids(labels)
# ----------------------------------------------
# --- theano end
# ----------------------------------------------
# create a LOT to go from sums to lith id's
node_to_layer_LOT = tpn.get_lith_lot(labels, 2, 5)

###############################################################################
# # Topology dimension implementation

reload(tpn)

""
# %%timeit
edges, centroids = tpn.compute_topology(geo_model)

""


""


""


""
lb, fb = tpn.lithblock_to_lb_fb(geo_model)
n_faults = 2
n_liths = 5
n_shift = 1
res = geo_model.grid.regular_grid.resolution

topology_labels = tpn.get_topology_labels(lb, fb, n_liths)
shift_xyz_block = tpn.topology_shift(topology_labels, res, n_shift=n_shift)
labels = tpn.bitstack_topology_labels(topology_labels).reshape(*res)
edges = tpn.get_edges(shift_xyz_block, labels, res, n_shift)
centroids = tpn.get_centroids(labels)

""
np.argwhere(np.array(list(np.binary_repr(e1).zfill(9))).astype(int) == 1)

""
for e1, e2 in edges:
    print(np.binary_repr(e1).zfill(9), np.binary_repr(e2).zfill(9))

""
import gempy.assets.topology as tp

""
# %%timeit
G, c = tp.compute_topology(geo_model, filter_rogue=True, compute_areas=False)

""
fig = plt.figure(figsize=(10,10))
plt.imshow(geo_model.solutions.block_matrix[-1].reshape(50,50, 50)[:, 24, :].T, origin="lower", cmap="YlOrRd_r")

for e1, e2 in edges:
    p1, p2 = centroids.get(e1), centroids.get(e2)
    x, y = (p1[0], p2[0]), (p1[2], p2[2])
    plt.plot(x, y, c="k", linewidth=1)
    
# for node in np.unique(labels):
#     p = centroids.get(node)
#     plt.scatter(p[0], p[2], c="k", s=500)
#     plt.text(p[0], p[2], str(node_to_layer_LOT[node]), c="w")

""
shift_xyz_block

""


""


""


""


""
f_labels = tpn.get_fault_labels(n_faults)
# print(f_labels)

fault_labels_bin = tpn.get_fault_label_comb_bin(f_labels)
# print(bin_str_faults)

lith_labels_bin = tpn.get_lith_labels_bin(n_liths)
# print(bin_str_lith)

adj_matrix_labels = tpn.get_adj_matrix_labels(lith_labels_bin, fault_labels_bin)
adj_matrix_labels

adj_matrix = tpn.get_adj_matrix(edges, adj_matrix_labels, labels)

plt.imshow(adj_matrix)

""
tri_i, tri_j = np.tril_indices(adj_matrix.shape[0])
topology_vector = adj_matrix[tri_i, tri_j]
topology_vector

""
topology_vector.shape

""
import networkx as nx
G = nx.from_numpy_matrix(adj_matrix)

""
nx.draw_networkx(G)

""
for edge in G.edges:
    print(edge)

###############################################################################
# # Other

###############################################################################
# # labels, olabels = tpn.get_labels_block(geo_model)
# lb, fb = tpn.lithblock_to_lb_fb(geo_model)
# labels = tpn.get_labels_block(lb, fb).reshape(50, 50, 50)
# shift_blocks = tpn.get_topo_block(labels, n_shift=1)
#
# fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(15,8))
#
# for ax, img in zip(axs, shift_blocks):
#     ax.imshow(img[:, 24, :].T, origin="lower", cmap="hsv")

""
###############################################################################
# logger.setLevel(logging.CRITICAL)
#
# topo_block = copy(shift_blocks[-1])
# topo_block_filtered = copy(shift_blocks[-1])
# topo_block_filtered[np.isin(topo_block_filtered, 2*np.unique(labels))] = 0
#
# diff_ctrs = tpn.get_centroids(topo_block)
# diff_ctrs_filtered = tpn.get_centroids(topo_block_filtered)
#
# fig, axs = plt.subplots(figsize=(25,10), ncols=3)
#
# labels_ctrs = tpn.get_centroids(labels)
# ax = axs[0]
# for k, v in labels_ctrs.items():
#     ax.scatter(v[0], v[2], c="k", s=250)
#     ax.text(v[0]-0.4, v[2]-0.3, np.binary_repr(k).zfill(9), c="w")
# ax.set_title("unique labels block")
# ax.imshow(labels[:, 24, :].T, origin="lower", cmap="jet")
#
# ax = axs[1]
# for k, v in diff_ctrs.items():
#     if k == 0:
#         continue
#     ax.scatter(v[0], v[2], c="k", s=250)
#     ax.text(v[0]-0.4, v[2]-0.3, np.binary_repr(k).zfill(9), c="w")
# ax.set_title("sum of Z-shifted labels block")
# ax.imshow(topo_block[:, 24, :].T, origin="lower", cmap="jet")
#
# ax = axs[2]
# for k, v in diff_ctrs_filtered.items():
#     if k == 0:
#         continue
#     ax.scatter(v[0], v[2], c="k", s=250)
#     ax.text(v[0]-0.4, v[2]-0.3, np.binary_repr(k).zfill(9), c="w")
# ax.set_title("filtered sum block")
# ax.imshow(topo_block_filtered[:, 24, :].T, origin="lower", cmap="jet")

""
###############################################################################
# ulabel_LOT = tpn.get_node_label_sum_lot(ulabels)

""
###############################################################################
# # create layer id LOT
# layer_ids = {np.binary_repr(2**i).zfill(9):i for i in range(4, 9)}
# print(layer_ids)

###############################################################################
# # Speed comparison

logger.setLevel(logging.CRITICAL)

""
# %%timeit
lb, fb = tpn.lithblock_to_lb_fb(geo_model)
labels = tpn.get_labels_block(lb, fb).reshape(50, 50, 50)
shift_blocks = tpn.get_topo_block(labels, n_shift=1)
shift_blocks[np.isin(shift_blocks, 2*np.unique(labels))] = 0
edges = tpn.get_edges(shift_blocks, labels, 1)
centroids = tpn.get_centroids(labels)
node_to_layer_LOT = tpn.get_lith_lot(labels, 2)

""
# %%timeit
G, c = tp.compute_topology(geo_model, filter_rogue=True, compute_areas=False)

""
print(f"Speed-up: {375  / 50.7:.02f}x" )

""
fig = plt.figure(figsize=(10,10))
plt.imshow(geo_model.solutions.block_matrix[-1].reshape(50,50, 50)[:, 24, :].T, origin="lower", cmap="YlOrRd_r")

for edge in edges:
    e1, e2 = edge
    p1 = centroids.get(e1)
    p2 = centroids.get(e2)
    x = (p1[0], p2[0])
    y = (p1[2], p2[2])
    plt.plot(x, y, c="k", linewidth=1)
    
for node in ulabels:
    p = centroids.get(node)
    plt.scatter(p[0], p[2], c="k", s=500)
    plt.text(p[0], p[2], str(node_to_layer_LOT[node]), c="w")

""
from gempy.assets import topology as tp
G, c = tp.compute_topology(geo_model, filter_rogue=True)
gp.plot.plot_section(geo_model, 24)
gp.plot.plot_topology(geo_model, G, c)

""


""


""


""


""


""


""

