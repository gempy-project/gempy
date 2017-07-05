"""
@author: Alexander Schaaf
"""
from skimage.future import graph
from skimage.measure import label
from skimage.measure import regionprops
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Topology:
    """
    3D-Topology analysis class.
    """
    # TODO: Implement Topology plotting
    def __init__(self, block, fault_block):
        self.block = block.astype(int)
        self.fault_block = fault_block.astype(int)
        self.ublock = (self.block.max() + 1) * self.fault_block + self.block
        self.lithologies = np.unique(self.block)
        self.labels, self.n_labels = self._get_labels()
        self.labels_unique = np.unique(self.labels)
        self.G = graph.RAG(self.labels)
        self.centroids_2d, self.centroids_3d = self._get_centroids()
        self.lith_to_labels_lot = self._lithology_labels_lot()
        self.labels_to_lith_lot = self._labels_lithology_lot()

    def _get_labels(self, neighbors=4, background=999, return_num=True):
        """Get label block."""
        return label(self.ublock,
                     neighbors, return_num, background)

    def _get_centroids(self):
        """Get node centroids in 2d and 3d."""
        _rprops = regionprops(self.labels)
        centroids_2d = {}
        centroids_3d = {}
        for rp in _rprops:
            # centroid coordinates seem to be not x,y,z but rather x,z,y
            centroids_2d[rp.label] = [rp.centroid[0], rp.centroid[2]]
            centroids_3d[rp.label] = [rp.centroid[0], rp.centroid[2], rp.centroid[1]]
        return centroids_2d, centroids_3d

    def _lithology_labels_lot(self, verbose=0):
        """Create LOT from lithology id to label."""
        lot = {}
        for lith in self.lithologies:
            lot[str(lith)] = {}
        for l in self.labels_unique:
            _x, _y, _z = np.where(self.labels == l)
            lith_id = np.unique(self.block[_x, _y, _z])[0]
            if verbose:
                print("label:", l)
                print("lith:", lith_id)
            lot[str(lith_id)][str(l)] = {}
        return lot

    def _labels_lithology_lot(self, verbose=0):
        """Create LOT from label to lithology id."""
        lot = {}
        for l in self.labels_unique:
            _x, _y, _z = np.where(self.labels == l)
            lith_id = np.unique(self.block[_x, _y, _z])[0]
            if verbose:
                print(l)
            lot[l] = str(lith_id)
        if verbose:
            print(lot)
        return lot

    def draw_section(self, n, plane="y", labels=None):
        """Rudimentary plotting function for debugging"""
        if plane == "y":
            nx.draw_networkx(self.G,
                             pos=self.centroids_2d,
                             labels=labels)
            plt.imshow(self.block[:, n, :].T, origin="lower")
        else:
            pass

    def check_adjacency(self, n1, n2):
        """Check if n2 is adjacent/shares edge with n1."""
        if n2 in self.G.adj[n1]:
            return True
        else:
            return False
