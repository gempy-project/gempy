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
from skimage.future import graph
from skimage.measure import label
from skimage.measure import regionprops
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# TODO: Across-fault edge identification
# TODO: Across-unconformity edge identification


class Topology:
    """
    3D-Topology analysis class.
    """
    # TODO: Implement Topology plotting
    def __init__(self, block, fault_block, section=None):
        """

        :param block:
        :param fault_block:
        :param section: y-section (int)
        """
        self.section = section
        if self.section is None:
            self.block = np.copy(block.astype(int))
            self.fault_block = fault_block.astype(int)
        else:
            self.block = np.copy(block.astype(int))[:,self.section,:]
            self.fault_block = fault_block.astype(int)[:,self.section,:]

        self.ublock = (self.block.max() + 1) * self.fault_block + self.block

        self.lithologies = np.unique(self.block)
        self.labels, self.n_labels = self._get_labels()
        self.labels_unique = np.unique(self.labels)
        self.G = graph.RAG(self.labels)
        self.centroids = self._get_centroids()
        self.lith_to_labels_lot = self._lithology_labels_lot()
        self.labels_to_lith_lot = self._labels_lithology_lot()

        self._classify_edges()

    def _get_labels(self, neighbors=8, background=999, return_num=True):
        """Get label block."""
        return label(self.ublock, neighbors, return_num, background)

    def _classify_edges(self):
        # loop over every node in adjacency dictionary
        for n1 in self.G.adj:
            # loop over every node that it is connected with
            for n2 in self.G.adj[n1]:
                # get centroid coordinates
                if n2 == 0 or n1 == 0:
                    continue
                n1_c = self.centroids[n1]
                n2_c = self.centroids[n2]
                # get fault block values at node positions
                if self.section is None:
                    n1_fb_val = self.fault_block[int(n1_c[0]), int(n1_c[1]), int(n1_c[2])]
                    n2_fb_val = self.fault_block[int(n2_c[0]), int(n2_c[1]), int(n2_c[2])]
                else:
                    n1_fb_val = self.fault_block[int(n1_c[0]), int(n1_c[2])]
                    n2_fb_val = self.fault_block[int(n2_c[0]), int(n2_c[2])]

                if n1_fb_val == n2_fb_val:
                    # both are in the same fault entity
                    self.G.adj[n1][n2] = {"edge_type": "stratigraphic"}
                else:
                    self.G.adj[n1][n2] = {"edge_type": "fault"}

    def _get_centroids(self):
        """Get node centroids in 2d and 3d."""
        _rprops = regionprops(self.labels)
        centroids = {}
        for rp in _rprops:
            if self.section is None:
                centroids[rp.label] = [rp.centroid[0], rp.centroid[1], rp.centroid[2]]
            else:
                centroids[rp.label] = [rp.centroid[0], self.section, rp.centroid[1]]
        return centroids

    def _lithology_labels_lot(self, verbose=0):
        """Create LOT from lithology id to label."""
        lot = {}
        for lith in self.lithologies:
            lot[str(lith)] = {}
        for l in self.labels_unique:
            if self.section is None:
                _x, _y, _z = np.where(self.labels == l)
                lith_id = np.unique(self.block[_x, _y, _z])[0]
            else:
                _x, _z = np.where(self.labels == l)
                lith_id = np.unique(self.block[_x, _z])[0]

            if verbose:
                print("label:", l)
                print("lith:", lith_id)
            lot[str(lith_id)][str(l)] = {}
        return lot

    def _labels_lithology_lot(self, verbose=0):
        """Create LOT from label to lithology id."""
        lot = {}
        for l in self.labels_unique:
            if self.section is None:
                _x, _y, _z = np.where(self.labels == l)
                lith_id = np.unique(self.block[_x, _y, _z])[0]
            else:
                _x, _z = np.where(self.labels == l)
                lith_id = np.unique(self.block[_x, _z])[0]
            if verbose:
                print(l)
            lot[l] = str(lith_id)
        if verbose:
            print(lot)
        return lot

    def check_adjacency(self, n1, n2):
        """Check if n2 is adjacent/shares edge with n1."""
        if n2 in self.G.adj[n1]:
            return True
        else:
            return False
