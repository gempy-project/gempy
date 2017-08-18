"""
@author: Alexander Schaaf, Miguel de la Varga
"""
import pymc
import theano
import numpy as np
import networkx as nx


class Posterior:
    """Posterior database analysis for GemPy-pymc2 hdf5 databases."""

    def __init__(self, dbname, topology=False, verbose=False):
        self.verbose = verbose
        # load db
        self.db = pymc.database.hdf5.load(dbname)
        # get trace names
        self.trace_names = self.db.trace_names[0]
        # get gempy block models
        try:
            self.sols = self.db.gempy_model.gettrace()
        except AttributeError:
            print("No GemPy block models tallied.")
            self.sols = None

        if topology:
            # load graphs
            topo_trace = self.db.gempy_topo.gettrace()
            self.graphs = topo_trace[:, 0]
            # load centroids
            self.centroids = topo_trace[:, 1]
            del topo_trace

        # load input data
        self.input_data = self.db.input_data.gettrace()

    def change_input_data(self, interp_data, i):
        """Changes input data in interp_data to posterior input data at iteration i."""

        # replace interface data
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        interp_data.geo_data_res.foliations[["G_x", "G_y", "G_z", "X", "Y", "Z"]] = self.input_data[i][1]
        # do all the ugly updating stuff
        # interp_data.interpolator.tg.final_potential_field_at_formations = theano.shared(np.zeros(
        #     interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        # interp_data.interpolator.tg.final_potential_field_at_faults = theano.shared(np.zeros(
        #     interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        # interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")

    def plot_topology_graph(self, i):
        # get centroid values into list
        centroid_values = [triplet for triplet in self.centroids[i].values()]
        # unzip them into seperate lists of x,y,z coordinates
        centroids_x, centroids_y, centroids_z = list(zip(*centroid_values))
        # create new 2d pos dict for plot
        pos_dict = {}
        for j in range(len(centroids_x)):  # TODO: Change this directly to use zip?
            pos_dict[j + 1] = [centroids_x[j], centroids_y[j]]
        # draw
        nx.draw_networkx(self.graphs[i], pos=pos_dict)


