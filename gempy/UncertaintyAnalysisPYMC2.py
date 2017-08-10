"""
@author: Alexander Schaaf, Miguel de la Varga
"""
import pymc
import theano
import numpy as np
import networkx as nx


class Posterior:
    """Posterior database analysis for pymc2."""

    def __init__(self, db_name, topology=True, verbose=False):
        # load db
        self.db = pymc.database.hdf5.load(db_name)
        # get trace names
        self.trace_names = self.db.trace_names[0]
        # get gempy block models
        try:
            self.sols = self.db.gempy_model.gettrace()
        except AttributeError:
            print("No GemPy models tallied.")
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
        # [self.dips_position_all, self.dip_angles_all, self.azimuth_all,
        # self.polarity_all, self.ref_layer_points_all, self.rest_layer_points_all]

    def change_input_data(self, interp_data, i):
        """Changes input data in interp_data to posterior input data at iteration i."""
        # TODO: accomodate for new input_data format
        interp_data.geo_data_res = self.input_data[i]
        interp_data.interpolator.tg.final_potential_field_at_formations = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.interpolator.tg.final_potential_field_at_faults = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.update_interpolator()

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


