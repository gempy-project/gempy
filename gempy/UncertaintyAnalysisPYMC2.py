"""
@author: Alexander Schaaf, Miguel de la Varga
"""
import pymc
import theano
import numpy as np
import networkx as nx
import gempy as gp
import matplotlib.pyplot as plt
import seaborn as sns

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
            print("No GemPy model trace tallied.")
            self.sols = None

        try:
            # load graphs
            topo_trace = self.db.gempy_topo.gettrace()
            self.topo_graphs = topo_trace[:, 0]
            # load centroids
            self.topo_centroids = topo_trace[:, 1]
            self.topo_labels_unique = topo_trace[:, 2]
            self.topo_lith_to_labels_lot = topo_trace[:, 3]
            self.topo_labels_to_lith_lot = topo_trace[:, 4]
            del topo_trace
        except AttributeError:
            print("No Topology trace tallied.")

        # load input data
        self.input_data = self.db.input_data.gettrace()

    def change_input_data(self, interp_data, i):
        """Changes input data in interp_data to posterior input data at iteration i."""

        # replace interface data
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        interp_data.geo_data_res.foliations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = self.input_data[i][1]
        # do all the ugly updating stuff
        interp_data.interpolator.tg.final_potential_field_at_formations = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.interpolator.tg.final_potential_field_at_faults = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")

    def plot_topology_graph(self, i):
        # get centroid values into list
        centroid_values = [triplet for triplet in self.topo_centroids[i].values()]
        # unzip them into seperate lists of x,y,z coordinates
        centroids_x, centroids_y, centroids_z = list(zip(*centroid_values))
        # create new 2d pos dict for plot
        pos_dict = {}
        for j in range(len(centroids_x)):  # TODO: Change this directly to use zip?
            pos_dict[j + 1] = [centroids_x[j], centroids_y[j]]
        # draw
        nx.draw_networkx(self.topo_graphs[i], pos=pos_dict)

    def compute_posterior_model(self, interp_data, i):
        self.change_input_data(interp_data, i)
        return gp.compute_model(interp_data)

    def plot_section(self, interp_data, i, dim, plot_data=False, plot_topo=False):
        self.change_input_data(interp_data, i)
        lith_block, fault_block = gp.compute_model(interp_data)
        #plt.imshow(lith_block[-1, 0,:].reshape(dim[0], dim[1], dim[2])[:, 0, :].T, origin="lower",
        #           cmap=gp.colors.cmap, norm=gp.colors.norm)
        gp.plot_section(interp_data.geo_data_res, lith_block[0], 0, plot_data=plot_data)

        rs = interp_data.rescaling_factor
        #if plot_data:
        #    plt.scatter(interp_data.geo_data_res.interfaces["X"].values,
        #                interp_data.geo_data_res.interfaces["Z"].values)

        if plot_topo:
            self.topo_plot_graph(i)

    def topo_plot_graph(self, i):
        pos_2d = {}
        for key in self.topo_centroids[i].keys():
            pos_2d[key] = [self.topo_centroids[i][key][0], self.topo_centroids[i][key][2]]
        nx.draw_networkx(self.topo_graphs[i], pos=pos_2d)


def compare_graphs(G1, G2):
    intersection = 0
    union = G1.number_of_edges()

    for edge in G1.edges_iter():
        if G2.has_edge(edge[0], edge[1]):
            intersection += 1
        else:
            union += 1

    return intersection / union


class Plane:
    def __init__(self, group_id, data_obj):
        self.group_id = group_id
        self.data_obj = data_obj

        # create dataframe bool filters for convenience
        self.fol_f = self.data_obj.foliations["group_id"] == self.group_id
        self.interf_f = self.data_obj.interfaces["group_id"] == self.group_id

        # get indices for both foliations and interfaces
        self.interf_i = self.data_obj.interfaces[self.interf_f].index
        self.fol_i = self.data_obj.foliations[self.fol_f].index[0]

        # normal
        self.normal = None
        # centroid
        self.centroid = None
        self.refresh()

    # method: give dip, change interfaces accordingly
    def interf_recalc(self, dip):
        """Changes the dip of plane and recalculates Z coordinates for the points belonging to it."""
        # modify the foliation
        self.data_obj.foliations.set_value(self.fol_i, "dip", dip)
        # get azimuth
        az = float(self.data_obj.foliations.iloc[self.fol_i]["azimuth"])
        # set polarity according to dip
        if -90 < dip < 90:
            polarity = 1
        else:
            polarity = -1
        self.data_obj.foliations.set_value(self.fol_i, "polarity", polarity)
        # modify gradient
        self.data_obj.foliations.set_value(self.fol_i, "G_x",
                                           np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * polarity)
        self.data_obj.foliations.set_value(self.fol_i, "G_y",
                                           np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * polarity)
        self.data_obj.foliations.set_value(self.fol_i, "G_z", np.cos(np.deg2rad(dip)) * polarity)

        # update normal
        self.normal = self.get_normal()
        # modify points (Z only so far)
        a, b, c = self.normal
        d = -a * self.centroid[0] - b * self.centroid[1] - c * self.centroid[2]
        for i, row in self.data_obj.interfaces[self.interf_f].iterrows():
            # iterate over each point and recalculate Z, set Z
            # x, y, z = row["X"], row["Y"], row["Z"]
            Z = (a * row["X"] + b * row["Y"] + d) / -c
            self.data_obj.interfaces.set_value(i, "Z", Z)

    def refresh(self):
        # normal
        self.normal = self.get_normal()
        # centroid
        self.centroid = [float(self.data_obj.foliations[self.fol_f]["X"]),
                         float(self.data_obj.foliations[self.fol_f]["Y"]),
                         float(self.data_obj.foliations[self.fol_f]["Z"])]

    def get_normal(self):
        """Just returns updated normal vector (values from dataframe)."""
        normal = [float(self.data_obj.foliations.iloc[self.fol_i]["G_x"]),
                  float(self.data_obj.foliations.iloc[self.fol_i]["G_y"]),
                  float(self.data_obj.foliations.iloc[self.fol_i]["G_z"])]
        return normal


