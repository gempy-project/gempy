"""
@author: Alexander Schaaf, Miguel de la Varga
"""
import warnings
try:
    import pymc
except ImportError:
    warnings.warn("pymc (v2) package is not installed. No support for stochastic simulation posterior analysis.")
import theano
import numpy as np
# try:
#     import networkx as nx
# except ImportError:
#     warnings.warn("networkx package is not installed. No topology graph visualization possible.")
import gempy as gp


class Posterior:
    """Posterior database analysis for GemPy-pymc2 hdf5 databases."""

    def __init__(self, dbname, pymc_model_f="gempy_model", topology=False, verbose=False):
        self.verbose = verbose
        # load db
        self.db = pymc.database.hdf5.load(dbname)

        # number iter
        self.n_iter = self.db.getstate()['sampler']['_iter']
        # get trace names
        self.trace_names = self.db.trace_names[0]
        # get gempy block models
        try:
            self.lb, self.fb = self.db.trace(pymc_model_f)[:]
        except KeyError:
            print("No GemPy model trace tallied.")
            self.lb = None
            self.fb = None

        if topology:
            # load graphs
            topo_trace = self.db.gempy_topo.gettrace()
            self.topo_graphs = topo_trace[:, 0]
            # load centroids
            self.topo_centroids = topo_trace[:, 1]
            self.topo_labels_unique = topo_trace[:, 2]
            self.topo_lith_to_labels_lot = topo_trace[:, 3]
            self.topo_labels_to_lith_lot = topo_trace[:, 4]
            del topo_trace

        self.topo_unique, self.topo_unique_freq, self.topo_unique_ids, self.topo_unique_prob = (None, None, None, None)

        # load input data
        self.input_data = self.db.input_data.gettrace()

        self.lith_prob = None
        self.ie = None
        self.ie_total = None

    def change_input_data(self, interp_data, i):
        """Changes input data in interp_data to posterior input data at iteration i."""

        i = int(i)
        # replace interface data
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        interp_data.geo_data_res.foliations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = self.input_data[i][1]
        # do all the ugly updating stuff
        interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")
        return interp_data

    # def plot_topology_graph(self, i):
    #     # get centroid values into list
    #     centroid_values = [triplet for triplet in self.topo_centroids[i].values()]
    #     # unzip them into seperate lists of x,y,z coordinates
    #     centroids_x, centroids_y, centroids_z = list(zip(*centroid_values))
    #     # create new 2d pos dict for plot
    #     pos_dict = {}
    #     for j in range(len(centroids_x)):
    #         pos_dict[j + 1] = [centroids_x[j], centroids_y[j]]
    #     # draw
    #     nx.draw_networkx(self.topo_graphs[i], pos=pos_dict)

    def compute_posterior_model(self, interp_data, i):
        self.change_input_data(interp_data, i)
        return gp.compute_model(interp_data)

    def plot_section(self, interp_data, i, ext, res, plot_data=False, plot_topo=False):
        """Deprecated."""
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
            self.topo_plot_graph(interp_data, i, ext, res)

    # def topo_plot_graph(self, interp_data, i, ext, res):
    #     pos_2d = {}
    #     for key in self.topo_centroids[i].keys():
    #         pos_2d[key] = [self.topo_centroids[i][key][0] * ext[1]/res[0] / interp_data.rescaling_factor, self.topo_centroids[i][key][2] * ext[5]/res[2] / interp_data.rescaling_factor]
    #     nx.draw_networkx(self.topo_graphs[i], pos=pos_2d)

    def compute_posterior_models_all(self, interp_data, n=None, calc_fb=True):
        """Computes block models from stored input parameters for all iterations."""
        if self.lb is None:
            # create the storage array
            lb, fb = self.compute_posterior_model(interp_data, 1)
            lb = lb[0]
            fb = fb[0]
            self.lb = np.empty_like(lb)
            if calc_fb:
                self.fb = np.empty_like(lb)

            # compute model for every iteration
            if n is None:
                n = self.db.getstate()["sampler"]["_iter"]
            for i in range(n):
                if i == 0:
                    lb, fb = self.compute_posterior_model(interp_data, i)
                    self.lb = lb[0]
                    self.fb = fb[0]
                else:
                    lb, fb = self.compute_posterior_model(interp_data, i)
                    self.lb = np.vstack((self.lb, lb[0]))
                    if calc_fb:
                        self.fb = np.vstack((self.fb, fb[0]))
        else:
            print("self.lb already filled with something.")

    def compute_entropy(self, interp_data):
        """Computes the voxel information entropy of stored block models."""
        if self.lb is None:
            return "No models stored in self.lb, please run 'self.compute_posterior_models_all' to generate block" \
                   " models for all iterations."

        self.lith_prob = compute_prob_lith(self.lb)
        self.ie = calcualte_ie_masked(self.lith_prob)
        self.ie_total = calculate_ie_total(self.ie)
        print("Information Entropy successfully calculated. Stored in self.ie and self.ie_total")

    def check_adjacency_freq(self, n1, n2):
        count = 0
        for G in self.topo_graphs:
            count += check_adjacency(G, n1, n2)
        return count

    def topo_analyze(self):
        if self.verbose:
            print("Starting topology analysis. This could take a while (depending on # iterations).")
        self.topo_unique, self.topo_unique_freq, self.topo_unique_ids = get_unique_topo(self.topo_graphs)
        self.topo_unique_prob = self.topo_unique_freq / np.sum(self.topo_unique_freq)
        if self.verbose:
            print("Topology analysis completed.")


def find_first_match(t, topo_u):
    index = 0
    for t2 in topo_u:
        if compare_graphs(t, t2) == 1:
            return index  # the models match
        index += 1

    return -1


def get_unique_topo(topo_l):
    # create list for our unique topologies
    topo_u = []
    topo_u_freq = []
    topo_u_ids = np.empty_like(topo_l)

    for n, t in enumerate(topo_l):
        i = find_first_match(t, topo_u)
        if i == -1:  # is a yet unobserved topology, so append it and initiate frequency
            topo_u.append(t)
            topo_u_freq.append(1)
            topo_u_ids[n] = len(topo_u) - 1
        else:  # is a known topology
            topo_u_freq[i] += 1  # 1-up the corresponding frequency
            topo_u_ids[n] = i

    return topo_u, topo_u_freq, topo_u_ids

def check_adjacency(G, n1, n2):
    """Check if n2 is adjacent/shares edge with n1."""
    if n2 in G.adj[n1]:
        return True
    else:
        return False


def compute_prob_lith(lith_blocks):
    """Blocks must be just the lith blocks!"""
    lith_id = np.unique(lith_blocks)
    lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
    for i, l_id in enumerate(lith_id):
        lith_count[i] = np.sum(lith_blocks == l_id, axis=0)
    lith_prob = lith_count / len(lith_blocks)
    return lith_prob


def calcualte_ie_masked(lith_prob):
    ie = np.zeros_like(lith_prob[0])
    for l in lith_prob:
        pm = np.ma.masked_equal(l, 0)  # mask where layer prob is 0
        ie -= (pm * np.ma.log2(pm)).filled(0)
    return ie


def calculate_ie_total(ie, absolute=False):
    if absolute:
        return np.sum(ie)
    else:
        return np.sum(ie) / np.size(ie)


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
    def interf_recalc_Z(self, dip):
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


