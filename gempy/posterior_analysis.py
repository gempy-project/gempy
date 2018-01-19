"""
@author: Alexander Schaaf, Miguel de la Varga
"""
import warnings
try:
    import pymc
except ImportError:
    warnings.warn("pymc (v2) package is not installed. No support for stochastic simulation posterior analysis.")
import numpy as np
import pandas as pn
import gempy as gp
try:
    import tqdm
except ImportError:
    warnings.warn("tqdm package not installed. No support for dynamic progress bars.")


class Posterior:
    def __init__(self, dbname, pymc_model_f="gempy_model", pymc_topo_f="gempy_topo",
                 topology=False, verbose=False):
        """
        Posterior database analysis for GemPy-pymc2 hdf5 databases.
        Args:
            dbname (str): Path of the hdf5 database.
            pymc_model_f (str, optional): name of the model output function used (default: "gempy_model).
            pymc_topo_f (str, optional): name of the topology output function used (default: "gempy_topo).
            topology (bool, optional):  if a topology trace should be loaded from the database (default: False).
            verbose (bool, optional): Verbosity switch.
        """
        self.verbose = verbose
        # load db
        self.db = pymc.database.hdf5.load(dbname)

        # number iter n = self.db.getstate()["sampler"]["_iter"]
        self.n_iter = self.db.getstate()['sampler']['_iter'] - self.db.getstate()["sampler"]["_burn"]
        # get trace names
        self.trace_names = self.db.trace_names[0]
        # get gempy block models
        try:
            self.lb = self.db.trace(pymc_model_f)[:, :2, :]
            self.fb = self.db.trace(pymc_model_f)[:, 2:, :]
        except KeyError:
            print("No GemPy model trace tallied.")
            self.lb = None
            self.fb = None

        if topology:  # load topology data from database
            topo_trace = self.db.trace(pymc_topo_f)[:]
            # load graphs
            self.topo_graphs = topo_trace[:, 0]
            # load centroids
            self.topo_centroids = topo_trace[:, 1]
            # unique labels
            self.topo_labels_unique = topo_trace[:, 2]
            # get the look-up-tables
            self.topo_lith_to_labels_lot = topo_trace[:, 3]
            self.topo_labels_to_lith_lot = topo_trace[:, 4]
            del topo_trace

            self.topo_unique, self.topo_unique_freq, self.topo_unique_ids, self.topo_unique_prob = (None, None, None, None)
            self.topo_count_dict = None

            self.topo_analyze()

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
        interp_data.geo_data_res.foliations[["X", "Y", "Z", "dip", "azimuth", "polarity"]] = self.input_data[i][1]

        recalc_gradients(interp_data.geo_data_res.foliations)

        # update interpolator
        interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")
        return interp_data

    def compute_posterior_model(self, interp_data, i):
        """Computes the model with the respective posterior input data. Returns lith block, fault block."""
        self.change_input_data(interp_data, i)
        return gp.compute_model(interp_data)

    def compute_posterior_models_all(self, interp_data, r=None):
        """Computes block models from stored input parameters for all iterations.

        Args:
            interp_data: GemPy interpolator object
            r (optional):

        Returns: Stores calculated posterior models in self.lb and self.lb

        """

        if r is None:  # compute model for every iteration
            r = range(self.n_iter)
        else:  # use the given slice
            r = range(r[0], r[1])

        for i in tqdm.tqdm(r):
            lb, fb = self.compute_posterior_model(interp_data, 1)
            if i == 0 or i == r[0]:
                self.lb = np.expand_dims(lb, 0)
                self.fb = np.expand_dims(fb, 0)
            else:
                self.lb = np.concatenate((self.lb, np.expand_dims(lb, 0)), axis=0)
                self.fb = np.concatenate((self.fb, np.expand_dims(fb, 0)), axis=0)

        # if self.lb is None:
        #     # create the storage array
        #     lb, fb = self.compute_posterior_model(interp_data, 1)
        #     lb = lb[0]
        #
        #     self.lb = np.empty_like(lb)
        #     if calc_fb:
        #         self.fb = np.empty_like(lb)
        #
        #     if r is None:  # compute model for every iteration
        #         r = range(self.n_iter)
        #     else:  # use the given slice
        #         r = range(r[0], r[1])
        #     try:
        #         for i in tqdm.tqdm(r):
        #             output = self.compute_posterior_model(interp_data, i)
        #             if i == 0:
        #                 self.lb = lb[0].astype("int32")
        #                 self.fb = fb[0].astype("int32")
        #             else:
        #                 lb, fb = self.compute_posterior_model(interp_data, i)
        #                 self.lb = np.vstack((self.lb, lb[0].astype("int32")))
        #                 if calc_fb:
        #                     self.fb = np.vstack((self.fb, fb[0].astype("int32")))
        #     except NameError:
        #         for i in range(r):
        #             if i == 0:
        #                 lb, fb = self.compute_posterior_model(interp_data, i)
        #                 self.lb = lb[0].astype("int32")
        #                 self.fb = fb[0].astype("int32")
        #             else:
        #                 lb, fb = self.compute_posterior_model(interp_data, i)
        #                 self.lb = np.vstack((self.lb, lb[0].astype("int32")))
        #                 if calc_fb:
        #                     self.fb = np.vstack((self.fb, fb[0].astype("int32")))
        # else:
        #     print("self.lb already filled with something. If you want to override, set self.lb to 'None'")

    def compute_posterior_model_avrg(self, interp_data):
        """Computes average posterior model."""
        list_interf = []
        list_fol = []
        for i in range(self.n_iter):
            list_interf.append(self.input_data[i][0])
            list_fol.append(self.input_data[i][1])

        interf_avrg = pn.concat(list_interf).groupby(level=0).mean()
        fol_avrg = pn.concat(list_fol).groupby(level=0).mean()

        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = interf_avrg
        interp_data.geo_data_res.foliations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = fol_avrg
        interp_data.update_interpolator()
        return gp.compute_model(interp_data)

    def compute_entropy(self):
        """Computes the voxel information entropy of stored block models."""
        if self.lb is None:
            return "No models stored in self.lb, please run 'self.compute_posterior_models_all' to generate block" \
                   " models for all iterations."

        self.lith_prob = compute_prob_lith(self.lb[:, 0, :])
        self.ie = calcualte_ie_masked(self.lith_prob)
        self.ie_total = calculate_ie_total(self.ie)
        print("Information Entropy successfully calculated. Stored in self.ie and self.ie_total")

    def topo_count_connection(self, n1, n2):
        """Counts the amount of times connection between nodes n1 and n2 in all of the topology graphs."""
        count = 0
        for G in self.topo_graphs:
            count += check_adjacency(G, n1, n2)
        return count

    def topo_count_connection_array(self, n1, n2):
        count = []
        for G in self.topo_graphs:
            count.append(check_adjacency(G, n1, n2))
        return count

    def topo_count_total_number_of_nodes(self):
        """Counts the amount of topology graphs with a certain amount of total nodes."""
        self.topo_count_dict = {}
        for g in self.topo_graphs:
            c = len(g.adj.keys())
            if c in self.topo_count_dict.keys():
                self.topo_count_dict[c] += 1
            else:
                self.topo_count_dict[c] = 1

    def topo_analyze(self):
        """Analysis of the tallied topology distribution."""
        if self.verbose:
            print("Starting topology analysis. This could take a while (depending on # iterations).")
        self.topo_unique, self.topo_unique_freq, self.topo_unique_ids = get_unique_topo(self.topo_graphs)
        self.topo_unique_prob = self.topo_unique_freq / np.sum(self.topo_unique_freq)
        # count unique node numbers
        self.topo_count_total_number_of_nodes()

        self.topo_sort = np.argsort(self.topo_unique_freq)[::-1]

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
    # lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
    lith_count = np.zeros((len(np.unique(lith_blocks)), lith_blocks.shape[1]))
    for i, l_id in enumerate(lith_id):
        lith_count[i] = np.sum(lith_blocks == l_id, axis=0)
    lith_prob = lith_count / len(lith_blocks)
    return lith_prob


def calcualte_ie_masked(lith_prob):
    """Calculates information entropy for the given probability array."""
    ie = np.zeros_like(lith_prob[0])
    for l in lith_prob:
        pm = np.ma.masked_equal(l, 0)  # mask where layer prob is 0
        ie -= (pm * np.ma.log2(pm)).filled(0)
    return ie


def calculate_ie_total(ie, absolute=False):
    """Calculate total information entropy (float) from an information entropy array."""
    if absolute:
        return np.sum(ie)
    else:
        return np.sum(ie) / np.size(ie)


def compare_graphs(G1, G2):
    """Compare two NetworkX graphs to obtain the Jaccard index (e.g. to compare topology graphs).

    Args:
        G1 (:obj:): Graph
        G2 (:obj:): Another graph

    Returns: (float) Jaccard index

    """
    intersection = 0
    union = G1.number_of_edges()

    for edge in G1.edges_iter():
        if G2.has_edge(edge[0], edge[1]):
            intersection += 1
        else:
            union += 1

    return intersection / union


def modify_plane_dip(dip, group_id, data_obj):
    """Modify a dip angle of a plane identified by a group_id, recalculate the gradient and move the points vertically.
    Currently only supports the modification of dip angle - azimuth and polarity will stay the same.

    Args:
        dip (float): Desired dip angle of the plane.
        group_id (str): Group id identifying the data points belonging to the plane.
        data_obj (:obj:): Data object to be modified (geo_data or interp_data.geo_data_res)

    Returns:
        Directly modifies the given data object.
    """
    # get foliation and interface data points ids
    fol_f = data_obj.foliations["group_id"] == group_id
    interf_f = data_obj.interfaces["group_id"] == group_id

    # get indices
    interf_i = data_obj.interfaces[interf_f].index
    fol_i = data_obj.foliations[fol_f].index[0]

    # update dip value for foliations
    data_obj.foliations.set_value(fol_i, "dip", dip)
    # get azimuth and polarity
    az = float(data_obj.foliations.iloc[fol_i]["azimuth"])
    pol = data_obj.foliations.iloc[fol_i]["polarity"]

    # calculate gradient/normal and modify
    gx, gy, gz = calculate_gradient(dip, az, pol)
    data_obj.foliations.set_value(fol_i, "G_x", gx)
    data_obj.foliations.set_value(fol_i, "G_y", gy)
    data_obj.foliations.set_value(fol_i, "G_z", gz)

    normal = [gx, gy, gz]
    centroid = np.array([float(data_obj.foliations[fol_f]["X"]),
                         float(data_obj.foliations[fol_f]["Y"]),
                         float(data_obj.foliations[fol_f]["Z"])])
    # move points vertically to fit plane
    move_plane_points(normal, centroid, data_obj, interf_f)


def move_plane_points(normal, centroid, data_obj, interf_f):
    """Moves interface points to fit plane of given normal and centroid in data object."""
    a, b, c = normal
    d = -a * centroid[0] - b * centroid[1] - c * centroid[2]
    for i, row in data_obj.interfaces[interf_f].iterrows():
        # iterate over each point and recalculate Z, set Z
        Z = (a * row["X"] + b * row["Y"] + d) / -c
        data_obj.interfaces.set_value(i, "Z", Z)


def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


def recalc_gradients(folations_dataframe):
    folations_dataframe["G_x"] = np.sin(np.deg2rad(folations_dataframe["dip"].astype('float'))) * \
                             np.sin(np.deg2rad(folations_dataframe["azimuth"].astype('float'))) * \
                             folations_dataframe["polarity"].astype('float')
    folations_dataframe["G_y"] = np.sin(np.deg2rad(folations_dataframe["dip"].astype('float'))) * \
                             np.cos(np.deg2rad(folations_dataframe["azimuth"].astype('float'))) *\
                             folations_dataframe["polarity"].astype('float')
    folations_dataframe["G_z"] = np.cos(np.deg2rad(folations_dataframe["dip"].astype('float'))) *\
                             folations_dataframe["polarity"].astype('float')

# DEP PLANE CLASS SHIT
# class Plane:
#     def __init__(self, group_id, data_obj):
#         self.group_id = group_id
#         self.data_obj = data_obj
#
#         # create dataframe bool filters for convenience
#         self.fol_f = self.data_obj.foliations["group_id"] == self.group_id
#         self.interf_f = self.data_obj.interfaces["group_id"] == self.group_id
#
#         # get indices for both foliations and interfaces
#         self.interf_i = self.data_obj.interfaces[self.interf_f].index
#         self.fol_i = self.data_obj.foliations[self.fol_f].index[0]
#
#         # normal
#         self.normal = None
#         # centroid
#         self.centroid = None
#         self.refresh()
#
#     # method: give dip, change interfaces accordingly
#     def interf_recalc_Z(self, dip):
#         """Changes the dip of plane and recalculates Z coordinates for the points belonging to it."""
#         # set the foliation dip in df
#         self.data_obj.foliations.set_value(self.fol_i, "dip", dip)
#         # get azimuth
#         az = float(self.data_obj.foliations.iloc[self.fol_i]["azimuth"])
#
#         # set polarity according to dip
#         #if -90 < dip < 90:
#         #    polarity = 1
#         #else:
#         #    polarity = -1
#
#         #self.data_obj.foliations.set_value(self.fol_i, "polarity", polarity)
#         polarity = self.data_obj.foliations.iloc[self.fol_i]["polarity"]
#
#         # modify gradient
#         self.data_obj.foliations.set_value(self.fol_i, "G_x",
#                                            np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * polarity)
#         self.data_obj.foliations.set_value(self.fol_i, "G_y",
#                                            np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * polarity)
#         self.data_obj.foliations.set_value(self.fol_i, "G_z", np.cos(np.deg2rad(dip)) * polarity)
#
#         # update normal
#         self.normal = self.get_normal()
#
#         # modify points (Z only so far)
#         a, b, c = self.normal
#         d = -a * self.centroid[0] - b * self.centroid[1] - c * self.centroid[2]
#         for i, row in self.data_obj.interfaces[self.interf_f].iterrows():
#             # iterate over each point and recalculate Z, set Z
#             Z = (a * row["X"] + b * row["Y"] + d) / -c
#             self.data_obj.interfaces.set_value(i, "Z", Z)
#
#     def refresh(self):
#         # normal
#         self.normal = self.get_normal()
#         # centroid
#         self.centroid = [float(self.data_obj.foliations[self.fol_f]["X"]),
#                          float(self.data_obj.foliations[self.fol_f]["Y"]),
#                          float(self.data_obj.foliations[self.fol_f]["Z"])]
#
#     def get_normal(self):
#         """Just returns updated normal vector (values from dataframe)."""
#         normal = [float(self.data_obj.foliations.iloc[self.fol_i]["G_x"]),
#                   float(self.data_obj.foliations.iloc[self.fol_i]["G_y"]),
#                   float(self.data_obj.foliations.iloc[self.fol_i]["G_z"])]
#         return normal


