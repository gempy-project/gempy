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


def change_input_data(db, interp_data, i):
    """
    Changes input data in interp_data to posterior input data at iteration i.

    Args:
        interp_data (gempy.data_management.InterpolationData): An interp_data object with the structure we want to
        compute.
        i (int): Iteration we want to recompute

    Returns:
         gempy.data_management.InterpolationData: interp_data with the data of the given iteration
    """
    i = int(i)
    # replace interface data
    interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = db.trace("input_interf")[i]
    # replace foliation data
    try:
        interp_data.geo_data_res.orientations[["X", "Y", "Z", "dip", "azimuth", "polarity"]] = db.trace("input_orient")[i]
    except ValueError:
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = db.trace("input_orient")[i]

    recalc_gradients(interp_data.geo_data_res.orientations)

    # interp_data.geo_data_res = interp_data.rescale_data(interp_data.geo_data_res)

    # update interpolator
    interp_data.update_interpolator()

    return interp_data


def _change_input_data_old(db, interp_data, i, tracename="input_data"):
    """
    Changes input data in interp_data to posterior input data at iteration i.

    Args:
        interp_data (gempy.data_management.InterpolationData): An interp_data object with the structure we want to
        compute.
        i (int): Iteration we want to recompute

    Returns:
         gempy.data_management.InterpolationData: interp_data with the data of the given iteration
    """
    i = int(i)
    # replace interface data
    interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = db.trace(tracename)[i][0]
    # replace foliation data
    try:
        interp_data.geo_data_res.orientations[["X", "Y", "Z", "dip", "azimuth", "polarity"]] = db.trace(tracename)[i][1]
    except ValueError:
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = db.trace(tracename)[i][1]

    recalc_gradients(interp_data.geo_data_res.orientations)

    # interp_data.geo_data_res = interp_data.rescale_data(interp_data.geo_data_res)

    # update interpolator
    interp_data.update_interpolator()

    return interp_data


def compute_posterior_models_all(db, interp_data, indices, u_grade=None, get_potential_at_interfaces=False):
    """Computes block models from stored input parameters for all iterations.

    Args:
        db (): loaded pymc database (e.g. hdf5)
        interp_data  (gp.data_management.InterpolatorData): GemPy interpolator object
        indices (list or np.array): Trace indices specifying which models from the database will be calculated.
        u_grade (list, optional):
        get_potential_at_interfaces:

    Returns:

    """

    for i in tqdm.tqdm(indices):
        interp_data_loop = change_input_data(db, interp_data, i)
        lb, fb = gp.compute_model(interp_data_loop, output="geology", u_grade=u_grade, get_potential_at_interfaces=get_potential_at_interfaces)
        if i == 0 or i == indices[0]:
            lbs = np.expand_dims(lb, 0)
            fbs = np.expand_dims(fb, 0)
        else:
            lbs = np.concatenate((lbs, np.expand_dims(lb, 0)), axis=0)
            fbs = np.concatenate((fbs, np.expand_dims(fb, 0)), axis=0)

    return lbs, fbs


def compute_probability_lithology(lith_blocks):
    """Blocks must be just the lith blocks!"""
    lith_id = np.unique(lith_blocks)
    # lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
    lith_count = np.zeros((len(np.unique(lith_blocks)), lith_blocks.shape[1]))
    for i, l_id in enumerate(lith_id):
        lith_count[i] = np.sum(lith_blocks == l_id, axis=0)
    lith_prob = lith_count / len(lith_blocks)
    return lith_prob


def calcualte_information_entropy(lith_prob):
    """Calculates information entropy for the given probability array."""
    ie = np.zeros_like(lith_prob[0])
    for l in lith_prob:
        pm = np.ma.masked_equal(l, 0)  # mask where layer prob is 0
        ie -= (pm * np.ma.log2(pm)).filled(0)
    return ie


def calculate_information_entropy_total(ie, absolute=False):
    """Calculate total information entropy (float) from an information entropy array."""
    if absolute:
        return np.sum(ie)
    else:
        return np.sum(ie) / np.size(ie)


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
        # TODO: Add a method to set the lith_block and fault_block


        self.verbose = verbose
        # load db
        self.db = pymc.database.hdf5.load(dbname)

        self.n_iter = self.db.getstate()['sampler']['_iter'] - self.db.getstate()["sampler"]["_burn"]
        # get trace names
        self.trace_names = self.db.trace_names[0]

        # TODO DEP
        # get gempy block models
        # try:
        #     self.lb = self.db.trace(pymc_model_f)[:, :2, :]
        #     self.fb = self.db.trace(pymc_model_f)[:, 2:, :]
        # except KeyError:
        #     print("No GemPy model trace tallied.")
        #     self.lb = None
        #     self.fb = None

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
        """
        Changes input data in interp_data to posterior input data at iteration i.

        Args:
            interp_data (gempy.data_management.InterpolationData): An interp_data object with the structure we want to
            compute.
            i (int): Iteration we want to recompute

        Returns:
             gempy.data_management.InterpolationData: interp_data with the data of the given iteration
        """
        i = int(i)
        # replace interface data
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = self.input_data[i][1]

        recalc_gradients(interp_data.geo_data_res.orientations)

        # update interpolator
        interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")
        return interp_data

    # TODO: DEP Use gp.compute_model instead
    # def compute_posterior_model(self, interp_data, i):
    #     """Computes the model with the respective posterior input data. Returns lith block, fault block."""
    #     self.change_input_data(interp_data, i)
    #     return gp.compute_model(interp_data)



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
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = fol_avrg
        interp_data.update_interpolator()
        return gp.compute_model(interp_data)

    def compute_entropy(self):
        """Computes the voxel information entropy of stored block models."""
        if self.lb is None:
            return "No models stored in self.lb, please run 'self.compute_posterior_models_all' to generate block" \
                   " models for all iterations."

        self.lith_prob = compute_probability_lithology(self.lb[:, 0, :])
        self.ie = calcualte_information_entropy(self.lith_prob)
        self.ie_total = calculate_information_entropy_total(self.ie)
        print("Information Entropy successfully calculated. Stored in self.ie and self.ie_total")

    def topo_count_connection(self, n1, n2):
        """Counts the amount of times connection between nodes n1 and n2 in all of the topology graphs."""
        count = 0
        for G in self.topo_graphs:
            count += gp.topology.check_adjacency(G, n1, n2)
        return count

    def topo_count_connection_array(self, n1, n2):
        count = []
        for G in self.topo_graphs:
            count.append(gp.topology.check_adjacency(G, n1, n2))
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
        if gp.topology.compare_graphs(t, t2) == 1:
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


def get_unique_jaccard(js):
    j_u = np.unique(js)  # unique topology states



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
    fol_f = data_obj.orientations["group_id"] == group_id
    interf_f = data_obj.interfaces["group_id"] == group_id

    # get indices
    interf_i = data_obj.interfaces[interf_f].index
    fol_i = data_obj.orientations[fol_f].index[0]

    # update dip value for orientations
    data_obj.orientations.set_value(fol_i, "dip", dip)
    # get azimuth and polarity
    az = float(data_obj.orientations.iloc[fol_i]["azimuth"])
    pol = data_obj.orientations.iloc[fol_i]["polarity"]

    # calculate gradient/normal and modify
    gx, gy, gz = calculate_gradient(dip, az, pol)
    data_obj.orientations.set_value(fol_i, "G_x", gx)
    data_obj.orientations.set_value(fol_i, "G_y", gy)
    data_obj.orientations.set_value(fol_i, "G_z", gz)

    normal = [gx, gy, gz]
    centroid = np.array([float(data_obj.orientations[fol_f]["X"]),
                         float(data_obj.orientations[fol_f]["Y"]),
                         float(data_obj.orientations[fol_f]["Z"])])
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
