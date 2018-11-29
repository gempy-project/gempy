
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
import matplotlib.pyplot as plt

class Posterior:
    def __init__(self, dbname, verbose=False, entropy=False, interp_data = None):
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
        self.interp_data = interp_data
        self.verbose = verbose
        # load db
        self.db = pymc.database.hdf5.load(dbname)

        self.n_iter = self.db.getstate()['sampler']['_iter'] - self.db.getstate()["sampler"]["_burn"]
        # get trace names
        self.trace_names = self.db.trace_names[0]

        # load input data
        self.input_data = self.db.input_data.gettrace()

        self.lith_prob = None
        self.ie = None
        self.ie_total = None

        if entropy is True:
            self.lbs, self.fbs = self.all_post_models()

            self.lith_prob = self.compute_prob(np.round(self.lbs).astype(int))
            self.fault_prob = self.compute_prob(np.round(self.fbs).astype(int))

            self.lb_ie = self.calculate_ie_masked(self.lith_prob)
            self.fb_ie = self.calculate_ie_masked(self.fault_prob)

            self.ie_total = self.calculate_ie_total()

    def plot_lith_entropy(self, resolution):
        '''plots information entropy in middle of block model in y-direction'''
        y = int(resolution[1] / 2)
        ie_reshaped = self.lb_ie.reshape(resolution)
        plt.figure(figsize=(15, 5))
        fig, ax = plt.subplots(1)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.imshow(ie_reshaped[:, y, :].T, origin="lower", cmap="viridis")
        plt.colorbar()
        return fig

    def plot_fault_entropy(self, resolution):
        '''plots information entropy in middle of block model in y-direction'''
        y = int(resolution[1] / 2)
        # print(y, resolution)
        ie_reshaped = self.fb_ie.reshape(resolution)
        # print(ie_reshaped.shape)
        plt.figure(figsize=(15, 5))
        fig, ax = plt.subplots(1)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.imshow(ie_reshaped[:, y, :].T, origin="lower", cmap="viridis")
        plt.colorbar()
        return fig

    def change_input_data(self, i):
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
        self.interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        self.interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = \
        self.input_data[i][1]

        # recalc_gradients(interp_data.geo_data_res.orientations)

        # update interpolator
        self.interp_data.update_interpolator()
        if self.verbose:
            print("interp_data parameters changed.")
        return self.interp_data

    def all_post_models(self):
        lbs = []
        fbs = []
        for i in range(0, self.n_iter):
            # print(i)
            self.change_input_data(i)
            lith_block, fault_block = gp.compute_model(self.interp_data)
            lbs.insert(i, lith_block[0])
            fbs.insert(i, fault_block[0])
        return lbs, fbs

    def compute_prob(self, lith_blocks):
        """Blocks must be just the lith blocks!"""
        lith_id = np.unique(lith_blocks)
        # print(len(lith_id))
        # lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
        lith_count = np.zeros((len(np.unique(lith_blocks)), lith_blocks.shape[1]))
        # print(lith_count)
        for i, l_id in enumerate(lith_id):
            # print(i, l_id)
            lith_count[i] = np.sum(lith_blocks == l_id, axis=0)
        lith_prob = lith_count / len(lith_blocks)
        # print(lith_prob)
        return lith_prob

    def calculate_ie_masked(self, lith_prob):
        ie = np.zeros_like(lith_prob[0])
        for l in lith_prob:
            # print(l)
            pm = np.ma.masked_equal(l, 0)  # mask where layer prob is 0
            # print(pm.shape)
            # print(pm * np.ma.log2(pm))
            ie -= (pm * np.ma.log2(pm)).filled(0)
        return ie

    def calculate_ie_total(self, absolute=False):
        if absolute:
            return np.sum(self.lb_ie)
        else:
            return np.sum(self.lb_ie) / np.size(self.lb_ie)


