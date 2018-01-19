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

Module with classes and methods to perform kriging of elements (and at some point exploit the potential field to
choose the directions of the variograms)

Tested on Ubuntu 16

Created on 1/5/2017

@author: Miguel de la Varga
"""

import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import pandas as pn

from bokeh.io import show
import bokeh.layouts as bl
import bokeh.plotting as bp


def choose_lithology_elements(df, litho, elem=None, coord = True):
    """
    litho(str): Name of the lithology-domain
    elem(list): List of strings with elements you want to analyze
    """
    # Choosing just the opx litology
    if elem is not None:
        if coord:
            domain = df[df['Lithology'] == litho][np.append(['X', 'Y', 'Z'], elem)]
        else:
            domain = df[df['Lithology'] == litho][elem]
        # Drop negative values
        domain = domain[(domain[elem] > 0).all(1)]
    else:
        domain = df[df['Lithology'] == litho][['X', 'Y', 'Z']]

    return domain


def select_segmented_grid(df, litho, grid, block):

    block = np.squeeze(block)
    assert grid.shape[0] == block.shape[0], 'The grid you want to use for kriging and the grid used for the layers ' \
                                            'segmentation are not the same'

    litho_num = df['Lithology Number'][df["Lithology"] == litho].iloc[0]
    segm_grid = grid[block == litho_num]
    return segm_grid


def transform_data(df_o, n_comp=1, log10=False):
    """
    Method to improve the normality of the input data before perform krigin
    Args:
        df_o: Dataframe with the data to interpolate
        n_comp: Number of component in case of multimodal data
        log10 (bool): If true return the log in base 10 of the properties:

    Returns:
        pandas.core.frame.DataFrame: Data frame with the transformed data
    """

    import copy
    df = copy.deepcopy(df_o)

    # Take log to try to aproximate better the normal distributions
    if log10:
        print('computing log')
        df[df.columns.difference(['X', 'Y', 'Z'])] = np.log10(df[df.columns.difference(['X', 'Y', 'Z'])])

    # Finding n modes in the data
    if n_comp > 1:
        from sklearn import mixture
        gmm = mixture.GaussianMixture(n_components=n_comp,
                                      covariance_type='full').fit(df[df.columns.difference(['X', 'Y', 'Z'])])

        # Adding the categories to the pandas frame
        labels_all = gmm.predict(df[df.columns.difference(['X', 'Y', 'Z'])])
        df['cluster'] = labels_all
    return df


def theano_sed():
    """
    Function to create a theano function to compute the euclidian distances efficiently
    Returns:
        theano.compile.function_module.Function: Compiled function

    """

    theano.config.compute_test_value = "ignore"

    # Set symbolic variable as matrix (with the XYZ coords)
    coord_T_x1 = T.dmatrix()
    coord_T_x2 = T.dmatrix()

    # Euclidian distances function
    def squared_euclidean_distances(x_1, x_2):
        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        return sqd

    # Compiling function
    f = theano.function([coord_T_x1, coord_T_x2],
                        squared_euclidean_distances(coord_T_x1, coord_T_x2),
                        allow_input_downcast=False)
    return f


# This is extremily ineficient. Try to vectorize it in theano, it is possible to gain X100
def compute_variogram(df, properties, euclidian_distances, tol=10, lags=np.logspace(0, 2.5, 100), plot=[]):
    """
    Compute the experimental variogram and cross variogram for a par of properties
    Args:
        df (pandas.core.frame.DataFrame): Dataframe with the properties and coordinates used in the experimental
         variogram computation
        properties (list): List of the two properties to compute the semivariogram.
        euclidian_distances (numpy.array): Precomputed distances of the euclidian distances
        tol (float): Tolerance
        lags (list): List of lags to compute the experimental variogram
        plot (bool): If true plot the experimental variogram after computed

    Returns:
        list: semvariance aor cross-semivariance
    """

    # Tiling the properties to a square matrix
    element = (df[properties[0]].as_matrix().reshape(-1, 1) -
               np.tile(df[properties[1]], (df[properties[1]].shape[0], 1))) ** 2

    # Semivariance computation
    semivariance_lag = []

    # Looping every lag to compute the semivariance
    for i in lags:
        # Selecting the points at the given lag and tolerance
        points_at_lag = ((euclidian_distances > i - tol) * (euclidian_distances < i + tol))

        # Extracting the values of the properties of the selected lags
        var_points = element[points_at_lag]

        # Appending the semivariance
        semivariance_lag = np.append(semivariance_lag, np.mean(var_points) / 2)

    if "experimental" in plot:
        # Visualizetion of the experimental variogram
        plt.plot(lags, semivariance_lag, 'o')

    return semivariance_lag


def exp_lags(max_range, exp=2, n_lags=100):
    """
    Function to create a more specific exponential distance between the lags in case that log10 gives too much weight
    to the smaller lags
    Args:
        max_range(float): Maximum distance
        exp (float): Exponential degree
        n_lags (int): Number of lags

    Returns:
        list: lags

    """
    lags = np.empty(0)
    for i in range(n_lags):
        lags = np.append(lags, i ** exp)
    lags = lags / lags.max() * max_range
    return lags


def compute_crossvariogram(df, properties_names, euclidian_distances=None, **kwargs):
    """
    Compute the experimental crossvariogram of all properties given
    Args:
        df (pandas.core.frame.DataFrame): Dataframe with the properties and coordinates used in the experimental
         variogram computation
        properties_names (list str): List of strings with the properties to compute
        euclidian_distances (numpy.array): Precomputed euclidian distances. If None they are computed inplace
        Keyword Args:
            - lag_exp: Degree of the exponential. If None log10
            - lag_range: Maximum distance to compute a lag
            - n_lags: Number of lags

    Returns:
        pandas.core.frame.DataFrame: Every experimental cross-variogram
    """

    lag_exp = kwargs.get('lag_exp', None)
    lag_range = kwargs.get('lag_range', 500)
    n_lags = kwargs.get('n_lags', 100)

    # Choosing the lag array
    if lag_exp is not None:
        lags = exp_lags(lag_range, lag_exp, n_lags)
    else:
        lags = np.logspace(0, np.log10(lag_range), n_lags)

    # Compute euclidian distance
    if not euclidian_distances:
        euclidian_distances = theano_sed()(df[['X', 'Y', 'Z']], df[['X', 'Y', 'Z']])

    # Init dataframe to store the results
    experimental_variograms_frame = pn.DataFrame()

    # This is extremily ineficient. Try to vectorize it in theano, it is possible to gain X100
    # Nested loop. DEPRECATED enumerate
    for i in properties_names:
        for j in properties_names:
            col_name = i + '-' + j
            values = compute_variogram(df, [i, j], euclidian_distances, lags=lags)
            experimental_variograms_frame[col_name] = values

    # Add lags column for plotting mainly
    experimental_variograms_frame['lags'] = lags

    return experimental_variograms_frame


class SGS(object):
    """
    Class that contains the necessary methods to perform a sequential gaurssian simmulation from experimental variograms
    """
    def __init__(self, exp_var, properties=None,  n_exp=2, n_gauss=2,
                 data=None, grid=None, grid_to_inter=None, lithology=None, geomodel=None):
        """

        Args:
            exp_var (pandas.core.frame.DataFrame): Dataframe with the experimental variograms. Columns are the variables
            while raws the lags
            properties (list strings): Variables to be used
            n_exp (int): number of exponential basic functions
            n_gauss (int): number of gaussian basic functions
        """

        # Analytical covariance
        # ---------------------
        self.exp_var_raw = exp_var
        if not properties:
            self.properties = exp_var.columns[['ppm' in i for i in exp_var.columns]]
        else:
            self.properties = properties
        self.n_properties = len(self.properties)
        self.lags = self.exp_var_raw['lags']
        self.exp_var, self.nuggets = self.preprocess()

        self.n_exp = n_exp
        self.n_gauss = n_gauss
        self.trace = None

        # Kriging
        # -------
        self.data = data
        self.data_to_inter = None
        self.grid = grid
        self.grid_to_inter = grid_to_inter
        self.lithology = lithology
        self.geomodel = geomodel

        # Theano functions
        # ----------------
        self.SED_f = theano_sed()

        self._recursion_check = 0

    def select_segmented_grid(self, grid=None, df=None, litho=None, block=None):
        """
        Extract from the full model the points belonging to a given domain (litho)
        Args:
            grid (numpy.arrau): Full model grid
            df (pandas.core.frame.DataFrame): with all data
            litho (str): Name of the lithology to interpolate
            block (numpy.array): 3D block model generated by gempy

        Returns:

        """

        # If we do not pass the attributes but are already a property of the object extract it
        if grid is None:
            grid = self.grid

        if block is not None:
            block = np.squeeze(block)
        else:
            block = self.geomodel

        if df is None:
            df = self.data

        if litho is None:
            litho = self.lithology

        # Check that the attributes have value
        assert all([any(block), ~df.empty, litho, np.any(grid)]), "Some of the attributes are not provided"
        assert grid.shape[0] == block.shape[0], 'The grid you want to use for kriging and the grid used for the layers ' \
                                                'segmentation are not the same'

        # Translate lithology name to lithology number, i.e. the value of the lithology in the gempy block
        litho_num = df['Lithology Number'][df["Lithology"] == litho].iloc[0]

        # Extract from the grid the XYZ coordinates where the gempy block model is the given lithology
        segm_grid = grid[block == litho_num]

        # Set the segmented grid
        self.grid_to_inter = segm_grid

        return segm_grid

    def choose_lithology_elements(self, df=None, litho=None, elem=None, coord=True):
        """
        Choose from the whole input data, those points which fall into the domain of interest
        litho(str): Name of the lithology-domain
        elem(list): List of strings with elements you want to analyze
        """

        # If we do not pass the attributes but are already a property of the object extract it
        if not df:
            df = self.data

        if not litho:
            litho = self.lithology

        # Check that the attributes have value
        assert all([~df.empty, litho]), "Some of the attributes are not provided"

        # Choosing just one lithology
        if elem is not None:
            if coord:
                domain = df[df['Lithology'] == litho][np.append(['X', 'Y', 'Z'], elem)]
            else:
                domain = df[df['Lithology'] == litho][elem]
            # Drop negative values
            domain = domain[(domain[elem] > 0).all(1)]
        else:
            domain = df[df['Lithology'] == litho][['X', 'Y', 'Z']]

        # Set the segmented data
        self.data_to_inter = domain

        return domain

    def set_data(self, data):
        """
        Data setter
        Args:
            data (pandas.core.frame.DataFrame): Dataframe with coordinates and values of the properties of interest
        """
        self.data = data

    def set_grid_to_inter(self, grid):
        """
        Grid setter
        Args:
            grid (numpy.array): Grid object from gempy with the points of interest of the model
        """
        self.grid_to_inter = grid

    def set_lithology(self, lithology):
        """
        Lithology setter
        Args:
            lithology (str): Name of the domain or lithology of interest
        """
        self.lithology = lithology

    def set_geomodel(self, geomodel):
        """
        gempy model setter
        Args:
            geomodel (numpy.array): gempy block model
        """
        self.geomodel = geomodel

    def set_trace(self, trace):
        """
        Analytical covarinace trace setter
        Args:
            trace (pymc3.trace): trace with the sill, range and weights of each property
        """
        self.trace = trace

    def preprocess(self):
        """
        Normalization of data between 0 and 1 and subtraction of the nuggets
        Returns:
            pandas.core.frame.DataFrame: Dataframe containing the transformed data
            pandas.core.frame.DataFrame: Containing the substracted nuggets

        """
        import sklearn.preprocessing as skp

        # Normalization
        scaled_data = pn.DataFrame(skp.minmax_scale(self.exp_var_raw[self.properties]), columns=self.properties)

        # Nuggets
        nuggets = scaled_data[self.properties].iloc[0]
        processed_data = scaled_data - nuggets
        return processed_data, nuggets

    def plot_experimental(self, transformed=False):
        """
        Plotting the experimental data either transformed (see preprocess doc) or not
        Args:
            transformed (bool): if true plot the transformed data

        Returns:
            matplotlib.plot: plot of experimental variogram
        """

        if transformed:
            plot = self.exp_var.plot(x=self.lags, y=self.properties, subplots=True, kind ='line',
                                     style='.',
                                     layout=(int(np.sqrt(self.n_properties)), int(np.sqrt(self.n_properties))),
                                     figsize=(16, 8))
        else:
            plot = self.exp_var_raw.plot(x=self.lags, y=self.properties, subplots=True, kind='line',
                                         style='.',
                                         layout=(int(np.sqrt(self.n_properties)), int(np.sqrt(self.n_properties))),
                                         figsize=(16, 8))
        return plot

    def fit_cross_cov(self, n_exp=2, n_gauss=2, range_mu=None):
        """
        Fit an analytical covariance to the experimental data.
        Args:
            n_exp (int): number of exponential basic functions
            n_gauss (int): number of gaussian basic functions
            range_mu: prior mean of the range. Default mean of the lags

        Returns:
            pymc.Model: PyMC3 model to be sampled using MCMC
        """
        self.n_exp = n_exp
        self.n_gauss = n_gauss
        n_var = self.n_properties
        df = self.exp_var
        lags = self.lags

        # Prior standard deviation for the error of the regression
        prior_std_reg = df.std(0).max() * 10

        # Prior value for the mean of the ranges
        if not range_mu:
            range_mu = lags.mean()

        # pymc3 Model
        with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
            # Define priors
            sigma = pm.HalfCauchy('sigma', beta=prior_std_reg, testval=1., shape=n_var)

            psill = pm.Normal('sill', prior_std_reg, sd=.5 * prior_std_reg, shape=(n_exp + n_gauss))
            range_ = pm.Normal('range', range_mu, sd=range_mu * .3, shape=(n_exp + n_gauss))

            lambda_ = pm.Uniform('weights', 0, 1, shape=(n_var * (n_exp + n_gauss)))

            # Exponential covariance
            exp = pm.Deterministic('exp',
                                   # (lambda_[:n_exp*n_var]*
                                   psill[:n_exp] *
                                   (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)),
                                                     (range_[:n_exp].reshape((1, n_exp)) / 3.) ** -1))))

            gauss = pm.Deterministic('gaus',
                                     psill[n_exp:] *
                                     (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)) ** 2,
                                                       (range_[n_exp:].reshape((1, n_gauss)) * 4 / 7.) ** -2))))

            # We stack the basic functions in the same matrix and tile it to match the number of properties we have
            func = pm.Deterministic('func', T.tile(T.horizontal_stack(exp, gauss), (n_var, 1, 1)))

            # We weight each basic function and sum them
            func_w = pm.Deterministic("func_w", T.sum(func * lambda_.reshape((n_var, 1, (n_exp + n_gauss))), axis=2))

            for e, cross in enumerate(df.columns):
                # Likelihoods
                pm.Normal(cross + "_like", mu=func_w[e], sd=sigma[e], observed=df[cross].as_matrix())
        return model

    def plot_cross_variograms(self, iter_plot=200, trace=None, experimental=None):
        """
        Plot the analytical cross-variogram of a given MCMC inference
        Args:
            iter_plot (int): Number of traces to plot
            trace (pymc3.trace): trace with the sill, range and weights of each property
            experimental (bool): if True plot the experimental variogram as well

        Returns:
            None
        """

        if not trace:
            trace = self.trace
        assert trace, 'set the trace to the object'

        n_exp = self.n_exp
        n_gauss = self.n_gauss
        lags = self.lags
        # DEP- n_equations = trace['weights'].shape[1]
        n_iter = trace['weights'].shape[0]
        lags_tiled = np.tile(lags, (iter_plot, 1))
        b_var = []
        for i in range(0, self.n_properties):  # DEP- n_equations, (n_exp+n_gaus)):
            # Init tensor
            b = np.zeros((len(lags), n_iter, 0))
            for i_exp in range(0, n_exp):

                b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gauss)] *
                               exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
            for i_gauss in range(n_exp, n_gauss + n_exp):
                b = np.dstack((b, trace['weights'][:, i_gauss + i * (n_exp + n_gauss)] *
                               gaus_vario(lags, trace['sill'][:, i_gauss], trace['range'][:, i_gauss])))

            # Sum the contributins of each function
            b_all = b.sum(axis=2)
            # Append each variable
            b_var.append(b_all[:, -iter_plot:].T)

        # Bokeh code to plot this
        p_all = []
        for e, el in enumerate(self.properties):
            p = bp.figure()#x_axis_type="log")
            p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)
            if experimental:
                p.scatter(self.lags, y=self.exp_var[el], color='navy', size=2)
            p.title.text = el
            p.xaxis.axis_label = "lags"
            p.yaxis.axis_label = "Semivariance"

            p_all = np.append(p_all, p)

        grid = bl.gridplot(list(p_all), ncols=5, plot_width=250, plot_height=150)

        show(grid)

    def exp_vario(self, lags, sill, range_):
        """
        Vectorial computation of exponential variogram
        Args:
            lags (numpy.array): Distances
            sill (numpy.array): Array of sills. The shape will be number of properties by number of exponential
             basic functions
            range_ (numpy.array): Array of ranges. The shape will be number of properties by number of exponential
             basic functions

        Returns:
            numpy.array: Exponential variogram for every lag and every sill and range.
        """
        return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) * 3, range_.reshape(1, -1) ** -1)))

    def gaus_vario(self, lags, sill, range_):
        """
        Vectorial computation of gauss variogram
        Args:
            lags (numpy.array): Distances
            sill (numpy.array): Array of sills. The shape will be number of properties by number of gauss
             basic functions
            range_ (numpy.array): Array of ranges. The shape will be number of properties by number of gauss
             basic functions

        Returns:
            numpy.array: gauss variogram for every lag and every sill and range.
        """
        return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) ** 2, (range_.reshape(1, -1) * 4 / 7) ** -2)))

    def plot_cross_covariance(self, nuggets=False, iter_plot=200):
        """
        Plot the cross covariance for the given properties
        Args:
            nuggets (numpy.array): subtracted nuggets
            iter_plot (int): number of traces to plot

        Returns:
            None
        """
        n_exp = self.n_exp
        n_gauss = self.n_gauss
        trace = self.trace
        lags = self.lags

        n_equations = trace['weights'].shape[1]
        n_iter = trace['weights'].shape[0]
        lags_tiled = np.tile(lags, (iter_plot, 1))
        b_var = []
        for i in range(0, self.n_properties):  # n_equations, (n_exp+n_gaus)):
            # Init tensor
            b = np.zeros((len(lags), n_iter, 0))
            for i_exp in range(0, n_exp):
                # print(i_exp, "exp")
                b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gauss)] *
                               exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
            for i_gaus in range(n_exp, n_gauss + n_exp):
                # print(i_gaus)
                b = np.dstack((b, trace['weights'][:, i_gaus + i * (n_exp + n_gauss)] *
                               gaus_vario(lags, trace['sill'][:, i_gaus], trace['range'][:, i_gaus])))
            # Sum the contributins of each function
            if nuggets:
                b_all = 1 - (b.sum(axis=2) + self.nuggets[i])
            else:
                b_all = 1 - (b.sum(axis=2))
            # Append each variable
            b_var.append(b_all[:, -iter_plot:].T)

        p_all = []
        for e, el in enumerate(self.properties):
            p = bp.figure(x_axis_type="log")
            p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)

            p.title.text = el
            p.xaxis.axis_label = "lags"
            p.yaxis.axis_label = "Semivariance"

            p_all = np.append(p_all, p)

        grid = bl.gridplot(list(p_all), ncols=5, plot_width=250, plot_height=150)

        show(grid)

    def solve_kriging(self, selection_A, selection_b, #selected_coord_data, selected_values_data, selected_grid_to_inter, trace,
                      nuggets=None, n_var=1, n_exp=2, n_gauss=2):
        """
        Solve the kriging system for n given point of the grid_to_inter property selected by selection_b using n
        points of data_to_inter given by selection_A
        Args:
            selection_A (bool): input points used in A matrix
            selection_b (bool): points to interpolate from the grid
            nuggets (numpy.array): Nugget effect of each cross variogram. If None take property
            n_var: number of variables
            n_exp (int): number of exponential basic functions
            n_gauss (int): number of gaussian basic functions

        Returns:

        """

        # Select input data and compute its euclidean distances
        selected_coord_data = self.data_to_inter[selection_A][['X', 'Y', 'Z']]
        selected_values_data = self.data_to_inter[selection_A][self.data_to_inter.columns.difference(['X', 'Y', 'Z'])].as_matrix()
        h_A = self.SED_f(selected_coord_data, selected_coord_data)

        # Select points of grid to interpolate and compute the euclidean distances respect the input data
        selected_grid_to_inter = self.grid_to_inter[selection_b]
        h_b = self.SED_f(selected_coord_data, selected_grid_to_inter)

        # Parameters for the covariances
        trace = self.trace
        nuggets = self.nuggets
        n_var = int(np.sqrt(self.n_properties))
        n_exp = self.n_exp
        n_gauss = self.n_gauss

        # Choose a random trace to conserve the covariance uncertainty of the regression
        sample = np.random.randint(trace['weights'].shape[0] - 1000, trace['weights'].shape[0])

       # Compute cross-covariances
        cov_h = cross_covariance(trace, h_A, sample=sample,
                                 nuggets=nuggets, n_var=n_var, n_exp=n_exp, n_gaus=n_gauss, ordinary=True)
        cov_b = cross_covariance(trace, h_b, sample=sample, nuggets=nuggets, n_var=n_var, n_exp=n_exp,
                                 n_gaus=n_gauss,
                                 ordinary=True)

        # Solve kriging system
        k_weights = np.linalg.solve(cov_h, cov_b)

        # Number of points to interpolate
        npti = selected_grid_to_inter.shape[0]

        # Repeat the input data for every point to interpolate
        svd_tmp = np.tile(np.repeat(selected_values_data, npti, axis=1), (n_var, 1))

        # Sol ordinary kriging mean
        k_mean = (svd_tmp * k_weights[:-n_var]).sum(axis=0)

        # Sol ordinary kriging std
        k_std = svd_tmp.std(axis=0) - (k_weights * cov_b)[:-n_var].sum(axis=0) +\
                (k_weights * cov_b)[-n_var:].sum(axis=0)

        assert all(k_std) > -10, "A standard deviation of kringing is really off. Check nothing is wrong"
        # Set negatives to 0
        k_std[k_std < 0] = 0.1

        # Check the results make sense else take another sample and recompute
        import scipy
        l_low, l_high = scipy.stats.norm.interval(.95, loc=np.mean(selected_values_data, axis=0),
                                                  scale=np.std(selected_values_data, axis=0))

        if not np.all((k_mean > l_low) * (k_mean < l_high)):

            k_mean, k_std, cov_h, cov_b, k_weights, sample = self.solve_kriging(selection_A, selection_b)
            self._recursion_check += 1
            assert self._recursion_check<500, 'Too many recursions. Probably something goes wrong'

        else:
            self._recursion_check = 0
            values_interp = np.random.normal(k_mean, k_std)

        return k_mean, k_std, cov_h, cov_b, k_weights, sample  # , k_std# - svd_tmp.std(axis=0)


def fit_cross_cov(df, lags, n_exp=2, n_gaus=2, range_mu=None):
    n_var = df.columns.shape[0]
    n_basis_f = n_var * (n_exp + n_gaus)
    prior_std_reg = df.std(0).max() * 10
    #
    if not range_mu:
        range_mu = lags.mean()

    # Because is a experimental variogram I am not going to have outliers
    nugget_max = df.values.max()
    # print(n_basis_f, n_var*n_exp, nugget_max, range_mu, prior_std_reg)
    # pymc3 Model
    with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy('sigma', beta=prior_std_reg, testval=1., shape=n_var)

        psill = pm.Normal('sill', prior_std_reg, sd=.5 * prior_std_reg, shape=(n_exp + n_gaus))
        range_ = pm.Normal('range', range_mu, sd=range_mu * .3, shape=(n_exp + n_gaus))
        #  nugget = pm.Uniform('nugget', 0, nugget_max, shape=n_var)

        lambda_ = pm.Uniform('weights', 0, 1, shape=(n_var * (n_exp + n_gaus)))

        # Exponential covariance
        exp = pm.Deterministic('exp',
                               # (lambda_[:n_exp*n_var]*
                               psill[:n_exp] *
                               (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)),
                                                 (range_[:n_exp].reshape((1, n_exp)) / 3.) ** -1))))

        gaus = pm.Deterministic('gaus',
                                psill[n_exp:] *
                                (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)) ** 2,
                                                  (range_[n_exp:].reshape((1, n_gaus)) * 4 / 7.) ** -2))))

        func = pm.Deterministic('func', T.tile(T.horizontal_stack(exp, gaus), (n_var, 1, 1)))

        func_w = pm.Deterministic("func_w", T.sum(func * lambda_.reshape((n_var, 1, (n_exp + n_gaus))), axis=2))
        #           nugget.reshape((n_var,1)))

        for e, cross in enumerate(df.columns):
            # Likelihoods
            pm.Normal(cross + "_like", mu=func_w[e], sd=sigma[e], observed=df[cross].as_matrix())
    return model


def exp_vario(lags, sill, range_):
    return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) * 3, range_.reshape(1, -1) ** -1)))


def gaus_vario(lags, sill, range_):
    return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) ** 2, (range_.reshape(1, -1) * 4 / 7) ** -2)))


def plot_cross_variograms(trace, lags, df, n_exp=2, n_gaus=2, iter_plot=200, experimental=None):
    n_equations = trace['weights'].shape[1]
    n_iter = trace['weights'].shape[0]
    lags_tiled = np.tile(lags, (iter_plot, 1))
    b_var = []
    for i in range(0, df.shape[1]):  # n_equations, (n_exp+n_gaus)):
        # Init tensor
        b = np.zeros((len(lags), n_iter, 0))
        for i_exp in range(0, n_exp):
            # print(i_exp, "exp")
            b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gaus)] *
                           exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
        for i_gaus in range(n_exp, n_gaus + n_exp):
            # print(i_gaus)
            b = np.dstack((b, trace['weights'][:, i_gaus + i * (n_exp + n_gaus)] *
                           gaus_vario(lags, trace['sill'][:, i_gaus], trace['range'][:, i_gaus])))
        # Sum the contributins of each function
        b_all = b.sum(axis=2)
        # Append each variable
        b_var.append(b_all[:, -iter_plot:].T)

    p_all = []
    for e, el in enumerate(df.columns):
        p = bp.figure(x_axis_type="log")
        p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)
        if experimental is not None:
            p.scatter(experimental['lags'], y=experimental[el], color='navy', size=2)
        p.title.text = el
        p.xaxis.axis_label = "lags"
        p.yaxis.axis_label = "Semivariance"

        p_all = np.append(p_all, p)

    grid = bl.gridplot(list(p_all), ncols=5, plot_width=200, plot_height=150)

    show(grid)


def plot_cross_covariance(trace, lags, df, n_exp=2, n_gaus=2, nuggets=None, iter_plot=200):
    n_equations = trace['weights'].shape[1]
    n_iter = trace['weights'].shape[0]
    lags_tiled = np.tile(lags, (iter_plot, 1))
    b_var = []
    for i in range(0, df.shape[1]):  # n_equations, (n_exp+n_gaus)):
        # Init tensor
        b = np.zeros((len(lags), n_iter, 0))
        for i_exp in range(0, n_exp):
            # print(i_exp, "exp")
            b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gaus)] *
                           exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
        for i_gaus in range(n_exp, n_gaus + n_exp):
            # print(i_gaus)
            b = np.dstack((b, trace['weights'][:, i_gaus + i * (n_exp + n_gaus)] *
                           gaus_vario(lags, trace['sill'][:, i_gaus], trace['range'][:, i_gaus])))
        # Sum the contributins of each function
        if nuggets is not None:
            b_all = 1 - (b.sum(axis=2) + nuggets[i])
        else:
            b_all = 1 - (b.sum(axis=2))
        # Append each variable
        b_var.append(b_all[:, -iter_plot:].T)

    p_all = []
    for e, el in enumerate(df.columns):
        p = bp.figure(x_axis_type="log")
        p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)

        p.title.text = el
        p.xaxis.axis_label = "lags"
        p.yaxis.axis_label = "Semivariance"

        p_all = np.append(p_all, p)

    grid = bl.gridplot(list(p_all), ncols=5, plot_width=250, plot_height=150)

    show(grid)


def cross_covariance(trace, sed, sample, nuggets=None, n_var=1, n_exp=2, n_gaus=2, ordinary=True):
    """

    Args:
        trace:
        sed:
        nuggets:
        n_var:
        n_exp:
        n_gaus:
        ordinary:

    Returns:

    """
    h = np.ravel(sed)
    n_points = len(h)
    n_points_r = sed.shape[0]
    n_points_c = sed.shape[1]
    #sample = np.random.randint(trace['weights'].shape[0]-200, trace['weights'].shape[0])
    n_eq = trace['weights'].shape[1]
    # Exp contribution
    exp_cont = (np.tile(
        exp_vario(h, trace['sill'][sample][:n_exp], trace['range'][sample][:n_exp]),
        n_var**2
    ) * trace['weights'][sample][np.linspace(0, n_eq-1, n_eq) % (n_exp + n_gaus) < n_exp]).reshape(n_points, n_exp, n_var**2, order="F")

    # Gauss contribution
    gaus_cont = (np.tile(
        gaus_vario(h, trace['sill'][sample][n_exp:], trace['range'][sample][n_exp:]),
        n_var**2
    ) * trace['weights'][sample][np.linspace(0, n_eq-1, n_eq) % (n_exp + n_gaus) >= n_exp]).reshape(n_points, n_gaus, n_var**2, order="F")

    # Stacking and summing
    conts = np.hstack((exp_cont, gaus_cont)).sum(axis=1)

    if nuggets is not None:
        conts += nuggets

    cov = 1 - conts

    cov_str = np.zeros((0, n_points_c*n_var))

    cov_aux = cov.reshape(n_points_r, n_points_c, n_var**2, order='F')

    # This is a incredibly convoluted way to reshape the cross covariance but I did not find anything better
    for i in range(n_var):
        cov_str = np.vstack((cov_str,
                             cov_aux[:, :, i*n_var:(i+1)*n_var].reshape(n_points_r, n_points_c*n_var, order='F')))

    if ordinary:
        ord = np.zeros((n_var, n_var*n_points_c+n_var))
        for i in range(n_var):
            ord[i, n_points_c * i:n_points_c * (i + 1)] = np.ones(n_points_c)
        cov_str = np.vstack((cov_str, ord[:, :-n_var]))

        # Stack the ordinary values to the C(h)
        if n_points_r == n_points_c:
             cov_str = np.hstack((cov_str, ord.T))

    return cov_str


def clustering_grid(grid_to_inter, n_clusters=50, plot=False):
    from sklearn.cluster import KMeans
    clust = KMeans(n_clusters=n_clusters).fit(grid_to_inter)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 1], clust.cluster_centers_[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    return clust


def select_points(df, grid_to_inter, cluster, SED_f = theano_sed(), n_rep=10):

    points_cluster = np.bincount(cluster.labels_)
  #  SED_f = theano_sed()

    for i in range(n_rep):
        for i_clust in range(cluster.n_clusters):
            cluster_bool = cluster.labels_ == i_clust
            cluster_grid = grid_to_inter[cluster_bool]
            # Mix the values of each cluster
            if i is 0:
                np.random.shuffle(cluster_grid)

            size_range = int(points_cluster[i_clust]/n_rep)
            selected_cluster_grid = cluster_grid[i * size_range:(i + 1) * size_range]
            dist = SED_f(df, selected_cluster_grid)

            # Checking the radio of the simulation
            for r in range(100, 1000, 100):
                select = (dist < r).any(axis=1)
                if select.sum() > 50:
                    break

            h_x0 = dist
            yield (h_x0, select, selected_cluster_grid)


def SGS_compute_points(selected_coord_data, selected_grid_to_inter, selected_values_data,
                trace, nuggets=None, n_var=1, n_exp=2, n_gaus=2):
    #SED_f = theano_sed()


    #SED = SED_f(selected_coord_data, selected_coord_data)

    npti = selected_grid_to_inter.shape[1]

    cov_h = cross_covariance(trace, selected_coord_data,
                             nuggets=nuggets, n_var=n_var, n_exp=n_exp, n_gaus=n_gaus, ordinary=True)
    cov_b = cross_covariance(trace, selected_grid_to_inter, nuggets=nuggets, n_var=n_var, n_exp=n_exp, n_gaus=n_gaus,
                             ordinary=True)

    k_weights = np.linalg.solve(cov_h, cov_b)
    svd_tmp = np.tile(np.repeat(selected_values_data, npti, axis=1), (n_var, 1))

    # Sol ordinary kriging mean
    k_mean = (svd_tmp * k_weights[:-n_var]).sum(axis=0)

    # Sol ordinary kriging std
    k_std = svd_tmp.std(axis=0) - (k_weights * cov_b)[:-n_var].sum(axis=0) + (k_weights * cov_b)[-n_var:].sum(axis=0)

    assert all(k_std) > -10, "A standard deviation of kringing is really off. Check nothing is wrong"

    # Set negatives to 0
    k_std[k_std < 0] = 0.1

    values_interp = np.random.normal(k_mean, k_std)
  #  for point in range(npti-1):
   #     values_data = np.vstack((values_data, values_interp[point::npti]))

    #coord_data = np.vstack((coord_data, grid_interpolating))
    return values_interp#, k_std# - svd_tmp.std(axis=0)


def SGS_run(df, grid_to_inter, cluster,
            trace, nuggets=None, n_var=1, n_exp=2, n_gaus=2,
            n_rep=10, verbose = 0):

    points_cluster = np.bincount(cluster.labels_)
    coord_data = df[['X', 'Y', 'Z']].as_matrix()
    values_data = df[df.columns.difference(['X', 'Y', 'Z'])].as_matrix()

    SED_f = theano_sed()

    for i in range(n_rep):
        for i_clust in range(cluster.n_clusters):
            cluster_bool = cluster.labels_ == i_clust
            cluster_grid = grid_to_inter[cluster_bool]
            # Mix the values of each cluster
           # if i is 0:
           #     np.random.shuffle(cluster_grid)

            size_range = int(points_cluster[i_clust]/n_rep)

            # Select points where interpolate
            selected_cluster_grid = cluster_grid[i * size_range:(i + 1) * size_range]
            npti = selected_cluster_grid.shape[0]

            # Euclidiand distances
            h_x0 = SED_f(coord_data, selected_cluster_grid)
            # Drop any point already simulated
            #print((h_x0==0).sum())
          #  h_x0 = h_x0[~np.any(h_x0 == 0, axis=1)]
            h_xi = SED_f(coord_data, coord_data)

            # Checking the radio of the simulation
            for r in range(50, 1000, 1):
                select = (h_x0 < r).any(axis=1)
                if select.sum() > 50:
                    break
            if verbose > 2:
                print("sel", select.shape)
                print("val", values_data.shape)
            if verbose > 0:
                print("Number of points used and number of points to interpolate", select.sum(), npti)
            h_xi_sel = h_xi[select][:, select]
            h_x0_sel = h_x0[select]
            values_data_sel = values_data[select]

            values_interpolated = SGS_compute(h_xi_sel, h_x0_sel, values_data_sel,
                                              trace, nuggets, n_var, n_exp, n_gaus)

            # Append the coordinates of the interpolated values to the values coord data since they will be interpolated
            coord_data = np.vstack((coord_data, selected_cluster_grid))

            # Setting negative values to 0
            values_data[values_data < 0] = 0

            # Append interpolated values to the initial values
            for point in range(npti-1):
                values_data = np.vstack((values_data, values_interpolated[point::npti]))

    return coord_data, values_data

# def SGS(coord_data, values_data, h_x0, select,
#         trace, nuggets=None, n_var=1, n_exp=2, n_gaus=2):
#     SED_f = theano_sed()
#
#     #for h_x0, select, grid_interpolating in selector:
#
#     SED = SED_f(coord_data, coord_data)
#
#     values_interp, npti = SGS_compute(SED[select], h_x0[select], values_data[select],
#                                       trace, nuggets, n_var, n_exp, n_gaus)
#
#     for point in range(npti-1):
#         values_data = np.vstack((values_data, values_interp[point::npti]))
#
#     coord_data = np.vstack((coord_data, grid_interpolating))

