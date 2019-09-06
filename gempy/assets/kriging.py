"""
This file is part of gempy.

Created on 07.08.2019

@author: Jan von Harten
"""

import warnings
try:
    from scipy.spatial.distance import cdist
except ImportError:
    warnings.warn('scipy.spatial package is not installed.')

import numpy as np
import pandas as pd
from gempy.plot import visualization_2d, plot, helpers
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class kriging_model(object):

    # some kind of _init_ function ? how exactly, what should be done, set all defaults?

    # properties of class:
    # domain (grid, lith-block, scalar field, resolution)
    # data (coordinates and values) - allow for csv to be loaded
    # variogram model + parameters (range, sill, nugget, whatever else)
    # Kriging type (default OK, SK, maybe UK?)
    # distance metric (default euclidian, maybe later non-euclidian)
    # moving neighbourhood (n closest = 20, within range, none (dangerous))
    # flag if calculation is up-to-date (boolean) - if parameters were changed without recalculation

    def __init__(self, model, domain=None, data=None, kriging_type=None,
                 distance_type=None, variogram_model=None, moving_neighbourhood=None):
        '''
            Args:
                model (gempy.core.solution.Solution) = solution of a gempy model
                domain (np.array)(x,)                = array containing all surfaces of interest from the gempy model that
                                                       the operation should be performed in
                data (np.array)(x,4)                 = array of input data (conditioning) with [:,0]=x coordinate, [:,1]=y coordinate
                                                       [:,2]=z coordinate and [:,3]=value of measured property
                kriging_type (string)                = string to define type of kriging type used (OK = ordinary kriging, SK = simple
                                                       kriging, UK = universal kriging)
                distance type (string)               = string to define distance type used
                                                       (euclidian only option as of now)
                variogram_model                      = ??? not sure how to best define this as of now.
                                                       Should be allowed to enter own function or choose one of predefined set.
                moving_neighbourhood(string)         = string containing type of moving neighbourhood
                                                       (either n_closest, range or all)
        '''
        #set model from a gempy solution
        # TODO: Check if I actually need all this or if its easier to just get grid and lith of the solution
        self.sol = model

        # set kriging surfaces, basically in which lithologies to do all this, default is everything
        # TODO: Maybe also allow to pass a gempy regular grid object
        if domain is None:
            domain = np.unique(self.sol.lith_block)
        self.set_domain(domain)

        # set data, default is None
        # TODO: need to figure out a way to then set mean and variance for the SGS and SK
        if data is None:
            data = None
        self.set_data(data)

        # basic statistics of data
        # TODO: allow to set this  for SK ???
        self.inp_mean = np.mean(data[:, 3])
        self.inp_var = np.var(data[:, 3])
        self.inp_std = np.sqrt(self.inp_var)

        # set default kriging type to OK
        if kriging_type is None:
            kriging_type = 'OK'
        self.set_kriging_type(kriging_type)

        # set default distance type to euclidian
        if distance_type is None:
            distance_type = 'euclidian'
        self.set_distance_type(distance_type)

        # set default moving neigghbourhood
        if moving_neighbourhood is None:
            moving_neighbourhood = 'n_closest'
        self.set_moving_neighbourhood(moving_neighbourhood, n_closest_points=20)

        # TODO: Better way to manage whole variogram stuff
        if variogram_model is None:
            variogram_model = 'exponential'
        self.variogram_model = variogram_model

        # preset as of now
        self.range_ = 100
        self.sill = 10
        self.nugget = 1

    def set_domain(self, domain):
        """
        Method to cut domain by array of surfaces. Simply masking the lith_block with array of input lithologies
        applying mask to grid.
        Args:
            domain (np.array)(x,) = array containing all surfaces of interest from the gempy model that
                                        the operation should be performed in
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        # set domain to variable of class
        self.domain = domain

        # mask by array of input surfaces (by id, can be from different series)
        self.mask = np.isin(self.sol.lith_block, self.domain)

        # Apply mask to lith_block and grid
        self.krig_lith = self.sol.lith_block[self.mask]
        self.krig_grid = self.sol.grid.values[self.mask]

    def set_data(self, data):
        """
        Method to set input data from csv or numpy array.
        Args:
            data (np.array)(x,4)  = array of input data (conditioning) with [:,0]=x coordinate, [:,1]=y coordinate
                                        [:,2]=z coordinate and [:,3]=value of measured property
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        # set domain to variable of class
        self.data = data

        # create dataframe of input data for calling
        d = {'X': data[:, 0], 'Y': data[:, 1], 'Z': data[:, 2], 'property': data[:, 3]}
        self.data_df = pd.DataFrame(data=d)

    def set_kriging_type(self, kriging_type):
        """
        Method to choose kriging type.
        Args:
            kriging_type(string)  = string containing kriging type (either "OK", "SK" or "UK")
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        if kriging_type in ('SK', 'OK', 'UK'):
            self.kriging_type = kriging_type
        else:
            print("Kriging type not understood, choose either SK, OK or UK - defaulted to OK")
            self.kriging_type = 'OK'

    def set_distance_type(self, distance_type):
        """
        Method to choose distance type.
        Args:
            distance_type(string)  = string containing distance type (atm only "euclidian")
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        if distance_type in ('euclidian'):
            self.distance_type = distance_type
        else:
            print("Distance type not understood - defaulted to euclidian (only supported optin as of now)")
            self.distance_type = 'euclidian'

    def set_moving_neighbourhood(self, moving_neighbourhood, n_closest_points=None):
        """
        Method to choose type and extent of moving neighbourhood.
        Args:
            moving_neighbourhood(string)  = string containing type of moving neighbourhood
            (either n_closest, range or all)
            n_closest_points (int)        = number of considered closest points for n_closest (defaults to 20)
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        if moving_neighbourhood in ('n_closest', 'range', 'all'):
            self.moving_neighbourhood = moving_neighbourhood
        else:
            print("Moving neighbourhood type not understood - defaulted to 20 closest points")
            self.moving_neighbourhood = 'n_closest'

        if n_closest_points is not None:
            self.n_closest_points = n_closest_points

    # seems better now by changing psill in covariance model
    def exponential_variogram_model(self, d):
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-(np.absolute(d) / (self.range_)))) + self.nugget
        return gamma

    def exponential_covariance_model(self, d):
        psill = self.sill - self.nugget
        cov = psill * (np.exp(-(np.absolute(d) / (self.range_))))
        return cov

    def variogram_model():
        # define a model for the spatial correlation of the estiamted process, by:
        # a) default estimating it from data with a certain theoretical model
        # b) choosing a predefined variogram function from included set of models
        # c) allowing to set parameters (range, sill, nugget) manually
        # d) allowing user to enter different function on their own

        # Bonus: directly calcualte covariance model based on this
        return None

    # TODO: check with new ordianry kriging and nugget effect
    def simple_kriging(self, a, b, prop):
        '''
        Method for simple kriging calculation.
        Args:
            a (np.array): distance matrix containing all distances between target point and moving neighbourhood
            b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
            prop (np.array): array containing scalar property values of locations in moving neighbourhood
        Returns:
            result (float?): single scalar property value estimated for target location
            std_ok (float?): single scalar variance value for estimate at target location
        '''

        # empty matrix building
        shape = len(a)
        C = np.zeros((shape, shape))
        c = np.zeros((shape))
        w = np.zeros((shape))

        # Filling matrices with covariances based on calculated distances
        C[:shape, :shape] = self.exponential_covariance_model(b)
        c[:shape] = self.exponential_covariance_model(a)

        # nugget effect for simple kriging - dont remember why i set this actively, should be the same
        #np.fill_diagonal(C, self.sill)

        # TODO: find way to check quality of matrix and solutions for instability
        # Solve Kriging equations
        w = np.linalg.solve(C, c)

        # calculating estimate and variance for kriging
        pred_var = self.sill - np.sum(w * c)
        result = self.inp_mean + np.sum(w * (prop - self.inp_mean))

        return result, pred_var

    def ordinary_kriging(self, a, b, prop):
        '''
        Method for ordinary kriging calculation.
        Args:
            a (np.array): distance matrix containing all distances between target point and moving neighbourhood
            b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
            prop (np.array): array containing scalar property values of locations in moving neighbourhood
        Returns:
            result (float?): single scalar property value estimated for target location
            std_ok (float?): single scalar variance value for estimate at target location
        '''

        # empty matrix building for OK
        shape = len(a)
        C = np.zeros((shape + 1, shape + 1))
        c = np.zeros((shape + 1))
        w = np.zeros((shape + 1))

        # filling matirces based on model for spatial correlation
        C[:shape, :shape] = self.exponential_variogram_model(b)
        c[:shape] = self.exponential_variogram_model(a)

        # matrix setup - compare pykrige, special for OK
        np.fill_diagonal(C, 0)  # this needs to be done as semivariance for distance 0 is 0 by definition
        C[shape, :] = 1.0
        C[:, shape] = 1.0
        C[shape, shape] = 0.0
        c[shape] = 1.0

        # This is if we want exact interpolator
        # but be aware that it strictly forces estimates to go through data points
        # c[c == self.nugget] = 0

        # TODO: find way to check quality of matrix and solutions for instability
        # Solve Kriging equations
        w = np.linalg.solve(C, c)

        # calculating estimate and variance for kriging
        pred_var = w[shape] + np.sum(w[:shape] * c[:shape])
        result = np.sum(w[:shape] * prop)

        return result, pred_var

    def create_kriged_field(self):
        '''
        Method to create a kriged field over the defined grid of the gempy solution depending on the defined
        input data (conditioning).
        Returns:
             self.results_df (pandas dataframe):   Dataframe containing coordinates, kriging estimate
                                                    and kriging variance for each grid point
        '''
        # empty arrays for results (estimated values and variances)
        self.kriging_result_vals = np.zeros(len(self.krig_grid))
        self.kriging_result_vars = np.zeros(len(self.krig_grid))

        # Start with distance calculation
        # 1) all grid points to all data points
        # 2) all data points among each other
        if self.distance_type == 'euclidian':
            # calculate distances between all input data points
            dist_all_to_all = cdist(self.data[:, :3], self.data[:, :3])
            # calculate distances between all grid points and all input data points
            dist_grid_to_all = cdist(self.krig_grid, self.data[:, :3])

        # Main loop that goes through whole domain (grid)
        for i in range(len(self.krig_grid)):

            # STEP 1: Multiple if elif conditions to define moving neighbourhood:
            if self.moving_neighbourhood == 'all':
                # cutting matrices and properties based on moving neighbourhood
                a = dist_grid_to_all[i]
                b = dist_all_to_all
                prop = self.data[:, 3]

            elif self.moving_neighbourhood == 'n_closest':
                # cutting matrices and properties based on moving neighbourhood
                a = np.sort(dist_grid_to_all[i])
                a = a[:self.n_closest_points]
                aux = np.argsort(dist_grid_to_all[i])
                prop = self.data[:, 3][aux]
                prop = prop[:self.n_closest_points]
                aux = aux[:self.n_closest_points]
                b = dist_all_to_all[np.ix_(aux, aux)]

            elif self.moving_neighbourhood == 'range':
                # cutting matrices and properties based on moving neighbourhood
                aux = np.where(dist_grid_to_all[i] <= self.range_)[0]
                a = dist_grid_to_all[i][aux]
                prop = self.data[:, 3][aux]
                b = dist_all_to_all[np.ix_(aux, aux)]

            else:
                print("FATAL ERROR: Moving neighbourhood not understood")

            # STEP 2: Multiple if elif conditions to calculate kriging at point
            if self.kriging_type == 'OK':
                val, var = self.ordinary_kriging(a, b, prop)
            elif self.kriging_type == 'SK':
                val, var = self.simple_kriging(a, b, prop)
            elif self.kriging_type == 'UK':
                print("Universal Kriging not implemented")
            else:
                print("FATAL ERROR: Kriging type not understood")

            # STEP 3: Save results
            self.kriging_result_vals[i] = val
            self.kriging_result_vars[i] = var

        # create dataframe of results data for calling
        d = {'X': self.krig_grid[:, 0], 'Y': self.krig_grid[:, 1], 'Z': self.krig_grid[:, 2],
             'est_value': self.kriging_result_vals, 'est_variance': self.kriging_result_vars}

        self.results_df = pd.DataFrame(data=d)

    def create_gaussian_field(self):
        '''
        Method to create a kriged field over the defined grid of the gempy solution depending on the defined
        input data (conditioning).
        Returns:
            self.results_df (pandas dataframe):   Dataframe containing coordinates, kriging estimate
                                                            and kriging variance for each grid point
        '''
        # perform SGS with same options as kriging
        # TODO: set options for no starting points (Gaussian field) - mean and variance

        # set random path through all unknown locations
        shuffled_grid = self.krig_grid
        np.random.shuffle(shuffled_grid)

        # append shuffled grid to input locations
        sgs_locations = np.vstack((self.data[:,:3],shuffled_grid))
        # create array for input properties
        sgs_prop_updating = self.data[:,3] # use this and then always stack new ant end

        # container for estimation variances
        estimation_var = np.zeros(len(shuffled_grid))

        # - distance calculation (stays the same)
        # 1) all points to all points in order of path
        # 2) known locations at beginning?
        if self.distance_type == 'euclidian':
            # calculate distances between all input data points
            dist_all_to_all = cdist(sgs_locations, sgs_locations)

        # set counter og active data (start=input data, grwoing by 1 newly calcualted point each run)
        active_data = len(sgs_prop_updating)

        # Main loop that goes through whole domain (grid)
        for i in range(len(self.krig_grid)):
            # STEP 1: cut update distance matrix to correct size
            # HAVE TO CHECK IF THIS IS REALLY CORRECT
            active_distance_matrix = dist_all_to_all[:active_data,:active_data]
            active_distance_vector = dist_all_to_all[:,active_data] #basically next point to be simulated
            active_distance_vector = active_distance_vector[:active_data] #cut to left or diagonal

            # TODO: NEED PART FOR ZERO INPUT OR NO POINTS IN RANGE OR LESS THAN N POINTS

            # STEP 2: Multiple if elif conditions to define moving neighbourhood:
            if self.moving_neighbourhood == 'all':
                # cutting matrices and properties based on moving neighbourhood
                a = active_distance_vector
                b = active_distance_matrix
                prop = sgs_prop_updating

            elif self.moving_neighbourhood == 'n_closest':
                # cutting matrices and properties based on moving neighbourhood

                # This seems to work
                if len(sgs_prop_updating) <= self.n_closest_points:
                    a = active_distance_vector[:active_data]
                    b = active_distance_matrix[:active_data,:active_data]
                    prop = sgs_prop_updating

                # this does not # DAMN THIS STILL HAS ITSELF RIGHT? PROBLEM!
                else:
                    a = np.sort(active_distance_vector)
                    a = a[:self.n_closest_points]
                    aux = np.argsort(active_distance_vector)
                    prop = sgs_prop_updating[aux]
                    prop = prop[:self.n_closest_points]
                    aux = aux[:self.n_closest_points]
                    b = active_distance_matrix[np.ix_(aux, aux)]

            elif self.moving_neighbourhood == 'range':
                # cutting matrices and properties based on moving neighbourhood
                aux = np.where(active_distance_vector <= self.range_)[0]
                a = active_distance_vector[aux]
                prop = sgs_prop_updating[aux]
                b = active_distance_matrix[np.ix_(aux, aux)]

            else:
                print("FATAL ERROR: Moving neighbourhood not understood")

            # STEP 3: Multiple if elif conditions to calculate kriging at point
            if self.kriging_type == 'OK':
                val, var = self.ordinary_kriging(a, b, prop)
            elif self.kriging_type == 'SK':
                val, var = self.simple_kriging(a, b, prop)
            elif self.kriging_type == 'UK':
                print("Universal Kriging not implemented")
            else:
                print("FATAL ERROR: Kriging type not understood")

            # STEP 4: Draw from random distribution
            std_ = np.sqrt(var)
            estimate = np.random.normal(val, scale=std_)

            # append to prop:
            sgs_prop_updating = np.append(sgs_prop_updating, estimate)
            estimation_var[i]= var

            # at end of loop: include simulated point for next step
            active_data += 1

        # delete original input data from results
        simulated_prop = sgs_prop_updating[len(self.data[:,3]):] # check if this works like intented

        # create dataframe of results data for calling
        d = {'X': shuffled_grid[:, 0], 'Y': shuffled_grid[:, 1], 'Z': shuffled_grid[:, 2],
             'sim_value': simulated_prop, 'est_variance': estimation_var}

        self.results_sim_df = pd.DataFrame(data=d)

    def plot_results_dep():
        # probably set of functions for visualization
        # some brainstorming:
        # 1) 3D and 2D options for domain with colormaps of property
        # 2) options to switch between smooth contour plot and pixelated
        # 3) options to show only stuff in certain layer and above certain cut-off grade
        # 4) options to plot variances
        # ...
        return None

    def plot_results(self, geo_data, prop='val', direction='y', cell_number=0, contour=False,
                     cmap='viridis', alpha=0, legend=False):
        """
        TODO WRITE DOCSTRING
        Args:
            geo_data:
            prop:
            direction:
            cell_number:
            contour:
            cmap:
            alpha:
            legend:

        Returns:

        """
        a = np.full_like(self.mask, np.nan, dtype=np.double)
        est_vals = self.results_df['est_value'].values
        est_var = self.results_df['est_variance'].values

        if prop == 'val':
            a[np.where(self.mask == True)] = est_vals
        elif prop == 'var':
            a[np.where(self.mask == True)] = est_var
        elif prop == 'both':
            a[np.where(self.mask == True)] = est_vals
            b = np.full_like(self.mask, np.nan, dtype=np.double)
            b[np.where(self.mask == True)] = est_var
        else:
            print('prop must be val var or both')

        p = visualization_2d.PlotSolution(geo_data)
        _a, _b, _c, extent_val, x, y = p._slice(direction, cell_number)[:-2]

        cmap = cm.get_cmap(cmap)
        cmap.set_bad(color='w', alpha=alpha)

        if prop is not 'both':

            plot.plot_section(geo_data, direction=direction, cell_number=cell_number)
            if contour == True:
                im = plt.contourf(a.reshape(self.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                  origin='lower',
                                  extent=extent_val)
            else:
                im = plt.imshow(a.reshape(self.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                origin='lower',
                                extent=extent_val)
                if legend:
                    helpers.add_colorbar(im, location='right')

        else:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].title.set_text('Estimated value')
            im1 = ax[0].imshow(a.reshape(self.sol.grid.regular_grid.resolution)[:, 2, :].T, cmap=cmap,
                               origin='lower',
                               extent=self.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im1)
            ax[1].title.set_text('Variance')
            im2 = ax[1].imshow(b.reshape(self.sol.grid.regular_grid.resolution)[:, 2, :].T, cmap=cmap,
                               origin='lower',
                               extent=self.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im2)
            plt.tight_layout()
