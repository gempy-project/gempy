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
                domain (np.array)(x,) = array containing all surfaces of interest from the gempy model that
                                        the operation should be performed in
                data (np.array)(x,4)  = array of input data (conditioning) with [:,0]=x coordinate, [:,1]=y coordinate
                                        [:,2]=z coordinate and [:,3]=value of measured property
                kriging_type (string) = string to define type of kriging type used (OK = ordinary kriging, SK = simple
                                        kriging, UK = universal kriging)
                distance type (string)= string to define distance type used (euclidian only option as of now)
                variogram_model       = ??? not sure how to best define this as of now.
                                        Should be allowed to enter own function or choose one of predefined set.
        '''
        #set model from a gempy solution
        self.sol = model

        # set kriging surfaces, basically in which lithologies to do all this, default is everything
        # TODO: Maybe also allwo to pass a gempy regular grid object
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

        # TODO: Better way to managa whole variogram stuff
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
        mask = np.isin(self.sol.lith_block, self.domain)

        # Apply mask to lith_block and grid
        self.krig_lith = self.sol.lith_block[mask]
        self.krig_grid = self.sol.grid.values[mask]

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

    def exponential_variogram_model(self, d):
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-(np.absolute(d) / (self.range_)))) + self.nugget
        return gamma

    def variogram_model():
        # define a model for the spatial correlation of the estiamted process, by:
        # a) default estimating it from data with a certain theoretical model
        # b) choosing a predefined variogram function from included set of models
        # c) allowing to set parameters (range, sill, nugget) manually
        # d) allowing user to enter different function on their own

        # Bonus: directly calcualte covariance model based on this
        return None

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

        # TODO: find way to check quality of matrix and solutions for instability
        # Solve Kriging equations
        w = np.linalg.solve(C, c)

        # calculating estiamte and variance for kriging
        pred_var = w[shape] + np.sum(w[:shape] * c[:shape])
        result = np.sum(w[:shape] * prop)

        return result, pred_var

    def create_kriged_field(self):

        # perform kriging allowing for
        # a) Simple Kriging
        # b) Ordinary Kriging
        # (c) Universal Kriging)
        # and later maybe more options

        # empty arrays for results (estimated values and variances)
        self.kriging_result_vals = np.zeros(len(self.krig_grid))
        self.kriging_result_vars = np.zeros(len(self.krig_grid))

        # - Start with distance calculation
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

            # elif self.kriging_type == 'SK':
            # val, var = self.simple_kriging(a, b, prop)

            # elif self.kriging_type == 'UK':
            # val, var = self.universal_kriging(a, b, prop)

            else:
                print("FATAL ERROR: Kriging type not understood")

            # STEP 3: Save results
            self.kriging_result_vals[i] = val
            self.kriging_result_vars[i] = var

            # STEP 4: Create results dataframe:

        # create dataframe of input data for calling
        d = {'X': self.krig_grid[:, 0], 'Y': self.krig_grid[:, 1], 'Z': self.krig_grid[:, 2],
             'est_value': self.kriging_result_vals, 'est_variance': self.kriging_result_vars}

        self.results_df = pd.DataFrame(data=d)

    def SGS():
        # perform SGS with same options as kriging

        # add data locations to grid locations

        # set random path through all unknown locations

        # - distance calculation
        # 1) all points to all points in order of path
        # 2) known locations at beginning?

        # perform SGS
        # allow for moving neighbourhood:
        # a) limited number of closest points
        # b) only points within range
        # - here this will be a little trickier (as input data updates)

        # set options for no starting points (Gaussian field) - mean and variance

        # return kriging estimate and variance
        return None

    def plot_results():
        # probably set of functions for visualization
        # some brainstorming:
        # 1) 3D and 2D options for domain with colormaps of property
        # 2) options to switch between smooth contour plot and pixelated
        # 3) options to show only stuff in certain layer and above certain cut-off grade
        # 4) options to plot variances
        # ...
        return None
