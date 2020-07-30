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
from gempy.plot import _visualization_2d, _plot, helpers
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from copy import deepcopy

class domain(object):

    def __init__(self, model, domain=None, data=None, set_mean=None):

        # set model from a gempy solution
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
            data = None  # why do you do this, data is none already if it is none?
        self.set_data(data)

        # basic statistics of data
        # TODO: allow to set this  for SK ???
        if set_mean is None:
            set_mean = np.mean(data[:, 3])
        self.inp_mean = set_mean

        self.inp_var = np.var(data[:, 3])
        self.inp_std = np.sqrt(self.inp_var)

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


class variogram_model(object):

    # class containing all the variogram functionality

    def __init__(self, theoretical_model=None, range_=1, sill=1, nugget=0):

        if theoretical_model is None:
            theoretical_model = 'exponential'
        self.theoretical_model = theoretical_model

        # default
        self.range_ = range_
        self.sill = sill
        self.nugget = nugget

    def calculate_semivariance(self, d):

        if self.theoretical_model == 'exponential':
            gamma = self.exponential_variogram_model(d)
        elif self.theoretical_model == 'gaussian':
            gamma = self.gaussian_variogram_model(d)
        elif self.theoretical_model == 'spherical':
            gamma = self.spherical_variogram_model(d)
        else:
            print('theoretical varigoram model not understood')
        return gamma

    def calculate_covariance(self, d):

        if self.theoretical_model == 'exponential':
            gamma = self.exponential_covariance_model(d)
        elif self.theoretical_model == 'gaussian':
            gamma = self.gaussian_covariance_model(d)
        elif self.theoretical_model == 'spherical':
            gamma = self.spherical_covariance_model(d)
        else:
            print('theoretical varigoram model not understood')
        return gamma

    # TODO: Add more options
    # seems better now by changing psill in covariance model
    def exponential_variogram_model(self, d):
        '''Exponential variogram model, effective range approximately 3r, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-(np.absolute(d) / (self.range_)))) + self.nugget
        return gamma

    def exponential_covariance_model(self, d):
        '''Exponential covariance model, effective range approximately 3r, valid in R3'''
        psill = self.sill - self.nugget
        cov = psill * (np.exp(-(np.absolute(d) / (self.range_))))
        return cov

    def gaussian_variogram_model(self, d):
        '''Gaussian variogram model, effective range approximately sqrt(3r),
        deprecated due to reverse curvature near orgin, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-d ** 2. / (self.range_) ** 2.)) + self.nugget
        return gamma

    def gaussian_covariance_model(self, d):
        '''Gaussian covariance model, effective range approximately sqrt(3r),
        deprecated due to reverse curvature near orgin, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (np.exp(-d ** 2. / (self.range_) ** 2.))
        return gamma

    def spherical_variogram_model(self, d):
        '''Spherical variogram model, effective range equals range parameter, valid in R3'''
        psill = self.sill - self.nugget
        d = d.astype(float)
        gamma = np.piecewise(d, [d <= self.range_, d > self.range_],
                             [lambda d:
                              psill * ((3. * d) / (2. * self.range_)
                                       - (d ** 3.) / (2. * self.range_ ** 3.)) + self.nugget,
                              lambda d: self.sill])
        return gamma

    def spherical_covariance_model(self, d):
        '''Spherical covariance model, effective range equals range parameter, valid in R3'''
        psill = self.sill - self.nugget
        d = d.astype(float)
        gamma = np.piecewise(d, [d <= self.range_, d > self.range_],
                             [lambda d:
                              psill * (1 - ((3. * d) / (2. * self.range_)
                                            - (d ** 3.) / (2. * self.range_ ** 3.))),
                              lambda d: 0])
        return gamma

    # TODO: Make this better and nicer and everything
    # option for covariance
    # display range, sill, nugget, practical range etc.
    def plot(self, type_='variogram', show_parameters=True):

        if show_parameters == True:
            plt.axhline(self.sill, color='black', lw=1)
            plt.text(self.range_*2, self.sill, 'sill', fontsize=12, va='center', ha='center', backgroundcolor='w')
            plt.axvline(self.range_, color='black', lw=1)
            plt.text(self.range_, self.sill/2, 'range', fontsize=12, va='center', ha='center', backgroundcolor='w')

        if type_ == 'variogram':
            d = np.arange(0, self.range_*4, self.range_/1000)
            plt.plot(d, self.calculate_semivariance(d), label=self.theoretical_model + " variogram model")
            plt.ylabel('semivariance')
            plt.title('Variogram model')
            plt.legend()

        if type_ == 'covariance':
            d = np.arange(0, self.range_*4, self.range_/1000)
            plt.plot(d, self.calculate_covariance(d), label=self.theoretical_model + " covariance model")
            plt.ylabel('covariance')
            plt.title('Covariance model')
            plt.legend()

        if type_ == 'both':
            d = np.arange(0, self.range_*4, self.range_/1000)
            plt.plot(d, self.calculate_semivariance(d), label=self.theoretical_model + " variogram model")
            plt.plot(d, self.calculate_covariance(d), label=self.theoretical_model + " covariance model")
            plt.ylabel('semivariance/covariance')
            plt.title('Models of spatial correlation')
            plt.legend()

        plt.xlabel('lag distance')
        plt.ylim(0-self.sill/20, self.sill+self.sill/20)
        plt.xlim(0, self.range_*4)



class field_solution(object):

    def __init__(self, domain, variogram_model, results, field_type):

        self.results_df = results
        self.variogram_model = deepcopy(variogram_model)
        self.domain = deepcopy(domain)
        self.field_type = field_type

    def plot_results(self, geo_data, prop='val', direction='y', result='interpolation', cell_number=0, contour=False,
                     cmap='viridis', alpha=0, legend=False, interpolation='nearest', show_data=True):
        """
        TODO WRITE DOCSTRING
        Args:
            geo_data:
            prop: property that should be plotted - "val", "var" or "both"
            direction: x, y or z
            cell_number:
            contour:
            cmap:
            alpha:
            legend:

        Returns:

        """
        a = np.full_like(self.domain.mask, np.nan, dtype=np.double) #array like lith_block but with nan if outside domain

        est_vals = self.results_df['estimated value'].values
        est_var = self.results_df['estimation variance'].values

        # set values
        if prop == 'val':
            a[np.where(self.domain.mask == True)] = est_vals
        elif prop == 'var':
            a[np.where(self.domain.mask == True)] = est_var
        elif prop == 'both':
            a[np.where(self.domain.mask == True)] = est_vals
            b = np.full_like(self.domain.mask, np.nan, dtype=np.double)
            b[np.where(self.domain.mask == True)] = est_var
        else:
            print('prop must be val var or both')

        #create plot object
        p = _visualization_2d.PlotSolution(geo_data)
        _a, _b, _c, extent_val, x, y = p._slice(direction, cell_number)[:-2]

        #colors
        cmap = cm.get_cmap(cmap)
        cmap.set_bad(color='w', alpha=alpha) #define color and alpha for nan values

        # plot
        if prop is not 'both':
            if show_data:
                plt.scatter(self.domain.data_df[x].values, self.domain.data_df[y].values, marker='*', s=9, c='k')

            _plot.plot_section(geo_data, direction=direction, cell_number=cell_number)
            if contour == True:
                im = plt.contourf(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                  origin='lower', levels=25,
                                  extent=extent_val, interpolation=interpolation)
                if legend:
                    ax = plt.gca()
                    helpers.add_colorbar(axes=ax, label='prop', cs=im)
            else:
                im = plt.imshow(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                origin='lower',
                                extent=extent_val, interpolation=interpolation)
                if legend:
                    helpers.add_colorbar(im, label='property value', location='right')

        else:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].title.set_text('Estimated value')
            im1 = ax[0].imshow(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                               origin='lower', interpolation=interpolation,
                               extent=self.domain.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im1, label='property value')
            ax[1].title.set_text('Variance')
            im2 = ax[1].imshow(b.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                               origin='lower', interpolation=interpolation,
                               extent=self.domain.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im2, label='variance[]')
            plt.tight_layout()

# TODO: check with new ordianry kriging and nugget effect
def simple_kriging(a, b, prop, var_mod, inp_mean):
    '''
    Method for simple kriging calculation.
    Args:
        a (np.array): distance matrix containing all distances between target point and moving neighbourhood
        b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop (np.array): array containing scalar property values of locations in moving neighbourhood
        var_mod: variogram model object
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
    C[:shape, :shape] = var_mod.calculate_covariance(b) #? cov or semiv
    c[:shape] = var_mod.calculate_covariance(a) #? cov or semiv

    # nugget effect for simple kriging - dont remember why i set this actively, should be the same
    #np.fill_diagonal(C, self.sill)

    # TODO: find way to check quality of matrix and solutions for instability
    # Solve Kriging equations
    w = np.linalg.solve(C, c)

    # calculating estimate and variance for kriging
    pred_var = var_mod.sill - np.sum(w * c)
    # Note that here the input mean is required, if kriged mean equivalent to OK
    result = inp_mean + np.sum(w * (prop - inp_mean))

    return result, pred_var

def ordinary_kriging(a, b, prop, var_mod):
    '''
    Method for ordinary kriging calculation.
    Args:
        a (np.array): distance matrix containing all distances between target point and moving neighbourhood
        b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop (np.array): array containing scalar property values of locations in moving neighbourhood
        var_mod: variogram model object
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
    C[:shape, :shape] = var_mod.calculate_semivariance(b)
    c[:shape] = var_mod.calculate_semivariance(a)

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

def create_kriged_field(domain, variogram_model, distance_type='euclidian',
                        moving_neighbourhood='all', kriging_type='OK', n_closest_points=20):
    '''
    Method to create a kriged field over the defined grid of the gempy solution depending on the defined
    input data (conditioning).
    Returns:
        self.results_df (pandas dataframe):   Dataframe containing coordinates, kriging estimate
                                                    and kriging variance for each grid point
    '''
    # empty arrays for results (estimated values and variances)
    kriging_result_vals = np.zeros(len(domain.krig_grid))
    kriging_result_vars = np.zeros(len(domain.krig_grid))

    # Start with distance calculation
    # 1) all grid points to all data points
    # 2) all data points among each other
    if distance_type == 'euclidian':
        # calculate distances between all input data points
        dist_all_to_all = cdist(domain.data[:, :3], domain.data[:, :3])
        # calculate distances between all grid points and all input data points
        dist_grid_to_all = cdist(domain.krig_grid, domain.data[:, :3])

    # Main loop that goes through whole domain (grid)
    for i in range(len(domain.krig_grid)):

        # STEP 1: Multiple if elif conditions to define moving neighbourhood:
        if moving_neighbourhood == 'all':
            # cutting matrices and properties based on moving neighbourhood
            a = dist_grid_to_all[i]
            b = dist_all_to_all
            prop = domain.data[:, 3]

        elif moving_neighbourhood == 'n_closest':
            # cutting matrices and properties based on moving neighbourhood
            a = np.sort(dist_grid_to_all[i])
            a = a[:n_closest_points]
            aux = np.argsort(dist_grid_to_all[i])
            prop = domain.data[:, 3][aux]
            prop = prop[:n_closest_points]
            aux = aux[:n_closest_points]
            b = dist_all_to_all[np.ix_(aux, aux)]

        elif moving_neighbourhood == 'range':
            # cutting matrices and properties based on moving neighbourhood
            aux = np.where(dist_grid_to_all[i] <= variogram_model.range_)[0]
            a = dist_grid_to_all[i][aux]
            prop = domain.data[:, 3][aux]
            b = dist_all_to_all[np.ix_(aux, aux)]

        else:
            print("FATAL ERROR: Moving neighbourhood not understood")

        # STEP 2: Multiple if elif conditions to calculate kriging at point
        if kriging_type == 'OK':
            val, var = ordinary_kriging(a, b, prop, variogram_model)
        elif kriging_type == 'SK':
            val, var = simple_kriging(a, b, prop, variogram_model, domain.inp_mean)
        elif kriging_type == 'UK':
            print("Universal Kriging not implemented")
        else:
            print("FATAL ERROR: Kriging type not understood")

        # STEP 3: Save results
        kriging_result_vals[i] = val
        kriging_result_vars[i] = var

    # create dataframe of results data for calling
    d = {'X': domain.krig_grid[:, 0], 'Y': domain.krig_grid[:, 1], 'Z': domain.krig_grid[:, 2],
        'estimated value': kriging_result_vals, 'estimation variance': kriging_result_vars}

    results_df = pd.DataFrame(data=d)

    return field_solution(domain, variogram_model, results_df, field_type='interpolation')

def create_gaussian_field(domain, variogram_model, distance_type='euclidian',
                        moving_neighbourhood='all', kriging_type='OK', n_closest_points=20):
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
    shuffled_grid = domain.krig_grid
    np.random.shuffle(shuffled_grid)

    # append shuffled grid to input locations
    sgs_locations = np.vstack((domain.data[:,:3],shuffled_grid))
    # create array for input properties
    sgs_prop_updating = domain.data[:,3] # use this and then always stack new ant end

    # container for estimation variances
    estimation_var = np.zeros(len(shuffled_grid))

    # - distance calculation (stays the same)
    # 1) all points to all points in order of path
    # 2) known locations at beginning?
    if distance_type == 'euclidian':
        # calculate distances between all input data points
        dist_all_to_all = cdist(sgs_locations, sgs_locations)

    # set counter og active data (start=input data, grwoing by 1 newly calcualted point each run)
    active_data = len(sgs_prop_updating)

    # Main loop that goes through whole domain (grid)
    for i in range(len(domain.krig_grid)):
        # STEP 1: cut update distance matrix to correct size
        # HAVE TO CHECK IF THIS IS REALLY CORRECT
        active_distance_matrix = dist_all_to_all[:active_data,:active_data]
        active_distance_vector = dist_all_to_all[:,active_data] #basically next point to be simulated
        active_distance_vector = active_distance_vector[:active_data] #cut to left or diagonal

        # TODO: NEED PART FOR ZERO INPUT OR NO POINTS IN RANGE OR LESS THAN N POINTS

        # STEP 2: Multiple if elif conditions to define moving neighbourhood:
        if moving_neighbourhood == 'all':
            # cutting matrices and properties based on moving neighbourhood
            a = active_distance_vector
            b = active_distance_matrix
            prop = sgs_prop_updating

        elif moving_neighbourhood == 'n_closest':
            # cutting matrices and properties based on moving neighbourhood

            # This seems to work
            if len(sgs_prop_updating) <= n_closest_points:
                a = active_distance_vector[:active_data]
                b = active_distance_matrix[:active_data,:active_data]
                prop = sgs_prop_updating

            # this does not # DAMN THIS STILL HAS ITSELF RIGHT? PROBLEM!
            else:
                a = np.sort(active_distance_vector)
                a = a[:n_closest_points]
                aux = np.argsort(active_distance_vector)
                prop = sgs_prop_updating[aux]
                prop = prop[:n_closest_points]
                aux = aux[:n_closest_points]
                b = active_distance_matrix[np.ix_(aux, aux)]

        elif moving_neighbourhood == 'range':
            # cutting matrices and properties based on moving neighbourhood
            aux = np.where(active_distance_vector <= variogram_model.range_)[0]
            a = active_distance_vector[aux]
            prop = sgs_prop_updating[aux]
            b = active_distance_matrix[np.ix_(aux, aux)]

        else:
            print("FATAL ERROR: Moving neighbourhood not understood")

        # STEP 3: Multiple if elif conditions to calculate kriging at point
        # TODO: Cover case of data location and grid point coinciding
        if kriging_type == 'OK':
            val, var = ordinary_kriging(a, b, prop, variogram_model)
        elif kriging_type == 'SK':
            val, var = simple_kriging(a, b, prop, variogram_model, domain.inp_mean)
        elif kriging_type == 'UK':
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
    simulated_prop = sgs_prop_updating[len(domain.data[:,3]):] # check if this works like intended

    # create dataframe of results data for calling
    d = {'X': shuffled_grid[:, 0], 'Y': shuffled_grid[:, 1], 'Z': shuffled_grid[:, 2],
         'estimated value': simulated_prop, 'estimation variance': estimation_var}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.sort_values(['X','Y','Z'])

    return field_solution(domain, variogram_model, results_df, field_type='simulation')



