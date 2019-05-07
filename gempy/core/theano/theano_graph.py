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


DEP-- I need to update this string
Function that generates the symbolic code to perform the interpolation. Calling this function creates
 both the theano functions for the potential field and the block.

Returns:
    theano function for the potential field
    theano function for the block
"""
import theano
import theano.tensor as T
import numpy as np
import sys

theano.config.openmp_elemwise_minsize = 10000
theano.config.openmp = True

theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float64'
theano.config.on_opt_error = 'ignore'

theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'
theano.config.profile_memory = False
theano.config.scan.debug = False
theano.config.profile = False


class TheanoGraph(object):
    """
    This class is used to help to divide the construction of the graph into sensical parts. All its methods buildDEP2 a part
    of the graph. Every method can be seen as a branch and collection of branches until the last method that will be the
    whole tree. Every part of the graph could be compiled separately but as we increase the complexity the input of each
    of these methods is more and more difficult to provide (if you are in a branch close to the trunk you need all the
    results of the branches above)
    """
    def __init__(self, output='geology', optimizer='fast_compile', verbose=[0], dtype='float32',
                 is_fault=None, is_lith=None):
        """
        In the init we need to create all the symbolic parameters that are used in the process. Most of the variables
        are shared parameters initialized with random values. At this stage we only care about the type and shape of the
        parameters. After we have the graph built we can update the value of these shared parameters to our data (in the
        interpolatorClass).

        Args:
            u_grade: grade of the drift to compile the right graph. I found out that we can make a graph that takes this
            as variable so this argument will be deprecated soon
            verbose (list): name of the nodes you want to print
            dtype (str): type of float either 32 or 64
        """

        # Pass the verbose list as property

        # OPTIONS
        # -------
        if verbose is np.nan:
            self.verbose = [None]
        else:
            self.verbose = verbose
        self.dot_version = False

        theano.config.floatX = dtype
        theano.config.optimizer = optimizer

        # Creation of symbolic parameters
        # =============
        # Constants
        # =============

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        self.len_i_0 = 0
        self.len_i_1 = 1

        # =================
        # INITIALIZE SHARED
        # =================

        # SEMI-VARIABLES
        # --------------
        self.grid_val_T = theano.shared(np.cast[dtype](np.zeros((2, 200))), 'Coordinates of the grid '
                                                                            'points to interpolate')
        # Shape is 9x2, 9 drift funcitons and 2 points
        self.universal_grid_matrix_T = theano.shared(np.cast[dtype](np.zeros((9, 9)))) #TODO future DEP

        # FORMATIONS
        # ----------
        self.n_surface = theano.shared(np.arange(2, 5, dtype='int32'), "ID of the surface")
        self.n_surface_op = self.n_surface
        self.surface_values = theano.shared((np.arange(2, 4, dtype=dtype).reshape(2, -1)), "Value of the surface to compute")
        self.n_surface_op_float = self.surface_values

        # FAULTS
        # ------
        # Init fault relation matrix
        self.fault_relation = theano.shared(np.array([[0, 1, 0, 1],
                                                      [0, 0, 1, 1],
                                                      [0, 0, 0, 1],
                                                      [0, 0, 0, 0]]), 'fault relation matrix')

        self.inf_factor = theano.shared(np.ones(200, dtype='int32') * 10, 'Arbitrary scalar to make df infinite')

        # KRIGING
        # -------
        self.a_T = theano.shared(np.cast[dtype](-1.), "Range")
        self.c_o_T = theano.shared(np.cast[dtype](-1.), 'Covariance at 0')
        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of gradients')
        self.nugget_effect_scalar_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of scalar')
        self.n_universal_eq_T = theano.shared(np.zeros(5, dtype='int32'), "Grade of the universal drift")
        self.n_universal_eq_T_op = theano.shared(3)

        # STRUCTURE
        # ---------
        # This parameters give me the shape of the different groups of data. I pass all data together and I threshold it
        # using these values to the different potential fields and surfaces
        self.is_fault = is_fault
        self.is_lith = is_lith
        self.n_faults = theano.shared(0, 'Number of df')
        self.n_surfaces_per_series = theano.shared(np.arange(2, dtype='int32'), 'List with the number of surfaces')

        # This is not accumulative
        self.number_of_points_per_surface_T = theano.shared(np.zeros(3, dtype='int32')) #TODO is DEP?
        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T
        # This is accumulative
        self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')
        self.npf_op = self.npf[[0, -2]]
        self.len_series_i = theano.shared(np.arange(2, dtype='int32'), 'Length of surface_points in every series')
        self.len_series_f = theano.shared(np.arange(2, dtype='int32'), 'Length of foliations in every series')

        # VARIABLES
        # ---------
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")

        self.surface_points = T.matrix("All the surface_points points at once")
        #self.ref_layer_points_all = T.matrix("Reference points for every layer") # TODO: This should be DEP
        #self.rest_layer_points_all = T.matrix("Rest of the points of the layers") # TODO: This should be DEP
        self.len_points = self.surface_points.shape[0] - self.number_of_points_per_surface_T.shape[0]
        # Tiling dips to the 3 spatial coordinations
        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series. I initialized them as the whole arrays but then they will take
        # the data of every potential field
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all

        self.ref_layer_points_all = self.set_rest_ref_matrix()[0]
        self.rest_layer_points_all = self.set_rest_ref_matrix()[1]

        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

        # SOLUTION
        # --------
        self.final_block = theano.shared(np.cast[dtype](np.zeros((1, 3))), "Final block computed")

        self.final_scalar_field_at_surfaces = theano.shared(
            np.zeros(self.n_surfaces_per_series.get_value().sum(), dtype=dtype))
        self.final_scalar_field_at_faults = theano.shared(
            np.zeros(self.n_surfaces_per_series.get_value().sum(), dtype=dtype))

        self.final_scalar_field_at_surfaces_op = self.final_scalar_field_at_surfaces
        self.final_potential_field_at_faults_op = self.final_scalar_field_at_faults

        # Init Results
        # Init lithology block. Here we store the block and potential field results
        self.lith_block_init = T.zeros((2, self.grid_val_T.shape[0] + 2 * self.len_points))
        self.lith_block_init.name = 'final block of lithologies init'

        # Init df block. Here we store the block and potential field results of one iteration
        self.fault_block_init = T.zeros((2, self.grid_val_T.shape[0] + 2 * self.len_points))
        self.fault_block_init.name = 'final block of df init'
        self.yet_simulated = T.nonzero(T.eq(self.fault_block_init[0, :], 0))[0]

        # Init gradient block.
        self.gradient_block_init = T.zeros((3, self.grid_val_T.shape[0] + 2 * self.len_points))
        self.gradient_block_init.name = 'final block of gradient init'
        self.gradients = []

        # Here we store the value of the potential field at surface_points
        self.pfai_fault = T.zeros((0, self.n_surfaces_per_series[-1]))
        self.pfai_lith = T.zeros((0, self.n_surfaces_per_series[-1]))

        self.fault_matrix = T.zeros((0, self.grid_val_T.shape[0] + 2 * self.len_points))

        # GRAVITY
        # -------
        if output is 'gravity':
            self.densities = theano.shared(np.cast[dtype](np.zeros(3)), "List with the densities")
            self.tz = theano.shared(np.cast[dtype](np.zeros((1, 3))), "Component z")
            self.select = theano.shared(np.cast['int8'](np.zeros(3)), "Select nearby cells")
            # Init gray voxels for gravity
            self.weigths_weigths = theano.shared(np.ones(0))
            self.weigths_index = theano.shared(np.ones(0, dtype='int32'))

        self.weights = theano.shared(None)

    def set_rest_ref_matrix(self):
        ref_positions = T.cumsum(T.concatenate((T.stack(0), self.number_of_points_per_surface_T[:-1] + 1)))
        ref_points = T.repeat(self.surface_points[ref_positions], self.number_of_points_per_surface_T, axis=0)

        rest_mask = T.ones(T.stack(self.surface_points.shape[0]), dtype='int16')
        rest_mask = T.set_subtensor(rest_mask[ref_positions], 0)
        rest_points = self.surface_points[T.nonzero(rest_mask)[0]]
        return [ref_points, rest_points, rest_mask, T.nonzero(rest_mask)[0]]

    def input_parameters_list(self):
        """
        Create a list with the symbolic variables to use when we compile the theano function

        Returns:
            list: [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
                   self.ref_layer_points_all, self.rest_layer_points_all]
        """
        ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all, self.surface_points]
               #self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    @staticmethod
    def squared_euclidean_distances(x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Args:
            x_1 (theano.tensor.matrix): shape n_points x number dimension
            x_2 (theano.tensor.matrix): shape n_points x number dimension

        Returns:
            theano.tensor.matrix: Distancse matrix. shape n_points x n_points
        """

        # T.maximum avoid negative numbers increasing stability
        sqd = T.sqrt(T.maximum(
            (x_1**2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2**2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 1e-12
        ))

        if False:
            sqd = theano.printing.Print('sed')(sqd)

        return sqd

    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """

        # Calculating the dimensions of the
        length_of_CG = self.dips_position_tiled.shape[0]
        length_of_CGI = self.rest_layer_points.shape[0]
        length_of_U_I = self.n_universal_eq_T_op

        # Self fault matrix contains the block and the potential field (I am not able to split them). Therefore we need
        # to divide it by 2
        length_of_faults = T.cast(self.fault_matrix.shape[0], 'int32')
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_faults = theano.printing.Print("length_of_faults")(length_of_faults)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    def cov_surface_points(self):
        """
        Create covariance function for the surface_points

        Returns:
            theano.tensor.matrix: covariance of the surface_points. Shape number of points in rest x number of
            points in rest

        """

        # Compute euclidian distances
        sed_rest_rest = self.squared_euclidean_distances(self.rest_layer_points, self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distances(self.ref_layer_points, self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distances(self.rest_layer_points, self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distances(self.ref_layer_points, self.ref_layer_points)

        # Covariance matrix for surface_points
        C_I = (self.c_o_T * self.i_reescale * (
            (sed_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
             35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
             7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
             3 / 4 * (sed_rest_rest / self.a_T) ** 7) -
            ((sed_ref_rest < self.a_T) *  # Reference - Rest
             (1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
              35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
              7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
              3 / 4 * (sed_ref_rest / self.a_T) ** 7)) -
            ((sed_rest_ref < self.a_T) *  # Rest - Reference
             (1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
              35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
              7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
              3 / 4 * (sed_rest_ref / self.a_T) ** 7)) +
            ((sed_ref_ref < self.a_T) *  # Reference - References
             (1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
              35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
              7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
              3 / 4 * (sed_ref_ref / self.a_T) ** 7))))

        C_I += T.eye(C_I.shape[0]) * 2 * self.nugget_effect_scalar_T
        # Add name to the theano node
        C_I.name = 'Covariance SurfacePoints'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_I = theano.printing.Print('Cov surface_points')(C_I)

        return C_I

    def cov_gradients(self, verbose=0):
        """
         Create covariance function for the gradients

         Returns:
             theano.tensor.matrix: covariance of the gradients. Shape number of points in dip_pos x number of
             points in dip_pos

         """

        # Euclidean distances
        sed_dips_dips = self.squared_euclidean_distances(self.dips_position_tiled, self.dips_position_tiled)

        if 'sed_dips_dips' in self.verbose:
            sed_dips_dips = theano.printing.Print('sed_dips_dips')(sed_dips_dips)

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(self.dips_position[:, 0] - self.dips_position[:, 0].reshape((self.dips_position[:, 0].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 1] - self.dips_position[:, 1].reshape((self.dips_position[:, 1].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 2] - self.dips_position[:, 2].reshape((self.dips_position[:, 2].shape[0], 1)),
                   self.n_dimensions))

        # Transpose
        h_v = h_u.T

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance (block diagonal)
        perpendicularity_matrix = T.zeros_like(sed_dips_dips)

        # Cross-covariances of x
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:self.dips_position.shape[0], 0:self.dips_position.shape[0]], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0]:self.dips_position.shape[0] * 2,
            self.dips_position.shape[0]:self.dips_position.shape[0] * 2], 1)

        # Cross-covariances of z
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3,
            self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3], 1)

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(sed_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / sed_dips_dips ** 2) *
                ((
                     (sed_dips_dips < self.a_T) *  # first derivative
                     (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                     35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                     21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7))) +
                 (sed_dips_dips < self.a_T) *  # Second derivative
                 self.c_o_T * 7 * (9 * sed_dips_dips ** 5 - 20 * self.a_T ** 2 * sed_dips_dips ** 3 +
                                   15 * self.a_T ** 4 * sed_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                (perpendicularity_matrix *
                 (sed_dips_dips < self.a_T) *  # first derivative
                 self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                               35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag if I remove the condition
        C_G += T.eye(C_G.shape[0])*self.nugget_effect_grad_T

        # Add name to the theano node
        C_G.name = 'Covariance Gradient'

        if verbose > 1:
            theano.printing.pydotprint(C_G, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_G = theano.printing.Print('Cov Gradients')(C_G)

        return C_G

    def cov_interface_gradients(self):
        """
        Create covariance function for the gradiens
        Returns:
            theano.tensor.matrix: covariance of the gradients. Shape number of points in rest x number of
              points in dip_pos
        """

        # Euclidian distances
        sed_dips_rest = self.squared_euclidean_distances(self.dips_position_tiled, self.rest_layer_points)
        sed_dips_ref  = self.squared_euclidean_distances(self.dips_position_tiled, self.ref_layer_points)

        # Cartesian distances between dips and interface points
        # Rest
        hu_rest = T.vertical_stack(
            (self.dips_position[:, 0] - self.rest_layer_points[:, 0].reshape(
                (self.rest_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.rest_layer_points[:, 1].reshape(
                (self.rest_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.rest_layer_points[:, 2].reshape(
                (self.rest_layer_points[:, 2].shape[0], 1))).T
        )

        # Reference point
        hu_ref = T.vertical_stack(
            (self.dips_position[:, 0] - self.ref_layer_points[:, 0].reshape(
                (self.ref_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.ref_layer_points[:, 1].reshape(
                (self.ref_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.ref_layer_points[:, 2].reshape(
                (self.ref_layer_points[:, 2].shape[0], 1))).T
        )

        # Cross-Covariance gradients-surface_points
        C_GI = self.gi_reescale * (
            (hu_rest *
             (sed_dips_rest < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_rest / self.a_T ** 3 -
                              35 / 2 * sed_dips_rest ** 3 / self.a_T ** 5 +
                              21 / 4 * sed_dips_rest ** 5 / self.a_T ** 7))) -
            (hu_ref *
             (sed_dips_ref < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_ref / self.a_T ** 3 -
                              35 / 2 * sed_dips_ref ** 3 / self.a_T ** 5 +
                              21 / 4 * sed_dips_ref ** 5 / self.a_T ** 7)))
        ).T

        # Add name to the theano node
        C_GI.name = 'Covariance gradient interface'

        if str(sys._getframe().f_code.co_name)+'_g' in self.verbose:
            theano.printing.pydotprint(C_GI, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)
        return C_GI

    def universal_matrix(self):
        """
        Create the drift matrices for the potential field and its gradient

        Returns:
            theano.tensor.matrix: Drift matrix for the surface_points. Shape number of points in rest x 3**degree drift
            (except degree 0 that is 0)

            theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x 3**degree drift
            (except degree 0 that is 0)
        """

        # Condition of universality 2 degree
        # Gradients

        n = self.dips_position.shape[0]
        U_G = T.zeros((n * self.n_dimensions, 3 * self.n_dimensions))
        # x
        U_G = T.set_subtensor(U_G[:n, 0], 1)
        # y
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
        # z
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
        # x**2
        U_G = T.set_subtensor(U_G[:n, 3], 2 * self.gi_reescale * self.dips_position[:, 0])
        # y**2
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * self.gi_reescale * self.dips_position[:, 1])
        # z**2
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * self.gi_reescale * self.dips_position[:, 2])
        # xy
        U_G = T.set_subtensor(U_G[:n, 6], self.gi_reescale * self.dips_position[:, 1])  # This is y
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], self.gi_reescale * self.dips_position[:, 0])  # This is x
        # xz
        U_G = T.set_subtensor(U_G[:n, 7], self.gi_reescale * self.dips_position[:, 2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], self.gi_reescale * self.dips_position[:, 0])  # This is x
        # yz
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], self.gi_reescale * self.dips_position[:, 2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], self.gi_reescale * self.dips_position[:, 1])  # This is y

        # Interface
        U_I = - T.stack(
            (self.gi_reescale * (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
             self.gi_reescale * (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
             self.gi_reescale * (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 0] ** 2 - self.ref_layer_points[:, 0] ** 2),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 1] ** 2 - self.ref_layer_points[:, 1] ** 2),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 2] ** 2 - self.ref_layer_points[:, 2] ** 2),
             self.gi_reescale ** 2 * (
                 self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1] - self.ref_layer_points[:, 0] *
                 self.ref_layer_points[:, 1]),
             self.gi_reescale ** 2 * (
                 self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 0] *
                 self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (
                 self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 1] *
                 self.ref_layer_points[:, 2]),
             )).T

        if 'U_I' in self.verbose:
            U_I = theano.printing.Print('U_I')(U_I)

        if 'U_G' in self.verbose:
            U_G = theano.printing.Print('U_G')(U_G)

        if str(sys._getframe().f_code.co_name)+'_g' in self.verbose:
            theano.printing.pydotprint(U_I, outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)

            theano.printing.pydotprint(U_G, outfile="graphs/" + sys._getframe().f_code.co_name + "_g.png",
                                       var_with_name_simple=True)

        # Add name to the theano node
        if U_I:
            U_I.name = 'Drift surface_points'
            U_G.name = 'Drift foliations'

        return U_I[:, :self.n_universal_eq_T_op], U_G[:, :self.n_universal_eq_T_op]

    def faults_matrix(self):
        """
        This function creates the part of the graph that generates the df function creating a "block model" at the
        references and the rest of the points. Then this vector has to be appended to the covariance function

        Returns:

            list:

            - theano.tensor.matrix: Drift matrix for the surface_points. Shape number of points in rest x n df. This drif
              is a simple addition of an arbitrary number

            - theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x n df. For
              discrete values this matrix will be null since the derivative of a constant is 0
        """

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults = self.matrices_shapes()[:4]

        # self.fault_matrix contains the df volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
        if 'fault matrix' in self.verbose:
            self.fault_matrix = theano.printing.Print('self.fault_matrix')(self.fault_matrix)
        interface_loc = self.fault_matrix.shape[1] - 2 * self.len_points

        fault_matrix_at_surface_points_rest = self.fault_matrix[:,
                                          interface_loc + self.len_i_0: interface_loc + self.len_i_1]
        fault_matrix_at_surface_points_ref = self.fault_matrix[:,
                                         interface_loc + self.len_points + self.len_i_0: interface_loc + self.len_points + self.len_i_1]

        F_I = (fault_matrix_at_surface_points_ref - fault_matrix_at_surface_points_rest)+0.0001

        # As long as the drift is a constant F_G is null
        F_G = T.zeros((length_of_faults, length_of_CG)) + 0.0001

        if str(sys._getframe().f_code.co_name) in self.verbose:
            F_I = theano.printing.Print('Faults surface_points matrix')(F_I)
            F_G = theano.printing.Print('Faults gradients matrix')(F_G)

        return F_I, F_G

    def covariance_matrix(self):
        """
        Set all the previous covariances together in the universal cokriging matrix

        Returns:
            theano.tensor.matrix: Multivariate covariance
        """

        # Lengths
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        # Individual matrices
        C_G = self.cov_gradients()
        C_I = self.cov_surface_points()
        C_GI = self.cov_interface_gradients()
        U_I, U_G = self.universal_matrix()
        F_I, F_G = self.faults_matrix()

        # =================================
        # Creation of the Covariance Matrix
        # =================================
        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        # Set C_G
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)
        # Set CGI
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)
        # Set UG
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG,
                                   length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I], U_G)
        # Set FG. I cannot use -index because when is -0 is equivalent to 0
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG+length_of_CGI+length_of_U_I:], F_G.T)
        # Second row of matrices
        # Set C_IG
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        # Set C_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG:length_of_CG + length_of_CGI], C_I)
        # Set U_I
        #if not self.u_grade_T.get_value() == 0:
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I], U_I)
        # Set F_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG+length_of_CGI+length_of_U_I:], F_I.T)
        # Third row of matrices
        # Set U_G
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I, 0:length_of_CG], U_G.T)
        # Set U_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I, length_of_CG:length_of_CG + length_of_CGI], U_I.T)
        # Fourth row of matrices
        # Set F_G
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI+length_of_U_I:, 0:length_of_CG], F_G)
        # Set F_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI+length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], F_I)
        # Add name to the theano node
        C_matrix.name = 'Block Covariance Matrix'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_matrix = theano.printing.Print('cov_function')(C_matrix)

        return C_matrix

    def b_vector(self):
        """
        Creation of the independent vector b to solve the kriging system

        Args:
            verbose: -deprecated-

        Returns:
            theano.tensor.vector: independent vector
        """

        length_of_C = self.matrices_shapes()[-1]
        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(self.dip_angles)) * T.sin(T.deg2rad(self.azimuth)) * self.polarity
        G_y = T.sin(T.deg2rad(self.dip_angles)) * T.cos(T.deg2rad(self.azimuth)) * self.polarity
        G_z = T.cos(T.deg2rad(self.dip_angles)) * self.polarity

        G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros((length_of_C,))
        b = T.set_subtensor(b[0:G.shape[0]], G)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            b = theano.printing.Print('b vector')(b)

        # Add name to the theano node
        b.name = 'b vector'
        return b

    def solve_kriging(self):
        """
        Solve the kriging system. This has to get substituted by a more efficient and stable method QR
        decomposition in all likelihood

        Returns:
            theano.tensor.vector: Dual kriging parameters

        """
        C_matrix = self.covariance_matrix()
        b = self.b_vector()
        # Solving the kriging system
        import theano.tensor.slinalg
        b2 = T.tile(b, (1, 1)).T
        DK_parameters = theano.tensor.slinalg.solve(C_matrix, b2)
        DK_parameters = DK_parameters.reshape((DK_parameters.shape[0],))

        # Add name to the theano node
        DK_parameters.name = 'Dual Kriging parameters'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            DK_parameters = theano.printing.Print(DK_parameters.name)(DK_parameters)
        return DK_parameters

    def x_to_interpolate(self, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        surface_points). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.

        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """

        grid_val = T.concatenate([self.grid_val_T, self.rest_layer_points_all,
                                  self.ref_layer_points_all])[self.yet_simulated, :]

        if verbose > 1:
            theano.printing.pydotprint(grid_val, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if 'grid_val' in self.verbose:
            grid_val = theano.printing.Print('Points to interpolate')(grid_val)

        return grid_val

    def extend_dual_kriging(self):
        """
        Tile the dual kriging vector to cover all the points to interpolate.So far I just make a matrix with the
        dimensions len(DK)x(grid) but in the future maybe I have to try to loop all this part so consume less memory

        Returns:
            theano.tensor.matrix: Matrix with the Dk parameters repeated for all the points to interpolate
        """

        grid_val = self.x_to_interpolate()
        if self.weights.get_value() is None:
            DK_parameters = self.solve_kriging()
        else:
            DK_parameters = self.weights
        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        # TODO IMP: Change the tile by a simple dot op -> The DOT version in gpu is slower
        DK_weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        if self.dot_version:
            DK_weights = DK_parameters

        return DK_weights

    def contribution_gradient_interface(self, grid_val=None, weights=None):
        """
        Computation of the contribution of the foliations at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG = self.matrices_shapes()[0]

        # Cartesian distances between the point to simulate and the dips
        hu_SimPoint = T.vertical_stack(
            (self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        # Euclidian distances
        sed_dips_SimPoint = self.squared_euclidean_distances(self.dips_position_tiled, grid_val)
        # Gradient contribution
        sigma_0_grad = T.sum(
            (weights[:length_of_CG] *
             self.gi_reescale *
             (-hu_SimPoint *
              (sed_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        if self.dot_version:
            sigma_0_grad = T.dot(
                weights[:length_of_CG] ,
                 self.gi_reescale *
                 (-hu_SimPoint *
                  (sed_dips_SimPoint < self.a_T) *  # first derivative
                  (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                                   35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                                   21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7))))

        # Add name to the theano node
        sigma_0_grad.name = 'Contribution of the foliations to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            sigma_0_grad = theano.printing.Print('interface_gradient_contribution')(sigma_0_grad)

        return sigma_0_grad

    def contribution_interface_gradient(self, direction='x', grid_val=None, weights=None):
        """
        Computation of the contribution of the foliations at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
        """

        if direction == 'x':
            dir_val = 0
        elif direction == 'y':
            dir_val = 1
        elif direction == 'z':
            dir_val = 2
        else:
            raise AttributeError('Directions muxt be x, y or z')

        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Cartesian distances between the point to simulate and the dips
        hu_rest = (- self.rest_layer_points[:, dir_val] + grid_val[:, dir_val].reshape((grid_val[:, dir_val].shape[0], 1)))
        hu_ref = (- self.ref_layer_points[:, dir_val] + grid_val[:, dir_val].reshape((grid_val[:, dir_val].shape[0], 1)))

        # Euclidian distances

        sed_grid_rest = self.squared_euclidean_distances(grid_val, self.rest_layer_points)
        sed_grid_ref = self.squared_euclidean_distances(grid_val, self.ref_layer_points)

        # Gradient contribution
        self.gi_reescale = 2

        sigma_0_grad = T.sum(
            (weights[length_of_CG:length_of_CG + length_of_CGI] *
             self.gi_reescale * (
                 (hu_rest *
                  (sed_grid_rest < self.a_T) *  # first derivative
                  (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_grid_rest / self.a_T ** 3 -
                                   35 / 2 * sed_grid_rest ** 3 / self.a_T ** 5 +
                                   21 / 4 * sed_grid_rest ** 5 / self.a_T ** 7))) -
            (hu_ref *
             (sed_grid_ref < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_grid_ref / self.a_T ** 3 -
                              35 / 2 * sed_grid_ref ** 3 / self.a_T ** 5 +
                              21 / 4 * sed_grid_ref ** 5 / self.a_T ** 7)))).T),
            axis=0)

        return sigma_0_grad

    def contribution_interface(self, grid_val=None, weights=None):
        """
          Computation of the contribution of the surface_points at every point to interpolate

          Returns:
              theano.tensor.vector: Contribution of all surface_points (input) at every point to interpolate
          """

        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()
        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Euclidian distances
        sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points, grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points, grid_val)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o_T * self.i_reescale * (
                    (sed_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                    (1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                     35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                     7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                     3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                    ((sed_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                     (1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                      35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                      7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                      3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7)))), axis=0))

        if self.dot_version:
            sigma_0_interf = (
                T.dot(-weights[length_of_CG:length_of_CG + length_of_CGI],
                      (self.c_o_T * self.i_reescale * (
                              (sed_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                              (1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                               35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                               7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                               3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                              ((sed_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                               (1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                                35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                                7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                                3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7))))))

        # Add name to the theano node
        sigma_0_interf.name = 'Contribution of the surface_points to the potential field at every point of the grid'

        return sigma_0_interf

    def contribution_gradient(self, direction='x', grid_val=None, weights=None):

        if direction == 'x':
           direction_val = 0
        if direction == 'y':
            direction_val = 1
        if direction == 'z':
            direction_val = 2
        self.gi_reescale = theano.shared(1)


        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG = self.matrices_shapes()[0]

        # Cartesian distances between the point to simulate and the dips
        # TODO optimize to compute this only once?
        # Euclidean distances
        sed_dips_SimPoint = self.squared_euclidean_distances(grid_val, self.dips_position_tiled).T

        if 'sed_dips_SimPoint' in self.verbose:
            sed_dips_SimPoint = theano.printing.Print('sed_dips_SimPoint')(sed_dips_SimPoint)

        # Cartesian distances between dips positions
        h_u = T.tile(self.dips_position[:, direction_val] - grid_val[:, direction_val].reshape((grid_val[:, direction_val].shape[0], 1)), 3)
        h_v = T.horizontal_stack(
            T.tile(self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1)),
                   1),
            T.tile(self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1)),
                   1),
            T.tile(self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1)),
                   1))

        perpendicularity_vector = T.zeros(T.stack(length_of_CG))
        perpendicularity_vector = T.set_subtensor(
            perpendicularity_vector[self.dips_position.shape[0]*direction_val:self.dips_position.shape[0]*(direction_val+1)], 1)

        sigma_0_grad = T.sum(
            (weights[:length_of_CG] * (
             ((-h_u * h_v).T/ sed_dips_SimPoint ** 2) *
             ((
                      (sed_dips_SimPoint < self.a_T) *  # first derivative
                      (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                                      35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                                      21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7))) +
              (sed_dips_SimPoint < self.a_T) *  # Second derivative
              self.c_o_T * 7 * (9 * sed_dips_SimPoint ** 5 - 20 * self.a_T ** 2 * sed_dips_SimPoint ** 3 +
                                   15 * self.a_T ** 4 * sed_dips_SimPoint - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                (perpendicularity_vector.reshape((-1, 1)) *
                                ((sed_dips_SimPoint < self.a_T) *  # first derivative
                 self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))

              ))
        , axis=0)

        return sigma_0_grad

    def contribution_universal_drift(self, grid_val=None, weights=None, a=0, b=100000000):
        """
        Computation of the contribution of the universal drift at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of the universal drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        # Universal drift contribution
        # universal_grid_surface_points_matrix = self.universal_grid_matrix_T[:, self.fault_mask[a: b]]

        # Universal drift contribution
        # Universal terms used to calculate f0
        # Here I create the universal terms for rest and ref. The universal terms for the grid are done in python
        # and append here. The idea is that the grid is kind of constant so I do not have to recompute it every
        # time
        # _universal_terms_surface_points_rest = T.horizontal_stack(
        #     self.rest_layer_points_all,
        #     (self.rest_layer_points_all ** 2),
        #     T.stack((self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 1],
        #              self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 2],
        #              self.rest_layer_points_all[:, 1] * self.rest_layer_points_all[:, 2]), axis=1))
        #
        # _universal_terms_surface_points_ref = T.horizontal_stack(
        #     self.ref_layer_points_all,
        #     (self.ref_layer_points_all ** 2),
        #     T.stack((self.ref_layer_points_all[:, 0] * self.ref_layer_points_all[:, 1],
        #              self.ref_layer_points_all[:, 0] * self.ref_layer_points_all[:, 2],
        #              self.ref_layer_points_all[:, 1] * self.ref_layer_points_all[:, 2]), axis=1),
        # )
        #
        # # I append rest and ref to grid
        # # universal_grid_surface_points_matrix = T.horizontal_stack(
        # #     (self.universal_grid_matrix_T * self.fault_mask).nonzero_values().reshape((9, -1)),
        # #     T.vertical_stack(_universal_terms_surface_points_rest, _universal_terms_surface_points_ref).T)
        #
        # universal_grid_surface_points_matrix = T.horizontal_stack(
        #     self.universal_grid_matrix_T.reshape((9, -1)),
        #     T.vertical_stack(_universal_terms_surface_points_rest, _universal_terms_surface_points_ref).T)

        universal_grid_surface_points_matrix = T.horizontal_stack(
             grid_val,
            (grid_val ** 2),
            T.stack((grid_val[:, 0] * grid_val[:, 1],
                     grid_val[:, 0] * grid_val[:, 2],
                     grid_val[:, 1] * grid_val[:, 2]), axis=1)).T


        # These are the magic terms to get the same as geomodeller
        i_rescale_aux = T.repeat(self.gi_reescale, 9)
        i_rescale_aux = T.set_subtensor(i_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(i_rescale_aux[:self.n_universal_eq_T_op], (grid_val.shape[0], 1)).T

        # Drif contribution
        f_0 = (T.sum(
            weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_reescale * _aux_magic_term *
            universal_grid_surface_points_matrix[:self.n_universal_eq_T_op]
            , axis=0))

        if self.dot_version:
            f_0 = T.dot(
                weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] , self.gi_reescale * _aux_magic_term *
                universal_grid_surface_points_matrix[:self.n_universal_eq_T_op])

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

        return f_0

    def contribution_universal_drift_d(self, direction='x', grid_val=None, weights=None, a=0, b=100000000):
        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        self.gi_reescale = theano.shared(2)

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        # These are the magic terms to get the same as geomodeller
        i_rescale_aux = T.repeat(self.gi_reescale, 9)
        i_rescale_aux = T.set_subtensor(i_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(i_rescale_aux[:self.n_universal_eq_T_op], (grid_val.shape[0], 1))

        n = self.dips_position.shape[0]
        n = grid_val.shape[0]
        U_G = T.zeros((n, 9))

        if direction == 'x':

            # x
            U_G = T.set_subtensor(U_G[:, 0], 1)
            # x**2
            U_G = T.set_subtensor(U_G[:, 3], 2 * self.gi_reescale * grid_val[:, 0])

            # xy
            U_G = T.set_subtensor(U_G[:, 6], self.gi_reescale * grid_val[:, 1])  # This is y
            # xz
            U_G = T.set_subtensor(U_G[:, 7], self.gi_reescale * grid_val[:, 2])  # This is z

        if direction == 'y':

            # y
            U_G = T.set_subtensor(U_G[:, 1], 1)
            # y**2
            U_G = T.set_subtensor(U_G[:, 4], 2 * self.gi_reescale * grid_val[:, 1])
            # xy
            U_G = T.set_subtensor(U_G[:, 6], self.gi_reescale * grid_val[:, 0])  # This is x
            # yz
            U_G = T.set_subtensor(U_G[:, 8], self.gi_reescale * grid_val[:, 2])  # This is z

        if direction == 'z':
            # z
            U_G = T.set_subtensor(U_G[:, 2], 1)

            # z**2
            U_G = T.set_subtensor(U_G[:, 5], 2 * self.gi_reescale * grid_val[:, 2])

            #xz
            U_G = T.set_subtensor(U_G[:, 7], self.gi_reescale * grid_val[:, 0])  # This is x

            # yz
            U_G = T.set_subtensor(U_G[:, 8], self.gi_reescale * grid_val[:, 1])  # This is y

        # Drif contribution
        f_0 = (T.sum(
            weights[
            length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] *
            U_G.T[:self.n_universal_eq_T_op]
            , axis=0))

        return f_0

    def faults_contribution(self, weights=None, a=0, b=100000000):
        """
        Computation of the contribution of the df drift at every point to interpolate. To get these we need to
        compute a whole block model with the df data

        Returns:
            theano.tensor.vector: Contribution of the df drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        fault_matrix_selection_non_zero = (self.fault_matrix[:, self.yet_simulated[a:b]]+1)

        f_1 = T.sum(
            weights[length_of_CG + length_of_CGI + length_of_U_I:, :] * fault_matrix_selection_non_zero, axis=0)

        if self.dot_version:
            f_1 = T.dot(
                weights[length_of_CG + length_of_CGI + length_of_U_I:], fault_matrix_selection_non_zero)

        # Add name to the theano node
        f_1.name = 'Faults contribution'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_1 = theano.printing.Print('Faults contribution')(f_1)

        return f_1

    def scalar_field_loop(self, a, b, Z_x, grid_val, weights, val):

        sigma_0_grad = self.contribution_gradient_interface(grid_val[a:b], weights[:, a:b])
        sigma_0_interf = self.contribution_interface(grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift(grid_val[a:b], weights[:, a:b], a, b)
        f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1 + 50 - (10 * val[0]))
        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def gradient_field_loop_x(self, a, b, Z_x, grid_val, weights, val):
        direction = 'x'
        sigma_0_grad = self.contribution_gradient(direction, grid_val[a:b], weights[:, a:b])
        sigma_0_interf_gradient = self.contribution_interface_gradient(direction, grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift_d(direction, grid_val[a:b], weights[:, a:b], a, b)
        #f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf_gradient + f_0)

        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def gradient_field_loop_y(self, a, b, Z_x, grid_val, weights, val):
        direction = 'y'
        sigma_0_grad = self.contribution_gradient(direction, grid_val[a:b], weights[:, a:b])
        sigma_0_interf_gradient = self.contribution_interface_gradient(direction, grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift_d(direction, grid_val[a:b], weights[:, a:b], a, b)
        #f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf_gradient + f_0)

        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def gradient_field_loop_z(self, a, b, Z_x, grid_val, weights, val):
        direction = 'z'
        sigma_0_grad = self.contribution_gradient(direction, grid_val[a:b], weights[:, a:b])
        sigma_0_interf_gradient = self.contribution_interface_gradient(direction, grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift_d(direction, grid_val[a:b], weights[:, a:b], a, b)
        #f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf_gradient + f_0)

        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def scalar_field_at_all(self, weights=None):
        """
        Compute the potential field at all the interpolation points, i.e. grid plus rest plus ref
        Returns:
            theano.tensor.vector: Potential fields at all points

        """
        grid_val = self.x_to_interpolate()

        if weights is None:
            weights = self.extend_dual_kriging()

        grid_shape = T.stack(grid_val.shape[0])
        Z_x_init = T.zeros(grid_shape, dtype='float32')
        if 'grid_shape' in self.verbose:
            grid_shape = theano.printing.Print('grid_shape')(grid_shape)

        steps = 1e13/self.matrices_shapes()[-1]/grid_shape
        slices = T.concatenate((T.arange(0, grid_shape[0], steps[0], dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        Z_x_loop, updates3 = theano.scan(
            fn=self.scalar_field_loop,
            outputs_info=[Z_x_init],
            sequences=[dict(input=slices, taps=[0, 1])],
            non_sequences=[grid_val, weights, self.n_surface_op],
            profile=False,
            name='Looping grid',
            return_list=True)

        Z_x = Z_x_loop[-1][-1]
        Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)

        return Z_x

    def gradient_field_at_all(self, weights=None, gradients=[]):

        grid_val = self.x_to_interpolate()
        if weights is None:
            weights = self.extend_dual_kriging()

        grid_shape = T.stack(grid_val.shape[0])

        Z_x_init = T.zeros(grid_shape, dtype='float32')
        if 'grid_shape' in self.verbose:
            grid_shape = theano.printing.Print('grid_shape')(grid_shape)

        steps = 1e13 / self.matrices_shapes()[-1] / grid_shape
        slices = T.concatenate((T.arange(0, grid_shape[0], steps[0], dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        G_field = T.zeros((3, self.grid_val_T.shape[0]))

        if 'Gx' in gradients:
            Gx_loop, updates5 = theano.scan(
                fn=self.gradient_field_loop_x,
                outputs_info=[Z_x_init],
                sequences=[dict(input=slices, taps=[0, 1])],
                non_sequences=[grid_val, weights, self.n_surface_op],
                profile=False,
                name='Looping grid x',
                return_list=True)

            Gx = Gx_loop[-1][-1]
            Gx.name = 'Value of the gradient field X at every point'
            G_field = T.set_subtensor(G_field[0, :], Gx)

        if 'Gy' in gradients:
            Gy_loop, updates6 = theano.scan(
                fn=self.gradient_field_loop_y,
                outputs_info=[Z_x_init],
                sequences=[dict(input=slices, taps=[0, 1])],
                non_sequences=[grid_val, weights, self.n_surface_op],
                profile=False,
                name='Looping grid y',
                return_list=True)

            Gy = Gy_loop[-1][-1]
            Gy.name = 'Value of the gradient field X at every point'
            G_field = T.set_subtensor(G_field[1, :], Gy)

        if 'Gz' in gradients:
            Gz_loop, updates7 = theano.scan(
                fn=self.gradient_field_loop_z,
                outputs_info=[Z_x_init],
                sequences=[dict(input=slices, taps=[0, 1])],
                non_sequences=[grid_val, weights, self.n_surface_op],
                profile=False,
                name='Looping grid z',
                return_list=True)

            Gz = Gz_loop[-1][-1]
            Gz.name = 'Value of the gradient field X at every point'
            G_field = T.set_subtensor(G_field[2, :], Gz)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)

        return G_field

    def compare(self, a, b, slice_init, Z_x, l, n_surface, drift):
        """
        Treshold of the points to interpolate given 2 potential field values. TODO: This function is the one we
        need to change for a sigmoid function

        Args:
            a (scalar): Upper limit of the potential field
            b (scalar): Lower limit of the potential field
            n_surface (scalar): Value given to the segmentation, i.e. lithology number
            Zx (vector): Potential field values at all the interpolated points

        Returns:
            theano.tensor.vector: segmented values
        """

        if True:

            slice_init = slice_init
            n_surface_0 = n_surface[:, slice_init:slice_init + 1]
            n_surface_1 = n_surface[:, slice_init + 1:slice_init + 2]
            drift = drift[:, slice_init:slice_init + 1]

            if 'compare' in self.verbose:
                a = theano.printing.Print("a")(a)
                b = theano.printing.Print("b")(b)
                # l = 200/ (a - b)
                slice_init = theano.printing.Print("slice_init")(slice_init)
                n_surface_0 = theano.printing.Print("n_surface_0")(n_surface_0)
                n_surface_1 = theano.printing.Print("n_surface_1")(n_surface_1)
                drift = theano.printing.Print("drift[slice_init:slice_init+1][0]")(drift)

            # drift = T.switch(slice_init == 0, n_surface_1, n_surface_0)
            #    drift = T.set_subtensor(n_surface[0], n_surface[1])

            # The 5 rules the slope of the function
            sigm = (-n_surface_0.reshape((-1, 1)) / (1 + T.exp(-l * (Z_x - a)))) - \
                   (n_surface_1.reshape((-1, 1)) / (1 + T.exp(l * (Z_x - b)))) + drift.reshape((-1, 1))
            if False:
                sigm = theano.printing.Print("middle point")(sigm)
            #      n_surface = theano.printing.Print("n_surface")(n_surface)
            return sigm

        else:
            return T.le(Zx, a) * T.ge(Zx, b) * n_surface_0

    def select_finite_faults(self):
        # get data points of fault
        fault_points = T.vertical_stack(T.stack(self.ref_layer_points[0]), self.rest_layer_points).T
        # compute centroid of fault points
        centroid = T.mean(fault_points, axis=1)
        # compute difference of fault points from centroid
        x = fault_points - centroid.reshape((-1, 1))
        M = T.dot(x, x.T)  # same as np.cov(x) * 2
        U = T.nlinalg.svd(M)  # is this the normal of the plane?
        # overall this looks like some sort of plane fit to me
        rotated_x = T.dot(self.x_to_interpolate(), U[0])  # this rotates ALL grid points that need to be interpolated
        # rotated_x = T.dot(rotated_x, U[-1])  # rotate them with both rotation matrices
        rotated_fault_points = T.dot(fault_points.T, U[0])  # same with fault points
        # rotated_fault_points = T.dot(rotated_fault_points, U[-1])  # same
        rotated_ctr = T.mean(rotated_fault_points, axis=0)  # and compute centroid of rotated points
        # a factor: horizontal vector of ellipse of normal fault
        a_radio = (rotated_fault_points[:, 0].max() - rotated_fault_points[:, 0].min()) / 2 \
                  + self.inf_factor[self.n_surface_op[0] - 1]
        # b_factor: vertical vector of ellipse
        b_radio = (rotated_fault_points[:, 1].max() - rotated_fault_points[:, 1].min()) / 2 \
                  + self.inf_factor[self.n_surface_op[0] - 1]

        # sel = T.lt((rotated_x[:, 0] - rotated_ctr[0])**2 / a_radio**2 +
        #            (rotated_x[:, 1] - rotated_ctr[1])**2 / b_radio**2,
        #            1)

        # ellipse equation: (x, c_x)^2 / a^2 +  (y - c_y)^2 / b^2 <= 1 if in ellipse
        ellipse_factor = (rotated_x[:,0] - rotated_ctr[0])**2 / a_radio**2 + \
            (rotated_x[:, 1] - rotated_ctr[1])**2 / b_radio**2

        if "select_finite_faults" in self.verbose:
            ellipse_factor = theano.printing.Print("h")(ellipse_factor)

        # h_factor = 1 - h
        # if "select_finite_faults" in self.verbose:
        #     h_factor = theano.printing.Print("h_factor")(h_factor)

        # because we select all grid points as rotated_x, the selection here is
        # a boolean for all grid points: True if in ellipse, False if outside ellipse

        # if "select_finite_faults" in self.verbose:
        #     sel = theano.printing.Print("scalar_field_iter")(sel)
            # sum of boolean array sel is in my example: 38301
            # so I guess this selects all grid points affected by this finite fault

        return ellipse_factor  # sel

    def block_series(self, slope=5000, weights=None):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """
        # TODO: IMP set soft max in the borders

        # Graph to compute the potential field
        Z_x = self.scalar_field_at_all(weights)

        # Max and min values of the potential field.
        # max_pot = T.max(Z_x) + 1
        # min_pot = T.min(Z_x) - 1
        # max_pot += max_pot * 0.1
        # min_pot -= min_pot * 0.1

        # Value of the potential field at the surface_points of the computed series
        self.scalar_field_at_surface_points_values = Z_x[-2*self.len_points: -self.len_points][self.npf_op]

        max_pot = T.max(Z_x)
        #max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
   #     min_pot = theano.printing.Print("min_pot")(min_pot)


        max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot)*0.01
        l = slope / (max_pot - min_pot)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
                                           T.stack([max_pot + boundary_pad]),
                                           self.scalar_field_at_surface_points_values,
                                           T.stack([min_pot - boundary_pad])
                                            ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Loop to segment the distinct lithologies

        n_surface_op_float_sigmoid = T.repeat(self.n_surface_op_float, 2, axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!

        # This -1 makes that after the last interfaces the gradient goes on upwards
        n_formation_op_float_sigmoid = T.set_subtensor(n_formation_op_float_sigmoid[0], -1)
                                                    #- T.sqrt(T.square(n_formation_op_float_sigmoid[0] - n_formation_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
                                                    #- T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid")\
                (n_surface_op_float_sigmoid)

        partial_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]), T.arange(0, n_surface_op_float_sigmoid.shape[1],
                                                                            2, dtype='int64')],
            non_sequences=[Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        partial_block = partial_block.sum(axis=0)

        # Add name to the theano node
        partial_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            partial_block = theano.printing.Print(partial_block.name)(partial_block)

        return Z_x, partial_block

    def block_fault(self, slope=50):  #
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        # Graph to compute the potential field
        Z_x = self.scalar_field_at_all()

        # Max and min values of the potential field.
        # max_pot = T.max(Z_x) + 1
        # min_pot = T.min(Z_x) - 1
        # max_pot += max_pot * 0.1
        # min_pot -= min_pot * 0.1

        # Value of the potential field at the surface_points of the computed series
        self.scalar_field_at_surface_points_values = Z_x[-2 *(self.len_points): -self.len_points][self.npf_op]

        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        # min_pot = theano.printing.Print("min_pot")(min_pot)

        # max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        # min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        #l = slope / (max_pot - min_pot)  # (max_pot - min_pot)

        ellipse_factor = self.select_finite_faults()
        ellipse_factor_rectified = T.switch(ellipse_factor < 1., ellipse_factor, 1.)

        if "select_finite_faults" in self.verbose:
            ellipse_factor_rectified = theano.printing.Print("h_factor_rectified")(ellipse_factor_rectified)

        if "select_finite_faults" in self.verbose:
            min_pot = theano.printing.Print("min_pot")(min_pot)
            max_pot = theano.printing.Print("max_pot")(max_pot)

        self.not_l = theano.shared(50.)
        self.ellipse_factor_exponent = theano.shared(2)
        # sigmoid_slope = (self.not_l * (1 / ellipse_factor_rectified)**3) / (max_pot - min_pot)
        sigmoid_slope = 950 - 950 * ellipse_factor_rectified ** self.ellipse_factor_exponent + self.not_l
        # l = T.switch(self.select_finite_faults(), 5000 / (max_pot - min_pot), 50 / (max_pot - min_pot))

        if "select_finite_faults" in self.verbose:
            sigmoid_slope = theano.printing.Print("l")(sigmoid_slope)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad]),
            self.scalar_field_at_surface_points_values,
            T.stack([min_pot - boundary_pad])
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        n_surface_op_float_sigmoid = T.repeat(self.n_surface_op_float[[0], :], 2, axis=1)
        # TODO: instead -1 at the border look for the average distance of the input!

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, 1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        partial_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]),
                       T.arange(0, n_surface_op_float_sigmoid.shape[1], 2, dtype='int64')],
            non_sequences=[Z_x, sigmoid_slope, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        partial_block = partial_block.sum(axis=0)

        # Add name to the theano node
        partial_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            partial_block = theano.printing.Print(partial_block.name)(partial_block)

        return [Z_x, partial_block]

    def compute_a_fault(self,
                        len_i_0, len_i_1,
                        len_f_0, len_f_1,
                        n_form_per_serie_0, n_form_per_serie_1,
                        u_grade_iter,
                        fault_matrix, final_block
                        ):
        """
        Function that loops each fault, generating a potential field for each on them with the respective block model

        Args:
            len_i_0: Lenght of rest of previous series
            len_i_1: Lenght of rest for the computed series
            len_f_0: Lenght of dips of previous series
            len_f_1: Length of dips of the computed series
            n_form_per_serie_0: Number of surfaces of previous series
            n_form_per_serie_1: Number of surfaces of the computed series

        Returns:
            theano.tensor.matrix: block model derived from the df that afterwards is used as a drift for the "real"
            data
        """

        # THIS IS THE FAULTS BLOCK.
        # ==================
        # Preparing the data
        # ==================

        # compute the youngest fault and consecutively the others

        # Theano shared
        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_surface_op = self.n_surface[n_form_per_serie_0: n_form_per_serie_1]
        self.n_surface_op_float = self.surface_values[:, n_form_per_serie_0: n_form_per_serie_1 + 1]
        self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]
        if 'n_surface' in self.verbose:
            self.n_surface_op = theano.printing.Print('n_surface_fault')(self.n_surface_op)

        self.n_universal_eq_T_op = u_grade_iter

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # Theano Var
        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

        # Updating the interface points involved in the iteration. This is important for the fault drift
        self.len_i_0 = len_i_0
        self.len_i_1 = len_i_1

        if 'lengths' in self.verbose:
            self.len_i_0 = theano.printing.Print('len_i_0')(self.len_i_0)
            self.len_i_1 = theano.printing.Print('len_i_1')(self.len_i_1)
            self.len_points = theano.printing.Print('len_points')(self.len_points)

        # Extracting a the subset of the fault matrix to the scalar field of the current iterations
        faults_relation_op = self.fault_relation[:, T.cast(self.n_surface_op-1, 'int8')]
        faults_relation_rep = T.repeat(faults_relation_op, 1)

        if 'faults_relation' in self.verbose:
            faults_relation_rep = theano.printing.Print('SELECT')(faults_relation_rep)
        if len(self.gradients) is not 0:
            self.fault_matrix = fault_matrix[::5][T.nonzero(T.cast(faults_relation_rep, "int8"))[0], :]
        else:
            self.fault_matrix = fault_matrix[::2][T.nonzero(T.cast(faults_relation_rep, "int8"))[0], :]

        if 'fault_matrix_loop' in self.verbose:
            self.fault_matrix = theano.printing.Print('self fault matrix')(self.fault_matrix)

        # ================================
        # Computing the fault scalar field
        # ================================

        potential_field_values, faults_matrix = self.block_fault(slope=1000)

        # Update the block matrix
        final_block = T.set_subtensor(
                    final_block[0, :],
                    faults_matrix[0])#T.cast(T.cast(faults_matrix, 'bool'), 'int8'))

        # Update the potential field matrix
       # potential_field_values = self.scalar_field_at_all()

        final_block = T.set_subtensor(
                    final_block[1, :],
                    potential_field_values)

        # Store the potential field at the surface_points
        self.final_potential_field_at_faults_op = T.set_subtensor(self.final_potential_field_at_faults_op[self.n_surface_op-1],
                                                                  self.scalar_field_at_surface_points_values)

        aux_ind = T.max(self.n_surface_op, 0)

        if len(self.gradients) is not 0:
            weights = self.extend_dual_kriging()
            gradients = self.gradient_field_at_all(weights, self.gradients)
            final_block = T.set_subtensor(
                final_block[2:, :],
                gradients)
            # Setting the values of the fault matrix computed in the current iteration
            fault_matrix = T.set_subtensor(fault_matrix[(aux_ind - 1) * 5:aux_ind * 5, :], final_block)

        else:
            # Setting the values of the fault matrix computed in the current iteration
            fault_matrix = T.set_subtensor(fault_matrix[(aux_ind-1)*2:aux_ind*2, :], final_block)

        return fault_matrix, self.final_potential_field_at_faults_op,

    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         u_grade_iter,
                         final_block, scalar_field_at_form,
                         #fault_block
                         ):

        """
        Function that loops each series, generating a potential field for each on them with the respective block model

        Args:
             len_i_0: Lenght of rest of previous series
             len_i_1: Lenght of rest for the computed series
             len_f_0: Lenght of dips of previous series
             len_f_1: Length of dips of the computed series
             n_form_per_serie_0: Number of surfaces of previous series
             n_form_per_serie_1: Number of surfaces of the computed series

        Returns:
             theano.tensor.matrix: final block model
        """

        # Setting the fault contribution to kriging from the previous loop
       # self.fault_matrix = fault_block

        # THIS IS THE FINAL BLOCK. (DO I NEED TO LOOP THE FAULTS FIRST? Yes you do)
        # ==================
        # Preparing the data
        # ==================

        # Vector that controls the points that have been simulated in previous iterations
        self.yet_simulated = T.nonzero(T.le(final_block[1, :], scalar_field_at_form[n_form_per_serie_0 - 1]))[0] # This -1 comes to get the last scalar field value (the bottom) of the previous series
        self.yet_simulated.name = 'Yet simulated LITHOLOGY node'
        if 'fault_mask' in self.verbose:
            self.yet_simulated = theano.printing.Print('fault_mask')(self.yet_simulated)
            scalar_field_at_form = theano.printing.Print('scalar_field_at_form_out')(scalar_field_at_form)

        # Theano shared
        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_surface_op = self.n_surface[n_form_per_serie_0: n_form_per_serie_1]
        self.n_surface_op_float = self.surface_values[:, n_form_per_serie_0: (n_form_per_serie_1 + 1)]
        self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]

        self.n_universal_eq_T_op = u_grade_iter

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # Theano Var
        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

        if 'rest_layer_points' in self.verbose:
            self.rest_layer_points = theano.printing.Print('rest_layer_points')(self.rest_layer_points)

        if 'ref_layer_points' in self.verbose:
            self.ref_layer_points = theano.printing.Print('ref_layer_points')(self.ref_layer_points)

        # For the contribution of the df I did not find a better way
        self.len_i_0 = len_i_0
        self.len_i_1 = len_i_1

        # Printing
        if 'n_surface' in self.verbose:
            self.n_surface_op = theano.printing.Print('n_surface_series')(self.n_surface_op)

        if 'lengths' in self.verbose:
            self.len_i_0 = theano.printing.Print('len_i_0')(self.len_i_0)
            self.len_i_1 = theano.printing.Print('len_i_1')(self.len_i_1)
            self.len_points = theano.printing.Print('len_points')(self.len_points)
        # ====================
        # Computing the series
        # ====================

        weights = self.extend_dual_kriging()

        scalar_field_values, scalar_field_contribution = self.block_series()

        # Updating the block model with the lithology block

        # Set model id
        final_block = T.set_subtensor(
            final_block[0, self.yet_simulated],
            scalar_field_contribution[0])

        final_block = T.set_subtensor(
            final_block[0, -2*self.len_points:],
            0)

        #scalar_field_values = self.scalar_field_at_all()

        # Set scalar field
        final_block = T.set_subtensor(
            final_block[1, self.yet_simulated],
            scalar_field_values)

        final_block = T.set_subtensor(
            final_block[1, -2 * self.len_points:],
            0)

        # Set additional values
        final_block = T.set_subtensor(
            final_block[2:, self.yet_simulated],
            scalar_field_contribution[1:])

        # Reset scalar field at the surface_points to 0
        # final_block = T.set_subtensor(
        #     final_block[:, -2 * self.len_points:],
        #     0)

        # Store the potential field at the surface_points
        self.final_scalar_field_at_surfaces_op = T.set_subtensor(
            self.final_scalar_field_at_surfaces_op[self.n_surface_op - 1],
            self.scalar_field_at_surface_points_values)

        if len(self.gradients) is not 0:
            gradients = self.gradient_field_at_all(weights, self.gradients)
            final_block = T.set_subtensor(
                final_block[2:, self.yet_simulated],
                gradients)

        return final_block, self.final_scalar_field_at_surfaces_op

    def compute_geological_model(self, weights=None):

        # Change the flag to extend the graph in the compute fault and compute series function
        lith_matrix = T.zeros((0, 0, self.grid_val_T.shape[0] + 2 * self.len_points))

        # Init to matrix which contains the block and scalar field of every fault
        self.fault_matrix = T.zeros((self.n_faults * 2, self.grid_val_T.shape[0] + 2 * self.len_points))
        self.fault_matrix_f = T.zeros((self.n_faults * 2, self.grid_val_T.shape[0] + 2 * self.len_points))

        self.final_scalar_field_at_surfaces_op = self.final_scalar_field_at_surfaces
        self.final_potential_field_at_faults_op = self.final_scalar_field_at_faults

        # Init df block. Here we store the block and potential field results of one iteration
        self.fault_block_init = T.zeros((2, self.grid_val_T.shape[0] + 2 * self.len_points))
        self.fault_block_init.name = 'final block of df init'
        self.yet_simulated = T.nonzero(T.eq(self.fault_block_init[0, :], 0))[0]

        # Compute Faults
        if self.n_faults.get_value() != 0 or self.is_fault:

            # Looping
            fault_loop, updates3 = theano.scan(
                fn=self.compute_a_fault,
                    outputs_info=[
                              dict(initial=self.fault_matrix, taps=[-1]),
                              None],  # This line may be used for the df network
                sequences=[dict(input=self.len_series_i[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.len_series_f[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_surfaces_per_series[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_universal_eq_T[:self.n_faults + 1], taps=[0])],
                non_sequences=self.fault_block_init,
                name='Looping df',
                return_list=True,
                profile=False
            )

            # We return the last iteration of the fault matrix
            self.fault_matrix_f = fault_loop[0][-1]
            self.fault_matrix = self.fault_matrix_f[::2]
          #  fault_block = self.fault_matrix[:, :-2 * self.len_points]
            # For this we return every iteration since is each potential field at interface
            self.pfai_fault = fault_loop[1]

        # Check if there are lithologies to compute
        if len(self.len_series_f.get_value()) - 1 > self.n_faults.get_value() or self.is_lith:

            # The +1 is due to the scalar field
            self.lith_block_init = T.zeros((self.surface_values.shape[0] + 1,
                                            self.grid_val_T.shape[0] + 2 * self.len_points))

            # Compute Lithologies
            lith_loop, updates2 = theano.scan(
                 fn=self.compute_a_series,
                 outputs_info=[self.lith_block_init, self.final_scalar_field_at_surfaces_op],
                 sequences=[dict(input=self.len_series_i[self.n_faults:], taps=[0, 1]),
                            dict(input=self.len_series_f[self.n_faults:], taps=[0, 1]),
                            dict(input=self.n_surfaces_per_series[self.n_faults:], taps=[0, 1]),
                            dict(input=self.n_universal_eq_T[self.n_faults:], taps=[0])],
                # non_sequences=[self.fault_matrix],
                 name='Looping surface_points',
                 profile=False,
                 return_list=True
            )

            lith_matrix = lith_loop[0][-1]
            self.pfai_lith = lith_loop[1]

        pfai = T.vertical_stack(self.pfai_fault, self.pfai_lith)
        return [lith_matrix[:, :-2 * self.len_points], self.fault_matrix_f[:, :-2 * self.len_points], pfai]

    def compute_geological_model_gradient(self, gradients = [], weights=None):
        # TODO update it to several properties!!

        self.gradients = ['Gx', 'Gy', 'Gz']#theano.shared(['Gx', 'Gy', 'Gz'])

        # Change the flag to extend the graph in the compute fault and compute series function
        lith_matrix = T.zeros((0, 0, self.grid_val_T.shape[0] + 2 * self.len_points))

        # Init to matrix which contains the block and scalar field of every fault
        self.fault_matrix = T.zeros((self.n_faults*5, self.grid_val_T.shape[0]))
        self.fault_matrix_f = T.zeros((self.n_faults*5, self.grid_val_T.shape[0]))

        # TODO I think I just have to change the size of the lith_block
        self.lith_block_init = T.zeros((5, self.grid_val_T.shape[0]))
        self.fault_block_init = T.zeros((5, self.grid_val_T.shape[0]))
        self.fault_block_init.name = 'final block of df init'
        # Compute Faults
        if self.n_faults.get_value() != 0 or self.is_fault:

            # Looping
            fault_loop, updates3 = theano.scan(
                fn=self.compute_a_fault,
                    outputs_info=[
                              dict(initial=self.fault_matrix, taps=[-1]),
                              None],  # This line may be used for the df network
                sequences=[dict(input=self.len_series_i[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.len_series_f[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_surfaces_per_series[:self.n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_universal_eq_T[:self.n_faults + 1], taps=[0])],
                non_sequences=self.fault_block_init,
                name='Looping df',
                return_list=True,
                profile=False
            )

            # We return the last iteration of the fault matrix
            self.fault_matrix_f = fault_loop[0][-1]
            self.fault_matrix = self.fault_matrix_f[::5]
          #  fault_block = self.fault_matrix[:, :-2 * self.len_points]
            # For this we return every iteration since is each potential field at interface
            self.pfai_fault = fault_loop[1]

        # Check if there are lithologies to compute
        if len(self.len_series_f.get_value()) - 1 > self.n_faults.get_value() or self.is_lith:

            # Compute Lithologies
            lith_loop, updates2 = theano.scan(
                 fn=self.compute_a_series,
                 outputs_info=[self.lith_block_init, self.final_scalar_field_at_surfaces_op],
                 sequences=[dict(input=self.len_series_i[self.n_faults:], taps=[0, 1]),
                            dict(input=self.len_series_f[self.n_faults:], taps=[0, 1]),
                            dict(input=self.n_surfaces_per_series[self.n_faults:], taps=[0, 1]),
                            dict(input=self.n_universal_eq_T[self.n_faults:], taps=[0])],
                # non_sequences=[self.fault_matrix],
                 name='Looping surface_points',
                 profile=False,
                 return_list=True
            )

            lith_matrix = lith_loop[0][-1]
            self.pfai_lith = lith_loop[1]

        pfai = T.vertical_stack(self.pfai_fault, self.pfai_lith)
        return [lith_matrix[:, :-2 * self.len_points], self.fault_matrix_f[:, :-2 * self.len_points], pfai]

    # def compute_weights_op(self,
    #                        len_i_0, len_i_1,
    #                        len_f_0, len_f_1,
    #                        n_form_per_serie_0, n_form_per_serie_1,
    #                        u_grade_iter,
    #                        weights):
    #     # Theano shared
    #     self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T[
    #                                                n_form_per_serie_0: n_form_per_serie_1]
    #     self.n_surface_op = self.n_surface[n_form_per_serie_0: n_form_per_serie_1]
    #     self.n_surface_op_float = self.surface_values[n_form_per_serie_0: (n_form_per_serie_1 + 1)]
    #     self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]
    #
    #     self.n_universal_eq_T_op = u_grade_iter
    #
    #     self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
    #     self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))
    #
    #     # Theano Var
    #     self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
    #     self.azimuth = self.azimuth_all[len_f_0: len_f_1]
    #     self.polarity = self.polarity_all[len_f_0: len_f_1]
    #
    #     self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
    #     self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]
    #
    #     # For the contribution of the df I did not find a better way
    #     self.len_i_0 = len_i_0
    #     self.len_i_1 = len_i_1
    #
    #     weights = T.set_subtensor(weights[:, len_i_0: len_i_1], self.solve_kriging())
    #     return weights
    #
    # def compute_weights(self):
    #     weights_init = T.zeros((1, self.len_points + self.dips_position_tiled.shape[0]))
    #     # Compute Lithologies
    #     weights_loop, updates2 = theano.scan(
    #          fn=self.compute_weights_op,
    #          outputs_info=[weights_init],
    #          sequences=[dict(input=self.len_series_i, taps=[0, 1]),
    #                     dict(input=self.len_series_f, taps=[0, 1]),
    #                     dict(input=self.n_surfaces_per_series, taps=[0, 1]),
    #                     dict(input=self.n_universal_eq_T, taps=[0])],
    #         # non_sequences=[self.fault_matrix],
    #          name='Looping surface_points',
    #          profile=False,
    #          return_list=True
    #     )
    #
    #     return weights_loop

    # ==================================
    # Geophysics
    # ==================================

    def switch_densities(self, n_surface, density, density_block):

        density_block = T.switch(T.eq(density_block, n_surface), density, density_block)
        return density_block

    def compute_forward_gravity(self): # densities, tz, select,

        # TODO: Assert outside that densities is the same size as surfaces (minus df)
        # Compute the geological model
        lith_matrix, fault_matrix, pfai = self.compute_geological_model()

        # if n_faults == 0:
        #     surfaces = T.concatenate([self.n_surface[::-1], T.stack([0])])
        # else:
        #     surfaces = T.concatenate([self.n_surface[:n_faults-1:-1], T.stack([0])])
        #
        #     if False:
        #         surfaces = theano.printing.Print('surfaces')(surfaces)
        #
        # # Substitue lithologies by its density
        # density_block_loop, updates4 = theano.scan(self.switch_densities,
        #                             outputs_info=[lith_matrix[0]],
        #                              sequences=[surfaces, self.densities],
        #                             return_list = True
        # )

        # if False:
        #     density_block_loop_f = T.set_subtensor(density_block_loop[-1][-1][self.weigths_index], self.weigths_weigths)
        #
        # else:
        density_block_loop_f = lith_matrix[0]


        if 'density_block' in self.verbose:
            density_block_loop_f = theano.printing.Print('density block')(density_block_loop_f)

        n_measurements = self.tz.shape[0]
        # Tiling the density block for each measurent and picking just the closer to them. This has to be possible to
        # optimize

        #densities_rep = T.tile(density_block_loop[-1][-1], n_measurements)
        densities_rep = T.tile(density_block_loop_f, n_measurements)
        densities_selected = densities_rep[T.nonzero(T.cast(self.select, "int8"))[0]]
        densities_selected_reshaped = densities_selected.reshape((n_measurements, -1))
        #
        # # density times the component z of gravity
        grav = densities_selected_reshaped * self.tz

        #return [lith_matrix, self.fault_matrix, pfai, grav.sum(axis=1)]
        return [lith_matrix, fault_matrix, grav.sum(axis=1), pfai]


    def compute_grad(self, n_faults=None):
        sol = self.block_series()
        return theano.grad(sol.sum(), self.rest_layer_points_all)

    # def compute_grad2(self, n_faults=None):
    #     sol = self.compute_a_series(
    #         self.len_series_i[n_faults:][0], self.len_series_i[n_faults:][-1],
    #         self.len_series_f[n_faults:][0], self.len_series_f[n_faults:][-1],
    #         self.n_surfaces_per_series[n_faults:][0], self.n_surfaces_per_series[n_faults:][-1],
    #         self.n_universal_eq_T[n_faults:],
    #         self.lith_block_init, self.final_scalar_field_at_surfaces,
    #         self.fault_matrix
    #     )
    #     return theano.grad(sol[0].sum(), self.rest_layer_points_all)
    #
    # def compute_grad3(self, n_faults=None
    #                   ):
    #     lith_matrix, fault_matrix, pfai = self.compute_geological_model(n_faults=n_faults)
    #     return theano.grad(lith_matrix[0].sum(), self.rest_layer_points_all)
    #
    #


class TheanoOptions(object):
    def __init__(self,  output='geology', optimizer='fast_compile', verbose=[0], dtype='float32',
                 is_fault=None, is_lith=None):
        # OPTIONS
        # -------
        if verbose is np.nan:
            self.verbose = [None]
        else:
            self.verbose = verbose
        self.dot_version = False

        theano.config.floatX = dtype
        theano.config.optimizer = optimizer


class TheanoGeometry(TheanoOptions):
    def __init__(self, output='geology', optimizer='fast_compile', verbose=[0], dtype='float32',
                 is_fault=None, is_lith=None):

        # # OPTIONS
        # # -------
        # if verbose is np.nan:
        #     self.verbose = [None]
        # else:
        #     self.verbose = verbose
        # self.dot_version = False
        #
        # theano.config.floatX = dtype
        # theano.config.optimizer = optimizer
        super(TheanoGeometry, self).__init__(optimizer='fast_compile', verbose=[0], dtype='float32',)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        # This is not accumulative
        self.number_of_points_per_surface_T = theano.shared(np.zeros(3, dtype='int32')) #TODO is DEP?
        self.number_of_points_per_surface_T_op = T.vector('Number of points per surface used to split rest-ref',
                                                          dtype='int32')
        self.npf_op = T.cumsum(T.concatenate((T.stack(0), self.number_of_points_per_surface_T_op[:-1])))

        # # FORMATIONS
        # # ----------
        # self.n_surface = theano.shared(np.arange(2, 5, dtype='int32'), "ID of the surface")
        # self.n_surface_op = self.n_surface
        # self.surface_values = theano.shared((np.arange(2, 4, dtype=dtype).reshape(2, -1)), "Value of the surface to compute")
        # self.n_surface_op_float = self.surface_values


        # KRIGING
        # -------
        self.a_T = theano.shared(np.cast[dtype](-1.), "Range")
        self.c_o_T = theano.shared(np.cast[dtype](-1.), 'Covariance at 0')
        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of gradients')
        self.nugget_effect_scalar_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of scalar')
        self.n_universal_eq_T = theano.shared(np.zeros(5, dtype='int32'), "Grade of the universal drift")
        self.n_universal_eq_T_op = theano.shared(3)

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # VARIABLES
        # ---------
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")

        self.surface_points_all = T.matrix("All the surface_points points at once")
        self.len_points = self.surface_points_all.shape[0] - self.number_of_points_per_surface_T_op.shape[0]
        # Tiling dips to the 3 spatial coordinations
        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series. I initialized them as the whole arrays but then they will take
        # the data of every potential field
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all

        self.ref_layer_points_all = self.set_rest_ref_matrix()[0]
        self.rest_layer_points_all = self.set_rest_ref_matrix()[1]

        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

        self.fault_drift = T.matrix('Drift matrix due to faults')

    def set_rest_ref_matrix(self):
        ref_positions = T.cumsum(T.concatenate((T.stack(0), self.number_of_points_per_surface_T_op[:-1] + 1)))
        ref_points = T.repeat(self.surface_points_all[ref_positions], self.number_of_points_per_surface_T_op, axis=0)

        rest_mask = T.ones(T.stack(self.surface_points_all.shape[0]), dtype='int16')
        rest_mask = T.set_subtensor(rest_mask[ref_positions], 0)
        rest_points = self.surface_points_all[T.nonzero(rest_mask)[0]]
        return [ref_points, rest_points, rest_mask, T.nonzero(rest_mask)[0]]

    @staticmethod
    def squared_euclidean_distances(x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Args:
            x_1 (theano.tensor.matrix): shape n_points x number dimension
            x_2 (theano.tensor.matrix): shape n_points x number dimension

        Returns:
            theano.tensor.matrix: Distancse matrix. shape n_points x n_points
        """

        # T.maximum avoid negative numbers increasing stability
        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 1e-12
        ))

        if False:
            sqd = theano.printing.Print('sed')(sqd)

        return sqd

    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """

        # Calculating the dimensions of the
        length_of_CG = self.dips_position_tiled.shape[0]
        length_of_CGI = self.rest_layer_points.shape[0]
        length_of_U_I = self.n_universal_eq_T_op

        # Self fault matrix contains the block and the potential field (I am not able to split them). Therefore we need
        # to divide it by 2
        length_of_faults = T.cast(self.fault_drift.shape[0], 'int32')
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_faults = theano.printing.Print("length_of_faults")(length_of_faults)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
