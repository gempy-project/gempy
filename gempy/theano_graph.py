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
    This class is used to help to divide the construction of the graph into sensical parts. All its methods build a part
    of the graph. Every method can be seen as a branch and collection of branches until the last method that will be the
    whole tree. Every part of the graph could be compiled separately but as we increase the complexity the input of each
    of these methods is more and more difficult to provide (if you are in a branch close to the trunk you need all the
    results of the branches above)
    """
    def __init__(self, output='geology', optimizer='fast_compile', verbose=[0], dtype='float32'):
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
        self.verbose = verbose
        self.dot_version = False

        theano.config.floatX = dtype
        theano.config.optimizer = optimizer
        # Creation of symbolic parameters
        # =============
        # Constants
        # =============

        # Arbitrary values to get the same results that GeoModeller. These parameters are a mystery for me yet. I have
        # to ask. In my humble opinion they weight the contribution of the interfaces against the
        # foliations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        # =================
        # INITIALIZE SHARED
        # =================
        self.n_universal_eq_T = theano.shared(np.arange(2, dtype='int32'), "Grade of the universal drift")
        self.n_universal_eq_T_op = theano.shared(0)

        self.a_T = theano.shared(np.cast[dtype](1.), "Range")
        self.c_o_T = theano.shared(np.cast[dtype](1.), 'Covariance at 0')
        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](0.01), 'Nugget effect of gradients')
        self.nugget_effect_scalar_T = theano.shared(np.cast[dtype](1e-6), 'Nugget effect of scalar')

        self.grid_val_T = theano.shared(np.cast[dtype](np.zeros((2, 200))), 'Coordinates of the grid '
                                                                          'points to interpolate')
        # Shape is 9x2, 9 drift funcitons and 2 points
        self.universal_grid_matrix_T = theano.shared(np.cast[dtype](np.zeros((9, 2))))
        self.final_block = theano.shared(np.cast[dtype](np.zeros((1, 3))), "Final block computed")

        # This parameters give me the shape of the different groups of data. I pass all data together and I threshold it
        # using these values to the different potential fields and formations
        self.len_series_i = theano.shared(np.arange(2, dtype='int32'), 'Length of interfaces in every series')
        self.len_series_f = theano.shared(np.arange(2, dtype='int32'), 'Length of foliations in every series')
        self.n_formations_per_serie = theano.shared(np.arange(3, dtype='int32'), 'List with the number of formations')
        self.n_formation = theano.shared(np.arange(2,5, dtype='int32'), "Value of the formation")
        self.n_formation_float = theano.shared(np.arange(2, 5, dtype='float32'), "Value of the formation to compute")
        self.number_of_points_per_formation_T = theano.shared(np.zeros(3, dtype='int32'))
        self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per formation accumulative')
        # Init fault relation matrix
        self.fault_relation = theano.shared(np.array([[0, 1, 0, 1],
                                                      [0, 0, 1, 1],
                                                      [0, 0, 0, 1],
                                                      [0, 0, 0, 0]]), 'fault relation matrix')

        # ======================
        # VAR
        # ======================
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")
        self.ref_layer_points_all = T.matrix("Reference points for every layer")
        self.rest_layer_points_all = T.matrix("Rest of the points of the layers")

        # Tiling dips to the 3 spatial coordinations
        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series. I initialized them as the whole arrays but then they will take
        # the data of every potential field
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all
        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

        self.len_points = self.rest_layer_points_all.shape[0]
        self.len_i_0 = 0
        self.len_i_1 = 1
        self.final_scalar_field_at_formations = theano.shared(np.zeros(self.n_formations_per_serie.get_value().sum(), dtype=dtype))
        self.final_scalar_field_at_faults = theano.shared(np.zeros(self.n_formations_per_serie.get_value().sum(), dtype=dtype))

        self.final_scalar_field_at_formations_op = self.final_scalar_field_at_formations
        self.final_potential_field_at_faults_op = self.final_scalar_field_at_faults

        # Init Results
        # Init lithology block. Here we store the block and potential field results
        self.lith_block_init = T.zeros((2, self.grid_val_T.shape[0]))
        self.lith_block_init.name = 'final block of lithologies init'

        # Init faults block. Here we store the block and potential field results of one iteration
        self.fault_block_init = T.zeros((2, self.grid_val_T.shape[0]))
        self.fault_block_init.name = 'final block of faults init'
        self.yet_simulated = T.nonzero(T.eq(self.fault_block_init[0, :], 0))[0]

        # Here we store the value of the potential field at interfaces
        self.pfai_fault = T.zeros((0, self.n_formations_per_serie[-1]))
        self.pfai_lith = T.zeros((0, self.n_formations_per_serie[-1]))

        if output is 'gravity':
            self.densities = theano.shared(np.cast[dtype](np.zeros(3)), "List with the densities")
            self.tz = theano.shared(np.cast[dtype](np.zeros((1, 3))), "Component z")
            self.select = theano.shared(np.cast['int8'](np.zeros(3)), "Select nearby cells")
            # Init gray voxels for gravity
            self.weigths_weigths = theano.shared(np.ones(0))
            self.weigths_index = theano.shared(np.ones(0, dtype='int32'))

    def input_parameters_list(self):
        """
        Create a list with the symbolic variables to use when we compile the theano function

        Returns:
            list: [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
                   self.ref_layer_points_all, self.rest_layer_points_all]
        """
        ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
               self.ref_layer_points_all, self.rest_layer_points_all]
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
            2 * x_1.dot(x_2.T), 1e-21
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
        length_of_faults = T.cast(self.fault_matrix.shape[0]/2, 'int32')
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_faults = theano.printing.Print("length_of_faults")(length_of_faults)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    def cov_interfaces(self):
        """
        Create covariance function for the interfaces

        Returns:
            theano.tensor.matrix: covariance of the interfaces. Shape number of points in rest x number of
            points in rest

        """

        # Compute euclidian distances
        sed_rest_rest = self.squared_euclidean_distances(self.rest_layer_points, self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distances(self.ref_layer_points, self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distances(self.rest_layer_points, self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distances(self.ref_layer_points, self.ref_layer_points)

        # Covariance matrix for interfaces
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

        C_I += T.eye(C_I.shape[0])*self.nugget_effect_scalar_T
        # Add name to the theano node
        C_I.name = 'Covariance Interfaces'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_I = theano.printing.Print('Cov interfaces')(C_I)

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

        # Cross-Covariance gradients-interfaces
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
            theano.tensor.matrix: Drift matrix for the interfaces. Shape number of points in rest x 3**degree drift
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
            U_I.name = 'Drift interfaces'
            U_G.name = 'Drift foliations'

        return U_I[:, :self.n_universal_eq_T_op], U_G[:, :self.n_universal_eq_T_op]

    def faults_matrix(self):
        """
        This function creates the part of the graph that generates the faults function creating a "block model" at the
        references and the rest of the points. Then this vector has to be appended to the covariance function

        Returns:

            list:

            - theano.tensor.matrix: Drift matrix for the interfaces. Shape number of points in rest x n faults. This drif
              is a simple addition of an arbitrary number

            - theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x n faults. For
              discrete values this matrix will be null since the derivative of a constant is 0
        """

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults = self.matrices_shapes()[:4]

        # self.fault_matrix contains the faults volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
        interface_loc = self.fault_matrix.shape[1] - 2*self.len_points

        fault_matrix_at_interfaces_rest = self.fault_matrix[::2,
                                          interface_loc + self.len_i_0: interface_loc + self.len_i_1]
        fault_matrix_at_interfaces_ref = self.fault_matrix[::2,
                                         interface_loc+self.len_points+self.len_i_0: interface_loc+self.len_points+self.len_i_1]

        F_I = (fault_matrix_at_interfaces_ref - fault_matrix_at_interfaces_rest)+0.0001

        # As long as the drift is a constant F_G is null
        F_G = T.zeros((length_of_faults, length_of_CG)) + 0.0001

        if str(sys._getframe().f_code.co_name) in self.verbose:
            F_I = theano.printing.Print('Faults interfaces matrix')(F_I)
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
        C_I = self.cov_interfaces()
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
        interfaces). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.

        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """

        grid_val = self.grid_val_T[self.yet_simulated, :]

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
        DK_parameters = self.solve_kriging()

        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        # TODO IMP: Change the tile by a simple dot op
        DK_weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        if self.dot_version:
            DK_weights = DK_parameters

        return DK_weights

    def gradient_contribution(self, grid_val=None, weights=None):
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
            sigma_0_grad = theano.printing.Print('Universal terms contribution')(sigma_0_grad)

        return sigma_0_grad

    def interface_contribution(self, grid_val=None, weights=None):
        """
          Computation of the contribution of the interfaces at every point to interpolate

          Returns:
              theano.tensor.vector: Contribution of all interfaces (input) at every point to interpolate
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
        sigma_0_interf.name = 'Contribution of the interfaces to the potential field at every point of the grid'

        return sigma_0_interf

    def universal_drift_contribution(self, grid_val=None, weights=None, a=0, b=100000000):
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
        universal_grid_interfaces_matrix = self.universal_grid_matrix_T[:, self.yet_simulated[a: b]]

        # These are the magic terms to get the same as geomodeller
        gi_rescale_aux = T.repeat(self.gi_reescale, 9)
        gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(gi_rescale_aux[:self.n_universal_eq_T_op], (grid_val.shape[0], 1)).T

        # Drif contribution
        f_0 = (T.sum(
            weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_reescale * _aux_magic_term *
            universal_grid_interfaces_matrix[:self.n_universal_eq_T_op]
            , axis=0))

        if self.dot_version:
            f_0 = T.dot(
                weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] , self.gi_reescale * _aux_magic_term *
                universal_grid_interfaces_matrix[:self.n_universal_eq_T_op])

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

        return f_0

    def faults_contribution(self, weights=None, a=0, b=100000000):
        """
        Computation of the contribution of the faults drift at every point to interpolate. To get these we need to
        compute a whole block model with the faults data

        Returns:
            theano.tensor.vector: Contribution of the faults drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        fault_matrix_selection_non_zero = (self.fault_matrix[::2, self.yet_simulated[a:b]]+1)

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

        sigma_0_grad = self.gradient_contribution(grid_val[a:b], weights[:, a:b])
        sigma_0_interf = self.interface_contribution(grid_val[a:b], weights[:, a:b])
        f_0 = self.universal_drift_contribution(grid_val[a:b],weights[:, a:b], a, b)
        f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1 + 50 - (10 * val[0]))

        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def scalar_field_at_all(self):
        """
        Compute the potential field at all the interpolation points, i.e. grid plus rest plus ref
        Returns:
            theano.tensor.vector: Potential fields at all points

        """
        grid_val = self.x_to_interpolate()
        weights = self.extend_dual_kriging()

        grid_shape = T.stack(grid_val.shape[0])
        Z_x_init = T.zeros(grid_shape, dtype='float32')
        if 'grid_shape' in self.verbose:
            grid_shape =  theano.printing.Print('grid_shape')(grid_shape)

        steps = 1e13/self.matrices_shapes()[-1]/grid_shape
        slices = T.concatenate((T.arange(0, grid_shape[0], steps[0], dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        Z_x_loop, updates3 = theano.scan(
            fn=self.scalar_field_loop,
            outputs_info=[Z_x_init],
            sequences=[dict(input=slices, taps=[0, 1])],
            non_sequences=[grid_val, weights, self.n_formation_op_float],
            profile=False,
            name='Looping grid',
            return_list=True)

        Z_x = Z_x_loop[-1][-1]
        Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)

        return Z_x

    def block_series(self):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        # Graph to compute the potential field
        Z_x = self.scalar_field_at_all()

        # Max and min values of the potential field.
        # TODO this may be expensive because I guess that is a sort algorithm. We just need a +inf and -inf... I guess
        max_pot = 1000
        min_pot = -1000

        # Value of the potential field at the interfaces of the computed series
        self.scalar_field_at_interfaces_values = Z_x[-2*self.len_points: -self.len_points][self.npf_op]

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((T.stack([max_pot]),   self.scalar_field_at_interfaces_values, T.stack([min_pot])))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Loop to segment the distinct lithologies
        def compare(a, b, n_formation, Zx):
            """
            Treshold of the points to interpolate given 2 potential field values. TODO: This function is the one we
            need to change for a sigmoid function

            Args:
                a (scalar): Upper limit of the potential field
                b (scalar): Lower limit of the potential field
                n_formation (scalar): Value given to the segmentation, i.e. lithology number
                Zx (vector): Potential field values at all the interpolated points

            Returns:
                theano.tensor.vector: segmented values
            """

            return T.le(Zx, a) * T.ge(Zx, b) * n_formation

        partial_block, updates2 = theano.scan(
            fn=compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]), self.n_formation_op_float],
            non_sequences=Z_x,
            name='Looping compare',
            profile=False)

        # For every formation we get a vector so we need to sum compress them to one dimension
        partial_block = partial_block.sum(axis=0)

        # Add name to the theano node
        partial_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            partial_block = theano.printing.Print(partial_block.name)(partial_block)

        return partial_block

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
            n_form_per_serie_0: Number of formations of previous series
            n_form_per_serie_1: Number of formations of the computed series

        Returns:
            theano.tensor.matrix: block model derived from the faults that afterwards is used as a drift for the "real"
            data
        """

        # THIS IS THE FAULTS BLOCK.
        # ==================
        # Preparing the data
        # ==================

        # compute the youngest fault and consecutively the others

        # Theano shared
        self.number_of_points_per_formation_T_op = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op_float = self.n_formation_float[n_form_per_serie_0: n_form_per_serie_1]
        self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]
        if 'n_formation' in self.verbose:
            self.n_formation_op = theano.printing.Print('n_formation_fault')(self.n_formation_op)

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


        # Extracting a the subset of the fault matrix to the scalar field of the current iterations
        faults_relation_op =  self.fault_relation[:, T.cast(self.n_formation_op-1, 'int8')]
        faults_relation_rep = T.repeat(faults_relation_op, 2)

        if 'faults_relation' in self.verbose:
            faults_relation_rep = theano.printing.Print('SELECT')(faults_relation_rep)

        self.fault_matrix = fault_matrix[T.nonzero(T.cast(faults_relation_rep, "int8"))[0], :]

        if 'fault_matrix_loop' in self.verbose:
            self.fault_matrix = theano.printing.Print('self fault matrix')(self.fault_matrix)

        # ================================
        # Computing the fault scalar field
        # ================================

        faults_matrix = self.block_series()

        # Update the block matrix
        final_block = T.set_subtensor(
                    final_block[0, :],
                    faults_matrix)#T.cast(T.cast(faults_matrix, 'bool'), 'int8'))

        # Update the potential field matrix
        potential_field_values = self.scalar_field_at_all()


        final_block =  T.set_subtensor(
                    final_block[1, :],
                    potential_field_values)

        # Store the potential field at the interfaces
        self.final_potential_field_at_faults_op = T.set_subtensor(self.final_potential_field_at_faults_op[self.n_formation_op-1],
                                                                  self.scalar_field_at_interfaces_values)

        aux_ind = T.max(self.n_formation_op, 0)

        # Setting the values of the fault matrix computed in the current iteration
        fault_matrix = T.set_subtensor(fault_matrix[(aux_ind-1)*2:aux_ind*2, :], final_block)

        return fault_matrix, self.final_potential_field_at_faults_op,

    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         u_grade_iter,
                         final_block, fault_block):

        """
        Function that loops each series, generating a potential field for each on them with the respective block model

        Args:
             len_i_0: Lenght of rest of previous series
             len_i_1: Lenght of rest for the computed series
             len_f_0: Lenght of dips of previous series
             len_f_1: Length of dips of the computed series
             n_form_per_serie_0: Number of formations of previous series
             n_form_per_serie_1: Number of formations of the computed series

        Returns:
             theano.tensor.matrix: final block model
        """

        # Setting the fault contribution to kriging from the previous loop
        self.fault_matrix = fault_block

        # THIS IS THE FINAL BLOCK. (DO I NEED TO LOOP THE FAULTS FIRST? Yes you do)
        # ==================
        # Preparing the data
        # ==================
        # Vector that controls the points that have been simulated in previous iterations
        self.yet_simulated = T.nonzero(T.eq(final_block[0, :], 0))[0]
        self.yet_simulated.name = 'Yet simulated LITHOLOGY node'

        # Theano shared
        self.number_of_points_per_formation_T_op = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op_float = self.n_formation_float[n_form_per_serie_0: n_form_per_serie_1]
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

        # For the contribution of the faults I did not find a better way
        self.len_i_0 = len_i_0
        self.len_i_1 = len_i_1

        # Printing
        if 'yet_simulated' in self.verbose:
            self.yet_simulated = theano.printing.Print(self.yet_simulated.name)(self.yet_simulated)
        if 'n_formation' in self.verbose:
            self.n_formation_op = theano.printing.Print('n_formation_series')(self.n_formation_op)

        # ====================
        # Computing the series
        # ====================

        scalar_field_contribution = self.block_series()

        # Updating the block model with the lithology block
        final_block = T.set_subtensor(
            final_block[0, self.yet_simulated],
            scalar_field_contribution)

        final_block = T.set_subtensor(
            final_block[0, -2*self.len_points:],
            0)

        scalar_field_values = self.scalar_field_at_all()
        final_block = T.set_subtensor(
            final_block[1, self.yet_simulated],
            scalar_field_values)

        # Store the potential field at the interfaces
        self.final_scalar_field_at_formations_op = T.set_subtensor(
            self.final_scalar_field_at_formations_op[self.n_formation_op - 1],
            self.scalar_field_at_interfaces_values)

        return final_block, self.final_scalar_field_at_formations_op

    def compute_geological_model(self, n_faults=0):

        # Change the flag to extend the graph in the compute fault and compute series function
        lith_matrix = T.zeros((0, 0, self.grid_val_T.shape[0] + 2 * self.len_points))

        # Init to matrix which contains the block and scalar field of every fault
        self.fault_matrix = T.zeros((n_faults*2, self.grid_val_T.shape[0]))

        # Compute Faults
        if n_faults != 0:
            # --DEP--? Initialize yet simulated
           # self.yet_simulated = T.nonzero(T.eq(self.fault_block_init[0, :], 0))[0]#T.eq(self.fault_block_init[0, :-2 * self.len_points], 0)

            # Looping
            fault_loop, updates3 = theano.scan(
                fn=self.compute_a_fault,
                    outputs_info=[
                              dict(initial=self.fault_matrix, taps=[-1]),
                              None],  # This line may be used for the faults network
                sequences=[dict(input=self.len_series_i[:n_faults + 1], taps=[0, 1]),
                           dict(input=self.len_series_f[:n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_formations_per_serie[:n_faults + 1], taps=[0, 1]),
                           dict(input=self.n_universal_eq_T[:n_faults + 1], taps=[0])],
                non_sequences=self.fault_block_init,
                name='Looping faults',
                return_list=True,
                profile=False
            )

            # We return the last iteration of the fault matrix
            self.fault_matrix = fault_loop[0][-1]

            # For this we return every iteration since is each potential field at interface
            self.pfai_fault = fault_loop[1]

        # Check if there are lithologies to compute
        if len(self.len_series_f.get_value()) - 1 > n_faults:

             # Compute Lithologies
             lith_loop, updates2 = theano.scan(
                 fn=self.compute_a_series,
                 outputs_info=[self.lith_block_init, None],
                 sequences=[dict(input=self.len_series_i[n_faults:], taps=[0, 1]),
                            dict(input=self.len_series_f[n_faults:], taps=[0, 1]),
                            dict(input=self.n_formations_per_serie[n_faults:], taps=[0, 1]),
                            dict(input=self.n_universal_eq_T[n_faults:], taps=[0])],
                 non_sequences=[self.fault_matrix],
                 name='Looping interfaces',
                 profile=False,
                 return_list=True
             )

             lith_matrix = lith_loop[0][-1]
             self.pfai_lith = lith_loop[1]

        pfai = T.vertical_stack(self.pfai_fault, self.pfai_lith)

        return [lith_matrix[:, :-2 * self.len_points], self.fault_matrix[:, :-2 * self.len_points], pfai]

    # ==================================
    # Geophysics
    # ==================================

    def switch_densities(self, n_formation, density, density_block):

        density_block = T.switch(T.eq(density_block, n_formation), density, density_block)
        return density_block

    def compute_forward_gravity(self, n_faults=0): # densities, tz, select,

        # TODO: Assert outside that densities is the same size as formations (minus faults)
        # Compute the geological model
        lith_matrix, fault_matrix, pfai = self.compute_geological_model(n_faults=n_faults)

        if n_faults == 0:
            formations = T.concatenate([self.n_formation[::-1], T.stack([0])])
        else:
            formations = T.concatenate([self.n_formation[:n_faults-1:-1], T.stack([0])])

            if False:
                formations = theano.printing.Print('formations')(formations)

        # Substitue lithologies by its density
        density_block_loop, updates4 = theano.scan(self.switch_densities,
                                    outputs_info=[lith_matrix[0]],
                                     sequences=[formations, self.densities],
                                    return_list = True
        )

        density_block_loop_f = T.set_subtensor(density_block_loop[-1][-1][self.weigths_index], self.weigths_weigths)
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
        return [lith_matrix, fault_matrix, pfai, grav.sum(axis=1)]


