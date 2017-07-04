"""
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

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'
theano.config.floatX = 'float32'
theano.config.profile_memory = True


class TheanoGraph_pro(object):
    """
    This class is used to help to divide the construction of the graph into sensical parts. All its methods build a part
    of the graph. Every method can be seen as a branch and collection of branches until the last method that will be the
    whole tree. Every part of the graph could be compiled separately but as we increase the complexity the input of each
    of these methods is more and more difficult to provide (if you are in a branch close to the trunk you need all the
    results of the branches above)
    """
    def __init__(self, verbose=[0], dtype='float32'):
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
        self.compute_all = False




        # Creation of symbolic parameters
        # =============
        # Constants
        # =============

        # Arbitrary values to get the same results that GeoModeller. These parameters are a mystery for me yet. I have
        # to ask Gabi and Simon. In my humble opinion they weight the contribution of the interfaces against the
        # foliations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        # ======================
        # INITIALIZE SHARED
        # ====================
        self.u_grade_T = theano.shared(np.arange(2, dtype='int64'), "Grade of the universal drift")
        self.a_T = theano.shared(np.cast[dtype](1.), "Range")
        self.c_o_T = theano.shared(np.cast[dtype](1.), 'Covariance at 0')
        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](0.01))
        # -DEP-
        # self.c_resc = theano.shared(np.cast[dtype](1), "Rescaling factor")
        self.grid_val_T = theano.shared(np.cast[dtype](np.zeros((2, 3))), 'Coordinates of the grid '
                                                                          'points to interpolate')
        # Shape is 9x2, 9 drift funcitons and 2 points
        self.universal_grid_matrix_T = theano.shared(np.cast[dtype](np.zeros((9, 2))))

        # -DEP- Now I pass it as attribute when I create that part of the graph
        #self.n_faults = theano.shared(0, 'Number of faults to compute')#

        self.final_block = theano.shared(np.cast[dtype](np.zeros((1, 3))), "Final block computed")
        #self.yet_simulated = theano.shared(np.ones(3, dtype='int64'), "Points to be computed yet")

        # This parameters give me the shape of the different groups of data. I pass all data together and I threshold it
        # using these values to the different potential fields and formations
        self.len_series_i = theano.shared(np.arange(2, dtype='int64'), 'Length of interfaces in every series')
        self.len_series_f = theano.shared(np.arange(2, dtype='int64'), 'Length of foliations in every series')
        self.n_formations_per_serie = theano.shared(np.arange(3, dtype='int64'), 'List with the number of formations')
        self.n_formation = theano.shared(np.arange(2, dtype='int64'), "Value of the formation")
        self.number_of_points_per_formation_T = theano.shared(np.zeros(3, dtype='int64'))

        # ======================
        # VAR
        #=======================
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

        # Block model out of the faults. It is initialized with shape(0, grid+ ref+rest) so if I do not change it during
        # the computation it does not have any effect


        self.u_grade_T_op = theano.shared(0)
        self.len_points = self.rest_layer_points_all.shape[0]
        self.fault_matrix = T.zeros((0, self.grid_val_T.shape[0] + 2*self.len_points))


        self.len_i_0 = 0
        self.len_i_1 = 1




    # -DEP-
    #def testing(self):
    #    return self.rest_layer_points, self.ref_layer_points

    # -DEP-
    # def yet_simulated_func(self, block=None):
    #     if not block:
    #         self.yet_simulated = T.eq(self.final_block, 0)
    #     else:
    #         self.yet_simulated = T.eq(block, 0)
    #     return self.yet_simulated

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
            2 * x_1.dot(x_2.T), 0
        ))

        return sqd

    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix
        Returns: length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """

        # Calculating the dimensions of the
        length_of_CG = self.dips_position_tiled.shape[0]
        length_of_CGI = self.rest_layer_points.shape[0]
        # if self.u_grade_T.get_value() == 0:
        #     length_of_U_I = 0*self.u_grade_T
        # else:
        #     length_of_U_I = 3**self.u_grade_T
        length_of_U_I = self.u_grade_T_op
        length_of_faults = self.fault_matrix.shape[0]
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
              3 / 4 * (sed_ref_ref / self.a_T) ** 7)))) + 1e-6

        # Add name to the theano node
        C_I.name = 'Covariance Interfaces'
        return C_I

    def cov_gradients(self, verbose=0):
        """
         Create covariance function for the gradiens
         Returns:
             theano.tensor.matrix: covariance of the gradients. Shape number of points in dip_pos x number of
             points in dip_pos

         """

        # Euclidean distances
        sed_dips_dips = self.squared_euclidean_distances(self.dips_position_tiled, self.dips_position_tiled)

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
        C_G = T.fill_diagonal(C_G, -self.c_o_T * (-14 / self.a_T ** 2) + self.nugget_effect_grad_T)

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

    # -DEP-
    # def cartesian_dist_reference_to_rest(self):
    #     # Cartesian distances between reference points and rest
    #     hx = T.stack(
    #         (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
    #         (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
    #         (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2])
    #     ).T
    #
    #     return hx

    def universal_matrix(self):
        """
        Create the drift matrices for the potential field and its gradient
        Returns:
            theano.tensor.matrix: Drift matrix for the interfaces. Shape number of points in rest x 3**degree drift
            (except degree 0 that is 0)

            theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x 3**degree drift
            (except degree 0 that is 0)
        """

        # # Init
        # U_I = None
        # U_G = None
        #
        # if self.u_grade_T.get_value() == 1:
        #     # ==========================
        #     # Condition of universality 1 degree
        #
        #     # Gradients
        #     n = self.dips_position.shape[0]
        #     U_G = T.zeros((n * self.n_dimensions, self.n_dimensions))
        #     # x
        #     U_G = T.set_subtensor(
        #         U_G[:n, 0], 1)
        #     # y
        #     U_G = T.set_subtensor(
        #         U_G[n:n * 2, 1], 1
        #     )
        #     # z
        #     U_G = T.set_subtensor(
        #         U_G[n * 2: n * 3, 2], 1
        #     )
        #
        #     # Interface
        #     # Cartesian distances between reference points and rest
        #     hx = T.stack(
        #         (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
        #         (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
        #         (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2])
        #     ).T
        #
        #     U_I = - hx * self.gi_reescale


      #  elif self.u_grade_T.get_value() == 2:
        # ==========================
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

        return U_I[:, :self.u_grade_T_op], U_G[:, :self.u_grade_T_op]

    def faults_matrix(self):
        """
        This function creates the part of the graph that generates the faults function creating a "block model" at the
        references and the rest of the points. Then this vector has to be appended to the covariance function
        Returns:
            theano.tensor.matrix: Drift matrix for the interfaces. Shape number of points in rest x n faults. This drif
            is a simple addition of an arbitrary number

            theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x n faults. For
            discrete values this matrix will be null since the derivative of a constant is 0
        """

        lenght_of_CG, lenght_of_CGI = self.matrices_shapes()[:2]

        # self.fault_matrix contains the faults volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
      #  self.len_points = self.rest_layer_points_all.shape[0]
        interface_loc = self.fault_matrix.shape[1] - 2*self.len_points

        fault_matrix_at_interfaces_rest = self.fault_matrix[:, interface_loc+self.len_i_0: interface_loc+self.len_i_1]
        fault_matrix_at_interfaces_ref =  self.fault_matrix[:, interface_loc+self.len_points+self.len_i_0:
                                                               interface_loc+self.len_points+self.len_i_1]

       # len_points_i = 2*self.rest_layer_points_all.shape[0] + self.n_formation_op[0]-1
       # len_points_e = 2*self.rest_layer_points_all.shape[0] + self.n_formation_op[-1]-1

     #   F_I = (self.fault_matrix[:, -2*len_points:-len_points] - self.fault_matrix[:, -len_points:])[self.n_formation_op-1]

        F_I = fault_matrix_at_interfaces_ref - fault_matrix_at_interfaces_rest

        # As long as the drift is a constant F_G is null
        F_G = T.zeros((self.fault_matrix.shape[0], lenght_of_CG))

        if str(sys._getframe().f_code.co_name) in self.verbose:
            F_I = theano.printing.Print('Faults interfaces matrix')(F_I)

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

        # TODO see if this condition is necesary. I think that simply by choosing len = 0 of the universal should work
        # (as I do in the fautls)
        # Set UG
        #if not self.u_grade_T.get_value() == 0:
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
       # if not self.u_grade_T.get_value() == 0:
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I, 0:length_of_CG], U_G.T)

        # Set U_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI:length_of_CG+length_of_CGI+length_of_U_I, length_of_CG:length_of_CG + length_of_CGI], U_I.T)

        # Fourth row of matrices
        # Set F_G
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI+length_of_U_I:, 0:length_of_CG], F_G)

        # Set F_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG+length_of_CGI+length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], F_I)

        # TODO: deprecate
        # -DEP-
        #  self.C_matrix = C_matrix
        # Add name to the theano node
        C_matrix.name = 'Block Covariance Matrix'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_matrix = theano.printing.Print('cov_function')(C_matrix)

        return C_matrix

    def b_vector(self, verbose=0):
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
       # G = T.tile(G, (1, 1)
        b = T.set_subtensor(b[0:G.shape[0]], G)

        if verbose > 1:
            theano.printing.pydotprint(b, outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)

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
        # TODO: look for an eficient way to substitute nlianlg by a theano operation
        import theano.tensor.slinalg
        b2 = T.tile(b, (1, 1)).T
        DK_parameters = theano.tensor.slinalg.solve(C_matrix, b2)
        DK_parameters = DK_parameters.reshape((DK_parameters.shape[0],))
      #  DK_parameters = T.dot(T.nlinalg.matrix_inverse(C_matrix), b)
        # Add name to the theano node
        DK_parameters.name = 'Dual Kriging parameters'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            DK_parameters = theano.printing.Print(DK_parameters.name )(DK_parameters)
        return DK_parameters

    def x_to_interpolate(self, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        interfaces). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.
        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """
        #yet_simulated = self.yet_simulated_func()

        # Removing points no simulated
        pns = (self.grid_val_T * self.yet_simulated.reshape((self.yet_simulated.shape[0], 1))).nonzero_values()

        # Adding the rest interface points
        grid_val = T.vertical_stack(pns.reshape((-1, 3)), self.rest_layer_points_all)

        # Adding the ref interface points
        grid_val = T.vertical_stack(grid_val, self.ref_layer_points_all)

        # Removing points no simulated
       # grid_val = (grid_val* self.yet_simulated.reshape((self.yet_simulated.shape[0], 1))).nonzero_values()

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
        DK_weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        return DK_weights

    def gradient_contribution(self):
        """
        Computation of the contribution of the foliations at every point to interpolate
        Returns:
            theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
        """
        weights = self.extend_dual_kriging()
        length_of_CG = self.matrices_shapes()[0]
        grid_val = self.x_to_interpolate()

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
            (weights[:length_of_CG, :] *
             self.gi_reescale *
             (-hu_SimPoint *
              (sed_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        # Add name to the theano node
        sigma_0_grad.name = 'Contribution of the foliations to the potential field at every point of the grid'

        return sigma_0_grad

    def interface_contribution(self):
        """
          Computation of the contribution of the interfaces at every point to interpolate
          Returns:
              theano.tensor.vector: Contribution of all interfaces (input) at every point to interpolate
          """
        weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]
        grid_val = self.x_to_interpolate()

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

        # Add name to the theano node
        sigma_0_interf.name = 'Contribution of the interfaces to the potential field at every point of the grid'

        return sigma_0_interf

    def universal_drift_contribution(self):
        """
        Computation of the contribution of the universal drift at every point to interpolate
        Returns:
            theano.tensor.vector: Contribution of the universal drift (input) at every point to interpolate
        """
        weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()
        grid_val = self.x_to_interpolate()

        # Universal drift contribution
        # Universal terms used to calculate f0
        # -DEP-
        # _universal_terms_layers = T.horizontal_stack(
        #     self.rest_layer_points,
        #     (self.rest_layer_points ** 2),
        #     T.stack((self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1],
        #              self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2],
        #              self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2]), axis=1)).T

        # if self.u_grade_T.get_value() == 0:
        #     f_0 = 0
        # else:
        # Here I create the universal terms for rest and ref. The universal terms for the grid are done in python
        # and append here. The idea is that the grid is kind of constant so I do not have to recompute it every
        # time
        _universal_terms_interfaces_rest = T.horizontal_stack(
            self.rest_layer_points_all,
            (self.rest_layer_points_all ** 2),
            T.stack((self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 1],
                     self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 2],
                     self.rest_layer_points_all[:, 1] * self.rest_layer_points_all[:, 2]), axis=1))

        _universal_terms_interfaces_ref = T.horizontal_stack(
            self.ref_layer_points_all,
            (self.ref_layer_points_all ** 2),
            T.stack((self.ref_layer_points_all[:, 0] * self.ref_layer_points_all[:, 1],
                     self.ref_layer_points_all[:, 0] * self.ref_layer_points_all[:, 2],
                     self.ref_layer_points_all[:, 1] * self.ref_layer_points_all[:, 2]), axis=1),
        )

        # _universal_terms_interfaces = T.horizontal_stack(
        #     self.rest_layer_points_all,
        #     (self.rest_layer_points_all ** 2),
        #     T.stack((self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 1],
        #              self.rest_layer_points_all[:, 0] * self.rest_layer_points_all[:, 2],
        #              self.rest_layer_points_all[:, 1] * self.rest_layer_points_all[:, 2]), axis=1)).T

        # I append rest and ref to grid
        universal_grid_interfaces_matrix = T.horizontal_stack(
            (self.universal_grid_matrix_T * self.yet_simulated).nonzero_values().reshape((9, -1)),
            T.vertical_stack(_universal_terms_interfaces_rest, _universal_terms_interfaces_ref).T)
          #  T.tile(_universal_terms_interfaces, (1, 2)))

        # These are the magic terms to get the same as geomodeller
        gi_rescale_aux = T.repeat(self.gi_reescale, 9)
        gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(gi_rescale_aux[:self.u_grade_T_op], (grid_val.shape[0], 1)).T

        # Drif contribution
        f_0 = (T.sum(
            weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I, :] * self.gi_reescale * _aux_magic_term *
            universal_grid_interfaces_matrix[:self.u_grade_T_op]
            , axis=0))

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

        return f_0

    def faults_contribution(self):
        """
        Computation of the contribution of the faults drift at every point to interpolate. To get these we need to
        compute a whole block model with the faults data
        Returns:
            theano.tensor.vector: Contribution of the faults drift (input) at every point to interpolate
        """
        weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()
        grid_val = self.x_to_interpolate()

        # Contribution
        f_1 = T.sum(weights[length_of_CG+length_of_CGI+length_of_U_I:, :] * self.fault_matrix[:, :grid_val.shape[0]], axis=0)

        # Add name to the theano node
        f_1.name = 'Faults contribution'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_1 = theano.printing.Print('Faults contribution')(f_1)

        return f_1

    def potential_field_at_all(self):
        """
        Compute the potential field at all the interpolation points, i.e. grid plus rest plus ref
        Returns:
            theano.tensor.vector: Potential fields at all points

        """
        sigma_0_grad = self.gradient_contribution()
        sigma_0_interf = self.interface_contribution()
        f_0 = self.universal_drift_contribution()
        f_1 = self.faults_contribution()
        # -DEP-
        # length_of_CGI = self.matrices_shapes()[1]

        Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1)#[:-2*self.rest_layer_points_all.shape[0]]

        Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)
        return Z_x

    def potential_field_at_interfaces(self):
        """
        Potential field at interfaces. To avoid errors I take all the points of rest that belong to one interface
        and make the average
        Returns:
            theano.tensor.vector: Potential field values at the interfaces of a given series
        """
        sigma_0_grad = self.gradient_contribution()
        sigma_0_interf = self.interface_contribution()
        f_0 = self.universal_drift_contribution()
        f_1 = self.faults_contribution()
        # -DEP-
        #length_of_CGI = self.matrices_shapes()[1]

        potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0 + f_1)[-2*self.len_points:
                                                                                 -self.len_points]

        npf = T.cumsum(T.concatenate((T.stack(0), self.number_of_points_per_formation_T)))

        # Loop to obtain the average Zx for every intertace
        def average_potential(dim_a, dim_b, pfi):
            """
            Function to make the average of the potential field at an interface
            Args:
                dim: size of the rest values vector per formation
                pfi: the values of all the rest values potentials
            Return:
                theano.tensor.vector: average of the potential per formation
            """
            average = pfi[T.cast(dim_a, "int32"): T.cast(dim_b, "int32")].sum() / (dim_b - dim_a)
            return average

        potential_field_interfaces_unique, updates1 = theano.scan(
            fn=average_potential,
            outputs_info=None,
            sequences=dict(input=npf,
                           taps=[0, 1]),
            non_sequences=potential_field_interfaces)

        # Add name to the theano node
        potential_field_interfaces_unique.name = 'Value of the potential field at the interfaces'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            potential_field_interfaces_unique = theano.printing.Print(potential_field_interfaces_unique.name)\
                                                                      (potential_field_interfaces_unique)
        return potential_field_interfaces_unique

    def block_series(self):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)
        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        # Graph to compute the potential field
        Z_x = self.potential_field_at_all()

        # Max and min values of the potential field.
        # TODO this may be expensive because I guess that is a sort algorithm. We just need a +inf and -inf... I guess
        max_pot = T.max(Z_x)  #T.max(potential_field_unique) + 1
        min_pot = T.min(Z_x)   #T.min(potential_field_unique) - 1

        # Value of the potential field at the interfaces of the computed series
        potential_field_at_interfaces = self.potential_field_at_interfaces()[self.n_formation_op-1]

        # A tensor with the values to segment
        potential_field_iter = T.concatenate((T.stack([max_pot]),
                                              T.sort(potential_field_at_interfaces)[::-1],
                                              T.stack([min_pot])))

        if "potential_field_iter" in self.verbose:
            potential_field_iter = theano.printing.Print("potential_field_iter")(potential_field_iter)

        if "potential_field_at_interfaces":
            potential_field_at_interfaces = theano.printing.Print('Potential field')(
                self.potential_field_at_interfaces())
            potential_field_at_interfaces = theano.printing.Print('Selected pt')(
                potential_field_at_interfaces[self.n_formation_op - 1])

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
            sequences=[dict(input=potential_field_iter, taps=[0, 1]), self.n_formation_op],
            non_sequences=Z_x)

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
                        final_block
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

        # TODO in order to make faults networks I will have to activate the yet simulated. The idea is that first we
        # compute the youngest fault and consecutively the others

        # -DEP- Until I add network faults
      #  self.yet_simulated = T.eq(final_block[0, :], 0)
        self.yet_simulated = T.eq(final_block[0, :-2*self.len_points], 0)
        self.yet_simulated.name = 'Yet simulated FAULTS node'
        #self.yet_simulated.name = 'Yet simulated node'

        # Slice the matrices for the corresponding series

        # Theano shared
        self.number_of_points_per_formation_T_op = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]
        self.u_grade_T_op = u_grade_iter

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # Theano Var
        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

       # self.fault_matrix_at_rest = self.fault_matrix[]
       # self.fault_matrix_at_ref =


        if 'n_formation' in self.verbose:
            self.n_formation_op = theano.printing.Print('n_formation_fault')(self.n_formation_op)

        # ====================
        # Computing the series
        # ====================
       # faults_matrix = self.block_series()

        faults_matrix = self.block_series()
        aux_ones = T.ones([2*self.len_points])
        faults_select = T.concatenate((self.yet_simulated, aux_ones))

        block_matrix = T.set_subtensor(
                    final_block[0, T.nonzero(T.cast(faults_select, "int8"))[0]],
                    faults_matrix)


        # -DEP- Until I add network faults
        #final_block = T.set_subtensor(
        #    final_block[T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
        #    potential_field_contribution)

        return block_matrix

    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         u_grade_iter,
                         final_block):

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

        # THIS IS THE FINAL BLOCK. (DO I NEED TO LOOP THE FAULTS FIRST? Yes you do)
        # ==================
        # Preparing the data
        # ==================
        # Vector that controls the points that have been simulated in previous iterations
        self.yet_simulated = T.eq(final_block[0, :], 0)
        self.yet_simulated.name = 'Yet simulated LITHOLOGY node'

        # Theano shared
        self.number_of_points_per_formation_T_op = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation_op = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]
        self.u_grade_T_op = u_grade_iter

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
        potential_field_contribution = self.block_series()[:-2*self.len_points]

        final_block = T.set_subtensor(
            final_block[0, T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
            potential_field_contribution)

        if self.compute_all:
            potential_field_values = self.potential_field_at_all()[:-2*self.len_points]

            final_block = T.set_subtensor(
            final_block[1, T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
                potential_field_values)
            if self.is_fault:
                final_block = T.set_subtensor(
                    final_block[2, :],
                    self.fault_matrix[-1, :-2 * self.len_points])

            #final_block_out = T.vertical_stack(final_block, pf)

        return final_block

    # def compute_a_series_pf(self,
    #                      len_i_0, len_i_1,
    #                      len_f_0, len_f_1,
    #                      n_form_per_serie_0, n_form_per_serie_1,
    #                      u_grade_iter,
    #                     ):
    #
    #     """
    #     Function that loops each series, generating a potential field for each on them with the respective block model
    #     Args:
    #          len_i_0: Lenght of rest of previous series
    #          len_i_1: Lenght of rest for the computed series
    #          len_f_0: Lenght of dips of previous series
    #          len_f_1: Length of dips of the computed series
    #          n_form_per_serie_0: Number of formations of previous series
    #          n_form_per_serie_1: Number of formations of the computed series
    #
    #     Returns:
    #          theano.tensor.matrix: final block model
    #     """
    #
    #     # THIS IS THE FINAL BLOCK. (DO I NEED TO LOOP THE FAULTS FIRST? Yes you do)
    #     # ==================
    #     # Preparing the data
    #     # ==================
    #     # Vector that controls the points that have been simulated in previous iterations
    #     #self.yet_simulated = T.eq(final_block, 0)
    #     #self.yet_simulated.name = 'Yet simulated node'
    #
    #     # Theano shared
    #     self.number_of_points_per_formation_T_op = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
    #     self.n_formation_op = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]
    #     self.u_grade_T_op = u_grade_iter
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
    #     # Printing
    #   #  if 'yet_simulated' in self.verbose:
    #   #      self.yet_simulated = theano.printing.Print(self.yet_simulated.name)(self.yet_simulated)
    #     if 'n_formation' in self.verbose:
    #         self.n_formation_op = theano.printing.Print('n_formation')(self.n_formation_op)
    #
    #     # ====================
    #     # Computing the series
    #     # ====================
    #     potential_field_contribution = self.potential_field_at_all()[:-2*self.rest_layer_points_all.shape[0]]
    #     #final_block = T.set_subtensor(
    #     #    final_block[T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
    #     #    potential_field_contribution)
    #
    #     return potential_field_contribution

    def whole_block_model(self, n_faults=0, compute_all=True):

        """
        Final function that loops first all the faults, then uses that result in the final block and loops again the
        series
        Args:
            n_faults (int): Number of faults to extract the correct values from the big input matrices

        Returns:
            theano.tensor.vector: Final block model with the segmented lithologies
        """
        # TODO move this to init
        self.compute_all = False
        self.is_fault = False
        if n_faults != 0:
            self.is_fault=True


        # Check if there are faults and loop them to create the Faults block
        if n_faults != 0:
            # we initialize the final block
            fault_block_init = T.zeros((1, self.grid_val_T.shape[0]+2*self.len_points))  # self.final_block
            fault_block_init.name = 'final block of faults init'
            self.yet_simulated = T.eq(fault_block_init[0, :-2*self.len_points], 0)


            fault_matrix, updates3 = theano.scan(
                 fn=self.compute_a_fault,
                 outputs_info=fault_block_init,  #  This line may be used for the faults network
                 sequences=[dict(input=self.len_series_i[:n_faults+1], taps=[0, 1]),
                            dict(input=self.len_series_f[:n_faults+1], taps=[0, 1]),
                            dict(input=self.n_formations_per_serie[:n_faults+1], taps=[0, 1]),
                            dict(input=self.u_grade_T[:n_faults + 1], taps=[0])]
                 )
            # fault_matrix, updates3 = theano.scan(
            #     fn=self.compute_a_series,
            #     outputs_info=final_block_init,
            #     sequences=[dict(input=self.len_series_i[:n_faults + 1], taps=[0, 1]),
            #                dict(input=self.len_series_f[:n_faults+1], taps=[0, 1]),
            #                dict(input=self.n_formations_per_serie[:n_faults+1], taps=[0, 1]),
            #                dict(input=self.u_grade_T[:n_faults + 1], taps=[0])]
            #     )

            self.fault_matrix = fault_matrix[-1]

            if 'faults block' in self.verbose:
                self.fault_matrix = theano.printing.Print('I am outside the faults')(fault_matrix[-1])

       # self.u_grade_T = theano.printing.Print('drift degree')(self.u_grade_T)
       # self.a_T = theano.printing.Print('range')(self.a_T)
        self.compute_all = compute_all
        final_block_init = self.final_block
        final_block_init.name = 'final block of lithologies init'

        # Checking there are more potential fields in the data that the faults.
        if len(self.len_series_f.get_value())-1 > n_faults:
            if self.compute_all:
                final_block_init = T.vertical_stack(self.final_block, self.final_block, self.final_block)
                # Loop the series to create the Final block
                all_series, updates2 = theano.scan(
                    fn=self.compute_a_series,
                    outputs_info=final_block_init,
                    sequences=[dict(input=self.len_series_i[n_faults:], taps=[0, 1]),
                               dict(input=self.len_series_f[n_faults:], taps=[0, 1]),
                               dict(input=self.n_formations_per_serie[n_faults:], taps=[0, 1]),
                               dict(input=self.u_grade_T[n_faults:], taps=[0])]
                # all_series_pf, updates3 = theano.scan(
                #      fn=self.compute_a_series,
                #      outputs_info=final_block_init,
                #      sequences=[dict(input=self.len_series_i[n_faults:], taps=[0, 1]),
                #                 dict(input=self.len_series_f[n_faults:], taps=[0, 1]),
                #                 dict(input=self.n_formations_per_serie[n_faults:], taps=[0, 1]),
                #                 dict(input=self.u_grade_T[n_faults:], taps=[0])]
                )

            else:
                # Loop the series to create the Final block
                all_series, updates2 = theano.scan(
                    fn=self.compute_a_series,
                    outputs_info=final_block_init,
                    sequences=[dict(input=self.len_series_i[n_faults:], taps=[0, 1]),
                               dict(input=self.len_series_f[n_faults:], taps=[0, 1]),
                               dict(input=self.n_formations_per_serie[n_faults:], taps=[0, 1]),
                               dict(input=self.u_grade_T[n_faults:], taps=[0])]
                )
        else:
            # We just pass the faults block
            all_series = self.fault_matrix

        return all_series

    # ==================================
    # Geophysics
    # ==================================
    #
    def choose_cells(self, grid, measure_points):
        """
        Preprocessing to see which nearby cells we use for the gravity
        Args:
            grid:
            measure_points:

        Returns:

        """
        return bool


    # def slice_cells(self, selected_cells, block_lith):
    #
    #     # First change the bedrock to arbitrary value
    #     new_block = block-lith + 1
    #
    #     (new_block * selected_cells[:. T.newaxis?]).nonzero_values()
    #
    #
    # def set_densities(self, block_lith):
    #
    #     def switch_densities(lith, block):
    #         """
    #
    #         Args:
    #             lith:
    #             block:
    #
    #         Returns:
    #
    #         """
    #         densities = T.switch(
    #             T.eq(sed_dips_dips, 0),  # This is the condition
    #             0,  # If true it is equal to 0. This is how a direction affect another
    #             (  # else, following Chiles book
    #
    #         return
    #
    #     partial_block, updates2 = theano.scan(
    #         fn=compare,
    #         outputs_info=None,
    #         sequences=[dict(input=self.n_formation)],
    #         non_sequences=block_lith)

#class ForwardGravity(TheanoGraph_pro):


