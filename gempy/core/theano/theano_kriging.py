import theano
import theano.tensor as T
import numpy as np
import sys
from .theano_graph import TheanoGeometry 


class TheanoKriging(TheanoGeometry):
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
        
        super(TheanoKriging, self).__init__(output='geology', optimizer='fast_compile', verbose=[0], dtype='float32',
                 is_fault=None, is_lith=None)
        # Pass the verbose list as property

        # Creation of symbolic parameters
        # =============
        # Constants
        # =============




        # # This is not accumulative
        # self.number_of_points_per_surface_T = theano.shared(np.zeros(3, dtype='int32')) #TODO is DEP?
        # self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T
        # This is accumulative
        #self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')
        #self.npf_op = self.npf[[0, -2]]

    def input_parameters_kriging(self):
        """
        Create a list with the symbolic variables to use when we compile the theano function

        Returns:
            list: [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
                   self.ref_layer_points_all, self.rest_layer_points_all]
        """
        ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all, self.surface_points_all,
               self.fault_drift, self.number_of_points_per_surface_T_op]
        # self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

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
        C_G += T.eye(C_G.shape[0]) * self.nugget_effect_grad_T

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
        sed_dips_ref = self.squared_euclidean_distances(self.dips_position_tiled, self.ref_layer_points)

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

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
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

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
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

        # self.fault_drift contains the df volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
        if 'fault matrix' in self.verbose:
            self.fault_drift = theano.printing.Print('self.fault_drift')(self.fault_drift)
        interface_loc = self.fault_drift.shape[1] - 2 * self.len_points

        fault_drift_at_surface_points_rest = self.fault_drift
        fault_drift_at_surface_points_ref = self.fault_drift

        F_I = (fault_drift_at_surface_points_ref - fault_drift_at_surface_points_rest) + 0.0001

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
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I], U_G)
        # Set FG. I cannot use -index because when is -0 is equivalent to 0
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG + length_of_CGI + length_of_U_I:], F_G.T)
        # Second row of matrices
        # Set C_IG
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        # Set C_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG:length_of_CG + length_of_CGI], C_I)
        # Set U_I
        # if not self.u_grade_T.get_value() == 0:
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I], U_I)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG + length_of_CGI + length_of_U_I:], F_I.T)
        # Third row of matrices
        # Set U_G
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I, 0:length_of_CG], U_G.T)
        # Set U_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I,
                                   length_of_CG:length_of_CG + length_of_CGI], U_I.T)
        # Fourth row of matrices
        # Set F_G
        C_matrix = T.set_subtensor(C_matrix[length_of_CG + length_of_CGI + length_of_U_I:, 0:length_of_CG], F_G)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI + length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], F_I)
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
