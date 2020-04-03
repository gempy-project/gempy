import tensorflow as tf
import numpy as np
import sys


def constant32(k):
    return tf.constant(k, dtype=tf.float32)


@tf.function
def squared_euclidean_distance(x_1, x_2):
    sqd = tf.sqrt(tf.reshape(tf.reduce_sum(x_1**2, 1), shape=(x_1.shape[0], 1)) +
                  tf.reshape(tf.reduce_sum(x_2**2, 1), shape=(1, x_2.shape[0])) -
                  2*tf.tensordot(x_1, tf.transpose(x_2), 1))
    return sqd


class TFGraph:
    def __init__(self, dips_position, dip_angles, azimuth, polarity,
                 surface_points_coord, fault_drift, grid, values_properties,
                 number_of_points_per_surface, Range, C_o, nugget_effect_scalar,
                 nugget_effect_grad, rescalefactor, output=None):
        self.dtype = tf.float64
        self.lengh_of_faults = tf.constant(0, dtype=tf.int32)

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = tf.cast(tf.divide(Range, rescalefactor),self.dtype)
        self.a_T_surface = self.a_T
        self.c_o_T = tf.divide(C_o, rescalefactor)

        self.n_universal_eq_T = tf.ones(
            5, dtype=tf.int32, name="Grade of the universal drift")
        self.n_universal_eq_T_op = tf.constant(3)# 9 for 2nd order drift

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = tf.constant(4., dtype=self.dtype)
        self.gi_reescale = tf.constant(2., dtype=self.dtype)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        self.number_of_points_per_surface = number_of_points_per_surface

        self.nugget_effect_grad = nugget_effect_grad
        self.nugget_effect_scalar = nugget_effect_scalar

        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES
        # ---------

        self.dip_angles_all = tf.cast(dip_angles,self.dtype)
        self.azimuth_all = azimuth
        self.polarity_all = polarity

        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all

        self.surface_points_all = surface_points_coord

        # Tiling dips to the 3 spatial coordinations
        self.dips_position_all = dips_position
        self.dips_position_all_tiled = tf.tile(
            self.dips_position_all, [self.n_dimensions, 1])

        self.ref_layer_points, self.rest_layer_points, self.ref_nugget, self.rest_nugget = self.set_rest_ref_matrix()

        self.nugget_effect_scalar_ref_rest = tf.expand_dims(
            self.ref_nugget + self.rest_nugget, 1)

        self.len_points = self.surface_points_all.shape[0] - \
            self.number_of_points_per_surface.shape[0]

        self.grid_val = grid

        self.fault_matrix = fault_drift

        interface_loc = grid.shape[0]
        self.fault_drift_at_surface_points_rest = self.fault_matrix[
            :, interface_loc: interface_loc + self.len_points]
        self.fault_drift_at_surface_points_ref = self.fault_matrix[
            :, interface_loc + self.len_points:]

        self.values_properties_op = values_properties

        if output is None:
            output = ['geology']
        self.output = output

        self.compute_type = output

        self.input_parameters_block = [self.dips_position_all, self.dip_angles_all, self.azimuth_all,
                                       self.polarity_all, self.surface_points_all,
                                       self.fault_matrix, self.grid_val,
                                       self.values_properties_op]
        self.is_erosion = tf.constant([1., 0.], dtype=self.dtype)
        self.is_onlap = tf.constant([0, 1], dtype=self.dtype)

        self.offset = tf.constant(10., dtype=self.dtype)
        self.shift = 0

        if 'gravity' in self.compute_type:
            self.lg0 = tf.constant(np.array(0, dtype='int64'))
            self.lg1 = tf.constant(np.array(1, dtype='int64'))

            self.tz = tf.constant(np.empty(0, dtype=self.dtype))
            self.pos_density = tf.constant(np.array(1, dtype='int64'))

        self.n_surface = tf.range(
            1, 5000, dtype='int32', name='ID of surfaces')

    def set_rest_ref_matrix(self):
        # reference point: every first point of each layer
        ref_positions = tf.cumsum(
            tf.concat([[0], self.number_of_points_per_surface[:-1]+1], axis=0))

        ref_positions = tf.expand_dims(ref_positions, 1)
        ref_points = tf.gather_nd(self.surface_points_all, ref_positions)
        ref_nugget = tf.gather_nd(self.nugget_effect_scalar, ref_positions)

        # repeat the reference points (the number of persurface -1)  times
        ref_points_repeated = tf.repeat(
            ref_points, self.number_of_points_per_surface, 0)
        ref_nugget_repeated = tf.repeat(
            ref_nugget, self.number_of_points_per_surface, 0)

        mask = tf.one_hot(ref_positions, tf.reduce_sum(
            self.number_of_points_per_surface+tf.constant(1, tf.int32)), on_value=1, off_value=0., dtype=tf.float64)
        rest_mask = tf.squeeze(tf.reduce_sum(mask, 0))

        rest_points = tf.gather_nd(
            self.surface_points_all, tf.where(rest_mask == 0))

        rest_nugget = tf.gather_nd(
            self.nugget_effect_scalar, tf.where(rest_mask == 0))

        return ref_points_repeated, rest_points, ref_nugget_repeated, rest_nugget

    def squared_euclidean_distance(self, x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Arguments:
            x_1 {[Tensor]} -- shape n_points x number dimension
            x_2 {[Tensor]} -- shape n_points x number dimension

        Returns:
            [Tensor] -- Distancse matrix. shape n_points x n_points
        """
        # tf.maximum avoid negative numbers increasing stability

        sqd = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(x_1**2, 1), shape=(x_1.shape[0], 1)) +
                                 tf.reshape(tf.reduce_sum(x_2**2, 1), shape=(1, x_2.shape[0])) -
                                 2*tf.tensordot(x_1, tf.transpose(x_2), 1), 1e-12))
        return sqd

    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """
        length_of_CG = tf.constant(self.dips_position_all_tiled.shape[0])
        length_of_CGI = tf.constant(self.ref_layer_points.shape[0])
        length_of_U_I = self.n_universal_eq_T_op
        length_of_faults = self.lengh_of_faults

        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    def cov_surface_points(self):
        sed_rest_rest = self.squared_euclidean_distance(
            self.rest_layer_points, self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distance(
            self.ref_layer_points, self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distance(
            self.rest_layer_points, self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distance(
            self.ref_layer_points, self.ref_layer_points)

        C_I = self.c_o_T*self.i_reescale*(
            tf.where(sed_rest_rest < self.a_T, x=(1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
                                                  35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
                                                  7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
                                                  3 / 4 * (sed_rest_rest / self.a_T) ** 7), y=0) -
            tf.where(sed_ref_rest < self.a_T, x=(1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
                                                 35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
                                                 7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
                                                 3 / 4 * (sed_ref_rest / self.a_T) ** 7), y=0) -
            tf.where(sed_rest_ref < self.a_T, x=(1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
                                                 35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
                                                 7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
                                                 3 / 4 * (sed_rest_ref / self.a_T) ** 7), y=0) +
            tf.where(sed_ref_ref < self.a_T, x=(1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
                                                35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
                                                7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
                                                3 / 4 * (sed_ref_ref / self.a_T) ** 7), y=0))

        C_I = C_I + tf.eye(C_I.shape[0], dtype=self.dtype) * \
            self.nugget_effect_scalar_ref_rest
        return C_I

    def cov_gradients(self):

        sed_dips_dips = self.squared_euclidean_distance(
            self.dips_position_all_tiled, self.dips_position_all_tiled)

        h_u = tf.concat([
            tf.tile(self.dips_position_all[:, 0] - tf.reshape(
                self.dips_position_all[:, 0], [self.dips_position_all.shape[0], 1]), [1, 3]),
            tf.tile(self.dips_position_all[:, 1] - tf.reshape(
                self.dips_position_all[:, 1], [self.dips_position_all.shape[0], 1]), [1, 3]),
            tf.tile(self.dips_position_all[:, 2] - tf.reshape(self.dips_position_all[:, 2], [self.dips_position_all.shape[0], 1]), [1, 3])], axis=0)

        h_v = tf.transpose(h_u)

        sub_x = tf.concat([tf.ones([self.dips_position_all.shape[0], self.dips_position_all.shape[0]]), tf.zeros(
            [self.dips_position_all.shape[0], 2*self.dips_position_all.shape[0]])], axis=1)

        sub_y = tf.concat([tf.concat([tf.zeros([self.dips_position_all.shape[0], self.dips_position_all.shape[0]]), tf.ones(
            [self.dips_position_all.shape[0], 1*self.dips_position_all.shape[0]])], axis=1), tf.zeros([self.dips_position_all.shape[0], self.dips_position_all.shape[0]])], 1)
        sub_z = tf.concat([tf.zeros([self.dips_position_all.shape[0], 2*self.dips_position_all.shape[0]]),
                           tf.ones([self.dips_position_all.shape[0], self.dips_position_all.shape[0]])], axis=1)

        perpendicularity_matrix = tf.cast(
            tf.concat([sub_x, sub_y, sub_z], axis=0), dtype=tf.float64)

        condistion_fail = tf.math.divide_no_nan(h_u * h_v, sed_dips_dips ** 2)*(
            tf.where(sed_dips_dips < self.a_T,
                     x=(((-self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                         35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                         21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.))) +
                        self.c_o_T * 7. * (9. * sed_dips_dips ** 5. - 20. * self.a_T ** 2. * sed_dips_dips ** 3. +
                                          15. * self.a_T ** 4. * sed_dips_dips - 4.* self.a_T ** 5.) / (2. * self.a_T ** 7.)), y=0.)) -\
            perpendicularity_matrix*tf.where(sed_dips_dips < self.a_T, x=self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                                                                       35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                                                                       21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.), y=0.)

        C_G = tf.where(sed_dips_dips == 0, x=tf.constant(
            0., dtype=self.dtype), y=condistion_fail)
        C_G = C_G + tf.eye(C_G.shape[0],
                           dtype=self.dtype)*self.nugget_effect_grad

        return C_G

    def cartesian_dist(self, x_1, x_2):
        return tf.concat([
            tf.transpose(
                (x_1[:, 0] - tf.reshape(x_2[:, 0], [x_2.shape[0], 1]))),
            tf.transpose(
                (x_1[:, 1] - tf.reshape(x_2[:, 1], [x_2.shape[0], 1]))),
            tf.transpose((x_1[:, 2] - tf.reshape(x_2[:, 2], [x_2.shape[0], 1])))], axis=0)

    def cov_interface_gradients(self):

        sed_dips_rest = self.squared_euclidean_distance(
            self.dips_position_all_tiled, self.rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(
            self.dips_position_all_tiled, self.ref_layer_points)

        hu_rest = self.cartesian_dist(
            self.dips_position_all, self.rest_layer_points)
        hu_ref = self.cartesian_dist(
            self.dips_position_all, self.ref_layer_points)

        C_GI = self.gi_reescale*tf.transpose(hu_rest *
                                             tf.where(sed_dips_rest < self.a_T_surface, x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_rest / self.a_T_surface ** 3 -
                                                                                                           35 / 2 * sed_dips_rest ** 3 / self.a_T_surface ** 5 +
                                                                                                           21 / 4 * sed_dips_rest ** 5 / self.a_T_surface ** 7)), y=tf.constant(0., dtype=self.dtype)) -
                                             (hu_ref * tf.where(sed_dips_ref < self.a_T_surface, x=- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_ref / self.a_T_surface ** 3 -
                                                                                                                   35 / 2 * sed_dips_ref ** 3 / self.a_T_surface ** 5 +
                                                                                                                   21 / 4 * sed_dips_ref ** 5 / self.a_T_surface ** 7), y=tf.constant(0., dtype=self.dtype))))

        return C_GI

    def universal_matrix(self):

        n = self.dips_position_all.shape[0]

        sub_x = tf.tile(tf.constant([[1., 0., 0.]], self.dtype), [n, 1])
        sub_y = tf.tile(tf.constant([[0., 1., 0.]], self.dtype), [n, 1])
        sub_z = tf.tile(tf.constant([[0., 0., 1.]], self.dtype), [n, 1])
        sub_block1 = tf.concat([sub_x, sub_y, sub_z], 0)

        sub_x_2 = tf.reshape(2 * self.gi_reescale *
                             self.dips_position_all[:, 0], [n, 1])
        sub_y_2 = tf.reshape(2 * self.gi_reescale *
                             self.dips_position_all[:, 1], [n, 1])
        sub_z_2 = tf.reshape(2 * self.gi_reescale *
                             self.dips_position_all[:, 2], [n, 1])

        sub_x_2 = tf.pad(sub_x_2, [[0, 0], [0, 2]])
        sub_y_2 = tf.pad(sub_y_2, [[0, 0], [1, 1]])
        sub_z_2 = tf.pad(sub_z_2, [[0, 0], [2, 0]])
        sub_block2 = tf.concat([sub_x_2, sub_y_2, sub_z_2], 0)

        sub_xy = tf.reshape(tf.concat([self.gi_reescale * self.dips_position_all[:, 1],
                                       self.gi_reescale * self.dips_position_all[:, 0]], 0), [2*n, 1])
        sub_xy = tf.pad(sub_xy, [[0, 2], [0, 0]])
        sub_xz = tf.concat([tf.pad(tf.reshape(self.gi_reescale * self.dips_position_all[:, 2], [n, 1]), [
            [0, n], [0, 0]]), tf.reshape(self.gi_reescale * self.dips_position_all[:, 0], [n, 1])], 0)
        sub_yz = tf.reshape(tf.concat([self.gi_reescale * self.dips_position_all[:, 2],
                                       self.gi_reescale * self.dips_position_all[:, 1]], 0), [2*n, 1])
        sub_yz = tf.pad(sub_yz, [[2, 0], [0, 0]])

        sub_block3 = tf.concat([sub_xy, sub_xz, sub_yz], 1)

        U_G = tf.concat([sub_block1, sub_block2, sub_block3], 1)


        U_I = -tf.stack([self.gi_reescale * (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]), self.gi_reescale *
                         (self.rest_layer_points[:, 1] -
                          self.ref_layer_points[:, 1]),
                         self.gi_reescale *
                         (self.rest_layer_points[:, 2] -
                          self.ref_layer_points[:, 2]),
                         self.gi_reescale ** 2 *
                         (self.rest_layer_points[:, 0] ** 2 -
                          self.ref_layer_points[:, 0] ** 2),
                         self.gi_reescale ** 2 *
                         (self.rest_layer_points[:, 1] ** 2 -
                          self.ref_layer_points[:, 1] ** 2),
                         self.gi_reescale ** 2 *
                         (self.rest_layer_points[:, 2] ** 2 -
                          self.ref_layer_points[:, 2] ** 2),
                         self.gi_reescale ** 2 * (
            self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1] - self.ref_layer_points[:, 0] *
            self.ref_layer_points[:, 1]),
            self.gi_reescale ** 2 * (
            self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 0] *
            self.ref_layer_points[:, 2]),
            self.gi_reescale ** 2 * (
            self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 1] *
            self.ref_layer_points[:, 2])], 1)
        

        return U_G, U_I

    def faults_matrix(self, f_ref=None, f_res=None):
        length_of_CG, _, _, length_of_faults = self.matrices_shapes()[
            :4]

        # self.fault_drift_at_surface_points_rest = self.fault_matrix
        # self.fault_drift_at_surface_points_ref = self.fault_matrix

        F_I = (self.fault_drift_at_surface_points_ref -
               self.fault_drift_at_surface_points_rest) + 0.0001

        F_G = tf.zeros((length_of_faults, length_of_CG),
                       dtype=self.dtype) + 0.0001

        return F_I, F_G

    @tf.function
    def covariance_matrix(self):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        C_G = self.cov_gradients()
        C_I = self.cov_surface_points()
        C_GI = self.cov_interface_gradients()
        U_G, U_I = self.universal_matrix()
        U_G = U_G[:length_of_CG,:3]
        U_I = U_I[:length_of_CGI,:3]
        F_I, F_G = self.faults_matrix()

        A = tf.concat([tf.concat([C_G, tf.transpose(C_GI)], -1),
                       tf.concat([C_GI, C_I], -1)], 0)

        B = tf.concat([U_G, U_I], 0)

        AB = tf.concat([A, B], -1)

        B_T = tf.transpose(B)

        C = tf.pad(B_T, [[0, 0], [0, U_I.shape[1]]])

        C_matrix = tf.concat([AB, C], 0)


        C_matrix = tf.where(tf.logical_and(tf.abs(C_matrix) > 0, tf.abs(
            C_matrix) < 1e-9), tf.constant(0, dtype=self.dtype), y=C_matrix)

        return C_matrix

    def deg2rad(self, degree_matrix):
        return degree_matrix*tf.constant(0.0174533, dtype=self.dtype)

    @tf.function
    def b_vector(self, dip_angles_=None, azimuth_=None, polarity_=None):

        length_of_C = self.matrices_shapes()[-1]
        if dip_angles_ is None:
            dip_angles_ = self.dip_angles
        if azimuth_ is None:
            azimuth_ = self.azimuth
        if polarity_ is None:
            polarity_ = self.polarity

        G_x = tf.sin(self.deg2rad(dip_angles_)) * \
            tf.sin(self.deg2rad(azimuth_)) * polarity_
        G_y = tf.sin(self.deg2rad(dip_angles_)) * \
            tf.cos(self.deg2rad(azimuth_)) * polarity_
        G_z = tf.cos(self.deg2rad(dip_angles_)) * polarity_

        G = tf.concat([G_x, G_y, G_z], -1)

        G = tf.expand_dims(G, axis=1)
        b_vector = tf.pad(G, [[0, length_of_C - G.shape[0]], [0, 0]])

        return b_vector

    @tf.function
    def solve_kriging(self, b=None):

        C_matrix = self.covariance_matrix()

        b_vector = self.b_vector()

        DK = tf.linalg.solve(C_matrix, b_vector)

        return DK

    def x_to_interpolate(self, grid):
        grid_val = tf.concat([grid, self.rest_layer_points], 0)
        grid_val = tf.concat([grid_val, self.ref_layer_points], 0)

        return grid_val

    def extend_dual_kriging(self, weights, grid_shape):

        DK_parameters = weights

        DK_parameters = tf.tile(DK_parameters, [1, grid_shape])

        return DK_parameters

    def contribution_gradient_interface(self, grid_val=None, weights=None):

        length_of_CG = self.matrices_shapes()[0]

        hu_SimPoint = self.cartesian_dist(self.dips_position_all, grid_val)

        sed_dips_SimPoint = self.squared_euclidean_distance(
            self.dips_position_all_tiled, grid_val)

        sigma_0_grad = (weights[:length_of_CG] *
                        self.gi_reescale * (tf.negative(hu_SimPoint) *\
                                            # first derivative
                                            tf.where((sed_dips_SimPoint < self.a_T_surface),
                                                     x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T_surface ** 3 -
                                                                        35 / 2 * sed_dips_SimPoint ** 3 / self.a_T_surface ** 5 +
                                                                        21 / 4 * sed_dips_SimPoint ** 5 / self.a_T_surface ** 7)),
                                                     y=tf.constant(0, dtype=tf.float64))))

        return sigma_0_grad
    