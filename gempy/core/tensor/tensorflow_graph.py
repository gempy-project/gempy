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

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = tf.divide(Range, rescalefactor)
        self.a_T_surface = self.a_T
        self.c_o_T = tf.divide(C_o, rescalefactor)

        self.n_universal_eq_T = tf.ones(
            5, dtype=tf.int32, name="Grade of the universal drift")
        self.n_universal_eq_T_op = tf.constant(3)

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = tf.constant(4, dtype=self.dtype)
        self.gi_reescale = tf.constant(2, dtype=self.dtype)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        self.number_of_points_per_surface = number_of_points_per_surface

        self.nugget_effect_grad = nugget_effect_grad
        self.nugget_effect_scalar = nugget_effect_scalar

        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES
        # ---------

        self.dip_angles_all = dip_angles
        self.azimuth_all = azimuth
        self.polarity_all = polarity
        self.surface_points_all = surface_points_coord

        # Tiling dips to the 3 spatial coordinations
        self.dips_position_all = dips_position
        self.dips_position_all_tiled = tf.tile(
            self.dips_position_all, [self.n_dimensions, 1])

        self.ref_layer_points, self.rest_layer_points, self.ref_nugget, self.rest_nugget = self.set_rest_ref_matrix()

        self.nugget_effect_scalar_ref_rest = tf.expand_dims(
            self.ref_nugget + self.rest_nugget, 1)

        self.fault_matrix = fault_drift
        self.grid_val = grid

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

        condistion_fail = tf.math.divide_no_nan(h_u * h_v, sed_dips_dips ** 2)*(tf.where(sed_dips_dips < self.a_T, x=(((-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                                                                                                                       35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                                                                                                                       21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7))) +
                                                                                                                      self.c_o_T * 7 * (9 * sed_dips_dips ** 5 - 20 * self.a_T ** 2 * sed_dips_dips ** 3 +
                                                                                                                                        15 * self.a_T ** 4 * sed_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)), y=0)) -\
            perpendicularity_matrix*tf.where(sed_dips_dips < self.a_T, x=self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                                                                       35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                                                                       21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7), y=0)

        C_G = tf.where(sed_dips_dips == 0, x=0, y=condistion_fail)
        C_G = C_G + tf.eye(C_G.shape[0],
                           dtype=self.dtype)*self.nugget_effect_grad

        return C_G

    def cov_ubterface_gradients(self):

        sed_dips_rest = self.squared_euclidean_distance(
            self.dips_position_all_tiled, self.rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(
            self.dips_position_all_tiled, self.ref_layer_points)

        def cartesian_dist_no_tile(x_1, x_2):
            return tf.concat([
                tf.transpose(
                    (x_1[:, 0] - tf.reshape(x_2[:, 0], [x_2.shape[0], 1]))),
                tf.transpose(
                    (x_1[:, 1] - tf.reshape(x_2[:, 1], [x_2.shape[0], 1]))),
                tf.transpose((x_1[:, 2] - tf.reshape(x_2[:, 2], [x_2.shape[0], 1])))], axis=0)

        hu_rest = cartesian_dist_no_tile(
            self.dips_position_all, self.rest_layer_points)
        hu_ref = cartesian_dist_no_tile(
            self.dips_position_all, self.ref_layer_points)

        C_GI = self.gi_reescale*tf.transpose(hu_rest *
                                             tf.where(sed_dips_rest < self.a_T_surface, x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_rest / self.a_T_surface ** 3 -
                                                                                                           35 / 2 * sed_dips_rest ** 3 / self.a_T_surface ** 5 +
                                                                                                           21 / 4 * sed_dips_rest ** 5 / self.a_T_surface ** 7)), y=0) -
                                             (hu_ref * tf.where(sed_dips_ref < self.a_T_surface, x=- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_ref / self.a_T_surface ** 3 -
                                                                                                                   35 / 2 * sed_dips_ref ** 3 / self.a_T_surface ** 5 +
                                                                                                                   21 / 4 * sed_dips_ref ** 5 / self.a_T_surface ** 7), y=0)))

        return C_GI
