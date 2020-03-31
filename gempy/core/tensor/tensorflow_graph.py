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
    def __init__(self, dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties, output=None):
        self.dtype = tf.float64

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = tf.constant(-1, dtype=self.dtype, name="Range")
        self.a_T_surface = self.a_T
        self.c_o_T = tf.constant(-1, dtype=self.dtype, name="Covariance at 0")

        self.n_universal_eq_T = tf.ones(
            5, dtype=tf.int32, name="Grade of the universal drift")
        self.n_universal_eq_T_op = tf.constant(3)

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = tf.constant(4, dtype=self.dtype)
        self.gi_reescale = tf.constant(2, dtype=self.dtype)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3
        self.number_of_points_per_surface = tf.zeros(
            3, dtype=tf.int32, name='Number of points per surface used to split rest-ref')

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

        self.ref_layer_points, self.rest_layer_points = self.set_rest_ref_matrix(
            self.number_of_points_per_surface)

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

    def set_rest_ref_matrix(self, number_of_points_per_surface):
        # reference point: every first point of each layer
        ref_positions = tf.cumsum(
            tf.concat([[0], number_of_points_per_surface[:-1]+1], axis=0))

        ref_positions = tf.expand_dims(ref_positions, 1)
        ref_points = tf.gather_nd(self.surface_points_all, ref_positions)
        # repeat the reference points (the number of persurface -1)  times
        ref_points_repeated = tf.repeat(
            ref_points, number_of_points_per_surface, 0)

        mask = tf.one_hot(ref_positions, tf.reduce_sum(
            number_of_points_per_surface+tf.constant(1, tf.int32)), on_value=1, off_value=0., dtype=tf.float64)
        mask = tf.squeeze(tf.reduce_sum(mask, 0))
        rest_points = tf.gather_nd(
            self.surface_points_all, tf.where(mask == 0))
        return ref_points_repeated, rest_points

    def squared_euclidean_distance(self, x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Arguments:
            x_1 {[Tensor]} -- shape n_points x number dimension
            x_2 {[Tensor]} -- shape n_points x number dimension

        Returns:
            [Tensor] -- Distancse matrix. shape n_points x n_points
        """
        sqd = tf.sqrt(tf.reshape(tf.reduce_sum(x_1**2, 1), shape=(x_1.shape[0], 1)) +
                      tf.reshape(tf.reduce_sum(x_2**2, 1), shape=(1, x_2.shape[0])) -
                      2*tf.tensordot(x_1, tf.transpose(x_2), 1))
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

        return C_I
