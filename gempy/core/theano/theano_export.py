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
from .theano_graph import TheanoGeometry, TheanoOptions

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


class TheanoExport(TheanoGeometry):

    def __init__(self, weights_op = None, grid: np.ndarray=None, weights=None):
        """

        Args:
            grid: if grid is passed it becomes constant
            weights:
        """

        super().__init__()

        # self.grid_val_T = theano.shared(np.cast[dtype](np.zeros((2, 200))), 'Coordinates of the grid '
        #                                                                     'points to interpolate')

        self.grid_val_T = T.matrix('Coordinates of the grid points to interpolate')
        self.fault_matrix = T.matrix('Full block matrix for x_to_interpolate')
        # TODO: This is either shared or theanoOP
       # if weights_op is None:
       #     self.weights = theano.shared(np.ones(10000), 'Weights to compute')
       # else:
       #     self.weights = weights_op(self.input_parameters_weights())
       #  if weights_op is None:
       #      self.weights = T.vector('kriging weights')
       #  else:
       #      self.weights_op = weights_op
       #      self.weights = weights_op(*self.input_parameters_kriging())

        #self.fault_matrix = T.zeros((0, self.grid_val_T.shape[0] + 2 * self.len_points))

       # self.fault_drift = T.matrix('Block matrix at interface points')
      #  self.fault_mask = T.zeros((0, self.grid_val_T.shape[0] + 2 * self.len_points))

       # self.n_surface = theano.shared(np.arange(1, 5, dtype='int32'), "ID of the surface")
      #  self.n_surface_op = self.n_surface
        #self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')

        self.dot_version = False

    def set_weights(self, weights_op = None):
        if weights_op is None:
            self.weights = T.vector('kriging weights')
        else:
            self.weights_op = weights_op
            self.weights = weights_op(*self.input_parameters_kriging())


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

    def input_parameters_export(self):
        """
        Create a list with the symbolic variables to use when we compile the theano function

        Returns:
            list:
        """
        ipl = [self.dips_position_all, self.surface_points_all,
               self.fault_drift, self.fault_matrix, self.weights, self.grid_val_T,
               self.number_of_points_per_surface_T_op]
        # self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    def input_parameters_export_kriging(self):

        # ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
        #        self.surface_points_all,
        #        self.fault_drift, self.number_of_points_per_surface_T_op]
        #
        # ipl = [self.dips_position_all, self.surface_points_all,
        #        self.fault_drift, self.fault_matrix, self.grid_val_T,
        #        self.number_of_points_per_surface_T_op]

        ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
               self.surface_points_all,
               self.fault_drift,  self.fault_matrix, self.grid_val_T, self.number_of_points_per_surface_T_op]

        # self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    def x_to_interpolate(self, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        surface_points). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.

        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """

        grid_val = T.concatenate([self.grid_val_T, self.rest_layer_points_all,
                                  self.ref_layer_points_all])

        if verbose > 1:
            theano.printing.pydotprint(grid_val, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if 'grid_val' in self.verbose:
            grid_val = theano.printing.Print('Points to interpolate')(grid_val)

        return grid_val

    def extend_dual_kriging(self):
        # TODO Think what object is worth to save to speed up computation
        """
        Tile the dual kriging vector to cover all the points to interpolate.So far I just make a matrix with the
        dimensions len(DK)x(grid) but in the future maybe I have to try to loop all this part so consume less memory

        Returns:
            theano.tensor.matrix: Matrix with the Dk parameters repeated for all the points to interpolate
        """

        grid_val = self.x_to_interpolate()
        # if self.weights.get_value() is None:
        #     DK_parameters = self.solve_kriging()
        # else:
        #     DK_parameters = self.weights
        DK_parameters = self.weights
        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        # TODO IMP: Change the tile by a simple dot op -> The DOT version in gpu is slower
        DK_weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        if self.dot_version:
            DK_weights = DK_parameters

        return DK_weights


class TheanoExportGeo(TheanoExport):

    # def __init__(self):
    #
    #     super().__init__()

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
                weights[:length_of_CG],
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
            weights[
            length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_reescale * _aux_magic_term *
            universal_grid_surface_points_matrix[:self.n_universal_eq_T_op]
            , axis=0))

        if self.dot_version:
            f_0 = T.dot(
                weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I],
                self.gi_reescale * _aux_magic_term *
                universal_grid_surface_points_matrix[:self.n_universal_eq_T_op])

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

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

        fault_matrix_selection_non_zero = self.fault_matrix[a:b] #* self.fault_mask[a:b] + 1

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

    def scalar_field_loop(self, a, b, Z_x, grid_val, weights):

        sigma_0_grad = self.contribution_gradient_interface(grid_val[a:b], weights[:, a:b])
        sigma_0_interf = self.contribution_interface(grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift(grid_val[a:b], weights[:, a:b], a, b)
        f_1 = self.faults_contribution(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1)
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

        grid_shape = T.stack([grid_val.shape[0]], axis=0)
        Z_x_init = T.zeros(grid_shape, dtype='float32')
        if 'grid_shape' in self.verbose:
            grid_shape = theano.printing.Print('grid_shape')(grid_shape)

        steps = 1e13 / self.matrices_shapes()[-1] / grid_shape
        slices = T.concatenate((T.arange(0, grid_shape[0], steps[0], dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        Z_x_loop, updates3 = theano.scan(
            fn=self.scalar_field_loop,
            outputs_info=[Z_x_init],
            sequences=[dict(input=slices, taps=[0, 1])],
            non_sequences=[grid_val, weights],
            profile=False,
            name='Looping grid',
            return_list=True)

        self.Z_x = Z_x_loop[-1][-1]
        self.Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            self.Z_x = theano.printing.Print('Potential field at all points')(self.Z_x)

        return self.Z_x
    #
    # def get_scalar_field_at_surface_points_values(self):
    #
    #     # self.Z_x = self.scalar_field_at_all()
    #     self.scalar_field_at_surface_points_values = self.Z_x[-2*self.len_points: -self.len_points][self.npf]
    #     return self.scalar_field_at_surface_points_values
        # Z_x_grid = Z_x[:-2*self.len_points]
        # Z_x_sp = Z_x[-2*self.len_points:]
        # scalar_field_at_surface_points_values = Z_x_sp[: -self.len_points][self.npf]

       # return [Z_x_grid, Z_x_sp, scalar_field_at_surface_points_values]


class TheanoBlock(TheanoExport):
    def __init__(self, Z_x_op = None):

        if type(Z_x_op) == theano.compile.builders.OpFromGraph:
            super(TheanoBlock, self).__init__()
            self.Z_x_op = Z_x_op
            self.Z_x = Z_x_op(*self.input_parameters_export_kriging())
            self.Z_x.name = 'Scalar op'
        else:
            self.Z_x = T.vector('Scalar')
         #   self.grid_val_T = T.matrix('Coordinates of the grid points to interpolate')
         #   self.weights = T.vector('kriging weights')

        # elif Z_x is None:
        #     self.Z_x = self.scalar_field_at_all

        self.n_surface = theano.shared(np.arange(1, 5000, dtype='int32'), "ID of the surface")
        self.n_surface_op = self.n_surface

        self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')
      #  self.npf_op = T.vector('Number of points per surface accumulative', dtype='int32')#self.npf[[0, -2]]
      #  self.Z_x = T.vector('Value of the potential field at every x_to_intep point')
        self.scalar_field_at_surface_points_values = self.Z_x[-2*self.len_points: -self.len_points][self.npf_op]

       # self.scalar_field_at_surface_points_values = T.vector('')
        #self.len_points = T.vector('Length of rest or ref surface points arrays')
        self.is_finite = theano.shared(np.zeros(3, dtype='int32'), 'The series (fault) is finite')
        self.inf_factor = self.is_finite * 10
        self.is_fault = theano.shared(np.zeros(3, dtype='int32'), 'The series is fault')

        # TODO shared or variable
        self.values_properties_op = T.matrix('Values that the blocks are taking')

    def input_parameters_block_formations(self):
        ipl = [self.values_properties_op, self.Z_x,
               self.number_of_points_per_surface_T_op,
               self.surface_points_all]

        # self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    def input_parameters_block_faults(self):
        ipl = [self.values_properties_op, self.Z_x,
               self.number_of_points_per_surface_T_op,
               self.surface_points_all, self.grid_val_T]

        # self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    def select_finite_faults(self):
        fault_points = T.vertical_stack(T.stack([self.ref_layer_points[0]], axis=0), self.rest_layer_points).T
        ctr = T.mean(fault_points, axis=1)
        x = fault_points - ctr.reshape((-1, 1))
        M = T.dot(x, x.T)
        U = T.nlinalg.svd(M)[2]
        rotated_x = T.dot(self.x_to_interpolate(), U)
        rotated_fault_points = T.dot(fault_points.T, U)
        rotated_ctr = T.mean(rotated_fault_points, axis=0)
        a_radio = (rotated_fault_points[:, 0].max() - rotated_fault_points[:, 0].min()) / 2 + self.inf_factor[
            self.n_surface_op[0] - 1]
        b_radio = (rotated_fault_points[:, 1].max() - rotated_fault_points[:, 1].min()) / 2 + self.inf_factor[
            self.n_surface_op[0] - 1]
        sel = T.lt((rotated_x[:, 0] - rotated_ctr[0]) ** 2 / a_radio ** 2 + (
                    rotated_x[:, 1] - rotated_ctr[1]) ** 2 / b_radio ** 2,
                   1)

        if "select_finite_faults" in self.verbose:
            sel = theano.printing.Print("scalar_field_iter")(sel)

        return sel

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
        if 'sigma' in self.verbose:
            sigm = theano.printing.Print("middle point")(sigm)
        #      n_surface = theano.printing.Print("n_surface")(n_surface)
        return sigm

    def export_fault_block(self, Z_x = None, slope=50, offset_slope=5000):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        if Z_x is None:
            Z_x = self.Z_x

        # Max and min values of the potential field.
        # max_pot = T.max(Z_x) + 1
        # min_pot = T.min(Z_x) - 1
        # max_pot += max_pot * 0.1
        # min_pot -= min_pot * 0.1

        # Value of the potential field at the surface_points of the computed series
        # TODO timeit
        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        #     min_pot = theano.printing.Print("min_pot")(min_pot)

        # max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        # min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        #l = slope / (max_pot - min_pot)  # (max_pot - min_pot)

        # This is the different line with respect layers
        l = T.switch(self.select_finite_faults(), offset_slope / (max_pot - min_pot), slope / (max_pot - min_pot))
        #  l = theano.printing.Print("l")(l)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad], axis=0),
            self.scalar_field_at_surface_points_values,
            T.stack([min_pot - boundary_pad], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Here we just take the first element of values properties because at least so far we do not find a reason
        # to populate fault blocks with anything else

        n_surface_op_float_sigmoid = T.repeat(self.values_properties_op[[0], :], 2, axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!
        # TODO I think should be -> n_surface_op_float_sigmoid[:, 2] - n_surface_op_float_sigmoid[:, 1]
        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, 1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        fault_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]),
                       T.arange(0, n_surface_op_float_sigmoid.shape[1], 2, dtype='int64')],
            non_sequences=[Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        fault_block = fault_block.sum(axis=0)

        # Add name to the theano node
        fault_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            fault_block = theano.printing.Print(fault_block.name)(fault_block)

        return fault_block

    def export_formation_block(self, Z_x = None, slope=5000,weights=None):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """
        # TODO: IMP set soft max in the borders

        if Z_x is None:
            Z_x = self.Z_x

        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        #     min_pot = theano.printing.Print("min_pot")(min_pot)

        max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        l = slope / (max_pot - min_pot)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad], axis=0),
            self.scalar_field_at_surface_points_values,
            T.stack([min_pot - boundary_pad], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Loop to segment the distinct lithologies

        n_surface_op_float_sigmoid = T.repeat(self.values_properties_op, 2, axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!
        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        formations_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]), T.arange(0, n_surface_op_float_sigmoid.shape[1],
                                                                            2, dtype='int64')],
            non_sequences=[self.Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        formations_block = formations_block.sum(axis=0)

        # Add name to the theano node
        formations_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            formations_block = theano.printing.Print(formations_block.name)(formations_block)

        return formations_block


class TheanoLoop(TheanoExportGeo):
    def __init__(self, weights_op, Z_x_op, export_formation_op, export_fault_op):
        super(TheanoLoop, self).__init__()

        self.weights_op = weights_op
        self.Z_x_op = Z_x_op
        self.export_formation_op = export_formation_op
        self.export_fault_op = export_fault_op

        # FORMATIONS
        # ----------
        self.n_surface = theano.shared(np.arange(2, 5, dtype='int32'), "ID of the surface")
        self.n_surface_op = self.n_surface
        self.surface_values = theano.shared((np.arange(2, 4, dtype=float).reshape(2, -1)),
                                            "Value of the surface to compute")
        self.n_surface_op_float = self.surface_values

        # FAULTS
        # ------
        # Init fault relation matrix
        self.fault_relation = theano.shared(np.array([[0, 1, 0, 1],
                                                      [0, 0, 1, 1],
                                                      [0, 0, 0, 1],
                                                      [0, 0, 0, 0]]), 'fault relation matrix')

        self.compute_weights = T.vector('Vector controlling if weights must be recomputed', dtype=bool)
        self.compute_scalar = T.vector('Vector controlling if scalar matrix must be recomputed', dtype=bool)
        self.compute_block = T.vector('Vector controlling if block matrix must be recomputed', dtype=bool)

        self.inf_factor = theano.shared(np.ones(200, dtype='int32') * 10, 'Arbitrary scalar to make df infinite')

        # STRUCTURE
        # ---------
        # This parameters give me the shape of the different groups of data. I pass all data together and I threshold it
        # using these values to the different potential fields and surfaces
        #self.is_fault = is_fault
        #self.is_lith = i
        self.n_faults = theano.shared(0, 'Number of faults')
        self.n_surfaces_per_series = theano.shared(np.arange(2, dtype='int32'), 'List with the number of surfaces')
        self.n_universal_eq_T = theano.shared(np.zeros(5, dtype='int32'), "Grade of the universal drift")



        # This is not accumulative
        self.number_of_points_per_surface_T = theano.shared(np.zeros(3, dtype='int32'))  # TODO is DEP?
        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T
        # This is accumulative
        self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')
        self.npf_op = self.npf[[0, -2]]
        self.len_series_i = theano.shared(np.arange(2, dtype='int32'), 'Length of surface_points in every series')
        self.len_series_f = theano.shared(np.arange(2, dtype='int32'), 'Length of foliations in every series')

        self.weights_matrix = theano.shared(np.zeros((3, 10000)), 'Weights matrix')
        self.scalar_fields_matrix = theano.shared(np.zeros((3, 10000)), 'Scalar matrix')
        self.block_matrix = theano.shared(np.zeros((3,10000)), "block matrix")#T.zeros((self.n_surface.shape[0], self.grid_val_T.shape[0] + 2 * self.len_points))
        #T.zeros((self.n_surface.shape[0], self.grid_val_T.shape[0] + 2 * self.len_points))
        if 'initial matrices' in self.verbose:
            self.block_matrix = theano.printing.Print("block_matrix")(self.block_matrix)
            self.scalar_fields_matrix = theano.printing.Print("scalar_fields")(self.scalar_fields_matrix)

    def compute_all_series(self):
        # # Change the flag to extend the graph in the compute fault and compute series function
        # lith_matrix = T.zeros((0, 0, self.grid_val_T.shape[0] + 2 * self.len_points))
        #
        # # Init to matrix which contains the block and scalar field of every fault
        # self.fault_matrix = T.zeros((self.n_faults * 2, self.grid_val_T.shape[0] + 2 * self.len_points))
        # self.fault_matrix_f = T.zeros((self.n_faults * 2, self.grid_val_T.shape[0] + 2 * self.len_points))
        #
        # self.final_scalar_field_at_surfaces_op = self.final_scalar_field_at_surfaces
        # self.final_potential_field_at_faults_op = self.final_scalar_field_at_faults
        #
        # # Init df block. Here we store the block and potential field results of one iteration
        # self.fault_block_init = T.zeros((2, self.grid_val_T.shape[0] + 2 * self.len_points))
        # self.fault_block_init.name = 'final block of df init'
        # self.yet_simulated = T.nonzero(T.eq(self.fault_block_init[0, :], 0))[0]



        # Looping
        series, updates1 = theano.scan(
            fn=self.compute_a_series,
            outputs_info=[
                dict(initial=self.weights_matrix, taps=[0]),
                dict(initial=self.scalar_fields_matrix, taps=[0]),
                dict(initial=self.block_matrix, taps=[0])
                ],  # This line may be used for the df network
            sequences=[dict(input=self.len_series_i, taps=[0, 1]),
                       dict(input=self.len_series_f, taps=[0, 1]),
                       dict(input=self.n_surfaces_per_series, taps=[0, 1]),
                       dict(input=self.n_universal_eq_T, taps=[0]),
                       dict(input=self.compute_weights, taps=[0]),
                       dict(input=self.compute_scalar, taps=[0]),
                       dict(input=self.compute_block, taps=[0])],
           # non_sequences=self.fault_block_init,
            name='Looping',
            return_list=True,
            profile=False
        )

        return series

    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         u_grade_iter,
                         weights_matrix, scalar_fields, block_matrix,
                         compute_weight, compute_scalar, compute_block
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

        # # Theano Var
        # self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
        # self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        # self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.surface_points_op = self.surface_points[len_i_0: len_i_1, :]

        self.weights_op = self.weights[len_i_0: len_i_1]
        # self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        # self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

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
        # if len(self.gradients) is not 0:
        #     fault_matrix = block_matrix[::5][T.nonzero(T.cast(faults_relation_rep, "int8"))[0], :]
        # else:

        fault_matrix = block_matrix[T.nonzero(T.cast(faults_relation_rep, "int8"))[0], :]

        if 'fault_matrix_loop' in self.verbose:
            fault_matrix = theano.printing.Print('self fault matrix')(self.fault_matrix)

        # ================================
        # Computing the fault scalar field
        # ================================
        Z_x = self.Z_x_op(self.dips_position_tiled, self.surface_points,
                          self.fault_matrix[2 * self.len_points:], self.fault_matrix[:2 * self.len_points],
                          self.weights_op, self.grid_val_T)

        series_block = theano.ifelse.ifelse(self.is_fault[T.cast(self.n_surface_op-1, 'int8')][0],
                                            self.export_fault_block(),
                                            self.export_formation_block(),
                                            name="Is faults condition")

        aux_ind = T.max(self.n_surface_op, 0)

        block_matrix = T.set_subtensor(block_matrix[(aux_ind - 1), :], series_block[0])
        scalar_fields = T.set_subtensor(scalar_fields[(aux_ind-1), :], Z_x)

        return block_matrix, scalar_fields

       #
       #  potential_field_values, faults_matrix = self.block_fault(slope=1000)
       #
       #  # Update the block matrix
       #  final_block = T.set_subtensor(
       #              final_block[0, :],
       #              faults_matrix[0])#T.cast(T.cast(faults_matrix, 'bool'), 'int8'))
       #
       #  # Update the potential field matrix
       # # potential_field_values = self.scalar_field_at_all()
       #
       #  final_block = T.set_subtensor(
       #              final_block[1, :],
       #              potential_field_values)
       #
       #  # Store the potential field at the surface_points
       #  self.final_potential_field_at_faults_op = T.set_subtensor(self.final_potential_field_at_faults_op[self.n_surface_op-1],
       #                                                            self.scalar_field_at_surface_points_values)
       #
       #
       #
       #  if len(self.gradients) is not 0:
       #      weights = self.extend_dual_kriging()
       #      gradients = self.gradient_field_at_all(weights, self.gradients)
       #      final_block = T.set_subtensor(
       #          final_block[2:, :],
       #          gradients)
       #      # Setting the values of the fault matrix computed in the current iteration
       #      fault_matrix = T.set_subtensor(fault_matrix[(aux_ind - 1) * 5:aux_ind * 5, :], final_block)
       #
       #  else:
       #      # Setting the values of the fault matrix computed in the current iteration
       #      fault_matrix = T.set_subtensor(fault_matrix[(aux_ind-1)*2:aux_ind*2, :], final_block)
       #
       #  return fault_matrix, self.final_potential_field_at_faults_op,







#
# class TheanoMask(object):
#     def __init__(self):
#
#         self.Z_x = T.vector('Value of the potential field at every x_to_intep point')
#         self.is_onlap = theano.shared(np.array([False, False], dtype=bool))
#
#         self.npf = theano.shared(np.zeros(3, dtype='int32'), 'Number of points per surface accumulative')
#         self.len_points = T.vector('Length of rest or ref surface points arrays')
#
#     def get_scalar_field_at_surface_points(self):
#         self.scalar_field_at_surface_points_values = self.Z_x[-2*self.len_points: -self.len_points][self.npf]
#
#     def set_mask(self):
#
#
#
#         self.mask = T.nonzero(T.le(self.Z_x, self.scalar_field_at_surface_points_values]))[
#             0]  # This -1 comes to get the last scalar field value (the bottom) of the previous series
#         self.mask.name = 'Indices where the scalar field overprints'


class TheanoExportGradient(TheanoExport):

    def __init__(self):

        super().__init__()

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
        hu_rest = (- self.rest_layer_points[:, dir_val] + grid_val[:, dir_val].reshape(
            (grid_val[:, dir_val].shape[0], 1)))
        hu_ref = (- self.ref_layer_points[:, dir_val] + grid_val[:, dir_val].reshape(
            (grid_val[:, dir_val].shape[0], 1)))

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
        h_u = T.tile(self.dips_position[:, direction_val] - grid_val[:, direction_val].reshape(
            (grid_val[:, direction_val].shape[0], 1)), 3)
        h_v = T.horizontal_stack(
            T.tile(self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1)),
                   1),
            T.tile(self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1)),
                   1),
            T.tile(self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1)),
                   1))

        perpendicularity_vector = T.zeros(T.stack([length_of_CG], axis=0))
        perpendicularity_vector = T.set_subtensor(
            perpendicularity_vector[
            self.dips_position.shape[0] * direction_val:self.dips_position.shape[0] * (direction_val + 1)], 1)

        sigma_0_grad = T.sum(
            (weights[:length_of_CG] * (
                    ((-h_u * h_v).T / sed_dips_SimPoint ** 2) *
                    ((
                             (sed_dips_SimPoint < self.a_T) *  # first derivative
                             (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                                             35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                                             21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7))) +
                     (sed_dips_SimPoint < self.a_T) *  # Second derivative
                     self.c_o_T * 7 * (9 * sed_dips_SimPoint ** 5 - 20 * self.a_T ** 2 * sed_dips_SimPoint ** 3 +
                                       15 * self.a_T ** 4 * sed_dips_SimPoint - 4 * self.a_T ** 5) / (
                             2 * self.a_T ** 7)) -
                    (perpendicularity_vector.reshape((-1, 1)) *
                     ((sed_dips_SimPoint < self.a_T) *  # first derivative
                      self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                                    35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                                    21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))

            ))
            , axis=0)

        return sigma_0_grad

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

            # xz
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

    def gradient_field_at_all(self, weights=None, gradients=[]):

        grid_val = self.x_to_interpolate()
        if weights is None:
            weights = self.extend_dual_kriging()

        grid_shape = T.stack([grid_val.shape[0]], axis=0)

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