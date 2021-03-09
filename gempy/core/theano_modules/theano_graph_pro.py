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
from theano import tensor as T
from theano import sparse

import theano.ifelse as tif
import numpy as np
import sys

# check if skcuda is installed
try:
    import skcuda

    SKCUDA_IMPORT = True
except ImportError:
    SKCUDA_IMPORT = False

theano.config.openmp_elemwise_minsize = 50000
theano.config.openmp = True

theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float32'
theano.config.on_opt_error = 'ignore'

theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'
theano.config.profile_memory = False
theano.config.scan.debug = False
theano.config.profile = False


def as_sparse_variable(x, name=None):
    """
    Wrapper around SparseVariable constructor to construct
    a Variable with a sparse matrix with the same dtype and
    format.
    Parameters
    ----------
    x
        A sparse matrix.
    Returns
    -------
    object
        SparseVariable version of `x`.
    """

    # TODO
    # Verify that sp is sufficiently sparse, and raise a
    # warning if it is not

    if isinstance(x, theano.gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a "
                             "multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, theano.gof.Variable):
        if not isinstance(x.type, theano.sparse.type.SparseType):
            raise TypeError("Variable type field must be a SparseType.", x,
                            x.type)
        return x
    try:
        return theano.constant(x, name=name)
    except TypeError:
        raise TypeError("Cannot convert %s to SparseType" % x, type(x))


as_sparse = as_sparse_variable


class SolveSparse(T.Op):
    # itypes = [T.dvector]
    # otypes = [T.dvector]

    def make_node(self, x, y):
        x, y = as_sparse_variable(x), as_sparse_variable(y)
        assert x.format in ["csr", "csc"]
        assert y.format in ["csr", "csc"]
        out_dtype = theano.scalar.upcast(x.type.dtype, y.type.dtype)
        return theano.gof.Apply(self,
                                [x, y],
                                [T.tensor(out_dtype, broadcastable=(False,))])
        #   [theano.sparse.type.SparseType(dtype=out_dtype,
        #               format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        from scipy.sparse.linalg import spsolve

        (C, b) = inputs
        b.ndim = 1
        weights = spsolve(C, b)
        outputs[0][0] = np.array(weights)


solv = SolveSparse()


class TheanoGraphPro(object):
    def __init__(self, optimizer='fast_compile', verbose=None, dtype=None,
                 output=None, **kwargs):
        """
        Class to create the symbolic theano graph with all the interpolation-FW engine

        Args:
            optimizer ({fast_compile, fast_run}): Theano optimizer option. See theano docs
            verbose [list]: list of many parameters. If the name of the parameter is on the list the value of the
             parameter will be printed in run time.
            dtype [{float64, float32}]: Type of float
            ** kwargs:
                - gradient[bool]: If true adapt the graph for AD
                - max_speed[int]: As the number gets higher true graph will be adapted to return meaningful
                 gradients with AD
        """
        self.lenght_of_faults = T.cast(0, 'int32')
        self.pi = theano.shared(3.14159265359, 'pi')

        # OPTIONS
        # -------
        if output is None:
            output = ['geology']
        self.output = output

        if verbose is None:
            self.verbose = [None]
        else:
            self.verbose = verbose

        self.compute_type = output

        if dtype is None:
            dtype = 'float32' if theano.config.device == 'cuda' else 'float64'

        self.dtype = dtype

        # Trade speed for memory this will consume more memory
        self.max_speed = kwargs.get('max_speed', 1)
        self.sparse_version = kwargs.get('sparse_version', False)

        self.gradient = kwargs.get('gradient', False)
        self.device = theano.config.device
        theano.config.floatX = dtype
        theano.config.optimizer = optimizer

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------

        self.a_T = theano.shared(np.ones(3, dtype=dtype), "Range")

        self.a_T_scalar = self.a_T
        self.c_o_T = theano.shared(np.ones(3, dtype=dtype), 'Covariance at 0')
        self.c_o_T_scalar = self.c_o_T

        self.n_universal_eq_T = theano.shared(np.zeros(5, dtype='int32'),
                                              "Grade of the universal drift")
        self.n_universal_eq_T_op = theano.shared(3)

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        # This is not accumulative
        self.number_of_points_per_surface_T = theano.shared(
            np.zeros(3, dtype='int32'),
            'Number of points per surface used to split rest-ref')
        self.number_of_points_per_surface_T_op = T.vector(
            'Number of points per surface used to split rest-ref',
            dtype='int32')
        self.npf = T.cumsum(
            T.concatenate((T.stack([0]), self.number_of_points_per_surface_T[:-1])))
        self.npf_op = self.npf
        self.npf.name = 'Number of points per surfaces after rest-ref. ' \
                        'This is used for finding the different' \
                        'surface points withing a layer.'

        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](np.ones(4)),
                                                  'Nugget effect of gradients')
        self.nugget_effect_scalar_T = theano.shared(np.cast[dtype](np.ones(4)),
                                                    'Nugget effect of scalar')

        self.nugget_effect_grad_T_op = self.nugget_effect_grad_T

        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES
        # ---------
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")

        self.surface_points_all = T.matrix("All the surface_points points at once")

        self.len_points = self.surface_points_all.shape[0] - \
                          self.number_of_points_per_surface_T.shape[0]

        # Tiling dips to the 3 spatial coordinations
        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series. I initialized them as the whole arrays but then they will take
        # the data of every potential field
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all

        rest_ref_aux = self.set_rest_ref_matrix(self.number_of_points_per_surface_T)
        self.ref_layer_points_all = rest_ref_aux[0]
        self.rest_layer_points_all = rest_ref_aux[1]

        self.nugget_effect_scalar_T_ref_rest = self.set_nugget_surface_points(
            rest_ref_aux[2], rest_ref_aux[3],
            self.number_of_points_per_surface_T)

        self.nugget_effect_scalar_T_op = self.nugget_effect_scalar_T_ref_rest

        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

        self.fault_matrix = T.matrix(
            'Full block matrix for faults or drift. '
            'We take 2 times len points for the fault'
            'drift.')

        self.input_parameters_kriging = [self.dips_position_all, self.dip_angles_all,
                                         self.azimuth_all,
                                         self.polarity_all, self.surface_points_all,
                                         self.fault_matrix]

        # COMPUTE SCALAR FIELDS
        # ---------
        # VARIABLES
        # ---------
        self.grid_val_T = T.matrix('Coordinates of the grid points to interpolate')
        self.input_parameters_export = [self.dips_position_all,
                                        self.surface_points_all,
                                        self.fault_matrix,
                                        self.grid_val_T]

        self.input_parameters_kriging_export = [self.dips_position_all,
                                                self.dip_angles_all,
                                                self.azimuth_all,
                                                self.polarity_all,
                                                self.surface_points_all,
                                                self.fault_matrix, self.grid_val_T,
                                                ]

        # interface_loc = self.fault_matrix.shape[1] - 2 * self.len_points
        interface_loc = 0  # self.grid_val_T.shape[0]
        self.fault_drift_at_surface_points_rest = self.fault_matrix[
                                                  :,
                                                  interface_loc: interface_loc + self.len_points]
        self.fault_drift_at_surface_points_ref = self.fault_matrix[
                                                 :,
                                                 interface_loc + self.len_points:]

        # COMPUTE BLOCKS
        # ---------
        # VARIABLES
        # ---------

        if self.gradient:
            self.sig_slope = theano.shared(np.array(50, dtype=dtype),
                                           'Sigmoid slope for gradient')
        else:
            self.sig_slope = theano.shared(np.array(50000, dtype=dtype),
                                           'Sigmoid slope')
            self.not_l = theano.shared(np.array(50., dtype=dtype), 'Sigmoid Outside')
            self.ellipse_factor_exponent = theano.shared(np.array(2., dtype=dtype),
                                                         'Attenuation factor')

        # It is a matrix because of the values: porosity, sus, etc
        self.values_properties_op = T.matrix('Values that the blocks are taking')

        self.n_surface = T.arange(1, 5000, dtype='int32')
        self.n_surface.name = 'ID of surfaces'

        self.input_parameters_block = [self.dips_position_all, self.dip_angles_all,
                                       self.azimuth_all,
                                       self.polarity_all, self.surface_points_all,
                                       self.fault_matrix, self.grid_val_T,
                                       self.values_properties_op]
        # ------
        # Shared
        # ------
        self.is_finite_ctrl = theano.shared(np.zeros(3, dtype='int32'),
                                            'The series (fault) is finite')
        self.inf_factor = 0
        self.is_fault = theano.shared(np.zeros(5000, dtype=bool))

        # COMPUTE LOOP
        # ------
        # Shared
        # ------
        # Init fault relation matrix
        self.fault_relation = theano.shared(np.array([[0, 0],
                                                      [0, 0]]),
                                            'fault relation matrix')

        # Structure
        self.n_surfaces_per_series = theano.shared(np.arange(2, dtype='int32'),
                                                   'List with the number of surfaces')
        self.len_series_i = theano.shared(np.arange(2, dtype='int32'),
                                          'Length of surface_points in every series')
        self.len_series_o = theano.shared(np.arange(2, dtype='int32'),
                                          'Length of foliations in every series')
        self.len_series_w = theano.shared(np.arange(2, dtype='int32'),
                                          'Length of weights in every series')

        # Control flow
        self.compute_weights_ctrl = T.vector(
            'Vector controlling if weights must be recomputed', dtype='bool')
        self.compute_scalar_ctrl = T.vector(
            'Vector controlling if scalar matrix must be recomputed', dtype='bool')
        self.compute_block_ctrl = T.vector(
            'Vector controlling if block matrix must be recomputed', dtype='bool')
        self.is_finite_ctrl = theano.shared(np.zeros(3, dtype='int32'),
                                            'The series (fault) is finite')
        self.onlap_erode_ctrl = theano.shared(np.zeros(3, dtype='int32'),
                                              'Onlap erode')

        self.input_parameters_loop = [self.dips_position_all, self.dip_angles_all,
                                      self.azimuth_all,
                                      self.polarity_all, self.surface_points_all,
                                      self.fault_matrix, self.grid_val_T,
                                      self.values_properties_op,
                                      self.compute_weights_ctrl,
                                      self.compute_scalar_ctrl,
                                      self.compute_block_ctrl]

        self.is_erosion = theano.shared(np.array([1, 0]))
        self.is_onlap = theano.shared(np.array([0, 1]))

        self.offset = theano.shared(10.)
        self.shift = 0

        if 'gravity' in self.compute_type:
            self.lg0 = theano.shared(np.array(0, dtype='int64'),
                                     'arg_0 of the centered grid')
            self.lg1 = theano.shared(np.array(1, dtype='int64'),
                                     'arg_1 of the centered grid')

            self.tz = theano.shared(np.empty(0, dtype=self.dtype), 'tz component')
            self.pos_density = theano.shared(np.array(1, dtype='int64'),
                                             'position of the density on the values matrix')

        if 'magnetics' in self.compute_type:
            self.lg0 = theano.shared(np.array(0, dtype='int64'),
                                     'arg_0 of the centered grid')
            self.lg1 = theano.shared(np.array(1, dtype='int64'),
                                     'arg_1 of the centered grid')

            self.B_ext = theano.shared(50e-6,
                                       'External magnetic field in [T], in magnetic surveys this is the'
                                       ' geomagnetic field - varies temporaly.')
            self.incl = theano.shared(np.array(1, dtype=dtype),
                                      'Dip of the geomagnetic field in degrees- varies'
                                      ' spatially')
            self.decl = theano.shared(np.array(1, dtype=dtype),
                                      'Angle between magnetic and true North in degrees -'
                                      ' varies spatially')
            self.pos_magnetics = theano.shared(np.array(2, dtype='int64'),
                                               'position of sus. on the values matrix.')
            self.V = theano.shared(np.ones((6, 10), dtype=self.dtype),
                                   'Solutions to volume integrals -'
                                   ' same for each device')
        if 'topology' in self.compute_type:
            # Topology
            self.max_lith = theano.shared(np.array(10, dtype='int'),
                                          'Max id of the lithologies')

            # TODO: the position of topology is not properly implemented yet
            self.pos_topology_id = theano.shared(np.array(-1, dtype='int'),
                                                 'Position of the surface dataframe with the'
                                                 'right topology ids to be able to have unique'
                                                 'identifiers')

            self.regular_grid_res = theano.shared(np.ones(3, dtype='int'),
                                                  'Resolution of the regular grid')
            self.dxdydz = theano.shared(np.ones(3, dtype=dtype),
                                        'Size of the voxels in each direction.')

        # Results matrix
        self.weights_vector = theano.shared(np.cast[dtype](np.zeros(10000)),
                                            'Weights vector')
        self.scalar_fields_matrix = theano.shared(
            np.cast[dtype](np.zeros((3, 10000))), 'Scalar matrix')
        self.block_matrix = theano.shared(np.cast[dtype](np.zeros((3, 3, 10000))),
                                          "block matrix")
        self.mask_matrix = theano.shared(np.zeros((3, 10000), dtype='bool'),
                                         "mask matrix")
        self.sfai = T.zeros(
            [self.is_erosion.shape[0], self.n_surfaces_per_series[-1]])

        self.new_block = self.block_matrix
        self.new_weights = self.weights_vector
        self.new_scalar = self.scalar_fields_matrix
        self.new_mask = self.mask_matrix
        self.new_sfai = self.sfai

    def compute_weights(self):
        return self.solve_kriging()

    def compute_scalar_field(self, weights, grid, fault_matrix):
        grid_val = self.x_to_interpolate(grid)
        return self.scalar_field_at_all(weights, grid_val, fault_matrix)

    def compute_formation_block(self, Z_x, scalar_field_at_surface_points, values):
        return self.export_formation_block(Z_x, scalar_field_at_surface_points,
                                           values)

    def compute_fault_block(self, Z_x, scalar_field_at_surface_points, values,
                            n_series, grid):
        grid_val = self.x_to_interpolate(grid)
        finite_faults_ellipse = self.select_finite_faults(grid_val)
        return self.export_fault_block(Z_x, scalar_field_at_surface_points, values,
                                       finite_faults_ellipse)

    def compute_final_block(self, mask, block):

        # We add the axis 1 to the mask. Axis 1 is the properties values axis
        # Then we sum over the 0 axis. Axis 0 is the series
        final_model = T.sum(T.stack([mask], axis=1) * block, axis=0)
        return final_model

    def compute_series(self, grid=None, shift=None):
        if grid is None:
            grid = self.grid_val_T
        if shift is None:
            shift = self.shift

        # TODO I have to take this function to interpolator and making the behaviour the same as mask_matrix
        self.mask_matrix_f = T.zeros_like(self.mask_op2)
        self.fault_matrix = T.zeros_like(self.block_op)

        # Looping
        series, self.updates1 = theano.scan(
            fn=self.compute_a_series,
            outputs_info=[
                dict(initial=self.block_op),
                dict(initial=self.weights_op),
                dict(initial=self.scalar_op),
                dict(initial=self.sfai_op),
                dict(initial=self.mask_op2),
                dict(initial=self.mask_matrix_f),
                dict(initial=self.fault_matrix),
                dict(initial=T.cast(0, 'int64'))

            ],  # This line may be used for the df network
            sequences=[dict(input=self.len_series_i, taps=[0, 1]),
                       dict(input=self.len_series_o, taps=[0, 1]),
                       dict(input=self.len_series_w, taps=[0, 1]),
                       dict(input=self.n_surfaces_per_series, taps=[0, 1]),
                       dict(input=self.n_universal_eq_T, taps=[0]),
                       dict(input=self.compute_weights_ctrl, taps=[0]),
                       dict(input=self.compute_scalar_ctrl, taps=[0]),
                       dict(input=self.compute_block_ctrl, taps=[0]),
                       dict(input=self.is_finite_ctrl, taps=[0]),
                       dict(input=self.is_erosion, taps=[0]),
                       dict(input=self.is_onlap, taps=[0]),
                       dict(input=T.arange(0, 5000, dtype='int32'), taps=[0]),
                       dict(input=self.a_T, taps=[0]),
                       dict(input=self.c_o_T_scalar, taps=[0])
                       ],
            non_sequences=[grid, shift],
            name='Looping',
            return_list=True,
            profile=False
        )

        self.block_op = series[0][-1]
        self.weights_op = series[1][-1]
        self.scalar_op = series[2][-1]
        self.sfai_op = series[3][-1]

        mask = series[4][-1]
        mask_rev_cumprod = T.vertical_stack(mask[[-1]],
                                            T.cumprod(T.invert(mask[:-1]), axis=0))
        self.mask_op2 = mask_rev_cumprod
        block_mask = mask * mask_rev_cumprod

        fault_mask = series[5][-1]

        fault_block = self.compute_final_block(fault_mask, self.block_op)
        final_model = self.compute_final_block(block_mask, self.block_op)

        return [final_model, self.block_op, fault_block, self.weights_op,
                self.scalar_op,
                self.sfai_op, block_mask, fault_mask]

    def create_oct_voxels(self, xyz, level=1):
        x_ = T.repeat(T.stack((xyz[:, 0] - self.dxdydz[0] / level / 4,
                               xyz[:, 0] + self.dxdydz[0] / level / 4), axis=1), 4,
                      axis=1)
        y_ = T.tile(T.repeat(T.stack((xyz[:, 1] - self.dxdydz[1] / level / 4,
                                      xyz[:, 1] + self.dxdydz[1] / level / 4),
                                     axis=1),
                             2, axis=1), (1, 2))

        z_ = T.tile(T.stack((xyz[:, 2] - self.dxdydz[2] / level / 4,
                             xyz[:, 2] + self.dxdydz[2] / level / 4), axis=1),
                    (1, 4))

        return T.stack((x_.ravel(), y_.ravel(), z_.ravel())).T

    def create_oct_level_dense(self, unique_val, grid):

        uv_3d = T.cast(T.round(unique_val[0, :T.prod(self.regular_grid_res)].reshape(
            self.regular_grid_res, ndim=3)),
            'int32')

        new_shape = T.concatenate([self.regular_grid_res, T.stack([3])])
        xyz = grid[:T.prod(self.regular_grid_res)].reshape(new_shape, ndim=4)

        shift_x = uv_3d[1:, :, :] - uv_3d[:-1, :, :]
        shift_x_select = T.neq(shift_x, 0)
        x_edg = (xyz[:-1, :, :][shift_x_select] + xyz[1:, :, :][shift_x_select]) / 2

        shift_y = uv_3d[:, 1:, :] - uv_3d[:, :-1, :]
        shift_y_select = T.neq(shift_y, 0)
        y_edg = (xyz[:, :-1, :][shift_y_select] + xyz[:, 1:, :][shift_y_select]) / 2

        shift_z = uv_3d[:, :, 1:] - uv_3d[:, :, :-1]
        shift_z_select = T.neq(shift_z, 0)
        z_edg = (xyz[:, :, :-1][shift_z_select] + xyz[:, :, 1:][shift_z_select]) / 2
        new_xyz_edg = T.vertical_stack(x_edg, y_edg, z_edg)

        return self.create_oct_voxels(new_xyz_edg)

    def create_oct_level_sparse(self, unique_val, grid):
        xyz_8 = grid.reshape((-1, 8, 3))
        # uv_8 = T.round(unique_val[0, :-2 * self.len_points].reshape((-1, 8)))

        uv_8 = T.round(unique_val[0, :].reshape((-1, 8)))

        shift_x = uv_8[:, :4] - uv_8[:, 4:]
        shift_x_select = T.neq(shift_x, 0)
        x_edg = (xyz_8[:, :4, :][shift_x_select] + xyz_8[:, 4:, :][
            shift_x_select]) / 2

        shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
        shift_y_select = T.neq(shift_y, 0)
        y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_y_select] +
                 xyz_8[:, [2, 3, 6, 7], :][shift_y_select]) / 2

        shift_z = uv_8[:, ::2] - uv_8[:, 1::2]
        shift_z_select = T.neq(shift_z, 0)
        z_edg = (xyz_8[:, ::2, :][shift_z_select] + xyz_8[:, 1::2, :][
            shift_z_select]) / 2

        new_xyz_edg = T.vertical_stack(x_edg, y_edg, z_edg)
        return self.create_oct_voxels(new_xyz_edg, level=2)

    def compute_series_oct(self, n_levels=3):
        # self.shift = 0

        solutions = self.compute_series(self.grid_val_T)
        unique_val = solutions[0] + self.max_lith * solutions[2]

        # # ------------------------------------------------------
        shift = self.grid_val_T.shape[0] + self.shift

        g = self.create_oct_level_dense(solutions[0], self.grid_val_T)

        # I need to init the solution matrices
        oct_sol = self.compute_series(g, shift)

        solutions.append(g)
        solutions.append(oct_sol[0])

        # -----------------------------------------------------
        unique_val = oct_sol[0] + self.max_lith * oct_sol[2]

        g1 = self.create_oct_level_sparse(unique_val[:, shift: g.shape[0] + shift],
                                          g)
        shift2 = g.shape[0] + shift

        oct_sol_2 = self.compute_series(g1, shift2)

        solutions.append(g1)
        solutions.append(oct_sol_2[0])

        self.new_block = self.block_op
        self.new_weights = self.weights_op
        self.new_scalar = self.scalar_op
        self.new_sfai = self.sfai_op
        self.new_mask = self.mask_op2

        return solutions

    def theano_output(self):
        # Create the solutions op
        self.block_op = self.block_matrix
        self.weights_op = self.weights_vector
        self.scalar_op = self.scalar_fields_matrix
        self.mask_op2 = self.mask_matrix
        self.sfai_op = self.sfai

        solutions = [theano.shared(np.nan)] * 15
        solutions[0] = theano.shared(np.zeros((2, 2)))
        # self.compute_type = ['lithology', 'topology']
        if 'geology' in self.compute_type:
            solutions[:9] = self.compute_series()
            # solutions[:12] = self.compute_series_oct()
        if 'topology' in self.compute_type:
            # This needs new data, resolution of the regular grid, value max
            unique_val = solutions[0][self.pos_topology_id] + \
                         self.max_lith * solutions[2][self.pos_topology_id]

            solutions[9:12] = self.compute_topology(unique_val)

        if 'gravity' in self.compute_type:
            densities = solutions[0][self.pos_density, self.lg0:self.lg1]
            solutions[12] = self.compute_forward_gravity_pro(densities)

        if 'magnetics' in self.compute_type:
            k_vals = solutions[0][self.pos_magnetics, self.lg0:self.lg1]
            solutions[13] = self.compute_forward_magnetics(k_vals)
        return solutions

    def compute_topology(self, unique_val):
        uv_3d = T.cast(T.round(unique_val[0, :T.prod(self.regular_grid_res)].reshape(
            self.regular_grid_res, ndim=3)),
            'int32')

        uv_l = T.horizontal_stack(uv_3d[1:, :, :].reshape((1, -1)),
                                  uv_3d[:, 1:, :].reshape((1, -1)),
                                  uv_3d[:, :, 1:].reshape((1, -1)))

        uv_r = T.horizontal_stack(uv_3d[:-1, :, :].reshape((1, -1)),
                                  uv_3d[:, :-1, :].reshape((1, -1)),
                                  uv_3d[:, :, :-1].reshape((1, -1)))

        shift = uv_l - uv_r
        select_edges = T.neq(shift.reshape((1, -1)), 0)
        select_edges_dir = select_edges.reshape((3, -1))

        select_voxels = T.zeros_like(uv_3d)
        select_voxels = T.inc_subtensor(
            select_voxels[1:, :, :],
            select_edges_dir[0].reshape((
                    self.regular_grid_res - np.array([1, 0, 0])),
                ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:-1, :, :],
                                        select_edges_dir[0].reshape((
                                                self.regular_grid_res - np.array(
                                            [1, 0,
                                             0])),
                                            ndim=3))

        select_voxels = T.inc_subtensor(select_voxels[:, 1:, :],
                                        select_edges_dir[1].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 1,
                                             0])),
                                            ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:, :-1, :],
                                        select_edges_dir[1].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 1,
                                             0])),
                                            ndim=3))

        select_voxels = T.inc_subtensor(select_voxels[:, :, 1:],
                                        select_edges_dir[2].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 0,
                                             1])),
                                            ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:, :, :-1],
                                        select_edges_dir[2].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 0,
                                             1])),
                                            ndim=3))

        uv_lr = T.vertical_stack(uv_l.reshape((1, -1)), uv_r.reshape((1, -1)))
        uv_lr_boundaries = uv_lr[
            T.tile(select_edges.reshape((1, -1)), (2, 1))].reshape((2, -1)).T

        # a = T.bincount(uv_lr_boundaries)
        edges_id, count_edges = T.extra_ops.Unique(return_counts=True, axis=0)(
            uv_lr_boundaries)
        return select_voxels, edges_id, count_edges

    def get_boundary_voxels(self, unique_val):
        uv_3d = T.cast(T.round(unique_val[0, :T.prod(self.regular_grid_res)].reshape(
            self.regular_grid_res, ndim=3)),
            'int32')

        uv_l = T.horizontal_stack(uv_3d[1:, :, :].reshape((1, -1)),
                                  uv_3d[:, 1:, :].reshape((1, -1)),
                                  uv_3d[:, :, 1:].reshape((1, -1)))

        uv_r = T.horizontal_stack(uv_3d[:-1, :, :].reshape((1, -1)),
                                  uv_3d[:, :-1, :].reshape((1, -1)),
                                  uv_3d[:, :, :-1].reshape((1, -1)))

        shift = uv_l - uv_r
        select_edges = T.neq(shift.reshape((1, -1)), 0)
        select_edges_dir = select_edges.reshape((3, -1))

        select_voxels = T.zeros_like(uv_3d)
        select_voxels = T.inc_subtensor(select_voxels[1:, :, :],
                                        select_edges_dir[0].reshape((
                                                self.regular_grid_res - np.array(
                                            [1, 0,
                                             0])),
                                            ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:-1, :, :],
                                        select_edges_dir[0].reshape((
                                                self.regular_grid_res - np.array(
                                            [1, 0,
                                             0])),
                                            ndim=3))

        select_voxels = T.inc_subtensor(select_voxels[:, 1:, :],
                                        select_edges_dir[1].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 1,
                                             0])),
                                            ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:, :-1, :],
                                        select_edges_dir[1].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 1,
                                             0])),
                                            ndim=3))

        select_voxels = T.inc_subtensor(select_voxels[:, :, 1:],
                                        select_edges_dir[2].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 0,
                                             1])),
                                            ndim=3))
        select_voxels = T.inc_subtensor(select_voxels[:, :, :-1],
                                        select_edges_dir[2].reshape((
                                                self.regular_grid_res - np.array(
                                            [0, 0,
                                             1])),
                                            ndim=3))

        return select_voxels

    # region Geometry
    def repeat_list(self, val, r_0, r_1, repeated_array, n_col):
        """
        Repeat an array

        Args:
            val: element or list that you want to repeat
            r_0: initial slicing position on the final array
            r_1: final slicing position on the final array
            repeated_array: final array

        Returns:
            final array
        """
        repeated_array = T.set_subtensor(repeated_array[r_0: r_1],
                                         T.alloc(val, r_1 - r_0, n_col))
        return repeated_array

    def set_rest_ref_matrix(self, number_of_points_per_surface):
        ref_positions = T.cumsum(
            T.concatenate((T.stack([0]), number_of_points_per_surface[:-1] + 1)))
        cum_rep = T.cumsum(
            T.concatenate((T.stack([0]), number_of_points_per_surface)))

        ref_points_init = T.zeros((cum_rep[-1], 3))
        ref_points_loop, update_ = theano.scan(self.repeat_list,
                                               outputs_info=[ref_points_init],
                                               sequences=[self.surface_points_all[
                                                              ref_positions],
                                                          dict(input=cum_rep,
                                                               taps=[0, 1])],
                                               non_sequences=[T.as_tensor(3)],

                                               return_list=False)

        #   ref_points_loop = theano.printing.Print('loop')(ref_points_loop)
        ref_points = ref_points_loop[-1]
        #  ref_points = T.repeat(self.surface_points_all[ref_positions], number_of_points_per_surface, axis=0)

        rest_mask = T.ones(T.stack([self.surface_points_all.shape[0]]),
                           dtype='int16')
        rest_mask = T.set_subtensor(rest_mask[ref_positions], 0)
        rest_mask = T.nonzero(rest_mask)[0]
        rest_points = self.surface_points_all[rest_mask]
        return [ref_points, rest_points, ref_positions, rest_mask]

    def set_nugget_surface_points(self, ref_positions, rest_mask,
                                  number_of_points_per_surface):
        # ref_nugget = T.repeat(self.nugget_effect_scalar_T[ref_positions], number_of_points_per_surface)
        cum_rep = T.cumsum(
            T.concatenate((T.stack([0]), number_of_points_per_surface)))
        ref_nugget_init = T.zeros((cum_rep[-1], 1))
        ref_nugget_loop, update_ = theano.scan(self.repeat_list,
                                               outputs_info=[ref_nugget_init],
                                               sequences=[
                                                   self.nugget_effect_scalar_T[
                                                       ref_positions],
                                                   dict(input=cum_rep, taps=[0, 1])],
                                               non_sequences=[T.as_tensor(1)],
                                               return_list=False)

        # ref_nugget_loop = theano.printing.Print('loop')(ref_nugget_loop)
        ref_nugget = ref_nugget_loop[-1]

        rest_nugget = self.nugget_effect_scalar_T[rest_mask]
        nugget_rest_ref = ref_nugget.reshape((1, -1))[0] + rest_nugget
        return nugget_rest_ref

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
        length_of_faults = self.lenght_of_faults  # T.cast(self.fault_matrix.shape[0], 'int32')
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_faults = theano.printing.Print("length_of_faults")(
                length_of_faults)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    # endregion

    # region Kriging
    def cov_surface_points(self):
        """
        Create covariance function for the surface_points

        Returns:
            theano.tensor.matrix: covariance of the surface_points. Shape number of points in rest x number of
            points in rest

        """

        # Compute euclidian distances
        sed_rest_rest = self.squared_euclidean_distances(self.rest_layer_points,
                                                         self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distances(self.ref_layer_points,
                                                        self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distances(self.rest_layer_points,
                                                        self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distances(self.ref_layer_points,
                                                       self.ref_layer_points)

        # Covariance matrix for surface_points
        C_I = (self.c_o_T_scalar * self.i_reescale * (
                (sed_rest_rest < self.a_T_scalar) *  # Rest - Rest Covariances Matrix
                (1 - 7 * (sed_rest_rest / self.a_T_scalar) ** 2 +
                 35 / 4 * (sed_rest_rest / self.a_T_scalar) ** 3 -
                 7 / 2 * (sed_rest_rest / self.a_T_scalar) ** 5 +
                 3 / 4 * (sed_rest_rest / self.a_T_scalar) ** 7) -
                ((sed_ref_rest < self.a_T_scalar) *  # Reference - Rest
                 (1 - 7 * (sed_ref_rest / self.a_T_scalar) ** 2 +
                  35 / 4 * (sed_ref_rest / self.a_T_scalar) ** 3 -
                  7 / 2 * (sed_ref_rest / self.a_T_scalar) ** 5 +
                  3 / 4 * (sed_ref_rest / self.a_T_scalar) ** 7)) -
                ((sed_rest_ref < self.a_T_scalar) *  # Rest - Reference
                 (1 - 7 * (sed_rest_ref / self.a_T_scalar) ** 2 +
                  35 / 4 * (sed_rest_ref / self.a_T_scalar) ** 3 -
                  7 / 2 * (sed_rest_ref / self.a_T_scalar) ** 5 +
                  3 / 4 * (sed_rest_ref / self.a_T_scalar) ** 7)) +
                ((sed_ref_ref < self.a_T_scalar) *  # Reference - References
                 (1 - 7 * (sed_ref_ref / self.a_T_scalar) ** 2 +
                  35 / 4 * (sed_ref_ref / self.a_T_scalar) ** 3 -
                  7 / 2 * (sed_ref_ref / self.a_T_scalar) ** 5 +
                  3 / 4 * (sed_ref_ref / self.a_T_scalar) ** 7))))

        # self.nugget_effect_scalar_T_op = theano.printing.Print('nug scalar')(self.nugget_effect_scalar_T_op)

        C_I += T.eye(C_I.shape[0]) * self.nugget_effect_scalar_T_op
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
        sed_dips_dips = self.squared_euclidean_distances(self.dips_position_tiled,
                                                         self.dips_position_tiled)

        if 'sed_dips_dips' in self.verbose:
            sed_dips_dips = theano.printing.Print('sed_dips_dips')(sed_dips_dips)

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(self.dips_position[:, 0] - self.dips_position[:, 0].reshape(
                (self.dips_position[:, 0].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 1] - self.dips_position[:, 1].reshape(
                (self.dips_position[:, 1].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 2] - self.dips_position[:, 2].reshape(
                (self.dips_position[:, 2].shape[0], 1)),
                   self.n_dimensions))

        # Transpose
        h_v = h_u.T

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance (block diagonal)
        perpendicularity_matrix = T.zeros_like(sed_dips_dips)

        # Cross-covariances of x
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:self.dips_position.shape[0],
            0:self.dips_position.shape[0]], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[
            self.dips_position.shape[0]:self.dips_position.shape[0] * 2,
            self.dips_position.shape[0]:self.dips_position.shape[0] * 2], 1)

        # Cross-covariances of z
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[
            self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3,
            self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3], 1)

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(sed_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                    (h_u * h_v / sed_dips_dips ** 2) *
                    ((
                             (sed_dips_dips < self.a_T_scalar) *  # first derivative
                             (-self.c_o_T_scalar * ((
                                                            -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_dips / self.a_T_scalar ** 3 -
                                                    35 / 2 * sed_dips_dips ** 3 / self.a_T_scalar ** 5 +
                                                    21 / 4 * sed_dips_dips ** 5 / self.a_T_scalar ** 7))) +
                     (sed_dips_dips < self.a_T_scalar) *  # Second derivative
                     self.c_o_T_scalar * 7 * (
                             9 * sed_dips_dips ** 5 - 20 * self.a_T_scalar ** 2 * sed_dips_dips ** 3 +
                             15 * self.a_T_scalar ** 4 * sed_dips_dips - 4 * self.a_T_scalar ** 5) / (
                             2 * self.a_T_scalar ** 7)) -
                    (perpendicularity_matrix *
                     (sed_dips_dips < self.a_T_scalar) *  # first derivative
                     self.c_o_T_scalar * ((
                                                  -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_dips / self.a_T_scalar ** 3 -
                                          35 / 2 * sed_dips_dips ** 3 / self.a_T_scalar ** 5 +
                                          21 / 4 * sed_dips_dips ** 5 / self.a_T_scalar ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag if I remove the condition
        C_G += T.eye(C_G.shape[0]) * self.nugget_effect_grad_T_op

        # Add name to the theano node
        C_G.name = 'Covariance Gradient'

        if verbose > 1:
            theano.printing.pydotprint(C_G,
                                       outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
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
        sed_dips_rest = self.squared_euclidean_distances(self.dips_position_tiled,
                                                         self.rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distances(self.dips_position_tiled,
                                                        self.ref_layer_points)

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
                 (sed_dips_rest < self.a_T_scalar) *  # first derivative
                 (- self.c_o_T_scalar * ((
                                                 -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_rest / self.a_T_scalar ** 3 -
                                         35 / 2 * sed_dips_rest ** 3 / self.a_T_scalar ** 5 +
                                         21 / 4 * sed_dips_rest ** 5 / self.a_T_scalar ** 7))) -
                (hu_ref *
                 (sed_dips_ref < self.a_T_scalar) *  # first derivative
                 (- self.c_o_T_scalar * ((
                                                 -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_ref / self.a_T_scalar ** 3 -
                                         35 / 2 * sed_dips_ref ** 3 / self.a_T_scalar ** 5 +
                                         21 / 4 * sed_dips_ref ** 5 / self.a_T_scalar ** 7)))
        ).T

        # Add name to the theano node
        C_GI.name = 'Covariance gradient interface'

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
            theano.printing.pydotprint(C_GI,
                                       outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
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
        U_G = T.set_subtensor(U_G[:n, 3],
                              2 * self.gi_reescale * self.dips_position[:, 0])
        # y**2
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 4],
                              2 * self.gi_reescale * self.dips_position[:, 1])
        # z**2
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 5],
                              2 * self.gi_reescale * self.dips_position[:, 2])
        # xy
        U_G = T.set_subtensor(U_G[:n, 6], self.gi_reescale * self.dips_position[:,
                                                             1])  # This is y
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 6],
                              self.gi_reescale * self.dips_position[:,
                                                 0])  # This is x
        # xz
        U_G = T.set_subtensor(U_G[:n, 7], self.gi_reescale * self.dips_position[:,
                                                             2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 7],
                              self.gi_reescale * self.dips_position[:,
                                                 0])  # This is x
        # yz
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 8],
                              self.gi_reescale * self.dips_position[:,
                                                 2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2:n * 3, 8],
                              self.gi_reescale * self.dips_position[:,
                                                 1])  # This is y

        # Interface
        U_I = - T.stack(
            (self.gi_reescale * (
                    self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
             self.gi_reescale * (
                     self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
             self.gi_reescale * (
                     self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] ** 2 - self.ref_layer_points[:,
                                                         0] ** 2),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 1] ** 2 - self.ref_layer_points[:,
                                                         1] ** 2),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 2] ** 2 - self.ref_layer_points[:,
                                                         2] ** 2),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:,
                                                    1] - self.ref_layer_points[:,
                                                         0] *
                     self.ref_layer_points[:, 1]),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:,
                                                    2] - self.ref_layer_points[:,
                                                         0] *
                     self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:,
                                                    2] - self.ref_layer_points[:,
                                                         1] *
                     self.ref_layer_points[:, 2]),
             )).T

        if 'U_I' in self.verbose:
            U_I = theano.printing.Print('U_I')(U_I)

        if 'U_G' in self.verbose:
            U_G = theano.printing.Print('U_G')(U_G)

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
            theano.printing.pydotprint(U_I,
                                       outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)

            theano.printing.pydotprint(U_G,
                                       outfile="graphs/" + sys._getframe().f_code.co_name + "_g.png",
                                       var_with_name_simple=True)

        # Add name to the theano node
        if U_I:
            U_I.name = 'Drift surface_points'
            U_G.name = 'Drift foliations'

        return U_I[:, :self.n_universal_eq_T_op], U_G[:, :self.n_universal_eq_T_op]

    def faults_matrix(self, f_ref=None, f_res=None):
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

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults = self.matrices_shapes()[
                                                                       :4]

        # self.fault_drift contains the df volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
        if 'fault matrix' in self.verbose:
            self.fault_matrix = theano.printing.Print('self.fault_drift')(
                self.fault_matrix)
        # interface_loc = self.fault_drift.shape[1] - 2 * self.len_points
        #
        # fault_drift_at_surface_points_rest = self.fault_drift
        # fault_drift_at_surface_points_ref = self.fault_drift

        F_I = (
                      self.fault_drift_at_surface_points_ref - self.fault_drift_at_surface_points_rest) + 0.0001

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
        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI],
            C_GI.T)
        # Set UG
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG,
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I],
                                   U_G)
        # Set FG. I cannot use -index because when is -0 is equivalent to 0
        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG, length_of_CG + length_of_CGI + length_of_U_I:],
            F_G.T)
        # Second row of matrices
        # Set C_IG
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG],
            C_GI)
        # Set C_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI,
            length_of_CG:length_of_CG + length_of_CGI], C_I)
        # Set U_I
        # if not self.u_grade_T.get_value() == 0:
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI,
            length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I],
            U_I)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI,
            length_of_CG + length_of_CGI + length_of_U_I:], F_I.T)
        # Third row of matrices
        # Set U_G
        C_matrix = T.set_subtensor(
            C_matrix[
            length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I,
            0:length_of_CG], U_G.T)
        # Set U_I
        C_matrix = T.set_subtensor(C_matrix[
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I,
                                   length_of_CG:length_of_CG + length_of_CGI], U_I.T)
        # Fourth row of matrices
        # Set F_G
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI + length_of_U_I:, 0:length_of_CG],
            F_G)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI + length_of_U_I:,
            length_of_CG:length_of_CG + length_of_CGI], F_I)
        # Add name to the theano node
        C_matrix.name = 'Block Covariance Matrix'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_matrix = theano.printing.Print('cov_function')(C_matrix)

        return C_matrix

    def b_vector(self, dip_angles=None, azimuth=None, polarity=None):
        """
        Creation of the independent vector b to solve the kriging system

        Args:
            verbose: -deprecated-

        Returns:
            theano.tensor.vector: independent vector
        """

        length_of_C = self.matrices_shapes()[-1]
        if dip_angles is None:
            dip_angles = self.dip_angles
        if azimuth is None:
            azimuth = self.azimuth
        if polarity is None:
            polarity = self.polarity

        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(dip_angles)) * T.sin(T.deg2rad(azimuth)) * polarity
        G_y = T.sin(T.deg2rad(dip_angles)) * T.cos(T.deg2rad(azimuth)) * polarity
        G_z = T.cos(T.deg2rad(dip_angles)) * polarity

        G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros((length_of_C,))
        b = T.set_subtensor(b[0:G.shape[0]], G)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            b = theano.printing.Print('b vector')(b)

        # Add name to the theano node
        b.name = 'b vector'
        return b

    def solve_kriging(self, b=None):
        """
        Solve the kriging system. This has to get substituted by a more efficient and stable method QR
        decomposition in all likelihood

        Returns:
            theano.tensor.vector: Dual kriging parameters

        """
        C_matrix = self.covariance_matrix()
        if b is None:
            b = self.b_vector()
            # b = theano.printing.Print('b')(b)
        if self.sparse_version is True:
            b2 = T.tile(b, (1, 1)).T

            C_sparse = sparse.csr_from_dense(C_matrix)
            b_sparse = sparse.csr_from_dense(b2)
            DK = solv(C_sparse, b_sparse)
            return DK

        # Solving the kriging system
        elif self.device == 'cuda' and SKCUDA_IMPORT is True:
            import theano.gpuarray.linalg
            b2 = T.tile(b, (1, 1)).T
            DK_parameters = theano.gpuarray.linalg.gpu_solve(C_matrix, b2)

        else:
            import theano.tensor.slinalg

            DK_parameters = theano.tensor.slinalg.solve(C_matrix, b)

        DK_parameters = DK_parameters.reshape((DK_parameters.shape[0],))

        # Add name to the theano node
        DK_parameters.name = 'Dual Kriging parameters'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            DK_parameters = theano.printing.Print(DK_parameters.name)(DK_parameters)
        return DK_parameters

    # endregion

    # region Evaluate
    def x_to_interpolate(self, grid, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        surface_points). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.

        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """

        grid_val = T.concatenate([grid, self.rest_layer_points_all,
                                  self.ref_layer_points_all])

        if verbose > 1:
            theano.printing.pydotprint(grid_val,
                                       outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if 'grid_val' in self.verbose:
            grid_val = theano.printing.Print('Points to interpolate')(grid_val)

        return grid_val

    def extend_dual_kriging(self, weights, grid_shape):
        # TODO Think what object is worth to save to speed up computation
        """
        Tile the dual kriging vector to cover all the points to interpolate.So far I just make a matrix with the
        dimensions len(DK)x(grid) but in the future maybe I have to try to loop all this part so consume less memory

        Returns:
            theano.tensor.matrix: Matrix with the Dk parameters repeated for all the points to interpolate
        """
        DK_parameters = weights
        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        # TODO IMP: Change the tile by a simple dot op -> The DOT version in gpu is slower
        DK_weights = T.tile(DK_parameters, (grid_shape, 1)).T

        return DK_weights

    # endregion

    # region Evaluate Geology
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
            (self.dips_position[:, 0] - grid_val[:, 0].reshape(
                (grid_val[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - grid_val[:, 1].reshape(
                (grid_val[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - grid_val[:, 2].reshape(
                (grid_val[:, 2].shape[0], 1))).T
        )

        # Euclidian distances
        sed_dips_SimPoint = self.squared_euclidean_distances(
            self.dips_position_tiled, grid_val)

        if self.sparse_version is True:
            cov_aux = sparse.csr_from_dense(
                self.gi_reescale *
                (-hu_SimPoint *
                 (sed_dips_SimPoint < self.a_T_scalar) *  # first derivative
                 (- self.c_o_T_scalar * ((
                                                 -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T_scalar ** 3 -
                                         35 / 2 * sed_dips_SimPoint ** 3 / self.a_T_scalar ** 5 +
                                         21 / 4 * sed_dips_SimPoint ** 5 / self.a_T_scalar ** 7))))

            sliced_weights = weights[
                             0:length_of_CG]  # T.stack([weights[0, 0:length_of_CG]])#weights[0:length_of_CG]
            sigma_0_grad = sparse.dot(sliced_weights, cov_aux)

        else:
            # Gradient contribution
            sigma_0_grad = T.sum(
                (weights[:length_of_CG] *
                 self.gi_reescale *
                 (-hu_SimPoint *
                  (sed_dips_SimPoint < self.a_T_scalar) *  # first derivative
                  (- self.c_o_T_scalar * ((
                                                  -14 / self.a_T_scalar ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T_scalar ** 3 -
                                          35 / 2 * sed_dips_SimPoint ** 3 / self.a_T_scalar ** 5 +
                                          21 / 4 * sed_dips_SimPoint ** 5 / self.a_T_scalar ** 7)))),
                axis=0)

        # Add name to the theano node
        sigma_0_grad.name = 'Contribution of the foliations to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            sigma_0_grad = theano.printing.Print('interface_gradient_contribution')(
                sigma_0_grad)

        return sigma_0_grad

    def contribution_interface(self, grid_val, weights=None):
        """
          Computation of the contribution of the surface_points at every point to interpolate

          Returns:
              theano.tensor.vector: Contribution of all surface_points (input) at every point to interpolate
          """

        if weights is None:
            weights = self.compute_weights()

        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Euclidian distances
        sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points,
                                                             grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points,
                                                            grid_val)

        if self.sparse_version is True:
            cov_aux = sparse.csr_from_dense(self.c_o_T_scalar * self.i_reescale * (
                    (
                            sed_rest_SimPoint < self.a_T_scalar) *  # SimPoint - Rest Covariances Matrix
                    (1 - 7 * (sed_rest_SimPoint / self.a_T_scalar) ** 2 +
                     35 / 4 * (sed_rest_SimPoint / self.a_T_scalar) ** 3 -
                     7 / 2 * (sed_rest_SimPoint / self.a_T_scalar) ** 5 +
                     3 / 4 * (sed_rest_SimPoint / self.a_T_scalar) ** 7) -
                    ((sed_ref_SimPoint < self.a_T_scalar) *  # SimPoint- Ref
                     (1 - 7 * (sed_ref_SimPoint / self.a_T_scalar) ** 2 +
                      35 / 4 * (sed_ref_SimPoint / self.a_T_scalar) ** 3 -
                      7 / 2 * (sed_ref_SimPoint / self.a_T_scalar) ** 5 +
                      3 / 4 * (sed_ref_SimPoint / self.a_T_scalar) ** 7))))

            weights_sliced = -weights[length_of_CG:length_of_CG + length_of_CGI]

            sigma_0_interf = sparse.dot(weights_sliced, cov_aux)

        else:
            # Interface contribution
            sigma_0_interf = (T.sum(
                -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
                (self.c_o_T_scalar * self.i_reescale * (
                        (
                                sed_rest_SimPoint < self.a_T_scalar) *  # SimPoint - Rest Covariances Matrix
                        (1 - 7 * (sed_rest_SimPoint / self.a_T_scalar) ** 2 +
                         35 / 4 * (sed_rest_SimPoint / self.a_T_scalar) ** 3 -
                         7 / 2 * (sed_rest_SimPoint / self.a_T_scalar) ** 5 +
                         3 / 4 * (sed_rest_SimPoint / self.a_T_scalar) ** 7) -
                        ((sed_ref_SimPoint < self.a_T_scalar) *  # SimPoint- Ref
                         (1 - 7 * (sed_ref_SimPoint / self.a_T_scalar) ** 2 +
                          35 / 4 * (sed_ref_SimPoint / self.a_T_scalar) ** 3 -
                          7 / 2 * (sed_ref_SimPoint / self.a_T_scalar) ** 5 +
                          3 / 4 * (sed_ref_SimPoint / self.a_T_scalar) ** 7)))),
                axis=0))
        # Add name to the theano node
        sigma_0_interf.name = 'Contribution of the surface_points to the potential field at every point of the grid'

        return sigma_0_interf

    def contribution_universal_drift(self, grid_val, weights=None):
        """
        Computation of the contribution of the universal drift at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of the universal drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.compute_weights()

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        universal_grid_surface_points_matrix = T.horizontal_stack(
            grid_val,
            (grid_val ** 2),
            T.stack((grid_val[:, 0] * grid_val[:, 1],
                     grid_val[:, 0] * grid_val[:, 2],
                     grid_val[:, 1] * grid_val[:, 2]), axis=1)).T

        i_rescale_aux = T.tile(self.gi_reescale, 9)
        i_rescale_aux = T.set_subtensor(i_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(i_rescale_aux[:self.n_universal_eq_T_op],
                                 (grid_val.shape[0], 1)).T

        if self.sparse_version is True:  # self.dot_version:
            f_0 = T.dot(
                weights[
                length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I],
                (self.gi_reescale * _aux_magic_term *
                 universal_grid_surface_points_matrix[:self.n_universal_eq_T_op]))
        else:
            # Drif contribution
            f_0 = (T.sum(
                weights[
                length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_reescale *
                _aux_magic_term *
                universal_grid_surface_points_matrix[:self.n_universal_eq_T_op]
                , axis=0))

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

        return f_0

    def contribution_faults(self, weights=None, a=0, b=100000000, f_m=None):
        """
        Computation of the contribution of the df drift at every point to interpolate. To get these we need to
        compute a whole block model with the df data

        Returns:
            theano.tensor.vector: Contribution of the df drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.compute_weights()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        fault_matrix_selection_non_zero = f_m[:, a:b]

        if self.sparse_version is True:  # self.dot_version:
            f_1 = T.dot(
                weights[length_of_CG + length_of_CGI + length_of_U_I:], (
                    fault_matrix_selection_non_zero))
        else:
            f_1 = T.sum(
                weights[length_of_CG + length_of_CGI + length_of_U_I:,
                :] * fault_matrix_selection_non_zero, axis=0)

        # Add name to the theano node
        f_1.name = 'Faults contribution'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_1 = theano.printing.Print('Faults contribution')(f_1)

        return f_1

    def scalar_field_loop(self, a, b, Z_x, grid_val, weights, fault_matrix):

        if self.sparse_version is True:
            rang = 5
            tiled_weights = self.extend_dual_kriging(weights, rang)
            sigma_0_grad = self.contribution_gradient_interface(grid_val[a:b],
                                                                tiled_weights)
            sigma_0_interf = self.contribution_interface(grid_val[a:b],
                                                         tiled_weights)
            f_0 = self.contribution_universal_drift(grid_val[a:b], tiled_weights)
            f_1 = self.contribution_faults(tiled_weights, a, b)

            # Add an arbitrary number at the potential field to get unique values for each of them
            partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1)[0]

        else:
            rang = b - a
            tiled_weights = self.extend_dual_kriging(weights, rang)
            sigma_0_grad = self.contribution_gradient_interface(grid_val[a:b],
                                                                tiled_weights[:, :])
            sigma_0_interf = self.contribution_interface(grid_val[a:b],
                                                         tiled_weights[:, :])
            f_0 = self.contribution_universal_drift(grid_val[a:b],
                                                    tiled_weights[:, :])
            f_1 = self.contribution_faults(tiled_weights[:, :], a, b, fault_matrix)

            # Add an arbitrary number at the potential field to get unique values for each of them
            partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1)

        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def scalar_field_at_all(self, weights, grid_val, fault_matrix):
        """
        Compute the potential field at all the interpolation points, i.e. grid plus rest plus ref
        Returns:
            theano.tensor.vector: Potential fields at all points

        """
        #

        grid_shape = T.stack([grid_val.shape[0]], axis=0)
        Z_x_init = T.zeros(grid_shape)
        if 'grid_shape' in self.verbose:
            grid_shape = theano.printing.Print('grid_shape')(grid_shape)

        # If memory errors reduce this to 11
        steps = 5e6 / self.matrices_shapes()[-1]  # / grid_shape
        if 'steps' in self.verbose:
            steps = theano.printing.Print('steps')(steps)

        slices = T.concatenate(
            (T.arange(0, grid_shape[0], steps, dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        # Check if we loop the grid or not
        if self.sparse_version is True:
            self.dot_version = True
            Z_x = self.scalar_field_loop(0, 100000000, Z_x_init, grid_val, weights)

        elif self.max_speed < 2:
            # tiled_weights = self.extend_dual_kriging(weights, grid_val.shape[0])
            Z_x_loop, updates3 = theano.scan(
                fn=self.scalar_field_loop,
                outputs_info=[Z_x_init],
                sequences=[dict(input=slices, taps=[0, 1])],
                non_sequences=[grid_val, weights, fault_matrix],
                profile=False,
                name='Looping grid',
                return_list=True)

            Z_x = Z_x_loop[-1][-1]
        else:
            tiled_weights = self.extend_dual_kriging(weights, grid_val.shape[0])
            Z_x = self.scalar_field_loop(0, 100000000, Z_x_init, grid_val,
                                         tiled_weights)

        Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)

        return Z_x

    def get_scalar_field_at_surface_points(self, Z_x, npf_op=None):
        if npf_op is None:
            npf_op = self.npf_op

        if 'len_pints' in self.verbose:
            self.len_points = theano.printing.Print('len_points')(self.len_points)
        if 'npf_op' in self.verbose:
            npf_op = theano.printing.Print('npf_op')(npf_op)
        scalar_field_at_surface_points_values = \
            Z_x[-2 * self.len_points: -self.len_points][npf_op]
        if 'sfai' in self.verbose:
            scalar_field_at_surface_points_values = theano.printing.Print('sfai')(
                scalar_field_at_surface_points_values)

        return scalar_field_at_surface_points_values

    # endregion

    # region Block export
    def select_finite_faults(self, grid):
        fault_points = T.vertical_stack(T.stack([self.ref_layer_points[0]], axis=0),
                                        self.rest_layer_points).T
        ctr = T.mean(fault_points, axis=1)
        x = fault_points - ctr.reshape((-1, 1))
        M = T.dot(x, x.T)
        U, D, V = T.nlinalg.svd(M)
        rotated_x = T.dot(T.dot(grid, U), V)
        rotated_fault_points = T.dot(T.dot(fault_points.T, U), V)
        rotated_ctr = T.mean(rotated_fault_points, axis=0)
        a_radius = (rotated_fault_points[:, 0].max() - rotated_fault_points[:,
                                                       0].min()) / 2
        b_radius = (rotated_fault_points[:, 1].max() - rotated_fault_points[:,
                                                       1].min()) / 2

        ellipse_factor = (rotated_x[:, 0] - rotated_ctr[0]) ** 2 / a_radius ** 2 + \
                         (rotated_x[:, 1] - rotated_ctr[1]) ** 2 / b_radius ** 2

        if "select_finite_faults" in self.verbose:
            ellipse_factor = theano.printing.Print("h")(ellipse_factor)

        return ellipse_factor

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

        # The 5 rules the slope of the function
        sigm = (-n_surface_0.reshape((-1, 1)) / (1 + T.exp(-l * (Z_x - a)))) - \
               (n_surface_1.reshape((-1, 1)) / (
                       1 + T.exp(l * (Z_x - b)))) + drift.reshape((-1, 1))
        if 'sigma' in self.verbose:
            sigm = theano.printing.Print("middle point")(sigm)
        return sigm

    def export_fault_block(self, Z_x, scalar_field_at_surface_points,
                           values_properties_op, finite_faults_sel,
                           slope=50, offset_slope=950):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        # if Z_x is None:
        #     Z_x = self.Z_x

        # Max and min values of the potential field.
        # max_pot = T.max(Z_x) + 1
        # min_pot = T.min(Z_x) - 1
        # max_pot += max_pot * 0.1
        # min_pot -= min_pot * 0.1

        # Value of the potential field at the surface_points of the computed series
        # TODO timeit. I think the
        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        #     min_pot = theano.printing.Print("min_pot")(min_pot)

        # max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        # min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        # l = slope / (max_pot - min_pot)  # (max_pot - min_pot)

        # This is the different line with respect layers
        # l = T.switch(finite_faults_sel, offset_slope / (max_pot - min_pot), slope / (max_pot - min_pot))
        #  l = theano.printing.Print("l")(l)

        # -------------------------------
        # Alex Schaaf contribution:
        # ellipse_factor = self.select_finite_faults()
        ellipse_factor_rectified = T.switch(finite_faults_sel < 1.,
                                            finite_faults_sel, 1.)

        if "select_finite_faults" in self.verbose:
            ellipse_factor_rectified = theano.printing.Print("h_factor_rectified")(
                ellipse_factor_rectified)

        if "select_finite_faults" in self.verbose:
            min_pot = theano.printing.Print("min_pot")(min_pot)
            max_pot = theano.printing.Print("max_pot")(max_pot)

        # sigmoid_slope = (self.not_l * (1 / ellipse_factor_rectified)**3) / (max_pot - min_pot)
        sigmoid_slope = offset_slope - offset_slope * ellipse_factor_rectified ** self.ellipse_factor_exponent + self.not_l
        # l = T.switch(self.select_finite_faults(), 5000 / (max_pot - min_pot), 50 / (max_pot - min_pot))

        if "select_finite_faults" in self.verbose:
            sigmoid_slope = theano.printing.Print("l")(sigmoid_slope)
        # --------------------------------
        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad], axis=0),
            scalar_field_at_surface_points,
            T.stack([min_pot - boundary_pad], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(
                scalar_field_iter)

        # Here we just take the first element of values properties because at least so far we do not find a reason
        # to populate fault blocks with anything else

        n_surface_op_float_sigmoid = T.repeat(values_properties_op[[0], :], 2,
                                              axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!
        # TODO I think should be -> n_surface_op_float_sigmoid[:, 2] - n_surface_op_float_sigmoid[:, 1]
        n_surface_op_float_sigmoid = T.set_subtensor(
            n_surface_op_float_sigmoid[:, 1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(
            n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0],
                                n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print(
                "n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        fault_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]),
                       T.arange(0, n_surface_op_float_sigmoid.shape[1], 2,
                                dtype='int64')],
            non_sequences=[Z_x, sigmoid_slope, n_surface_op_float_sigmoid, drift],
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

    def export_formation_block(self, Z_x, scalar_field_at_surface_points,
                               values_properties_op):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """
        # TODO: IMP set soft max in the borders: Test
        # TODO: instead -1 at the border look for the average distance of the input!: Test

        slope = self.sig_slope
        if self.max_speed < 1:
            # max_pot = T.max(scalar_field_at_surface_points)
            # min_pot = T.min(scalar_field_at_surface_points)

            max_pot = T.max(Z_x)
            # max_pot = theano.printing.Print("max_pot")(max_pot)
            min_pot = T.min(Z_x)
            # min_pot = theano.printing.Print("min_pot")(min_pot)

            # max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
            # min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

            # boundary_pad = (max_pot - min_pot) * 0.01
            l = slope / (max_pot - min_pot)
        else:
            l = slope

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            #  T.stack([T.max(Z_x)], axis=0),
            T.stack([0], axis=0),  # somehow this also works. I do not remember why
            scalar_field_at_surface_points,
            T.stack([0], axis=0)
            #   T.stack([T.min(Z_x)], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(
                scalar_field_iter)

        # Loop to segment the distinct lithologies

        n_surface_op_float_sigmoid = T.repeat(values_properties_op, 2, axis=1)
        n_surface_op_float_sigmoid = T.set_subtensor(
            n_surface_op_float_sigmoid[:, 0], 0)
        n_surface_op_float_sigmoid = T.set_subtensor(
            n_surface_op_float_sigmoid[:, -1], 0)
        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0],
                                n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print(
                "n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        formations_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]),
                       T.arange(0, n_surface_op_float_sigmoid.shape[1], 2,
                                dtype='int64')],
            non_sequences=[Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        formations_block = formations_block.sum(axis=0)

        if self.gradient is True:
            ReLU_up = T.switch(Z_x < scalar_field_iter[1], 0,
                               - 0.01 * (Z_x - scalar_field_iter[1]))
            ReLU_down = T.switch(Z_x > scalar_field_iter[-2], 0,
                                 0.01 * T.abs_(Z_x - scalar_field_iter[-2]))

            if 'relu' in self.verbose:
                ReLU_up = theano.printing.Print('ReLU_up')(ReLU_up)
                ReLU_down = theano.printing.Print('ReLU_down')(ReLU_down)

            formations_block += ReLU_down + ReLU_up

        # Add name to the theano node
        formations_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            formations_block = theano.printing.Print(formations_block.name)(
                formations_block)

        return formations_block

    # endregion

    # region Compute model
    def compute_a_series(self,
                         len_i_0=0, len_i_1=None,
                         len_f_0=0, len_f_1=None,
                         len_w_0=0, len_w_1=None,
                         n_form_per_serie_0=0, n_form_per_serie_1=None,
                         u_grade_iter=3,
                         compute_weight_ctr=np.array(True),
                         compute_scalar_ctr=np.array(True),
                         compute_block_ctr=np.array(True),
                         is_finite=np.array(False), is_erosion=np.array(True),
                         is_onlap=np.array(False),
                         n_series=0,
                         range=10., c_o=10.,
                         block_matrix=None, weights_vector=None,
                         scalar_field_matrix=None, sfai=None, mask_matrix=None,
                         mask_matrix_f=None, fault_matrix=None, nsle=0, grid=None,
                         shift=None
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
        self.a_T_scalar = range
        self.c_o_T_scalar = c_o

        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T[
                                                 n_form_per_serie_0: n_form_per_serie_1]

        self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]
        n_surface_op = self.n_surface[n_form_per_serie_0: n_form_per_serie_1]

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]

        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

        self.nugget_effect_scalar_T_op = self.nugget_effect_scalar_T_ref_rest[
                                         len_i_0: len_i_1]

        # The gradients have been tiled outside
        self.nugget_effect_grad_T_op = self.nugget_effect_grad_T[
                                       len_f_0 * 3: len_f_1 * 3]

        self.n_universal_eq_T_op = u_grade_iter

        x_to_interpolate_shape = grid.shape[0] + 2 * self.len_points
        if 'x_to_interpolate_shape' in self.verbose:
            x_to_interpolate_shape = theano.printing.Print('x_to_interpolate_shape')(
                x_to_interpolate_shape)

        # Extracting faults matrices
        faults_relation_op = self.fault_relation[:, T.cast(n_series, 'int8')]
        fault_matrix_op = fault_matrix[
                          T.nonzero(T.cast(faults_relation_op, "int8"))[0],
                          0, shift:x_to_interpolate_shape + shift] * self.offset

        self.lenght_of_faults = T.cast(fault_matrix_op.shape[0], 'int32')

        if 'fault_matrix_loop' in self.verbose:
            self.fault_matrix = theano.printing.Print('self fault matrix')(
                self.fault_matrix)
        # TODO this is wrong

        interface_loc = grid.shape[0]

        if 'len_i' in self.verbose:
            len_i_0 = theano.printing.Print('len_i_0')(len_i_0)
            len_i_1 = theano.printing.Print('len_i_1')(len_i_1)

        self.fault_drift_at_surface_points_rest = fault_matrix_op[
                                                  :, interface_loc + len_i_0:
                                                     interface_loc + len_i_1]

        self.fault_drift_at_surface_points_ref = fault_matrix_op[
                                                 :,
                                                 interface_loc + self.len_points + len_i_0:
                                                 interface_loc + self.len_points + len_i_1]

        b = self.b_vector(self.dip_angles, self.azimuth, self.polarity)

        if self.sparse_version is True:
            weights = self.solve_kriging(b)
            Z_x = self.compute_scalar_field(weights, grid)
            weights = weights[0]

        else:
            weights = theano.ifelse.ifelse(compute_weight_ctr,
                                           self.solve_kriging(b),
                                           weights_vector[len_w_0:len_w_1])

            Z_x = tif.ifelse(compute_scalar_ctr,
                             self.compute_scalar_field(weights, grid,
                                                       fault_matrix_op),
                             scalar_field_matrix[n_series])

        if 'weights' in self.verbose:
            weights = theano.printing.Print('weights foo')(weights)

        scalar_field_at_surface_points = self.get_scalar_field_at_surface_points(Z_x,
                                                                                 self.npf_op)

        if 'sfai' in self.verbose:
            scalar_field_at_surface_points = theano.printing.Print('sfai')(
                scalar_field_at_surface_points)

        # TODO: add control flow for this side
        mask_e = tif.ifelse(is_erosion,  # If is erosion
                            T.gt(Z_x, T.min(scalar_field_at_surface_points)),
                            # It is True the values over the last surface
                            ~ self.is_fault[n_series] * T.ones_like(Z_x,
                                                                    dtype='bool'))  # else: all False if is Fault else all ones

        if 'mask_e' in self.verbose:
            mask_e = theano.printing.Print('mask_e')(mask_e)

        # Number of series since last erode: This is necessary in case there are
        # multiple consecutives onlaps

        # Erosion version
        is_erosion_ = self.is_erosion[:n_series + 1]
        args_is_erosion = T.nonzero(T.concatenate(([1], is_erosion_)))
        last_erode = T.argmax(args_is_erosion[0])

        # Onlap version
        is_onlap_or_fault = self.is_onlap[n_series] + self.is_fault[n_series]

        # This adds a counter  --- check series onlap-fault --- check the chain starts with onlap
        nsle = (nsle + is_onlap_or_fault) * is_onlap_or_fault *\
               self.is_onlap[n_series - nsle]
        nsle_op = nsle  # T.max([nsle, 1])

        if 'nsle' in self.verbose:
            nsle_op = theano.printing.Print('nsle_op')(nsle_op)

        mask_o = tif.ifelse(is_onlap,
                            T.gt(Z_x, T.max(scalar_field_at_surface_points)),
                            mask_matrix[n_series - 1,
                            shift:x_to_interpolate_shape + shift])

        mask_f = tif.ifelse(self.is_fault[n_series],
                            T.gt(Z_x, T.min(scalar_field_at_surface_points)),
                            T.zeros_like(Z_x, dtype='bool'))

        if self.gradient is False:
            block = tif.ifelse(
                compute_block_ctr,
                tif.ifelse(
                    is_finite,
                    self.compute_fault_block(
                        Z_x, scalar_field_at_surface_points,
                        self.values_properties_op[:,
                        n_form_per_serie_0: n_form_per_serie_1 + 1],
                        n_series, grid
                    ),
                    self.compute_formation_block(
                        Z_x, scalar_field_at_surface_points,
                        self.values_properties_op[:,
                        n_form_per_serie_0: n_form_per_serie_1 + 1])
                ),
                block_matrix[n_series, :]
            )
        else:
            block = tif.ifelse(compute_block_ctr,
                               self.compute_formation_block(
                                   Z_x, scalar_field_at_surface_points,
                                   self.values_properties_op[:,
                                   n_form_per_serie_0: n_form_per_serie_1 + 1]),
                               block_matrix[n_series, :]
                               )

        weights_vector = T.set_subtensor(weights_vector[len_w_0:len_w_1], weights)
        scalar_field_matrix = T.set_subtensor(
            scalar_field_matrix[n_series, shift:x_to_interpolate_shape + shift], Z_x)
        block_matrix = T.set_subtensor(
            block_matrix[n_series, :, shift:x_to_interpolate_shape + shift], block)
        fault_matrix = T.set_subtensor(
            fault_matrix[n_series, :, shift:x_to_interpolate_shape + shift], block)

        # LITH MASK
        mask_matrix = T.set_subtensor(mask_matrix[n_series - 1: n_series,
                                      shift:x_to_interpolate_shape + shift], mask_o)

        mask_matrix = T.set_subtensor(mask_matrix[n_series - nsle_op: n_series,
                                      shift:x_to_interpolate_shape + shift],
                                      T.cumprod(
                                          mask_matrix[n_series - nsle_op: n_series,
                                          shift:x_to_interpolate_shape + shift][
                                          ::-1], axis=0)[::-1])

        mask_matrix = T.set_subtensor(
            mask_matrix[n_series, shift:x_to_interpolate_shape + shift], mask_e)

        if 'mask_matrix_loop' in self.verbose:
            mask_matrix = theano.printing.Print('mask_matrix_loop')(mask_matrix)

        # FAULT MASK
        # This creates a matrix with Trues in the positive side of the faults. When is not faults is 0

        # This select the indices where is fault but are not offsetting
        # TODO having a better way to control the number of series than is_erosion
        idx_e = (self.is_fault * ~T.cast(faults_relation_op, 'bool'))[
                :self.is_erosion.shape[0]]
        idx_o = (self.is_fault * ~T.cast(
            self.fault_relation[:, T.cast(n_series - 1, 'int8')], 'bool'))[
                :self.is_erosion.shape[0]]

        mask_matrix_f = T.set_subtensor(
            mask_matrix_f[idx_e, shift:x_to_interpolate_shape + shift],
            mask_e + mask_f)
        # mask_matrix_f = T.set_subtensor(mask_matrix_f[idx_o, :], mask_o + mask_matrix_f[n_series-1])

        # Scalar field at interfaces
        sfai = T.set_subtensor(sfai[n_series, n_surface_op - 1],
                               scalar_field_at_surface_points)

        return block_matrix, weights_vector, scalar_field_matrix, sfai, mask_matrix,\
               mask_matrix_f, fault_matrix, nsle

    def compute_forward_gravity(self, densities=None,
                                pos_density=None):  # densities, tz, select,

        assert pos_density is not None or densities is not None, 'If a density block is not passed, you need to' \
                                                                 ' specify which interpolated value is density.' \
                                                                 ' See :class:`Surface`'

        if densities is None:
            final_model, new_block, new_weights, new_scalar, new_sfai, new_mask = self.compute_series()
            densities = final_model[pos_density,
                        self.lg0:self.lg1]  # - 2 * self.len_points]
        else:
            final_model, new_block, new_weights, new_scalar, new_sfai, new_mask = None, None, None, None, None, None

        if 'densities' in self.verbose:
            densities = theano.printing.Print('density')(densities)

        n_devices = T.cast((densities.shape[0] / self.tz.shape[0]), dtype='int32')

        if 'grav_devices' in self.verbose:
            n_devices = theano.printing.Print('n_devices')(n_devices)

        tz_rep = T.tile(self.tz, n_devices)

        # density times the component z of gravity
        grav = (densities * tz_rep).reshape((n_devices, -1)).sum(axis=1)

        return final_model, new_block, new_weights, new_scalar, new_sfai, new_mask, grav  # , model_sol.append(grav)

    def compute_forward_gravity_pro(self, densities=None):  # densities, tz, select,

        if 'densities' in self.verbose:
            densities = theano.printing.Print('density')(densities)

        n_devices = T.cast((densities.shape[0] / self.tz.shape[0]), dtype='int32')

        if 'grav_devices' in self.verbose:
            n_devices = theano.printing.Print('n_devices')(n_devices)

        tz_rep = T.tile(self.tz, n_devices)

        # density times the component z of gravity
        grav = (densities * tz_rep).reshape((n_devices, -1)).sum(axis=1)

        return grav

    def compute_forward_magnetics(self, k_vals):
        """
        Compute magnetics

        Args:
            k_vals: Susceptibility values per voxel [-] - varies per device! GemPy

        Returns:

        """

        def magnetic_direction(incl, decl):
            incl_rad = incl * 3.14159265359 / 180.  # np.deg2rad(incl)
            decl_rad = decl * 3.14159265359 / 180.  # np.deg2rad(decl)
            x = T.cos(incl_rad) * T.cos(decl_rad)
            y = T.cos(incl_rad) * T.sin(decl_rad)
            z = T.sin(incl_rad)
            return x, y, z

        if 'magnetics' in self.verbose:
            k_vals = theano.printing.Print('Sus. values')(k_vals)

        # get induced magnetisation [T]
        J = k_vals * self.B_ext  # k_vals contains susceptibility values of each voxel for all devices: [k1dev1,..,kndevn]

        # and the components:
        dir_x, dir_y, dir_z = magnetic_direction(self.incl, self.decl)
        Jx = dir_x * J
        Jy = dir_y * J
        Jz = dir_z * J

        n_devices = T.cast((k_vals.shape[0] / self.V.shape[1]), dtype='int32')
        if 'mag_devices' in self.verbose:
            n_devices = theano.printing.Print('n_devices')(n_devices)

        V = T.tile(self.V, (1, n_devices))  # repeat for each device

        # directional magnetic effect on one voxel (3.19)
        Tx = (Jx * V[0, :] + Jy * V[1, :] + Jz * V[2, :]) / (4 * self.pi)
        Ty = (Jx * V[1, :] + Jy * V[3, :] + Jz * V[4, :]) / (4 * self.pi)
        Tz = (Jx * V[2, :] + Jy * V[4, :] + Jz * V[5, :]) / (4 * self.pi)

        T2nT = 1e9  # to get result in [nT] - common for geophysical applications

        Tx = (T.sum(Tx.reshape((n_devices, -1)), axis=1)) * T2nT
        Ty = (T.sum(Ty.reshape((n_devices, -1)), axis=1)) * T2nT
        Tz = (T.sum(Tz.reshape((n_devices, -1)), axis=1)) * T2nT

        # -„Total field magnetometers can measure only that part of the anomalous field which is in the direction of
        # the Earths main field“ (SimPEG documentation)'
        dT = Tx * dir_x + Ty * dir_y + Tz * dir_z
        return dT  # , Tx, Ty, Tz
