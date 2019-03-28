import numpy as np
import theano


class Interpolator_pro(object):
    """
    Class that act as:
     1) linker between the data objects and the theano graph
     2) container of theano graphs + shared variables
     3) container of theano function

     Attributes:
        surface_points (SurfacePoints)
        orientaions (Orientations)
        grid (GridClass)
        surfaces (Surfaces)
        faults (Faults)
        additional_data (AdditionalData)
        dtype (['float32', 'float64']): float precision
        input_matrices (list[arrays])
            - dip positions XYZ
            - dip angles
            - azimuth
            - polarity
            - surface_points coordinates XYZ

        theano_graph: theano graph object with the properties from AdditionalData -> Options
        theano function: python function to call the theano code

    Args:
        surface_points (SurfacePoints)
        orientaions (Orientations)
        grid (GridClass)
        surfaces (Surfaces)
        faults (Faults)
        additional_data (AdditionalData)
        kwargs:
            - compile_theano: if true, the function is compile at the creation of the class
    """
    # TODO assert passed data is rescaled
    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "GridClass",
                 surfaces: "Surfaces", faults: "Faults", additional_data: "AdditionalData", **kwargs):

        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.surfaces = surfaces
        self.faults = faults

        self.dtype = additional_data.options.df.loc['values', 'dtype']
        self.theano_graph = self.create_theano_graph(additional_data, inplace=False)

        if 'compile_theano' in kwargs:
            self.theano_function = self.compile_th_fn(additional_data.options.df.loc['values', 'output'])
        else:
            self.theano_function = None

    def create_theano_graph(self, additional_data: "AdditionalData" = None, inplace=True):
        """
        create the graph accordingy to the options in the AdditionalData object
        Args:
            additional_data (AdditionalData):

        Returns:
            # TODO look for the right type in the theano library
            theano graph
        """
        import gempy.core.theano.theano_graph_pro as tg
        import importlib
        importlib.reload(tg)

        if additional_data is None:
            additional_data = self.additional_data

        #options = additional_data.options.df
        graph = tg.TheanoGraphPro(output=additional_data.options.df.loc['values', 'output'],
                               optimizer=additional_data.options.df.loc['values', 'theano_optimizer'],
                               dtype=additional_data.options.df.loc['values', 'dtype'],
                               verbose=additional_data.options.df.loc['values', 'verbosity'],
                               is_lith=additional_data.structure_data.df.loc['values', 'isLith'],
                               is_fault=additional_data.structure_data.df.loc['values', 'isFault'])

        return graph

    def set_theano_graph(self, th_graph):
        self.theano_graph = th_graph

    def set_theano_shared_kriging(self):
        # Range
        # TODO add rescaled range and co into the rescaling data df?
        self.theano_graph.a_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.df.loc['values', 'range'] /
                                                            self.additional_data.rescaling_data.df.loc[
                                                                'values', 'rescaling factor']))
        # Covariance at 0
        self.theano_graph.c_o_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.df.loc['values', '$C_o$'] /
                                                              self.additional_data.rescaling_data.df.loc[
                                                                  'values', 'rescaling factor']
                                                              ))
        # universal grades
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')))
        # nugget effect
        self.theano_graph.nugget_effect_grad_T.set_value(
            np.cast[self.dtype](self.additional_data.kriging_data.df.loc['values', 'nugget grad']))
        self.theano_graph.nugget_effect_scalar_T.set_value(
            np.cast[self.dtype](self.additional_data.kriging_data.df.loc['values', 'nugget scalar']))

    def set_theano_shared_structure_series(self):
        len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'] - 1)
        self.theano_graph.number_of_points_per_surface_T.set_value(len_rest_form.astype('int32'))


class InterpolatorWeights(Interpolator_pro):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "GridClass",
                 surfaces: "Surfaces", faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorWeights, self).__init__(surface_points, orientations, grid, surfaces, faults, additional_data,
                                                  **kwargs)

    def get_python_input_weights(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        if fault_drift is None:
            fault_drift = np.zeros((0, surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift)]
        return idl

    def compile_th_fn(self, inplace=False, debug=False):

        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_series()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_kriging
        print('Compiling theano function...')

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_weights(),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='warn',
                                allow_input_downcast=False,
                                profile=False)
        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn


class InterpolatorScalar(Interpolator_pro):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "GridClass",
                 surfaces: "Surfaces", faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorScalar, self).__init__(surface_points, orientations, grid, surfaces, faults, additional_data,
                                                  **kwargs)

    def get_python_input_zx(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r

        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid)]
        return idl

    def compile_th_fn(self, weights=None, grid=None, inplace=False, debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_series()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_kriging_export
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_scalar_field(weights,
                                                                       grid),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn


class InterpolatorModel(Interpolator_pro):
    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "GridClass",
                 surfaces: "Surfaces", faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorModel, self).__init__(surface_points, orientations, grid, surfaces, faults, additional_data,
                                                **kwargs)
        self.len_series_i = np.empty(0)
        self.len_series_o = np.empty(0)
        self.len_series_u = np.empty(0)
        self.len_series_f = np.empty(0)
        self.len_series_w = np.empty(0)

        self.compute_weights_ctrl = np.ones(10000)
        self.compute_scalar_ctrl = np.ones(10000)
        self.compute_bolck_ctrl = np.ones(10000)

    def set_theano_shared_structure_C(self):
        self.len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc['values', 'number surfaces per series'].astype('int32')

        self.len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype('int32')
        self.len_series_u = self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')
        self.len_series_f = self.faults.faults_relations_df.sum(axis=0).values.astype('int32')
        self.len_series_w = self.len_series_i + self.len_series_o + self.len_series_u + self.len_series_f

        self.theano_graph.len_series_i.set_value(np.insert(0, 0, self.theano_graph.len_series_i).cumsum())
        self.theano_graph.len_series_o.set_value(np.insert(0, 0, self.theano_graph.len_series_o).cumsum())
        self.theano_graph.len_series_w.set_value(np.insert(0, 0, self.theano_graph.len_series_w).cumsum())

    def set_initial_results(self):
        x_to_interp_shape = self.grid.values_r.shape[0] + self.surface_points.df.shape[0]
        n_series = self.additional_data.structure_data.df.loc['values', 'number surfaces']

        self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w[-1])))
        self.theano_graph.scalar_fields_matrix.set_value(
            np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
        self.theano_graph.block_matrix.set_value(np.zeros((n_series, self.surfaces.df.iloc[:, 4:].values.shape[0],
                                                           x_to_interp_shape), dtype=self.dtype))

    def get_python_input_block(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r
        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        values_properties = self.surfaces.df.iloc[:, 4:].values.astype(self.dtype).T

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid, values_properties)]
        return idl

    def compile_th_fn(self, inplace=False,
                      debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_series()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_loop
        print('Compiling theano function...')

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_series(),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function_formation = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')

        return th_fn


class InterpolatorBlock(Interpolator_pro):

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "GridClass",
                 surfaces: "Surfaces", faults: "Faults", additional_data: "AdditionalData", **kwargs):

        super(InterpolatorBlock, self).__init__(surface_points, orientations, grid, surfaces, faults, additional_data,
                                                  **kwargs)

    def get_python_input_block(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r
        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        values_properties = self.surfaces.df.iloc[:, 4:].values.astype(self.dtype).T

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid, values_properties)]
        return idl

    def compile_th_fn_formation_block(self, Z_x=None, weights=None, grid=None, values_properties=None, inplace=False,
                                  debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_series()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_block
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        if values_properties is None:
            values_properties = self.theano_graph.values_properties_op
        else:
            values_properties = theano.shared(values_properties)

        if Z_x is None:
            Z_x = self.theano_graph.compute_scalar_field(weights, grid)
        else:
            Z_x = theano.shared(Z_x)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_formation_block(
                                    Z_x,
                                    self.theano_graph.get_scalar_field_at_surface_points(Z_x),
                                    values_properties
                                ),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function_formation = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')

        return th_fn

    def compile_th_fn_fault_block(self, Z_x=None, weights=None, grid=None, values_properties=None, inplace=False, debug=False):
        """

        Args:
            weights: Constant weights
            grid:  Constant grids
            inplace:
            debug:

        Returns:

        """
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_series()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_block
        print('Compiling theano function...')

        if weights is None:
            weights = self.theano_graph.compute_weights()
        else:
            weights = theano.shared(weights)

        if grid is None:
            grid = self.theano_graph.grid_val_T
        else:
            grid = theano.shared(grid)

        if values_properties is None:
            values_properties = self.theano_graph.values_properties_op
        else:
            values_properties = theano.shared(values_properties)

        if Z_x is None:
            Z_x = self.theano_graph.compute_scalar_field(weights, grid)
        else:
            Z_x = theano.shared(Z_x)

        th_fn = theano.function(input_data_T,
                                self.theano_graph.compute_fault_block(
                                    Z_x,
                                    self.theano_graph.get_scalar_field_at_surface_points(Z_x),
                                    values_properties,
                                    0,
                                    grid
                               ),
                                # mode=NanGuardMode(nan_is_error=True),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)

        if inplace is True:
            self.theano_function_faults = th_fn

        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
        print('Compilation Done!')
        return th_fn

