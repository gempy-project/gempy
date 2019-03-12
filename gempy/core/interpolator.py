import numpy as np


class Interpolator(object):
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
        self.input_matrices = self.get_input_matrix()

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
        import gempy.core.theano_graph as tg
        import importlib
        importlib.reload(tg)

        if additional_data is None:
            additional_data = self.additional_data

        #options = additional_data.options.df
        graph = tg.TheanoGraph(output=additional_data.options.df.loc['values', 'output'],
                               optimizer=additional_data.options.df.loc['values', 'theano_optimizer'],
                               dtype=additional_data.options.df.loc['values', 'dtype'],
                               verbose=additional_data.options.df.loc['values', 'verbosity'],
                               is_lith=additional_data.structure_data.df.loc['values', 'isLith'],
                               is_fault=additional_data.structure_data.df.loc['values', 'isFault'])

        return graph

    def set_theano_graph(self, th_graph):
        self.theano_graph = th_graph

    def set_theano_function(self, th_function):
        self.theano_function = th_function

    def set_theano_shared_structure(self):
        # Size of every layer in rests. SHARED (for theano)
        len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'] - 1)
        self.theano_graph.number_of_points_per_surface_T.set_value(len_rest_form.astype('int32'))
        self.theano_graph.npf.set_value(
            np.cumsum(np.concatenate(([0], len_rest_form))).astype('int32'))  # Last value is useless
        # and breaks the basement
        # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
        self.theano_graph.len_series_i.set_value(
            np.insert(self.additional_data.structure_data.df.loc['values', 'len series surface_points'] -
                      self.additional_data.structure_data.df.loc['values', 'number surfaces per series'], 0,
                      0).cumsum().astype('int32'))
        # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
        self.theano_graph.len_series_f.set_value(
            np.insert(self.additional_data.structure_data.df.loc['values', 'len series orientations'], 0,
                      0).cumsum().astype('int32'))
        # Number of surfaces per series. The function is not pretty but the result is quite clear
        n_surfaces_per_serie = np.insert(
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series'].cumsum(), 0, 0). \
            astype('int32')
        self.theano_graph.n_surfaces_per_series.set_value(n_surfaces_per_serie)

        self.theano_graph.n_faults.set_value(self.additional_data.structure_data.df.loc['values', 'number faults'])
        # Set fault relation matrix
        self.theano_graph.fault_relation.set_value(self.faults.faults_relations_df.values.astype('int32'))

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

    def set_theano_shared_output_init(self):
        # Initialization of the block model
        self.theano_graph.final_block.set_value(np.zeros((1, self.grid.values_r.shape[0] + self.surface_points.df.shape[0]),
                                                         dtype=self.dtype))
        # Init the list to store the values at the surface_points. Here we init the shape for the given dataset
        self.theano_graph.final_scalar_field_at_surfaces.set_value(
            np.zeros(self.theano_graph.n_surfaces_per_series.get_value()[-1],
                     dtype=self.dtype))
        self.theano_graph.final_scalar_field_at_faults.set_value(
            np.zeros(self.theano_graph.n_surfaces_per_series.get_value()[-1],
                     dtype=self.dtype))

    def set_theano_share_input(self):
        self.theano_graph.grid_val_T.set_value(np.cast[self.dtype](self.grid.values_r + 10e-9))

        # Unique number assigned to each lithology
        n_surfaces = self.additional_data.structure_data.df.loc['values', 'number surfaces per series']
        if n_surfaces.size != 0:
            self.theano_graph.n_surface.set_value(np.arange(1, n_surfaces.sum() + 2, dtype='int32'))

        # Final values the lith block takes
        self.theano_graph.surface_values.set_value(
            self.surfaces.df.iloc[:, 5:].values.astype(self.dtype).T)

    def set_theano_shared_parameters(self):
        """
        Set theano shared variables from the other data objects.
        """

        # TODO: I have to split this one between structure_data and init data
        self.set_theano_shared_structure()
        self.set_theano_shared_kriging()
        self.set_theano_shared_output_init()
        self.set_theano_share_input()
        self.set_theano_inf_factor()

    def set_theano_inf_factor(self):
        """
        Set theano infinite factor so it knows!
        """
        inf_factor = self.faults.df[self.faults.df.isFault == True]["isFinite"].values
        self.theano_graph.inf_factor.set_value(np.invert(inf_factor).astype("int32"))

    def get_input_matrix(self) -> list:
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

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord)]
        return idl

    def compile_th_fn(self, output=None, inplace=True, debug=False, **kwargs):
        """
        Compile the theano function given the input_data data.

        Args:
            output (list['geology', 'gradients']): if output is gradients, the gradient field is also computed (in
            addition to the geology and properties)

        Returns:
            theano.function: Compiled function if C or CUDA which computes the interpolation given the input_data data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref surface_points, XYZ rest surface_points)
        """
        import theano
        self.set_theano_shared_parameters()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_list()
        if output is None:
            output = self.additional_data.options.df.loc['values', 'output']

        print('Compiling theano function...')

        if output == 'geology':
            # then we compile we have to pass the number of surfaces that are df!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_geological_model(),
                                    # mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output == 'gravity':
            # then we compile we have to pass the number of surfaces that are df!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_forward_gravity(),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output == 'gradients':

            gradients = kwargs.get('gradients', ['Gx', 'Gy', 'Gz'])
            self.theano_graph.gradients = gradients

            # then we compile we have to pass the number of surfaces that are df!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_geological_model_gradient(
                                        self.additional_data.structure_data['number faults']),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        else:
            raise SyntaxError('The output given does not exist. Please use geology, gradients or gravity ')

        if inplace is True:
            self.theano_function = th_fn
        if debug is True:
            print('Level of Optimization: ', theano.config.optimizer)
            print('Device: ', theano.config.device)
            print('Precision: ', self.dtype)
            print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])

        print('Compilation Done!')
        return th_fn
