class Interpolator(object):
    """Class that act as:
     1) linker between the data objects and the theano graph
     2) container of theano graphs + shared variables
     3) container of theano function

    Args:
        surface_points (SurfacePoints): [s0]
        orientations (Orientations): [s1]
        grid (Grid): [s2]
        surfaces (Surfaces): [s3]
        series (Series): [s4]
        faults (Faults): [s5]
        additional_data (AdditionalData): [s6]
        kwargs:
            - compile_theano: if true, the function is compile at the creation of the class

    Attributes:
        surface_points (SurfacePoints)
        orientations (Orientations)
        grid (Grid)
        surfaces (Surfaces)
        faults (Faults)
        additional_data (AdditionalData)
        dtype (['float32', 'float64']): float precision
        theano_graph: theano graph object with the properties from AdditionalData -> Options
        theano function: python function to call the theano code

    """
    # TODO assert passed data is rescaled

    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series: Series, faults: "Faults", additional_data: "AdditionalData", **kwargs):
        # Test
        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.surfaces = surfaces
        self.series = series
        self.faults = faults

        self.dtype = additional_data.options.df.loc['values', 'dtype']
        self.theano_graph = self.create_theano_graph(
            additional_data, inplace=False)
        self.theano_function = None

        self._compute_len_series()

    def _compute_len_series(self):
        self.len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
            self.additional_data.structure_data.df.loc['values',
                                                       'number surfaces per series']
        if self.len_series_i.shape[0] == 0:
            self.len_series_i = np.zeros(1, dtype=int)

        self.len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype(
            'int32')
        if self.len_series_o.shape[0] == 0:
            self.len_series_o = np.zeros(1, dtype=int)

        self.len_series_u = self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype(
            'int32')
        if self.len_series_u.shape[0] == 0:
            self.len_series_u = np.zeros(1, dtype=int)

        self.len_series_f = self.faults.faults_relations_df.sum(axis=0).values.astype('int32')[
            :self.additional_data.get_additional_data()['values']['Structure', 'number series']]
        if self.len_series_f.shape[0] == 0:
            self.len_series_f = np.zeros(1, dtype=int)

        self.len_series_w = self.len_series_i + self.len_series_o * \
            3 + self.len_series_u + self.len_series_f

    @setdoc_pro([AdditionalData.__doc__, ds.inplace, ds.theano_graph_pro])
    def create_theano_graph(self, additional_data: "AdditionalData" = None, inplace=True,
                            output=None, **kwargs):
        """
        Create the graph accordingly to the options in the AdditionalData object

        Args:
            additional_data (AdditionalData): [s0]
            inplace (bool): [s1]

        Returns:
            TheanoGraphPro: [s2]
        """
        if output is None:
            output = ['geology']

        import gempy.core.theano.theano_graph_pro as tg
        import importlib
        importlib.reload(tg)

        if additional_data is None:
            additional_data = self.additional_data

        graph = tg.TheanoGraphPro(optimizer=additional_data.options.df.loc['values', 'theano_optimizer'],
                                  verbose=additional_data.options.df.loc['values',
                                                                         'verbosity'],
                                  output=output,
                                  **kwargs)
        if inplace is True:
            self.theano_graph = graph
        else:
            return graph

    @setdoc_pro([ds.theano_graph_pro])
    def set_theano_graph(self, th_graph):
        """
        Attach an already create theano graph.

        Args:
            th_graph (TheanoGraphPro): [s0]

        Returns:
            True
        """
        self.theano_graph = th_graph
        return True

    def set_theano_shared_kriging(self):
        """
        Set to the theano_graph attribute the shared variables of kriging values from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        # Range
        # TODO add rescaled range and co into the rescaling data df?
        self.theano_graph.a_T.set_value(np.cast[self.dtype]
                                        (self.additional_data.kriging_data.df.loc['values', 'range'] /
                                         self.additional_data.rescaling_data.df.loc[
                                             'values', 'rescaling factor']))
        # Covariance at 0
        self.theano_graph.c_o_T.set_value(np.cast[self.dtype](
            self.additional_data.kriging_data.df.loc['values', '$C_o$'] /
            self.additional_data.rescaling_data.df.loc[
                'values', 'rescaling factor']
        ))
        # universal grades
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')[self.non_zero]))

        self.set_theano_shared_nuggets()

    def set_theano_shared_nuggets(self):
        # nugget effect
        # len_orientations = self.additional_data.structure_data.df.loc['values', 'len series orientations']
        # len_orientations_len = np.sum(len_orientations)

        self.theano_graph.nugget_effect_grad_T.set_value(
            np.cast[self.dtype](np.tile(
                self.orientations.df['smooth'], 3)))

        # len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'])
        # len_rest_len = np.sum(len_rest_form)
        self.theano_graph.nugget_effect_scalar_T.set_value(
            np.cast[self.dtype](self.surface_points.df['smooth']))
        return True

    def set_theano_shared_structure_surfaces(self):
        """
        Set to the theano_graph attribute the shared variables of structure from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        len_rest_form = (
            self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'] - 1)
        self.theano_graph.number_of_points_per_surface_T.set_value(
            len_rest_form.astype('int32'))
