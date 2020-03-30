import numpy as np
import tensorflow as tf


class Interpolator(object):
    """Class that act as:
    copy from theano Interpolator by Miguel
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
                 surfaces: "Surfaces", series: "Series", faults: "Faults", additional_data: "AdditionalData", **kwargs):
        # Test
        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.surfaces = surfaces
        self.series = series
        self.faults = faults

        self.dtype = additional_data.options.df.loc['values', 'dtype']

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
