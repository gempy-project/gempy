import numpy as np
import pandas as pn

from .surfaces import Surfaces


class Structure(object):
    """
    The structure_data class analyses the different lengths of subset in the interface and orientations categories_df
    to pass them to the aesara function.

    Attributes:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`gempy.Surfaces`): [s2]
        faults (:class:`Faults`): [s3]
        df (:class:`pn.DataFrame`):
            * len surfaces surface_points (list): length of each surface/fault in surface_points
            * len series surface_points (list) : length of each series in surface_points
            * len series orientations (list) : length of each series in orientations
            * number surfaces per series (list): number of surfaces per series
            * ...
    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`gempy.Surfaces`): [s2]
        faults (:class:`Faults`): [s3]

    """

    def __init__(self, surface_points, orientations, surfaces: Surfaces, faults):
        self.surface_points = surface_points
        self.orientations = orientations
        self.surfaces = surfaces
        self.faults = faults

        df_ = pn.DataFrame(np.array(['False', 'False', -1, -1, -1, -1, -1, -1, -1], ).reshape(1, -1),
                           index=['values'],
                           columns=['isLith', 'isFault',
                                    'number faults', 'number surfaces', 'number series',
                                    'number surfaces per series',
                                    'len surfaces surface_points', 'len series surface_points',
                                    'len series orientations'])

        self.df = df_.astype({'isLith': bool, 'isFault': bool, 'number faults': int,
                              'number surfaces': int, 'number series': int})

        self.update_structure_from_input()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def update_structure_from_input(self):
        """
        Update all fields dependent on the linked Data objects.

        Returns:
            bool: True
        """
        self.set_length_surfaces_i()
        self.set_series_and_length_series_i()
        self.set_length_series_o()
        self.set_number_of_surfaces_per_series()
        self.set_number_of_faults()
        self.set_number_of_surfaces()
        self.set_is_lith_is_fault()
        return True

    def set_length_surfaces_i(self):
        """
        Set the length of each **surface** on `SurfacePoints` i.e. how many data points are related to each surface

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # ==================
        # Extracting lengths
        # ==================
        # Array containing the size of every surface. SurfacePoints
        lssp = self.surface_points.df.groupby('id')['order_series'].count().values
        lssp_nonzero = lssp[np.nonzero(lssp)]

        self.df.at['values', 'len surfaces surface_points'] = lssp_nonzero

        return self.df

    def set_series_and_length_series_i(self):
        """
        Set the length of each **series** on `SurfacePoints` i.e. how many data points are related to each series. Also
        sets the number of series itself.

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        len_series = self.surfaces.series.df.shape[0]

        # Array containing the size of every series. SurfacePoints.
        points_count = self.surface_points.df['order_series'].value_counts(sort=False)
        len_series_i = np.zeros(len_series, dtype=int)
        len_series_i[points_count.index.astype('int') - 1] = points_count.values

        if len_series_i.shape[0] == 0:
            len_series_i = np.insert(len_series_i, 0, 0)

        self.df.at['values', 'len series surface_points'] = len_series_i
        self.df['number series'] = len(len_series_i)
        return self.df

    def set_length_series_o(self):
        """
        Set the length of each **series** on `Orientations` i.e. how many orientations are related to each series.

        Returns:
            :class:`pn.DataFrame`: df where the Structural data is stored

        """
        # Array containing the size of every series. orientations.

        len_series_o = np.zeros(self.surfaces.series.df.shape[0], dtype=int)
        ori_count = self.orientations.df['order_series'].value_counts(sort=False)
        len_series_o[ori_count.index.astype('int') - 1] = ori_count.values

        self.df.at['values', 'len series orientations'] = len_series_o

        return self.df

    def set_number_of_surfaces_per_series(self):
        """
        Set number of surfaces for each series

        Returns:
            :class:`pn.DataFrame`: df where the Structural data is stored

        """
        len_sps = np.zeros(self.surfaces.series.df.shape[0], dtype=int)
        surf_count = self.surface_points.df.groupby('order_series'). \
            surface.nunique()

        len_sps[surf_count.index.astype('int') - 1] = surf_count.values

        self.df.at['values', 'number surfaces per series'] = len_sps
        return self.df

    def set_number_of_faults(self):
        """
        Set number of faults series. This method in gempy v2 is simply informative

        Returns:
            :class:`pn.DataFrame`: df where the Structural data is stored

        """
        # Number of faults existing in the surface_points df
        self.df.at['values', 'number faults'] = self.faults.df['isFault'].sum()
        return self.df

    def set_number_of_surfaces(self):
        """
        Set the number of total surfaces

        Returns:
            :class:`pn.DataFrame`: df where the Structural data is stored

        """
        # Number of surfaces existing in the surface_points df
        self.df.at['values', 'number surfaces'] = self.surface_points.df['surface'].nunique()

        return self.df

    def set_is_lith_is_fault(self):
        """
        Check if there are lithologies in the data and/or df. This method in gempy v2 is simply informative

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored
        """
        self.df['isLith'] = True if self.df.loc['values', 'number series'] >= self.df.loc['values', 'number faults'] \
            else False
        self.df['isFault'] = True if self.df.loc['values', 'number faults'] > 0 else False

        return self.df
