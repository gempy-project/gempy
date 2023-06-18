import warnings

import numpy as np
import pandas as pn

from .surfaces import Surfaces
from .grid import Grid
from .structure import Structure

# This is for sphenix to find the packages
from ..utils.meta import _setdoc_pro

pn.options.mode.chained_assignment = None


class Options(object):
    """The class options contains the auxiliary user editable flags mainly independent to the model.

     Attributes:
        df (:class:`pn.DataFrame`): df containing the flags. All fields are pandas categories allowing the user to
         change among those categories.

     """

    def __init__(self):
        df_ = pn.DataFrame(np.array(['float32', 'geology', 'fast_compile', 'cpu', None]).reshape(1, -1),
                           index=['values'],
                           columns=['dtype', 'output', 'aesara_optimizer', 'device', 'verbosity'])

        self.df = df_.astype({'dtype'           : 'category', 'output': 'category',
                              'aesara_optimizer': 'category', 'device': 'category',
                              'verbosity'       : object})

        self.df['dtype'] = self.df['dtype'].cat.set_categories(['float32', 'float64'])
        self.df['aesara_optimizer'] = self.df['aesara_optimizer'].cat.set_categories(['fast_run', 'fast_compile'])
        self.df['device'] = self.df['device'].cat.set_categories(['cpu', 'cuda'])

        self.default_options()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_options(self, attribute, value):
        """Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.

        Returns:
            :class:`pandas.DataFrame`: df where options data is stored
        """

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)
        self.df.loc['values', attribute] = value
        return self.df

    def default_options(self):
        """Set default options.

        Returns:
            bool: True
        """
        import aesara
        self.df.loc['values', 'device'] = aesara.config.device

        if self.df.loc['values', 'device'] == 'cpu':
            self.df.loc['values', 'dtype'] = 'float64'
        else:
            self.df.loc['values', 'dtype'] = 'float32'

        self.df.loc['values', 'aesara_optimizer'] = 'fast_compile'
        return True


@_setdoc_pro([Grid.__doc__, Structure.__doc__])
class KrigingParameters(object):
    """
    Class that stores and computes the default values for the kriging parameters used during the interpolation.
    The default values will be computed from the :class:`Grid` and :class:`Structure` linked objects

    Attributes:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
        df (:class:`pn.DataFrame`): df containing the kriging parameters.

    Args:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
    """

    def __init__(self, grid: Grid, structure: Structure):
        self.structure = structure
        self.grid = grid

        df_ = pn.DataFrame(np.array([np.nan, np.nan, 3]).reshape(1, -1),
                           index=['values'],
                           columns=['range', '$C_o$', 'drift equations',
                                    ])

        self.df = df_.astype({'drift equations': object,
                              'range'          : object,
                              '$C_o$'          : object})
        self.set_default_range()
        self.set_default_c_o()
        self.set_u_grade()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_kriging_parameters(self, attribute: str, value, **kwargs):
        """Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.
            kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:
            :class:`pandas.DataFrame`: df where options data is stored
        """

        u_grade_sep = kwargs.get('u_grade_sep', ',')

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)

        if attribute == 'drift equations':
            value = np.asarray(value)
            print(value)

            if type(value) is str:
                value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
            try:
                assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
                print(value, attribute)
                self.df.at['values', attribute] = value
                print(self.df)

            except AssertionError:
                print('u_grade length must be the same as the number of series')

        else:
            self.df = self.df.astype({
                'drift equations': object,
                'range'          : object,
                '$C_o$'          : object})

            self.df.at['values', attribute] = value

    def str2int_u_grade(self, **kwargs):
        """
        Convert u_grade to ints

        Args:
            **kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:

        """
        u_grade_sep = kwargs.get('u_grade_sep', ',')
        value = self.df.loc['values', 'drift equations']
        if type(value) is str:
            value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
        try:
            assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
            self.df.at['values', 'drift equations'] = value

        except AssertionError:
            print('u_grade length must be the same as the number of series')

        return self.df

    def set_default_range(self, extent=None):
        """
        Set default kriging_data range

        Args:
            extent (Optional[float, np.array]): extent used to compute the default range--i.e. largest diagonal. If None
             extent of the linked :class:`Grid` will be used.

        Returns:

        """
        if extent is None:
            extent = self.grid.regular_grid.extent
            if np.sum(extent) == 0 and self.grid.values.shape[0] > 1:
                extent = np.concatenate((np.min(self.grid.values, axis=0),
                                         np.max(self.grid.values, axis=0)))[[0, 3, 1, 4, 2, 5]]

        try:
            range_var = np.sqrt(
                (extent[0] - extent[1]) ** 2 +
                (extent[2] - extent[3]) ** 2 +
                (extent[4] - extent[5]) ** 2)
        except TypeError:
            warnings.warn('The extent passed or if None the extent of the grid object has some '
                          'type of problem',
                          TypeError)
            range_var = np.array(np.nan)

        self.df['range'] = np.atleast_1d(range_var)

        return range_var

    def set_default_c_o(self, range_var=None):
        """
        Set default covariance at 0.

        Args:
            range_var (Optional[float, np.array]): range used to compute the default c_0--i.e. largest diagonal. If None
             the already computed range will be used.

        Returns:

        """
        if range_var is None:
            range_var = self.df.loc['values', 'range']

        if type(range_var) is list:
            range_var = np.atleast_1d(range_var)

        self.df.at['values', '$C_o$'] = range_var ** 2 / 14 / 3

        return self.df['$C_o$']

    def set_u_grade(self, u_grade: list = None):
        """
         Set default universal grade. Transform polynomial grades to number of equations

         Args:

             u_grade (list):

         Returns:

         """
        # =========================
        # Choosing Universal drifts
        # =========================
        if u_grade is None:

            len_series_i = self.structure.df.loc['values', 'len series surface_points']
            u_grade = np.ones_like(len_series_i)
            # u_grade[(len_series_i > 1)] = 1

        else:
            u_grade = np.array(u_grade)

        # Transforming grade to number of equations
        n_universal_eq = np.zeros_like(u_grade)
        n_universal_eq[u_grade == 0] = 0
        n_universal_eq[u_grade == 1] = 3
        n_universal_eq[u_grade == 2] = 9

        self.df.at['values', 'drift equations'] = n_universal_eq
        return self.df['drift equations']


class AdditionalData(object):
    """
    Container class that encapsulate :class:`Structure`, :class:`KrigingParameters`, :class:`Options` and
     rescaling parameters

    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]
        faults (:class:`Faults`): [s4]
        surfaces (:class:`Surfaces`): [s3]
        rescaling (:class:`RescaledData`): [s5]

    Attributes:
        structure_data (:class:`Structure`): [s6]
        options (:class:`Options`): [s8]
        kriging_data (:class:`Structure`): [s7]
        rescaling_data (:class:`RescaledData`):

    """

    def __init__(self, surface_points, orientations, grid: Grid,
                 faults, surfaces: Surfaces, rescaling):
        self.structure_data = Structure(surface_points, orientations, surfaces, faults)
        self.options = Options()
        self.kriging_data = KrigingParameters(grid, self.structure_data)
        self.rescaling_data = rescaling

    def __repr__(self):
        concat_ = self.get_additional_data()
        return concat_.to_string()

    def _repr_html_(self):
        concat_ = self.get_additional_data()
        return concat_.to_html()

    def get_additional_data(self):
        """
        Concatenate all linked data frames and transpose them for a nice visualization.

        Returns:
            pn.DataFrame: concatenated and transposed dataframe
        """
        concat_ = pn.concat([self.structure_data.df, self.options.df, self.kriging_data.df, self.rescaling_data.df],
                            axis=1, keys=['Structure', 'Options', 'Kriging', 'Rescaling'])
        return concat_.T

    def update_default_kriging(self):
        """
        Update default kriging values.
        """
        self.kriging_data.set_default_range()
        self.kriging_data.set_default_c_o()
        self.kriging_data.set_u_grade()

    def update_structure(self):
        """
        Update fields dependent on input data sucha as structure and universal kriging grade
        """
        self.structure_data.update_structure_from_input()
        if len(self.kriging_data.df.loc['values', 'drift equations']) < \
                self.structure_data.df.loc['values', 'number series']:
            self.kriging_data.set_u_grade()
