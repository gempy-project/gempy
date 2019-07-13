import os
import sys
import numpy as np
import pandas as pn
from typing import Union
import warnings

from gempy.core.data import AdditionalData, Faults, Grid, MetaData, Orientations, RescaledData, Series, SurfacePoints,\
    Surfaces, Options, Structure, KrigingParameters
from gempy.core.solution import Solution
from gempy.core.interpolator import InterpolatorModel, InterpolatorGravity
from gempy.utils.meta import setdoc, setdoc_pro
import gempy.utils.docstring as ds
from gempy.plot.decorators import *

pn.options.mode.chained_assignment = None


@setdoc_pro([Grid.__doc__, Faults.__doc__, Series.__doc__, Surfaces.__doc__, SurfacePoints.__doc__,
             Orientations.__doc__, RescaledData.__doc__, AdditionalData.__doc__, InterpolatorModel.__doc__,
             Solution.__doc__])
class DataMutation(object):
    """
    This class handles all the mutation of an object belonging to model and the update of every single object depend
    on that.

    Attributes:
        grid (:class:`Grid`): [s0]
        faults (:class:`Faults`): [s1]
        series (:class:`Series`): [s2]
        surfaces (:class:`Surfaces`): [s3]
        surface_points (:class:`SurfacePoints`): [s4]
        orientations (:class:`Orientations`): [s5]
        rescaling (:class:`Rescaling`): [s6]
        additional_data (:class:`AdditionalData`): [s7]
        interpolator (:class:`InterpolatorModel`): [s8]
        solutions (:class:`Solutions`): [s9]

     """

    def __init__(self):

        self.grid = Grid()
        self.faults = Faults()
        self.series = Series(self.faults)
        self.surfaces = Surfaces(self.series)
        self.surface_points = SurfacePoints(self.surfaces)
        self.orientations = Orientations(self.surfaces)

        self.rescaling = RescaledData(self.surface_points, self.orientations, self.grid)
        self.additional_data = AdditionalData(self.surface_points, self.orientations, self.grid, self.faults,
                                              self.surfaces, self.rescaling)

        self.interpolator = InterpolatorModel(self.surface_points, self.orientations, self.grid, self.surfaces,
                                              self.series, self.faults, self.additional_data)

        self.solutions = Solution(self.grid, self.surfaces, self.series)

    def _add_valid_idx_s(self, idx):
        if idx is None:
            idx = self.surface_points.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1
        else:
            assert isinstance(idx, (int, list, np.ndarray)), 'idx must be an int or a list of ints'

        return idx

    def _add_valid_idx_o(self, idx):
        if idx is None:
            idx = self.orientations.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1
        else:
            assert isinstance(idx, (int, list, np.ndarray)), 'idx must be an int or a list of ints'

        return idx

    @setdoc_pro([AdditionalData.update_structure.__doc__, InterpolatorModel.set_theano_shared_structure.__doc__,
                 InterpolatorModel.modify_results_matrices_pro.__doc__,
                 InterpolatorModel.modify_results_weights.__doc__])
    def update_structure(self, update_theano=None):
        """Update python and theano structure parameters.

        [s0]
        [s1]

        Args:
            update_theano: str{'matrices', 'weights'}:
                * matrices [s2]
                * weights [s3]
        """

        self.additional_data.update_structure()
        if update_theano == 'matrices':
            self.interpolator.modify_results_matrices_pro()
        elif update_theano == 'weights':
            self.interpolator.modify_results_weights()

        self.interpolator.set_theano_shared_structure()
        return self.additional_data.structure_data

    # region Grid
    def update_from_grid(self):
        self.rescaling.rescale_data()
        self.interpolator.set_initial_results_matrices()

    def set_active_grid(self, grid_name: Union[str, np.ndarray]):
        """
        Set active a given or several grids.

        Args:
            grid_name (str, list) {regular, custom, topography, centered}:

        """
        self.grid.deactivate_all_grids()
        self.grid.set_active(grid_name)
        self.update_from_grid()
        print(f'Active grids: {self.grid.grid_types[self.grid.active_grids]}')

        return self.grid

    def set_grid_object(self, grid: Grid, update_model=True):
        # TODO this should go to the api and let call all different grid types
        raise NotImplementedError

    @setdoc(Grid.set_regular_grid.__doc__)
    @setdoc_pro([ds.extent, ds.resolution])
    def set_regular_grid(self, extent, resolution):
        """
        Set a regular grid, rescale data and initialize theano solutions.

        Args:
            extent (np.ndarray): [s0]
            resolution (np.ndarray): [s1]

        Returns:
            Grid

        Set regular grid docs
        """
        self.grid.set_regular_grid(extent=extent, resolution=resolution)
        self.update_from_grid()
        print(f'Active grids: {self.grid.grid_types[self.grid.active_grids]}')
        return self.grid

    @setdoc(Grid.set_custom_grid.__doc__, )
    @setdoc_pro(ds.coord)
    def set_custom_grid(self, custom_grid):
        """
        Set custom grid, rescale gird and initialize theano solutions. foo

        Args:
            custom_grid (np.array): [s0]
        Returns:
            Grid

        Set custom grid Docs
        """
        self.grid.set_custom_grid(custom_grid)
        self.update_from_grid()
        print(f'Active grids: {self.grid.grid_types[self.grid.active_grids]}')
        return self.grid

    @plot_set_topography
    def set_topography(self, source='random', **kwargs):
        """
        Args:
            source:
                'gdal':     Load topography from a raster file.
                'random':   Generate random topography (based on a fractal grid).
                'saved':    Load topography that was saved with the topography.save() function.
                            This is useful after loading and saving a heavy raster file with gdal once or after saving a
                            random topography with the save() function. This .npy file can then be set as topography.
        Kwargs:
            if source = 'gdal:
                filepath:   path to raster file, e.g. '.tif', (for all file formats see https://gdal.org/drivers/raster/index.html)
            if source = 'random':
                fd:         fractal dimension, defaults to 2.0
                d_z:        maximum height difference. If none, last 20% of the model in z direction
                extent:     extent in xy direction. If none, geo_model.grid.extent
                resolution: desired resolution of the topography array. If none, geo_model.grid.resoution
            if source = 'saved':
                filepath:   path to the .npy file that was created using the topography.save() function

        Returns: :class:gempy.core.data.Topography
        """

        self.grid.set_topography(source, **kwargs)
        self.update_from_grid()
        print(f'Active grids: {self.grid.grid_types[self.grid.active_grids]}')
        return self.grid

    @setdoc(Grid.set_centered_grid.__doc__, )
    def set_centered_grid(self, centers, radio, resolution=None):
        self.grid.set_centered_grid(centers, radio, resolution=resolution)
        self.update_from_grid()
        print(f'Active grids: {self.grid.grid_types[self.grid.active_grids]}')
        return self.grid

    # endregion

    # region Series
    def set_series_object(self):
        """
        Not implemented yet. Exchange the series object of the Model object. foo

        Returns:

        """
        raise NotImplementedError

    @setdoc([Series.set_bottom_relation.__doc__], indent=False)
    def set_bottom_relation(self, series: Union[str, list], bottom_relation: Union[str, list]):
        """"""
        self.series.set_bottom_relation(series, bottom_relation)
        self.interpolator.set_theano_shared_relations()
        return self.series

    @setdoc(Series.add_series.__doc__, indent=False)
    def add_series(self, series_list: Union[str, list], reset_order_series=True):
        """ Add series, update the categories dependet on them and reset the flow control.
        """
        self.series.add_series(series_list, reset_order_series)
        self.surfaces.df['series'].cat.add_categories(series_list, inplace=True)
        self.surface_points.df['series'].cat.add_categories(series_list, inplace=True)
        self.orientations.df['series'].cat.add_categories(series_list, inplace=True)
        self.interpolator.set_flow_control()
        return self.series

    @setdoc(Series.delete_series.__doc__, indent=False)
    def delete_series(self, indices: Union[str, list], refactor_order_series=True):
        """Delete series, update the categories dependet on them and reset the flow control.
        """
        self.series.delete_series(indices, refactor_order_series)
        self.surfaces.df['series'].cat.remove_categories(indices, inplace=True)
        self.surface_points.df['series'].cat.remove_categories(indices, inplace=True)
        self.orientations.df['series'].cat.remove_categories(indices, inplace=True)
        self.map_geometric_data_df(self.surface_points.df)
        self.map_geometric_data_df(self.orientations.df)

        self.interpolator.set_theano_shared_relations()
        self.interpolator.set_flow_control()
        return self.series

    @setdoc(Series.rename_series.__doc__, indent=False)
    def rename_series(self, new_categories: Union[dict, list]):
        """Rename series and update the categories dependet on them."""
        self.series.rename_series(new_categories)
        self.surfaces.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.surface_points.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.orientations.df['series'].cat.rename_categories(new_categories, inplace=True)
        return self.series

    @setdoc(Series.modify_order_series.__doc__, indent=False)
    def modify_order_series(self, new_value: int, idx: str):
        """Modify order of the series. Reorder categories of the link Surfaces, sort surface (reset the basement layer)
        remap the Series and Surfaces to the corrspondent dataframes, sort Geometric objects, update structure and
        reset the flow control objects.

        """
        self.series.modify_order_series(new_value, idx)

        self.surfaces.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
                                                          ordered=False, inplace=True)

        self.surfaces.sort_surfaces()
        self.surfaces.set_basement()

        self.map_geometric_data_df(self.surface_points.df)
        self.surface_points.sort_table()
        self.map_geometric_data_df(self.orientations.df)
        self.orientations.sort_table()

        self.interpolator.set_flow_control()
        self.update_structure()
        return self.series

    @setdoc(Series.reset_order_series.__doc__, indent=False)
    def reorder_series(self, new_categories: Union[list, np.ndarray]):
        """Reorder series. Reorder categories of the link Surfaces, sort surface (reset the basement layer)
        remap the Series and Surfaces to the corrspondent dataframes, sort Geometric objects, update structure and
        reset the flow control objects.
        """
        self.series.reorder_series(new_categories)
        self.surfaces.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
                                                          ordered=False, inplace=True)

        self.surfaces.sort_surfaces()
        self.surfaces.set_basement()

        self.map_geometric_data_df(self.surface_points.df)
        self.surface_points.sort_table()
        self.map_geometric_data_df(self.orientations.df)
        self.orientations.sort_table()

        self.interpolator.set_flow_control()
        self.update_structure()
        return self.series

    # endregion

    # region Faults
    def set_fault_object(self):
        pass

    @setdoc([Faults.set_is_fault.__doc__], indent=False)
    def set_is_fault(self, series_fault: Union[str, list] = None, toggle: bool = False, offset_faults=False,
                     change_color: bool = True, twofins = False):
        """
        Set a series to fault and update all dependet objects of the Model.

        Args:
            change_color (bool): If True faults surfaces get the default fault color (light gray)

        Faults.set_is_fault Doc:

        """
        series_fault = np.atleast_1d(series_fault)
        if twofins is False:
            for fault in series_fault:
                assert np.sum(self.surfaces.df['series'] == fault) < 2,\
                    'Having more than one fault in a series is generally rather bad. Better go' \
                    ' back to the function map_series_to_surfaces and give each fault its own' \
                    ' series. If you are really sure what you are doing, you can set twofins to' \
                    ' True to suppress this error.'

        self.faults.set_is_fault(series_fault, toggle=toggle)

        if toggle is True:
            already_fault = self.series.df.loc[series_fault, 'BottomRelation'] == 'Fault'
            self.series.df.loc[series_fault[already_fault], 'BottomRelation'] = 'Erosion'
            self.series.df.loc[series_fault[~already_fault], 'BottomRelation'] = 'Fault'
        else:
            self.series.df.loc[series_fault, 'BottomRelation'] = 'Fault'

        self.additional_data.structure_data.set_number_of_faults()
        self.interpolator.set_theano_shared_relations()
        self.interpolator.set_theano_shared_loop()
        if change_color:
            print('Fault colors changed. If you do not like this behavior, set change_color to False.')
            self.surfaces.colors.make_faults_black(series_fault)
        self.update_structure(update_theano='matrices')
        return self.faults

    @setdoc([Faults.set_is_finite_fault.__doc__], indent=False)
    def set_is_finite_fault(self, series_fault=None, toggle: bool = False):
        """ """
        s = self.faults.set_is_finite_fault(series_fault, toggle)  # change df in Fault obj
        # change shared theano variable for infinite factor
        self.interpolator.set_theano_shared_is_finite()
        return s

    @setdoc([Faults.set_fault_relation.__doc__], indent=False)
    def set_fault_relation(self, rel_matrix):
        """"""
        self.faults.set_fault_relation(rel_matrix)

        # Updating
        self.interpolator.set_theano_shared_fault_relation()
        self.interpolator.set_theano_shared_weights()
        return self.faults.faults_relations_df

    # endregion

    # region Surfaces
    def set_surfaces_object(self):
        """
        Not implemented yet. Exchange the surface object of the Model object
        Returns:

        """
        raise NotImplementedError

    @setdoc(Surfaces.add_surface.__doc__, indent=False)
    def add_surfaces(self, surface_list: Union[str, list], update_df=True):
        self.surfaces.add_surface(surface_list, update_df)
        self.surface_points.df['surface'].cat.add_categories(surface_list, inplace=True)
        self.orientations.df['surface'].cat.add_categories(surface_list, inplace=True)
        self.update_structure()
        return self.surfaces

    @setdoc(Surfaces.delete_surface.__doc__, indent=False)
    def delete_surfaces(self, indices: Union[str, list, np.ndarray], update_id=True, remove_data=False):
        """Delete a surface and update all related object.

        Args:
            remove_data (bool): if true delete all GeometricData labeled with the given surface.

        Surface.delete_surface Doc:
        """
        indices = np.atleast_1d(indices)
        self.surfaces.delete_surface(indices, update_id)

        if indices.dtype == int:
            surfaces_names = self.surfaces.df.loc[indices, 'surface']
        else:
            surfaces_names = indices
        if remove_data:
            self.surface_points.del_surface_points(self.surface_points.df[self.surface_points.df.surface.isin(surfaces_names)].index)
            self.orientations.del_orientation(self.orientations.df[self.orientations.df.surface.isin(surfaces_names)].index)
            self.update_structure(update_theano='matrices')
            self.update_structure(update_theano='weights')
        self.surface_points.df['surface'].cat.remove_categories(surfaces_names, inplace=True)
        self.orientations.df['surface'].cat.remove_categories(surfaces_names, inplace=True)
        self.map_geometric_data_df(self.surface_points.df)
        self.map_geometric_data_df(self.orientations.df)
        return self.surfaces

    @setdoc(Surfaces.rename_surfaces.__doc__, indent=False)
    def rename_surfaces(self, to_replace: Union[dict], **kwargs):

        self.surfaces.rename_surfaces(to_replace, **kwargs)
        self.surface_points.df['surface'].cat.rename_categories(to_replace, inplace=True)
        self.orientations.df['surface'].cat.rename_categories(to_replace, inplace=True)
        return self.surfaces

    @setdoc(Surfaces.modify_order_surfaces.__doc__, indent=False)
    def modify_order_surfaces(self, new_value: int, idx: int, series_name: str = None):
        """"""
        self.surfaces.modify_order_surfaces(new_value, idx, series_name)

        self.map_geometric_data_df(self.surface_points.df)
        self.surface_points.sort_table()
        self.map_geometric_data_df(self.orientations.df)
        self.orientations.sort_table()

        self.update_structure()
        return self.surfaces

    @setdoc(Surfaces.add_surfaces_values.__doc__, indent=False)
    def add_surface_values(self,  values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        self.surfaces.add_surfaces_values(values_array, properties_names)
        return self.surfaces

    @setdoc(Surfaces.delete_surface_values.__doc__, indent=False)
    def delete_surface_values(self, properties_names: list):
        self.delete_surface_values(properties_names)
        return self.surfaces

    @setdoc(Surfaces.modify_surface_values.__doc__, indent=False)
    def modify_surface_values(self, idx, properties_names, values):
        self.surfaces.modify_surface_values(idx, properties_names, values)
        return self.surfaces

    def set_surface_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        self.surfaces.set_surfaces_values(values_array, properties_names)
        return self.surfaces

    @setdoc([Surfaces.map_series.__doc__], indent=False)
    def map_series_to_surfaces(self, mapping_object: Union[dict, pn.Categorical] = None,
                               set_series=True, sort_geometric_data: bool = True, remove_unused_series=True):
        """
        Map series to surfaces and update all related objects accordingly to the following arguments:

        Args:
            set_series (bool): if True, if mapping object has non existing series they will be created.
            sort_geometric_data (bool): If true geometric data will be sorted accordingly to the new order of the
             series
            remove_unused_series (bool): if true, if an existing series is not assigned with a surface, it will get
             removed from the Series object

        Returns:
            Surfaces

        Surfaces.map_series Doc:
        """
        # Add New series to the series df
        if set_series is True:
            if type(mapping_object) is dict:
                series_list = list(mapping_object.keys())
                self.series.add_series(series_list)
            elif isinstance(mapping_object, pn.Categorical):
                series_list = mapping_object['series'].values
                self.series.add_series(series_list)
            else:
                raise AttributeError(str(type(mapping_object)) + ' is not the right attribute type.')

        self.surfaces.map_series(mapping_object)

        # Here we remove the series that were not assigned to a surface
        if remove_unused_series is True:
            self.surfaces.df['series'].cat.remove_unused_categories(inplace=True)
            unused_cat = self.series.df.index[~self.series.df.index.isin(
                self.surfaces.df['series'].cat.categories)]
            self.series.delete_series(unused_cat)

        self.series.reset_order_series()

        self.update_from_surfaces()
        self.update_from_series()

        if sort_geometric_data is True:
            self.surface_points.sort_table()
            self.orientations.sort_table()

        if set_series is True and self.series.df.index.isin(['Basement']).any():
            aux = self.series.df.index.drop('Basement').get_values()
            self.reorder_series(np.append(aux, 'Basement'))

        return self.surfaces

    # endregion

    # region Surface_points

    def set_surface_points_object(self, surface_points: SurfacePoints, update_model=True):
        raise NotImplementedError

    @setdoc(SurfacePoints.set_surface_points.__doc__, indent=False, position='beg')
    def set_surface_points(self, table: pn.DataFrame, **kwargs):
        """
        Args:
            table (pn.Dataframe): table with surface points data.
            **kwargs:
                - add_basement (bool): add a basement surface to the df. foo

        """
        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        surface_name = kwargs.get('surface_name', "surface")
        update_surfaces = kwargs.get('update_surfaces', True)

        if update_surfaces is True:
            self.add_surfaces(table[surface_name].unique())

        c = np.array(self.surface_points._columns_i_1)
        surface_points_table = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
        self.surface_points.set_surface_points(surface_points_table[[coord_x_name, coord_y_name, coord_z_name]],
                                               surface=surface_points_table[surface_name])

        if 'add_basement' in kwargs:
            if kwargs['add_basement'] is True:
                self.surfaces.add_surface(['basement'])
                self.map_series_to_surfaces({'Basement': 'basement'}, set_series=True)
        self.map_geometric_data_df(self.surface_points.df)
        self.rescaling.rescale_data()
        self.additional_data.update_structure()
        self.additional_data.update_default_kriging()

    @setdoc_pro(ds.recompute_rf)
    @setdoc(SurfacePoints.add_surface_points.__doc__, indent=False, position='beg')
    @plot_add_surface_points
    def add_surface_points(self, X, Y, Z, surface, idx: Union[int, list, np.ndarray] = None,
                           recompute_rescale_factor=False):
        """
        Args:
            recompute_rescale_factor (bool): [s0]
        """

        surface = np.atleast_1d(surface)
        idx = self._add_valid_idx_s(idx)
        self.surface_points.add_surface_points(X, Y, Z, surface, idx)

        if recompute_rescale_factor is True or idx < 20:
            # This will rescale all data again
            self.rescaling.rescale_data()
            self.interpolator.set_theano_shared_kriging()
        else:
            # This branch only recompute the added point
            self.rescaling.set_rescaled_surface_points(idx)
        self.update_structure(update_theano='matrices')
        self.interpolator.set_theano_shared_nuggets()

        return self.surface_points, idx

    @setdoc(SurfacePoints.del_surface_points.__doc__, indent=False, position='beg')
    @plot_delete_surface_points
    def delete_surface_points(self, idx: Union[list, int, np.ndarray]):
        self.surface_points.del_surface_points(idx)
        self.update_structure(update_theano='matrices')
        return self.surface_points

    def delete_surface_points_basement(self):
        """Delete surface points belonging to the basement layer if any"""
        basement_name = self.surfaces.df['surface'][self.surfaces.df['isBasement']].values
        select = (self.surface_points.df['surface'] == basement_name)
        self.delete_surface_points(self.surface_points.df.index[select])
        return True

    @setdoc_pro(ds.recompute_rf)
    @setdoc(SurfacePoints.modify_surface_points.__doc__, indent=False, position='beg')
    @plot_move_surface_points
    def modify_surface_points(self, indices: Union[int, list], recompute_rescale_factor=False, **kwargs):
        """
        Args:
            recompute_rescale_factor: [s0]
        """
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface:
            assert (~self.surfaces.df[self.surfaces.df['isBasement']]['surface'].isin(
                np.atleast_1d(kwargs['surface']))).any(),\
                'Surface points cannot belong to Basement. Add a new surface.'

        self.surface_points.modify_surface_points(indices, **kwargs)

        if recompute_rescale_factor is True or np.atleast_1d(indices).shape[0] < 20:
            # This will rescale all data again
            self.rescaling.rescale_data()
            self.interpolator.set_theano_shared_kriging()
        else:
            # This branch only recompute the added point
            self.rescaling.set_rescaled_surface_points(indices)

        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface == True:
            self.update_structure(update_theano='matrices')

        if 'smooth' in kwargs:
            self.interpolator.set_theano_shared_nuggets()

        return self.surface_points

    # endregion

    # region Orientation
    def set_orientations_object(self, orientations: Orientations, update_model=True):
        raise NotImplementedError

    @setdoc_pro(ds.recompute_rf)
    @setdoc(Orientations.add_orientation.__doc__, indent=False, position='beg')
    @plot_add_orientation
    def add_orientations(self,  X, Y, Z, surface, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, idx=None, recompute_rescale_factor=False):
        """
        Args:
            recompute_rescale_factor: [s0]

        """

        surface = np.atleast_1d(surface)
        idx = self._add_valid_idx_o(idx)
        self.orientations.add_orientation(X, Y, Z, surface, pole_vector=pole_vector,
                                          orientation=orientation, idx=idx)
        if recompute_rescale_factor is True or idx < 5:
            # This will rescale all data again
            self.rescaling.rescale_data()
        else:
            # This branch only recompute the added point
            self.rescaling.set_rescaled_orientations(idx)
        self.update_structure(update_theano='weights')
        self.interpolator.set_theano_shared_nuggets()

        return self.orientations, idx

    @setdoc(Orientations.del_orientation.__doc__, indent=False, position='beg')
    @plot_delete_orientations
    def delete_orientations(self, idx: Union[list, int]):
        self.orientations.del_orientation(idx)
        self.update_structure(update_theano='weights')
        return self.orientations

    @setdoc(Orientations.modify_orientations.__doc__, indent=False, position='beg')
    @plot_move_orientations
    def modify_orientations(self, idx: list, **kwargs):

        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        self.orientations.modify_orientations(idx, **kwargs)
        self.rescaling.set_rescaled_orientations(idx)

        if is_surface:
            self.update_structure(update_theano='weights')

        return self.orientations
    # endregion

    # region Options
    @setdoc(Options.modify_options.__doc__, indent=False, position='beg')
    def modify_options(self, attribute, value):
        self.additional_data.options.modify_options(attribute, value)
        warnings.warn('You need to recompile the Theano code to make it the changes in options.')

    # endregion

    # region Kriging
    @setdoc(KrigingParameters.modify_kriging_parameters.__doc__, indent=False, position='beg')
    def modify_kriging_parameters(self, attribute, value, **kwargs):
        self.additional_data.kriging_data.modify_kriging_parameters(attribute, value, **kwargs)
        self.interpolator.set_theano_shared_kriging()
        if attribute == 'drift equations':
            self.interpolator.set_initial_results()

    # endregion

    # region rescaling
    @setdoc(RescaledData.modify_rescaling_parameters.__doc__, indent=False, position='beg')
    def modify_rescaling_parameters(self, attribute, value):
        self.additional_data.rescaling_data.modify_rescaling_parameters(attribute, value)
        self.additional_data.rescaling_data.rescale_data()
        self.additional_data.update_default_kriging()
    # endregion

    # ======================================
    # --------------------------------------
    # ======================================

    def set_default_surface_point(self, **kwargs):
        """
        Set a default surface point if the df is empty. This is necessary for some type of functionality such as qgrid

        Args:
            **kwargs: :meth:`add_surface_points` kwargs

        Returns:
            SurfacePoints
        """
        if self.surface_points.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0],
                                    recompute_rescale_factor=True, **kwargs)
        return self.surface_points

    def set_default_orientation(self, **kwargs):
        """
         Set a default orientation if the df is empty. This is necessary for some type of functionality such as qgrid

         Args:
             **kwargs: :meth:`add_orientation` kwargs

         Returns:
             Orientations
         """
        if self.orientations.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.add_orientations(.00001, .00001, .00001,
                                  self.surfaces.df['surface'].iloc[0],
                                  [0, 0, 1], recompute_rescale_factor=True, **kwargs)

    def set_default_surfaces(self):
        """
         Set two default surfaces if the df is empty. This is necessary for some type of functionality such as qgrid

         Returns:
             Surfaces
         """
        if self.surfaces.df.shape[0] == 0:
            self.add_surfaces(['surface1', 'surface2'])
        return self.surfaces

    @setdoc_pro(ds.extent)
    def set_extent(self, extent: Union[list, np.ndarray]):
        """
        Set project extent

        Args:
            extent: [s0]

        Returns:

        """
        extent = np.atleast_1d(extent)
        self.grid.extent = extent
        self.rescaling.set_rescaled_grid()
        return self.grid

    def update_from_series(self, rename_series: dict = None, reorder_series=True, sort_geometric_data=True,
                           update_interpolator=True):
        """
        Update all objects dependent on series.

        This method is a bit of a legacy and has been substituted by :meth:`rename_series` and :meth:`reorder_series`,
        however is useful if you want to make sure all objects are up to date with the latest changes on series.

        Args:
            rename_series (dict): DEP see :meth:`rename_series`
            reorder_series (bool): if True reorder all pandas categories accordingly to the series.df
            sort_geometric_data (bool): It True sort the geometric data after mapping the new order
            update_interpolator (bool): If True update the theano shared variables dependent on the structure

        Returns:
            True
        """
        # Add categories from series to surface
        # Updating surfaces['series'] categories
        if rename_series is None:
            self.surfaces.df['series'].cat.set_categories(self.series.df.index, inplace=True)
        else:
            self.surfaces.df['series'].cat.rename_categories(rename_series, inplace=True)

        if reorder_series is True:
            self.surfaces.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
                                                              ordered=False, inplace=True)
            self.series.df.index = self.series.df.index.reorder_categories(self.series.df.index.get_values(),
                                                                           ordered=False)
            self.surfaces.sort_surfaces()
            self.update_from_surfaces(set_categories_from_series=False, set_categories_from_surfaces=True,
                                      map_surface_points=False, map_orientations=False, update_structural_data=False)

        self.surfaces.set_basement()

        # Add categories from series
        self.surface_points.set_series_categories_from_series(self.series)
        self.orientations.set_series_categories_from_series(self.series)

        self.surface_points.map_data_from_series(self.series, 'order_series')
        self.orientations.map_data_from_series(self.series, 'order_series')

        if sort_geometric_data is True:
            self.surface_points.sort_table()
            self.orientations.sort_table()

        self.additional_data.update_structure()
        # For the drift equations.
        self.additional_data.update_default_kriging()

        if update_interpolator is True:
            self.interpolator.set_theano_shared_structure(reset_ctrl=True)

        return True

    def update_from_surfaces(self, set_categories_from_series=True, set_categories_from_surfaces=True,
                             map_surface_points=True, map_orientations=True, update_structural_data=True):
        """
        Update all objects dependt on surfaces.

        Args:
            set_categories_from_series (bool): If True update the pandas categories with the Series object
            set_categories_from_surfaces (bool): If True update the pandas categories with the surfaces object
            map_surface_points (bool): If True map the surface points fields with the Surfaces obejct
            map_orientations (bool): If True map the orientations fields with the Surfaces object
            update_structural_data (bool): If true update the Structure with the Surface object

        Returns:
            True
        """
        # Add categories from series
        if set_categories_from_series is True:
            self.surface_points.set_series_categories_from_series(self.surfaces.series)
            self.orientations.set_series_categories_from_series(self.surfaces.series)

        # Add categories from surfaces
        if set_categories_from_surfaces is True:
            self.surface_points.set_surface_categories_from_surfaces(self.surfaces)
            self.orientations.set_surface_categories_from_surfaces(self.surfaces)

        if map_surface_points is True:
            self.surface_points.map_data_from_surfaces(self.surfaces, 'series')
            self.surface_points.map_data_from_surfaces(self.surfaces, 'id')

        if map_orientations is True:
            self.orientations.map_data_from_surfaces(self.surfaces, 'series')
            self.orientations.map_data_from_surfaces(self.surfaces, 'id')

        if update_structural_data is True:
            self.additional_data.update_structure()

        return True

    # region Theano interface
    @setdoc(InterpolatorModel.__doc__)
    def set_theano_graph(self, interpolator: InterpolatorModel):
        """Pass a theano graph of a Interpolator instance other than the Model compose

        Use this method only if you know what are you doing!

        Args:
            interpolator (:class:`InterpolatorModel`): [s0]

        Returns:
             True """
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.update_to_interpolator()
        return True

    @setdoc(InterpolatorModel.__doc__)
    def set_theano_function(self, interpolator: InterpolatorModel):
        """
        Pass a theano function and its correspondent graph from an Interpolator instance other than the Model compose

        Args:
            interpolator (:class:`InterpolatorModel`): [s0]

        Returns:
             True
        """

        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.interpolator.set_all_shared_parameters()
        self.update_structure(update_theano='matrices')
        return True

    def update_to_interpolator(self, reset=True):
        """Update all shared parameters from the data objects

        Args:
            reset (bool): if True reset the flow control and initialize results arrays

        Returns:
            True
        """
        self.interpolator.set_all_shared_parameters()
        if reset is True:
            self.interpolator.reset_flow_control_initial_results()
        return True
    # endregion

    def map_geometric_data_df(self, d: pn.DataFrame):
        """
        Map a geometric data dataframe from the linked objects (at 07.2019 surfaces and series)

        Args:
            d (pn.DataFrame): Geometric data dataframe to be mapped

        Returns:
            DataFrame
        """
        d['series'] = d['surface'].map(self.surfaces.df.set_index('surface')['series'])
        d['id'] = d['surface'].map(self.surfaces.df.set_index('surface')['id'])
        d['order_series'] = d['series'].map(self.series.df['order_series'])
        return d

    def set_surface_order_from_solution(self):
        """
        Order the surfaces respect the last computation. Therefore if you call this method,
        after sorting surface_points without recomputing you may get wrong results.

        Returns:
            Surfaces
        """
        # TODO time this function
        spu = self.surface_points.df['surface'].unique()
        sps = self.surface_points.df['series'].unique()
        sel = self.surfaces.df['surface'].isin(spu)
        for e, name_series in enumerate(sps):
            try:
                sfai_series = self.solutions.scalar_field_at_surface_points[e]
                sfai_order_aux = np.argsort(sfai_series[np.nonzero(sfai_series)])
                sfai_order = (sfai_order_aux - sfai_order_aux.shape[0]) * -1
                # select surfaces which exist in surface_points
                group = self.surfaces.df[sel].groupby('series').get_group(name_series)
                idx = group.index
                surface_names = group['surface']

                self.surfaces.df.loc[idx, 'order_surfaces'] = self.surfaces.df.loc[idx, 'surface'].map(
                    pn.DataFrame(sfai_order, index=surface_names)[0])

            except IndexError:
                pass

        self.surfaces.sort_surfaces()
        self.surfaces.set_basement()
        self.surface_points.df['id'] = self.surface_points.df['surface'].map(
            self.surfaces.df.set_index('surface')['id'])
        self.surface_points.sort_table()
        self.update_structure()
        return self.surfaces


@setdoc([MetaData.__doc__, DataMutation.__doc__], indent=False)
class Model(DataMutation):
    """Container class of all objects that constitute a GemPy model.

    In addition the class provides the methods that act in more than one of this class. Model is a child class of
    :class:`DataMutation` and :class:`MetaData`.

    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        super().__init__()
        self.interpolator_gravity = None

    def __repr__(self):
        return self.meta.project_name + ' ' + self.meta.date

    def new_model(self, name_project='default_project'):
        """Reset the model object."""
        self.__init__(name_project)

    def save_model_pickle(self, path=False):
        """
        Short term model storage. Object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            True
        """
        # Deleting qi attribute otherwise doesnt allow to pickle
        if hasattr(self, 'qi'):
            self.__delattr__('qi')

        sys.setrecursionlimit(10000)

        if not path:
            path = './'+self.meta.project_name
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return True

    @staticmethod
    def load_model_pickle(path):
        """
        Read InputData object from python pickle.

        Args:
           path (str): path where save the pickle

        Returns:
            :class:`gempy.core.model.Model`

        """
        import pickle
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            model = pickle.load(f)
            return model

    def save_model(self, name=None, path=None):
        """
        Save model in new folder. Input data is saved as csv files. Solutions, extent and resolutions are saved as npy.

        Args:
            name (str): name of the newly created folder and the part of the files name
            path (str): path where save the model folder.

        Returns:
            True
        """
        if name is None:
            name = self.meta.project_name

        if not path:
            path = './'
        path = f'{path}/{name}'

        if os.path.isdir(path):
            print("Directory already exists, files will be overwritten")
        else:
            os.mkdir(f'{path}')

        # save dataframes as csv
        self.surface_points.df.to_csv(f'{path}/{name}_surface_points.csv')
        self.surfaces.df.to_csv(f'{path}/{name}_surfaces.csv')
        self.orientations.df.to_csv(f'{path}/{name}_orientations.csv')
        self.series.df.to_csv(f'{path}/{name}_series.csv')
        self.faults.df.to_csv(f'{path}/{name}_faults.csv')
        self.faults.faults_relations_df.to_csv(f'{path}/{name}_faults_relations.csv')
        self.additional_data.kriging_data.df.to_csv(f'{path}/{name}_kriging_data.csv')
        self.additional_data.rescaling_data.df.to_csv(f'{path}/{name}_rescaling_data.csv')
        self.additional_data.options.df.to_csv(f'{path}/{name}_options.csv')

        # # save resolution and extent as npy
        np.save(f'{path}/{name}_extent.npy', self.grid.regular_grid.extent)
        np.save(f'{path}/{name}_resolution.npy', self.grid.regular_grid.resolution)

        # # save solutions as npy
        # np.save(f'{path}/{name}_lith_block.npy' ,self.solutions.lith_block)
        # np.save(f'{path}/{name}_scalar_field_lith.npy', self.solutions.scalar_field_matrix)
        #
        # np.save(f'{path}/{name}_gradient.npy', self.solutions.gradient)
        # np.save(f'{path}/{name}_values_block.npy', self.solutions.matr)

        return True

    @setdoc([SurfacePoints.read_surface_points.__doc__, Orientations.read_orientations.__doc__])
    def read_data(self, path_i=None, path_o=None, add_basement=True, **kwargs):
        """
        Read data from a csv

        Args:
            path_i: Path to the data bases of surface_points. Default os.getcwd(),
            path_o: Path to the data bases of orientations. Default os.getcwd()
            add_basement (bool): if True add a basement surface. This wont be interpolated it just gives the values
            for the volume below the last surface.
            **kwargs:
                update_surfaces (bool): True

        Returns:
            True
        """
        if 'update_surfaces' not in kwargs:
            kwargs['update_surfaces'] = True

        if path_i:
            self.surface_points.read_surface_points(path_i, inplace=True, **kwargs)
        if path_o:
            self.orientations.read_orientations(path_o, inplace=True, **kwargs)
        if add_basement is True:
            self.surfaces.add_surface(['basement'])
            self.map_series_to_surfaces({'Basement': 'basement'}, set_series=True)
        self.rescaling.rescale_data()

        self.additional_data.update_structure()
        self.additional_data.update_default_kriging()
        return True

    def get_data(self, itype='data', numeric=False):
        """
        Method that returns the surface_points and orientations pandas Dataframes. Can return both at the same time or only
        one of the two

        Args:
            itype: input_data data type, either 'orientations', 'surface_points' or 'all' for both.
            numeric(bool): Return only the numerical values of the dataframe. This is much lighter database for storing
                traces
            verbosity (int): Number of properties shown

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # TODO adapt this

        show_par_f = self.orientations.df.columns
        show_par_i = self.surface_points.df.columns

        if numeric:
            show_par_f = self.orientations._columns_o_num
            show_par_i = self.surface_points._columns_i_num
            dtype = 'float'

        if itype == 'orientations':
            raw_data = self.orientations.df[show_par_f]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'surface_points':
            raw_data = self.surface_points.df[show_par_i]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'data':
            raw_data = pn.concat([self.surface_points.df[show_par_i],  # .astype(dtype),
                                  self.orientations.df[show_par_f]],  # .astype(dtype)],
                                 keys=['surface_points', 'orientations'],
                                 sort=False)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]

        elif itype == 'surfaces':
            raw_data = self.surfaces
        elif itype == 'series':
            raw_data = self.series
        elif itype == 'faults':
            raw_data = self.faults
        elif itype == 'faults_relations_df' or itype == 'faults_relations':
            raw_data = self.faults.faults_relations_df
        elif itype == 'additional data' or itype == 'additional_data':
            raw_data = self.additional_data
        elif itype == 'kriging':
            raw_data = self.additional_data.kriging_data
        else:
            raise AttributeError('itype has to be \'data\', \'additional data\', \'surface_points\', \'orientations\','
                                 ' \'surfaces\',\'series\', \'faults\' or \'faults_relations_df\'')

        return raw_data

    def get_additional_data(self):
        return self.additional_data.get_additional_data()

    @setdoc_pro([ds.compile_theano, ds.theano_optimizer])
    def set_gravity_interpolator(self, density_block=None,
                                 pos_density=None, tz=None, compile_theano: bool = True,
                                 theano_optimizer=None, verbose: list = None):
        """
        Method to create a graph and compile the theano code to compute forward gravity.

        Args:
            density_block (Optional[np.array]): numpy array of the size of the grid.values with the correspondent
             density to each of the voxels. If it is not passed the density block will be also computed at run time but you will need
             to specify with value of the Surface object is density.
            pos_density (Optional[int]): Only necessary when density block is not passed. Location on the Surfaces().df
             where density is located (starting on id being 0). TODO allow the user to pass the name of the column.
            tz (Optional[np.array]): Numpy array of the size of grid.values with each component z of the vector
             device-voxel. In None is passed it will be automatically computed on the self.grid.centered grid
            compile_theano (bool): [s0]
            theano_optimizer (str {'fast_run', 'fast_compile'}): [s1]
            verbose (list):

        Returns:
            :class:`Options`
        """

        assert self.grid.centered_grid is not None, 'First you need to set up a gravity grid to compile the graph'
        assert density_block is not None or pos_density is not None, 'If you do not pass the density block you need to'\
                                                                     ' pass the position of surface values where' \
                                                                     ' density is assigned'
        # TODO output is dep
        if theano_optimizer is not None:
            self.additional_data.options.df.at['values', 'theano_optimizer'] = theano_optimizer
        if verbose is not None:
            self.additional_data.options.df.at['values', 'verbosity'] = verbose

        self.interpolator_gravity = InterpolatorGravity(
            self.surface_points, self.orientations, self.grid, self.surfaces,
            self.series, self.faults, self.additional_data)

        # geo_model.interpolator.set_theano_graph(geo_model.interpolator.create_theano_graph())
        self.interpolator_gravity.create_theano_graph(self.additional_data, inplace=True)

        # set shared variables
        self.interpolator_gravity.set_theano_shared_tz_kernel(tz)
        self.interpolator_gravity.set_all_shared_parameters(reset_ctrl=True)

        if compile_theano is True:
            self.interpolator_gravity.compile_th_fn(density_block, pos_density, inplace=True)

        return self.additional_data.options
