import os
import sys
from os import path
import numpy as np
import pandas as pn
from typing import Union
import warnings

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import pandas as pn
pn.options.mode.chained_assignment = None
from .data import AdditionalData, Faults, Grid, MetaData, Orientations, RescaledData, Series, SurfacePoints,\
    Surfaces, Topography, Options, Structure, KrigingParameters
from .solution import Solution
from .interpolator import Interpolator
from .interpolator_pro import InterpolatorModel
from gempy.utils.meta import _setdoc
from gempy.plot.visualization_3d import vtkVisualization
from gempy.plot.decorators import *

class DataMutation_pro(object):
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

        self.solutions = Solution(self.additional_data, self.grid, self.surface_points, self.series, self.surfaces)

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

    def update_structure(self, update_theano=None):
        """Update python and theano structure paramteres
        Args:
            update_theano: str['matrices', 'weights']
        """

        self.additional_data.update_structure()
        if update_theano == 'matrices':
            self.interpolator.modify_results_matrices_pro()
        elif update_theano == 'weights':
            self.interpolator.modify_results_weights()
        self.interpolator.set_theano_shared_structure()

    # region Grid

    def set_grid_object(self, grid: Grid, update_model=True):
        pass
        # self.grid = grid
        # self.additional_data.grid = grid
        # self.rescaling.grid = grid
        # self.interpolator.grid = grid
        # self.solutions.grid = grid
        #
        # if update_model is True:
        #     self.update_from_grid()

    def set_regular_grid(self, extent, resolution):
        self.grid.set_regular_grid(extent, resolution)
        self.rescaling.rescale_data()
        self.interpolator.set_initial_results_matrices()

    def set_custom_grid(self):
        pass

    # endregion

    # region Series
    def set_series_object(self):
        """
        Not implemented yet. Exchange the series object of the Model object
        Returns:

        """
        pass

    def set_bottom_relation(self, series: Union[str, list], bottom_relation: Union[str, list]):
        self.series.set_bottom_relation(series, bottom_relation)
        self.interpolator.set_theano_shared_relations()
        return self.series

    def add_series(self, series_list: Union[str, list], update_order_series=True, **kwargs):
        self.series.add_series(series_list, update_order_series)
        self.surfaces.df['series'].cat.add_categories(series_list, inplace=True)
        self.surface_points.df['series'].cat.add_categories(series_list, inplace=True)
        self.orientations.df['series'].cat.add_categories(series_list, inplace=True)

        self.interpolator.set_flow_control()
        # self.interpolator.compute_weights_ctrl = np.append(self.interpolator.compute_weights_ctrl,
        #                                                    np.ones(series_list.shape[0]))
        # self.interpolator.compute_scalar_ctrl = np.append(self.interpolator.compute_scalar_ctrl,
        #                                                   np.ones(series_list.shape[0]))
        # self.interpolator.compute_block_ctrl = np.append(self.interpolator.compute_block_ctrl,
        #                                                  np.ones(series_list.shape[0]))
        #self.update_from_series()

    def delete_series(self, indices: Union[str, list], update_order_series=True):
        """

        Args:
            indices: name of the series
            update_order_series:

        Returns:

        """
        self.series.delete_series(indices, update_order_series)
        self.surfaces.df['series'].cat.remove_categories(indices, inplace=True)
        self.surface_points.df['series'].cat.remove_categories(indices, inplace=True)
        self.orientations.df['series'].cat.remove_categories(indices, inplace=True)
        self.map_data_df(self.surface_points.df)
        self.map_data_df(self.orientations.df)

        self.interpolator.set_theano_shared_relations()
        self.interpolator.set_flow_control()
       # self.update_from_series()

    def rename_series(self, new_categories: Union[dict, list]):
        self.series.rename_series(new_categories)
        self.surfaces.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.surface_points.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.orientations.df['series'].cat.rename_categories(new_categories, inplace=True)

    def modify_order_series(self, new_value: int, idx: str):
        self.series.modify_order_series(new_value, idx)

        self.surfaces.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
                                                          ordered=False, inplace=True)

        self.surfaces.sort_surfaces()
        self.surfaces.set_basement()

        self.map_data_df(self.surface_points.df)
        self.surface_points.sort_table()
        self.map_data_df(self.orientations.df)
        self.orientations.sort_table()

        self.interpolator.set_flow_control()
        self.update_structure()

    # endregion

    # region Faults
    def set_fault_object(self):
        pass

    @_setdoc([Faults.set_is_fault.__doc__])
    def set_is_fault(self, series_fault: Union[str, list] = None, toggle: bool = False, change_color: bool = True):
        series_fault = np.atleast_1d(series_fault)

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
        return self.faults
        # for series_as_faults in np.atleast_1d(series_fault):

        # TODO: Decide if this makes sense anymore
        # This code is to push faults up the pile
        # if self.faults.df.loc[series_fault[0], 'isFault'] is True:
        #     self.series.modify_order_series(self.faults.n_faults, series_as_faults)
        #     print('Fault series: ' + str(series_fault) + ' moved to the top of the surfaces.')
        # else:
        #     self.series.modify_order_series(self.faults.n_faults + 1, series_as_faults)
        #     print('Fault series: ' + str(series_fault) + ' moved to the top of the pile.')
        #
        # self.faults.set_is_fault(series_fault)
        # self.interpolator.set_theano_shared_relations()
        # # TODO this update from series is alsod related to the move in the pile
        # self.update_from_series()

    @_setdoc([Faults.set_is_fault.__doc__])
    def set_is_finite_fault(self, series_fault=None, toggle: bool = False):
        s = self.faults.set_is_finite_fault(series_fault, toggle)  # change df in Fault obj
        # change shared theano variable for infinite factor
        self.interpolator.set_theano_shared_is_finite()
        return s

    def set_fault_relation(self, rel_matrix):
        self.faults.set_fault_relation(rel_matrix)

        # Updating
        self.interpolator.set_theano_shared_fault_relation()
        self.interpolator.set_theano_shared_weights()

    # endregion

    # region Surfaces
    def set_surfaces_object(self):
        """
        Not implemented yet. Exchange the surface object of the Model object
        Returns:

        """

    def add_surfaces(self, surface_list: Union[str, list], update_df=True):
        self.surfaces.add_surface(surface_list, update_df)
        self.surface_points.df['surface'].cat.add_categories(surface_list, inplace=True)
        self.orientations.df['surface'].cat.add_categories(surface_list, inplace=True)
        self.update_structure()
        return self.surfaces

    def delete_surfaces(self, indices: Union[str, list, np.ndarray], update_id=True):
        indices = np.atleast_1d(indices)
        self.surfaces.delete_surface(indices, update_id)

        if indices.dtype == int:
            surfaces_names = self.surfaces.df.loc[indices, 'surface']
        else:
            surfaces_names = indices

        self.surface_points.df['surface'].cat.remove_categories(surfaces_names, inplace=True)
        self.orientations.df['surface'].cat.remove_categories(surfaces_names, inplace=True)
        self.map_data_df(self.surface_points.df)
        self.map_data_df(self.orientations.df)
        return self.surfaces

    def rename_surfaces(self, to_replace: Union[dict], **kwargs):

        self.surfaces.rename_surfaces(to_replace, **kwargs)
        self.surface_points.df['surface'].cat.rename_categories(to_replace, inplace=True)
        self.orientations.df['surface'].cat.rename_categories(to_replace, inplace=True)
        return self.surfaces

    def modify_order_surfaces(self, new_value: int, idx: int, series: str = None):
        """

        Args:
            new_value : New position of the surface
            idx: index of the surface. The surface should be unique all the time since the creation of the surface
            series:

        Returns:

        """
        self.surfaces.modify_order_surfaces(new_value, idx, series)

        self.map_data_df(self.surface_points.df)
        self.surface_points.sort_table()
        self.map_data_df(self.orientations.df)
        self.orientations.sort_table()

        self.update_structure()
        return self.surfaces

    def add_surface_values(self,  values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        self.surfaces.add_surfaces_values(values_array, properties_names)

    def delete_surface_values(self, properties_names: list):
        self.delete_surface_values(properties_names)

    def modify_surface_values(self, idx, properties_names, values):
        self.surfaces.modify_surface_values(idx, properties_names, values)

    def set_surface_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        self.surfaces.set_surfaces_values(values_array, properties_names)

    @_setdoc([Surfaces.map_series.__doc__])
    def map_series_to_surfaces(self, mapping_object: Union[dict, pn.Categorical] = None,
                               set_series=True, sort_geometric_data: bool = True, remove_unused_series=True):
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

        self.surfaces.update_sequential_pile()
        self.series.update_order_series()
        # self.surfaces.sort_surfaces()

        self.update_from_surfaces()
        self.update_from_series()

        if sort_geometric_data is True:
            self.surface_points.sort_table()
            self.orientations.sort_table()

        return self.surfaces.sequential_pile.figure

    # endregion

    # region Surface_points

    def set_surface_points_object(self, surface_points: SurfacePoints, update_model=True):
        pass
        # self.surface_points = surface_points
        # self.rescaling.surface_points = surface_points
        # self.interpolator.surface_points = surface_points
        #
        # if update_model is True:
        #     self.update_from_surface_points()

    @plot_add_surface_points
    def add_surface_points(self, X, Y, Z, surface, idx: Union[int, list, np.ndarray] = None,
                           recompute_rescale_factor=False, **kwargs):
        surface = np.atleast_1d(surface)
        idx = self._add_valid_idx_s(idx)
        self.surface_points.add_surface_points(X, Y, Z, surface, idx)

        if recompute_rescale_factor is True or idx < 20:
            # This will rescale all data again
            self.rescaling.rescale_data()
        else:
            # This branch only recompute the added point
            self.rescaling.set_rescaled_surface_points(idx)

        print(surface, surface.ndim)
        # Add results has to be called before we update the theano len_series_i
        # if surface.ndim == 1:
        #     self.interpolator.add_to_results(surface)
        # else:
        #     self.interpolator.set_initial_results()
        #self.interpolator.modify_results_matrices_pro()
        self.update_structure(update_theano='matrices')

        return self.surface_points, idx

    @plot_delete_surface_points
    def delete_surface_points(self, indices: Union[list, int], **kwargs):
        self.surface_points.del_surface_points(indices)
        self.update_structure(update_theano='matrices')
        return self.surface_points

    @plot_move_surface_points
    def modify_surface_points(self, indices: Union[int, list], recompute_rescale_factor=False, **kwargs):
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface:
            assert (~self.surfaces.df[self.surfaces.df['isBasement']]['surface'].isin(
                np.atleast_1d(kwargs['surface']))).any(),\
                'Surface points cannot belong to Basement. Add a new surface.'

        self.surface_points.modify_surface_points(indices, **kwargs)

        if recompute_rescale_factor is True or np.atleast_1d(indices)[0] < 20:
            # This will rescale all data again
            self.rescaling.rescale_data()
        else:
            # This branch only recompute the added point
            self.rescaling.set_rescaled_surface_points(indices)

        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface == True:
            self.update_structure(update_theano='matrices')
        return self.surface_points

    # endregion

    # region Orientation
    def set_orientations_object(self, orientations: Orientations, update_model=True):
        pass
        # self.orientations = orientations
        # self.rescaling.orientations = orientations
        # self.interpolator.orientations = orientations
        #
        # if update_model is True:
        #     self.update_from_orientations()

    @plot_add_orientation
    def add_orientations(self,  X, Y, Z, surface, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, idx=None, recompute_rescale_factor=False,
                         **kwargs):

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

        #self.interpolator.add_to_weights(surface, n_rows=3)
        #self.interpolator.modify_results_weights()
        self.update_structure(update_theano='weights')

        return self.orientations, idx

    @plot_delete_orientations
    def delete_orientations(self, indices: Union[list, int], **kwargs ):
        self.orientations.del_orientation(indices)
        self.update_structure(update_theano='weights')
        return self.orientations

    @plot_move_orientations
    def modify_orientations(self, indices: list, **kwargs):

        indices = np.array(indices, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        self.orientations.modify_orientations(indices, **kwargs)

        if is_surface:
            self.update_structure(update_theano='weights')

        return self.orientations
    # endregion

    # region Options
    def modify_options(self, property, value):
        self.additional_data.options.modify_options(property, value)
        warnings.warn('You need to recompile the Theano code to make it the changes in options.')

    # endregion

    # region Kriging
    def modify_kriging_parameters(self, property, value, **kwargs):
        self.additional_data.kriging_data.modify_kriging_parameters(property, value, **kwargs)

    # endregion

    # region rescaling
    def modify_rescaling_parameters(self, property, value):
        self.additional_data.rescaling_data.modify_rescaling_parameters(property, value)
        self.additional_data.rescaling_data.rescale_data()
        self.additional_data.update_default_kriging()
    # endregion

    # ======================================
    # --------------------------------------
    # ======================================

    def set_default_surface_point(self):
        if self.surface_points.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0],
                                    recompute_rescale_factor=True)

    def set_default_orientation(self):
        if self.orientations.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.add_orientations(.00001, .00001, .00001,
                                  self.surfaces.df['surface'].iloc[0],
                                  [0, 0, 1], recompute_rescale_factor=True)

    def set_default_surfaces(self):
        self.add_surfaces(['surface1', 'surface2'])
        # self.surfaces.set_default_surface_name()
        # self.update_from_surfaces()
        return self.surfaces


    # def update_from_grid(self):
    #     """
    #
    #     Note: update_from_grid does not have the inverse, i.e. update_to_grid, because GridClass is independent
    #     Returns:
    #
    #     """
    #     self.additional_data.update_default_kriging()  # TODO decide if this makes sense here. Probably is better to do
    #     #  it with a checker
    #     self.rescaling.set_rescaled_grid()
    #     # self.interpolator.set_theano_share_input()
    #
    #
    #
    def update_from_series(self, rename_series: dict = None, reorder_series=True, sort_geometric_data=True,
                           update_interpolator=True):
        """
        Note: update_from_series does not have the inverse, i.e. update_to_series, because Series is independent
        Returns:

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
            # self.surface_points.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
            #                                                   ordered=False, inplace=True)
            # self.orientations.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
            #                                                   ordered=False, inplace=True)

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
        # For the drift equations. TODO disentagle this property
        self.additional_data.update_default_kriging()

        if update_interpolator is True:
            self.interpolator.set_theano_shared_structure(reset=True)
    #
    #
    #
    def update_from_surfaces(self, set_categories_from_series=True, set_categories_from_surfaces=True,
                             map_surface_points=True, map_orientations=True, update_structural_data=True):
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
    #
    # def update_to_surfaces(self):
    #     # TODO decide if makes sense. I think it is quite independent as well. The only thing would be the categories of
    #     #   series?
    #     pass
    #
    # def update_from_faults(self):
    #     self.interpolator.set_theano_shared_faults()
    #
    #
    #
    # def update_to_orientations(self, idx: Union[list, np.ndarray] = None):
    #     # TODO debug
    #     if idx is None:
    #         idx = self.orientations.df.index
    #     idx = np.atleast_1d(idx)
    #     self.orientations.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
    #     self.orientations.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
    #     self.orientations.map_data_from_series(self.series, 'order_series', idx=idx)
    #     self.orientations.sort_table()
    #     return self.orientations
    #
    #
    # def update_from_orientations(self, idx: Union[list, np.ndarray] = None,  recompute_rescale_factor=False):
    #     # TODO debug
    #
    #     self.update_structure()
    #     if recompute_rescale_factor is False:
    #         self.rescaling.set_rescaled_orientations(idx=idx)
    #     else:
    #         self.rescaling.rescale_data()


    # region Theano interface
    def set_theano_graph(self, interpolator: Interpolator):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.update_to_interpolator()


    def set_theano_function(self, interpolator: Interpolator):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.interpolator.set_all_shared_parameters()

    def update_to_interpolator(self, reset=True):
        self.interpolator.set_all_shared_parameters()
        self.interpolator.reset_flow_control_initial_results()

    # endregion

    def map_data_df(self, d:pn.DataFrame):
        d['series'] = d['surface'].map(self.surfaces.df.set_index('surface')['series'])
        d['id'] = d['surface'].map(self.surfaces.df.set_index('surface')['id'])
        d['order_series'] = d['series'].map(self.series.df['order_series'])
      #  d['isFault'] = d['series'].map(self.faults.df['isFault'])

    def update_from_additional_data(self):
        pass

    def load_topography(self, source = 'random', filepath = None, **kwargs):
        """

        Args:
            mode: 'random': random topography is generated
                   'gdal'. filepath must be provided
            filepath: filepath to a raster file
            kwargs: gp.utils.create_topography.Load_DEM_artificial kwargs: z_ext, resolution

        Returns: :class:gempy.core.data.Topography

        """
        self.topography = Topography(self)
        if source == 'random':
            self.topography.load_random_hills(**kwargs)
        elif source == 'gdal':
            if filepath is not None:
                self.topography.load_from_gdal(filepath)
            else:
                print('to load a raster file, a path to the file must be provided')
        else:
            print('source must be either random or gdal')
        self.grid.mask_topo = self.topography._create_grid_mask()
        self.topography.show()

@_setdoc([MetaData.__doc__, Grid.__doc__])
class Model(DataMutation_pro):
    """
    Container class of all objects that constitute a GemPy model. In addition the class provides the methods that
    act in more than one of this class.
    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        super().__init__()
        self.topography = None

    def __repr__(self):
        return self.meta.project_name + ' ' + self.meta.date

    def new_model(self, name_project='default_project'):
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
            # TODO: Update default to meta name
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
        # TODO: UPDATE!!!!!!! TO new solution
        """
        Save model in new folder. Input data is saved as csv files. Solutions, extent and resolutions are saved as npy.

        Args:
            name (str): name of the newly created folder and the part of the files name
            path (str): path where save the model folder.

        Returns:
            True
        """
        if not path:
            path = './'
        path = f'{path}/{name}'

        if name is None:
            name = self.meta.project_name

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
        # np.save(f'{path}/{name}_extent.npy', self.grid.extent)
        # np.save(f'{path}/{name}_resolution.npy', self.grid.resolution)
        #
        # # save solutions as npy
        # np.save(f'{path}/{name}_lith_block.npy' ,self.solutions.lith_block)
        # np.save(f'{path}/{name}_scalar_field_lith.npy', self.solutions.scalar_field_matrix)
        #
        # np.save(f'{path}/{name}_gradient.npy', self.solutions.gradient)
        # np.save(f'{path}/{name}_values_block.npy', self.solutions.matr)

        return True

    @_setdoc([SurfacePoints.read_surface_points.__doc__, Orientations.read_orientations.__doc__])
    def read_data(self, path_i=None, path_o=None, add_basement=True, **kwargs):
        """

        Args:
            path_i:
            path_o:
            **kwargs:
                update_surfaces (bool): True

        Returns:

        """
        if 'update_surfaces' not in kwargs:
            kwargs['update_surfaces'] = True

        if path_i:
            self.surface_points.read_surface_points(path_i, inplace=True, **kwargs)
        if path_o:
            self.orientations.read_orientations(path_o, inplace=True, **kwargs)
        if add_basement is True:
            self.surfaces.add_surface(['basement'])

        self.rescaling.rescale_data()

        self.additional_data.update_structure()
        self.additional_data.update_default_kriging()

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
        # dtype = 'object'
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
        else:
            raise AttributeError('itype has to be \'data\', \'additional data\', \'surface_points\', \'orientations\','
                                 ' \'surfaces\',\'series\', \'faults\' or \'faults_relations_df\'')

        return raw_data

    def get_additional_data(self):
        return self.additional_data.get_additional_data()

    # def get_theano_input(self):
    #     pass





























