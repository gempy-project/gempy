import os
import sys
from os import path
import numpy as np
import pandas as pn
from typing import Union
import warnings

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
pn.options.mode.chained_assignment = None
from .data import AdditionalData, Faults, Grid, MetaData, Orientations, RescaledData, Series, SurfacePoints,\
    Surfaces, Options, Structure, KrigingParameters
from .solution import Solution
from .interpolator import InterpolatorModel, InterpolatorGravity
from gempy.utils.meta import _setdoc
from gempy.plot.decorators import *


class DataMutation(object):
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
        return self.additional_data.structure_data
    # region Grid

    def set_grid_object(self, grid: Grid, update_model=True):
        # TODO this should go to the api and let call all different grid types
        raise NotImplementedError

    def set_regular_grid(self, extent, resolution):
        self.grid.set_regular_grid(extent=extent, resolution=resolution)
        self.rescaling.rescale_data()
        self.interpolator.set_initial_results_matrices()
        return self.grid

    def set_custom_grid(self, custom_grid):
        self.grid.set_custom_grid(custom_grid)
        self.rescaling.set_rescaled_grid()
        self.interpolator.set_initial_results_matrices()
        return self.grid
    # endregion

    # region Series
    def set_series_object(self):
        """
        Not implemented yet. Exchange the series object of the Model object
        Returns:

        """
        raise NotImplementedError

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
        return self.series

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
        return self.series

    def rename_series(self, new_categories: Union[dict, list]):
        self.series.rename_series(new_categories)
        self.surfaces.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.surface_points.df['series'].cat.rename_categories(new_categories, inplace=True)
        self.orientations.df['series'].cat.rename_categories(new_categories, inplace=True)
        return self.series

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
        return self.series

    def reorder_series(self, new_categories: Union[list, np.ndarray]):
        self.series.reorder_series(new_categories)
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
        return self.series

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
        self.update_structure(update_theano='matrices')
        return self.faults

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
        return self.faults.faults_relations_df

    # endregion

    # region Surfaces
    def set_surfaces_object(self):
        """
        Not implemented yet. Exchange the surface object of the Model object
        Returns:

        """
        raise NotImplementedError

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
        return self.surfaces

    def delete_surface_values(self, properties_names: list):
        self.delete_surface_values(properties_names)
        return self.surfaces

    def modify_surface_values(self, idx, properties_names, values):
        self.surfaces.modify_surface_values(idx, properties_names, values)
        return self.surfaces

    def set_surface_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        self.surfaces.set_surfaces_values(values_array, properties_names)
        return self.surfaces

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

    @plot_add_surface_points
    def add_surface_points(self, X, Y, Z, surface, idx: Union[int, list, np.ndarray] = None,
                           recompute_rescale_factor=False):
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

        return self.surface_points, idx

    @plot_delete_surface_points
    def delete_surface_points(self, indices: Union[list, int]):
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
            self.interpolator.set_theano_shared_kriging()
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
        raise NotImplementedError

    @plot_add_orientation
    def add_orientations(self,  X, Y, Z, surface, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, idx=None, recompute_rescale_factor=False):

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

        return self.orientations, idx

    @plot_delete_orientations
    def delete_orientations(self, indices: Union[list, int]):
        self.orientations.del_orientation(indices)
        self.update_structure(update_theano='weights')
        return self.orientations

    @plot_move_orientations
    def modify_orientations(self, indices: list, **kwargs):

        indices = np.array(indices, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        self.orientations.modify_orientations(indices, **kwargs)
        self.rescaling.set_rescaled_orientations(indices)

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
        self.interpolator.set_theano_shared_kriging()
        if property == 'drift equations':
            self.interpolator.set_initial_results()

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

    def set_default_surface_point(self, **kwargs):
        if self.surface_points.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0],
                                    recompute_rescale_factor=True, **kwargs)

    def set_default_orientation(self, **kwargs):
        if self.orientations.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.add_orientations(.00001, .00001, .00001,
                                  self.surfaces.df['surface'].iloc[0],
                                  [0, 0, 1], recompute_rescale_factor=True, **kwargs)

    def set_default_surfaces(self):
        if self.surfaces.df.shape[0] == 0:
            self.add_surfaces(['surface1', 'surface2'])
        return self.surfaces

    def set_extent(self, extent: Union[list, np.ndarray]):
        extent = np.atleast_1d(extent)
        self.grid.extent = extent
        self.rescaling.set_rescaled_grid()

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
            self.interpolator.set_theano_shared_structure(reset=True)

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

    # region Theano interface
    def set_theano_graph(self, interpolator: InterpolatorModel):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.update_to_interpolator()

    def set_theano_function(self, interpolator: InterpolatorModel):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.interpolator.set_all_shared_parameters()
        self.update_structure(update_theano='matrices')

    def update_to_interpolator(self, reset=True):
        self.interpolator.set_all_shared_parameters()
        if reset is True:
            self.interpolator.reset_flow_control_initial_results()

    # endregion

    def map_data_df(self, d:pn.DataFrame):
        d['series'] = d['surface'].map(self.surfaces.df.set_index('surface')['series'])
        d['id'] = d['surface'].map(self.surfaces.df.set_index('surface')['id'])
        d['order_series'] = d['series'].map(self.series.df['order_series'])

    @plot_set_topography
    def set_topography(self, source='random', **kwargs):
        #
        """
        Args:
            mode: 'random': random topography is generated (based on a fractal grid).
                   'gdal': filepath must be provided to load topography from a raster file.
            filepath: path to raster file
            kwargs: only when mode is 'random', gp.utils.create_topography.Load_DEM_artificial kwargs:
                fd: fractal dimension, defaults to 2.0
                d_z: height difference. If none, last 20% of the model in z direction
                extent: extent in xy direction. If none, geo_model.grid.extent
                resolution: resolution of the topography array. If none, geo_model.grid.resoution

        Returns: :class:gempy.core.data.Topography
        """

        self.grid.set_topography(source, **kwargs)
        self.rescaling.set_rescaled_grid()
        self.interpolator.set_initial_results_matrices()

    def set_surface_order_from_solution(self):
        """
        Order the surfaces respect the last computation. Therefore if you call this method,
        after sorting surface_points without recomputing you may get wrong results
        Returns:

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


@_setdoc([MetaData.__doc__, Grid.__doc__])
class Model(DataMutation):
    """
    Container class of all objects that constitute a GemPy model. In addition the class provides the methods that
    act in more than one of this class.
    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        super().__init__()
        self.interpolator_gravity = None

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
        np.save(f'{path}/{name}_extent.npy', self.grid.extent)
        np.save(f'{path}/{name}_resolution.npy', self.grid.regular_grid.resolution)

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
            self.map_series_to_surfaces({'Basement': 'basement'}, set_series=True)
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

    def set_gravity_interpolator(self, density_block= None, pos_density=None, inplace=True, compile_theano: bool = True,
                                 theano_optimizer=None, verbose: list = None):
        """

        Args:
            geo_model:
            inplace:
            compile_theano

        Returns:

        """
        assert self.grid.gravity_grid is not None, 'First you need to set up a gravity grid to compile the graph'
        assert density_block is not None or pos_density is not None, 'If you do not pass the density block you need to pass' \
                                                                     'the position of surface values where density is' \
                                                                     ' assigned'

        # TODO Possibly this is only necessary when computing gravity
        self.grid.active_grids = np.zeros(4, dtype=bool)
        self.grid.set_active('gravity')
        self.interpolator.set_initial_results_matrices()

        # TODO output is dep
        if theano_optimizer is not None:
            self.additional_data.options.df.at['values', 'theano_optimizer'] = theano_optimizer
        if verbose is not None:
            self.additional_data.options.df.at['values', 'verbosity'] = verbose

        # TODO add kwargs
        self.rescaling.rescale_data()
        self.update_structure()

        # This two should be unnecessary now too
        self.surface_points.sort_table()
        self.orientations.sort_table()

        self.interpolator_gravity = InterpolatorGravity(
            self.surface_points, self.orientations, self.grid, self.surfaces,
            self.series, self.faults, self.additional_data)

        # geo_model.interpolator.set_theano_graph(geo_model.interpolator.create_theano_graph())
        self.interpolator_gravity.create_theano_graph(self.additional_data, inplace=True)

        # set shared variables
        self.interpolator_gravity.set_theano_shared_tz_kernel()
        self.interpolator_gravity.set_all_shared_parameters(reset=True)

        if compile_theano is True:
            self.interpolator_gravity.compile_th_fn(density_block, pos_density, inplace=inplace)

        return self.additional_data.options





























