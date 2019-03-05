import os
import sys
from os import path

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import pandas as pn
pn.options.mode.chained_assignment = None
from .data import *
from .interpolator import Interpolator
from gempy.utils.meta import _setdoc
from gempy.plot.plot import vtkPlot


class DataMutation(object):
    # TODO Add dummy input when a df is empty
    def __init__(self):

        self.grid = Grid()
        self.faults = Faults()
        self.series = Series(self.faults)
        self.formations = Formations(self.series)
        self.interfaces = Interfaces(self.formations)
        self.orientations = Orientations(self.formations)

        self.rescaling = RescaledData(self.interfaces, self.orientations, self.grid)
        self.additional_data = AdditionalData(self.interfaces, self.orientations, self.grid, self.faults,
                                              self.formations, self.rescaling)

    @_setdoc([Interfaces.read_interfaces.__doc__, Orientations.read_orientations.__doc__])
    def read_data(self, path_i=None, path_o=None, **kwargs):
        """

        Args:
            path_i:
            path_o:
            **kwargs:
                update_formations (bool): True

        Returns:

        """
        if 'update_formations' not in kwargs:
            kwargs['update_formations'] = True

        if path_i:
            self.interfaces.read_interfaces(path_i, inplace=True, **kwargs)
        if path_o:
            self.orientations.read_orientations(path_o, inplace=True, **kwargs)

        self.rescaling.rescale_data()

        self.additional_data.update_structure()
       # self.additional_data.update_rescaling_data()
        self.additional_data.update_default_kriging()

    @_setdoc([Formations.map_series.__doc__])
    def map_series_to_formations(self, mapping_object: Union[dict, pn.Categorical] = None,
                                 set_series=True, sort_data: bool = True):
        # TODO: decide if this method should just go to the api
        if set_series is True:
            if type(mapping_object) is dict:
                series_list = list(mapping_object.keys())
                self.series.add_series(series_list)
            elif isinstance(mapping_object, pn.Categorical):
                series_list = mapping_object['series'].values
                self.series.add_series(series_list)
            else:
                raise AttributeError(str(type(mapping_object)) + ' is not the right attribute type.')

        self.formations.map_series(mapping_object)
        self.formations.update_sequential_pile()
        self.update_from_formations()
        self.update_from_series()

        if sort_data is True:
            self.interfaces.sort_table()
            self.orientations.sort_table()

        return self.formations.sequential_pile.figure

    # ======================================
    # --------------------------------------
    # ======================================

    def set_grid_object(self, grid: Grid, update_model=True):
        self.grid = grid
        self.additional_data.grid = grid
        self.rescaling.grid = grid
        self.interpolator.grid = grid
        self.solutions.grid = grid

        if update_model is True:
            self.update_from_grid()

    def set_regular_grid(self, extent, resolution):
        self.grid.set_regular_grid(extent, resolution)
        self.update_from_grid()

    def set_default_interface(self):
        self.interfaces.set_default_interface()
        self.update_to_interfaces()
        self.update_from_interfaces()

    def set_default_orientation(self):
        self.orientations.set_default_orientation()
        self.update_to_orientations()
        self.update_from_orientations()

    def set_default_formations(self):
        self.formations.set_default_formation_name()
        self.update_from_formations()

    def update_from_grid(self):
        """

        Note: update_from_grid does not have the inverse, i.e. update_to_grid, because GridClass is independent
        Returns:

        """
        self.additional_data.update_default_kriging()  # TODO decide if this makes sense here. Probably is better to do
        #  it with a checker
        self.rescaling.set_rescaled_grid()
        self.interpolator.set_theano_share_input()

    def set_series_object(self):
        """
        Not implemented yet. Exchange the series object of the Model object
        Returns:

        """
        pass

    def update_from_series(self, rename_series: dict = None, reorder_series=True, sort_geometric_data=True):
        """
        Note: update_from_series does not have the inverse, i.e. update_to_series, because Series is independent
        Returns:

        """
        # Add categories from series to formation
        # Updating formations['series'] categories
        if rename_series is None:
            self.formations.df['series'].cat.set_categories(self.series.df.index, inplace=True)
        else:
            self.formations.df['series'].cat.rename_categories(rename_series)

        if reorder_series is True:
            self.formations.df['series'].cat.reorder_categories(self.series.df.index.get_values(),
                                                                ordered=False, inplace=True)
            self.formations.sort_formations()

        self.formations.set_basement()

        # Add categories from series
        self.interfaces.add_series_categories_from_series(self.series)
        self.orientations.add_series_categories_from_series(self.series)

        self.interfaces.map_data_from_series(self.series, 'order_series')
        self.orientations.map_data_from_series(self.series, 'order_series')

        if sort_geometric_data is True:
            self.interfaces.sort_table()
            self.orientations.sort_table()

        self.additional_data.update_structure()
        # For the drift equations. TODO disentagle this property
        self.additional_data.update_default_kriging()


    def set_formations_object(self):
        """
        Not implemented yet. Exchange the formation object of the Model object
        Returns:

        """

    def update_from_formations(self, add_categories_from_series=True, add_categories_from_formations=True,
                               map_interfaces=True, map_orientations=True, update_structural_data=True):
        # Add categories from series
        if add_categories_from_series is True:
            self.interfaces.add_series_categories_from_series(self.formations.series)
            self.orientations.add_series_categories_from_series(self.formations.series)

        # Add categories from formations
        if add_categories_from_formations is True:
            self.interfaces.add_surface_categories_from_formations(self.formations)
            self.orientations.add_surface_categories_from_formations(self.formations)

        if map_interfaces is True:
            self.interfaces.map_data_from_formations(self.formations, 'series')
            self.interfaces.map_data_from_formations(self.formations, 'id')

        if map_orientations is True:
            self.orientations.map_data_from_formations(self.formations, 'series')
            self.orientations.map_data_from_formations(self.formations, 'id')

        if update_structural_data is True:
            self.additional_data.update_structure()

    def update_to_formations(self):
        # TODO decide if makes sense. I think it is quite independent as well. The only thing would be the categories of
        #   series?
        pass

    @_setdoc([Faults.set_is_fault.__doc__])
    def set_is_fault(self, series_fault=None):
        for series_as_faults in np.atleast_1d(series_fault):
            if self.faults.df.loc[series_fault[0], 'isFault'] == True:
                self.series.modify_order_series(self.faults.n_faults, series_as_faults)
            else:
                self.series.modify_order_series(self.faults.n_faults + 1, series_as_faults)

            s = self.faults.set_is_fault(series_fault)
            print(s)
        self.update_from_series()
        return s

    @_setdoc([Faults.set_is_fault.__doc__])
    def set_is_finite_fault(self, series_fault=None):
        s = self.faults.set_is_finite_fault(series_fault)  # change df in Fault obj
        print(s)
        # change shared theano variable for infinite factor
        self.interpolator.set_theano_inf_factor()


    def set_interface_object(self, interfaces: Interfaces, update_model=True):
        self.interfaces = interfaces
        self.rescaling.interfaces = interfaces
        self.interpolator.interfaces = interfaces

        if update_model is True:
            self.update_from_interfaces()

    def update_to_interfaces(self, idx: Union[list, np.ndarray] = None):

        if idx is None:
            idx = self.interfaces.df.index
        idx = np.atleast_1d(idx)
        self.interfaces.map_data_from_formations(self.formations, 'series', idx=idx)
        self.interfaces.map_data_from_formations(self.formations, 'id', idx=idx)

        self.interfaces.map_data_from_series(self.series, 'order_series', idx=idx)
        self.interfaces.sort_table()
        return self.interfaces

    def update_to_orientations(self, idx: Union[list, np.ndarray] = None):
        # TODO debug
        if idx is None:
            idx = self.orientations.df.index
        idx = np.atleast_1d(idx)
        self.orientations.map_data_from_formations(self.formations, 'series', idx=idx)
        self.orientations.map_data_from_formations(self.formations, 'id', idx=idx)
        self.orientations.map_data_from_series(self.series, 'order_series', idx=idx)
        self.orientations.sort_table()
        return self.orientations

    def update_from_interfaces(self, idx: Union[list, np.ndarray] = None, recompute_rescale_factor=False):
        self.update_structure()
        if idx is None:
            idx = self.interfaces.df.index
        idx = np.atleast_1d(idx)

        if self.interfaces.df.loc[idx][['X_r', 'Y_r', 'Z_r']].isna().any().any():
            recompute_rescale_factor = True

        if recompute_rescale_factor is False:
            self.rescaling.set_rescaled_interfaces(idx=idx)
        else:
            self.rescaling.rescale_data()

    def set_orientations_object(self, orientations: Orientations, update_model=True):

        self.orientations = orientations
        self.rescaling.orientations = orientations
        self.interpolator.orientations = orientations

        if update_model is True:
            self.update_from_orientations()

    def update_from_orientations(self, idx: Union[list, np.ndarray] = None,  recompute_rescale_factor=False):
        # TODO debug

        self.update_structure()
        if recompute_rescale_factor is False:
            self.rescaling.set_rescaled_orientations(idx=idx)
        else:
            self.rescaling.rescale_data()

    def set_theano_graph(self, interpolator: Interpolator):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.update_to_interpolator()

    def set_theano_function(self, interpolator: Interpolator):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.interpolator.set_theano_shared_parameters()

    def map_data_df(self, d):
        d['series'] = d['formation'].map(self.formations.df.set_index('formation')['series'])
        d['id'] = d['formation'].map(self.formations.df.set_index('formation')['id'])
        d['order_series'] = d['series'].map(self.series.df['order_series'])
        d['isFault'] = d['series'].map(self.faults.df['isFault'])

    def update_structure(self):
        self.additional_data.update_structure()
        self.interpolator.set_theano_shared_structure()

    def update_plot(self, plot_object: Union[vtkPlot]):
        if plot_object is not None:
            if isinstance(plot_object, vtkPlot):
                if plot_object.vv.real_time is True:
                    plot_object.vv.update_surfaces_real_time()
                plot_object.vv.interactor.Render()

    # DEP
    # def modify_kriging_parameters(self, vtk_object: vtkPlot=None, **properties):
    #     d = pn.DataFrame(properties).T
    #     self.additional_data.kriging_data.loc[d.index, 'values'] = d
    #     self.update_plot(vtk_object)

    def add_interfaces_DEP(self, vtk_object: vtkPlot = None, **properties):

        d = pn.DataFrame(properties)
        d[['X_r', 'Y_r', 'Z_r']] = self.rescaling.rescale_data_point(d[['X', 'Y', 'Z']])
        try:
            self.map_data_df(d)
        except KeyError:
            pass

        for index, frame in d.iterrows():
            new_ind = self.interfaces.df.last_valid_index() + 1
            self.interfaces.df.loc[new_ind, d.columns] = frame

            if vtk_object is not None:
                vtk_object.render_add_interfaces(new_ind)

        self.interfaces.sort_table()
        self.update_structure()

    def add_interfaces(self, X, Y, Z, surface, idx=None):
        self.interfaces.add_interface(X, Y, Z, surface, idx)

        self.update_to_interfaces(idx)
        self.interfaces.sort_table()
        self.update_from_interfaces(idx, recompute_rescale_factor=True)

    def add_orientations(self,  X, Y, Z, surface, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, idx=None,
                         vtk_object: vtkPlot = None):
        self.orientations.add_orientation(X, Y, Z, surface, pole_vector=pole_vector,
                                          orientation=orientation, idx=idx)

        self.update_to_orientations(idx)
        self.orientations.sort_table()
        self.update_from_orientations(idx, recompute_rescale_factor=True)

        #TODO!!!!!! Update
        #
        # d = pn.DataFrame(properties)
        # d[['X_r', 'Y_r', 'Z_r']] = self.rescaling.rescale_data_point(d[['X', 'Y', 'Z']])
        # try:
        #     self.map_data_df(d)
        # except KeyError:
        #     pass
        #
        # for index, frame in d.iterrows():
        #     new_ind = self.orientations.df.last_valid_index() + 1
        #     self.orientations.df.loc[new_ind, d.columns] = frame
        #
        #     if vtk_object is not None:
        #         vtk_object.render_add_orientations(new_ind)
        #
        # self.orientations.sort_table()
        # _checker = 0
        #
        # if d.columns.isin(['G_x', "G_y", 'G_z']).sum() == 3:
        #     self.orientations.calculate_orientations()
        #     _checker += 1
        # elif d.columns.isin(['dip', 'azimuth', 'polarity']).sum() == 3:
        #     self.orientations.calculate_gradient()
        #     _checker += 1
        #     if _checker == 2:
        #         raise AttributeError(
        #             'add orientation only accept either orientation data [dip, azimuth, polarity] or'
        #             'gradient data [G_x, G_y, G_z]')
        # else:
        #     raise AttributeError(
        #         'Not enough angular data to calculate the gradients. Pass orientations or gradients')
        #
        # self.update_structure()

    def add_series(self, series_list: Union[pn.DataFrame, list], update_order_series=True, vtk_object: vtkPlot = None):
        self.series.add_series(series_list, update_order_series)
        self.update_from_series()

    def add_formations(self, formation_list: Union[pn.DataFrame, list], update_df=True):
        self.formations.add_formation(formation_list, update_df)
        self.update_from_formations()

    def delete_formations(self, indices, update_id=True):
        self.formations.delete_formation(indices, update_id)
        self.update_from_formations()

    def delete_series(self, indices, update_order_series=True):
        self.series.delete_series(indices, update_order_series)
        self.update_from_series()

    def delete_interfaces(self, indices: Union[list, int], vtk_object: vtkPlot = None):
        self.interfaces.del_interface(indices)
        if vtk_object is not None:
            vtk_object.render_delete_interfaes(indices)

        self.update_from_interfaces(indices)

    def delete_orientations(self, indices: Union[list, int], vtk_object: vtkPlot = None, ):
        self.orientations.del_orientation(indices)

        if vtk_object is not None:
            vtk_object.render_delete_orientations(indices)

        self.update_structure(indices)

    def modify_interfaces(self, indices: list, vtk_object: vtkPlot = None, **properties):
        indices = np.array(indices, ndmin=1)
        keys = list(properties.keys())
        is_formation = np.isin('surface', keys).all()
        self.interfaces.modify_interface(indices, **properties)

        if is_formation:
            self.update_to_interfaces(indices)
        self.update_from_interfaces(indices)

    def modify_orientations(self, indices: list, vtk_object: vtkPlot = None, **properties: list):

        indices = np.array(indices, ndmin=1)
        keys = list(properties.keys())
        is_formation = np.isin('surface', keys).all()
        self.orientations.modify_orientations(indices, **properties)

        if is_formation:
            self.update_to_orientations(indices)
        self.update_from_orientations(indices)

    def rename_formations(self, old, new):
        self.formations.rename_formations(old, new)
        self.update_from_formations()

    def reorder_formations(self, list_names):
        self.formations.reorder_formations(list_names)
        self.update_from_formations()

        #
        # xyz_check = ~np.isin(['X', 'Y', 'Z'], keys)
        # d = pn.DataFrame(properties, columns=np.append(np.array(['X', 'Y', 'Z'])[xyz_check], keys), index=indices)
        # is_formation = any(d.columns.isin(['formation']))
        #
        # if is_formation:
        #     self.map_data_df(d)
        #
        # # To be sure that we
        # xyz_exist = np.array(['X', 'Y', 'Z'])
        # xyz_res = np.array(['X_r', 'Y_r', 'Z_r'])
        # d[xyz_res] = self.rescaling.rescale_data_point(d[xyz_exist])
        # d.dropna(axis=1, inplace=True)
        #
        # assert indices.shape[0] == d.shape[
        #     0], 'The number of values passed in the properties does not match with the' \
        #         'length of indices.'
        #
        # self.interfaces.df.loc[indices, d.columns] = d.values
        # if is_formation:
        #     self.interfaces.sort_table()
        #     self.update_structure()

        if vtk_object is not None:
            vtk_object.render_move_interfaces(indices)


        #
        #
        #
        # indices = np.array(indices, ndmin=1)
        # keys = list(properties.keys())
        # xyz_check = ~np.isin(['X', 'Y', 'Z'], keys)
        #
        # d = pn.DataFrame(properties, columns=np.append(np.array(['X', 'Y', 'Z'])[xyz_check], keys), index=indices)
        # is_formation = any(d.columns.isin(['formation']))
        #
        # if is_formation:
        #     self.map_data_df(d)
        #
        # _checker = 0
        # if d.columns.isin(['G_x', "G_y", 'G_z']).sum() == 3:
        #     self.orientations.calculate_orientations()
        #     _checker += 1
        # elif d.columns.isin(['dip', 'azimuth', 'polarity']).sum() == 3:
        #     self.orientations.calculate_gradient()
        #     _checker += 1
        #     if _checker == 2:
        #         raise AttributeError(
        #             'add orientation only accept either orientation data [dip, azimuth, polarity] or'
        #             'gradient data [G_x, G_y, G_z]')
        #
        # # To be sure that we
        # xyz_exist = np.array(['X', 'Y', 'Z'])
        # xyz_res = np.array(['X_r', 'Y_r', 'Z_r'])
        # d[xyz_res] = self.rescaling.rescale_data_point(d[xyz_exist])
        # d.dropna(axis=1, inplace=True)
        #
        # assert indices.shape[0] == d.shape[
        #     0], 'The number of values passed in the properties does not match with the' \
        #         'length of indices.'
        #
        # self.orientations.df.loc[indices, d.columns] = d.values
        # if is_formation:
        #     self.orientations.sort_table()
        #     self.update_structure()

        if vtk_object is not None:
            vtk_object.render_move_orientations(indices)

    def modify_rescaling_parameters(self, property, value):
        self.additional_data.rescaling_data.modify_rescaling_parameters(property, value)
        self.additional_data.rescaling_data.rescale_data()
        self.additional_data.update_default_kriging()

    def modify_options(self, property, value):
        self.additional_data.options.modify_options(property, value)
        warnings.warn('You need to recompile the Theano code to make it the changes in options.')

    def modify_kriging_parameters(self, property, value):
        self.additional_data.kriging_data.modify_kriging_parameters(property, value)

    def modify_order_surfaces(self,  new_value: int, idx: int, series: str = None):
        self.formations.modify_order_surfaces(new_value, idx, series)
        self.update_from_formations(False, False, False, False, True)

    def modify_order_series(self, new_value: int, idx: str):
        self.series.modify_order_series(new_value, idx)
        self.update_from_series()

    def update_from_additional_data(self):
        pass

    def update_to_interpolator(self):
        self.interpolator.set_theano_shared_parameters()


@_setdoc([MetaData.__doc__, Grid.__doc__])
class Model(DataMutation):
    """
    Container class of all objects that constitute a GemPy model. In addition the class provides the methods that
    act in more than one of this class.
    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        super().__init__()

        self.interpolator = Interpolator(self.interfaces, self.orientations, self.grid, self.formations,
                                         self.faults, self.additional_data)
        self.solutions = Solution(self.additional_data, self.formations, self.grid)

    def __str__(self):
        return self.meta.project_name

    def new_model(self, name_project='default_project'):
        self.__init__(name_project)

    def save_model(self, path=False):
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
    def load_model(path):
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

    def save_model_long_term(self):
        # TODO saving the main attributes in a seriealize way independent on the package i.e. interfaces and
        # TODO orientations categories_df, grid values etc.
        pass

    def get_data(self, itype='data', numeric=False):
        """
        Method that returns the interfaces and orientations pandas Dataframes. Can return both at the same time or only
        one of the two

        Args:
            itype: input_data data type, either 'orientations', 'interfaces' or 'all' for both.
            numeric(bool): Return only the numerical values of the dataframe. This is much lighter database for storing
                traces
            verbosity (int): Number of properties shown

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # dtype = 'object'
        # TODO adapt this

        show_par_f = self.orientations.df.columns
        show_par_i = self.interfaces.df.columns

        if numeric:
            show_par_f = self.orientations._columns_o_num
            show_par_i = self.interfaces._columns_i_num
            dtype = 'float'

        if itype == 'orientations':
            raw_data = self.orientations.df[show_par_f]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'interfaces':
            raw_data = self.interfaces.df[show_par_i]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'data':
            raw_data = pn.concat([self.interfaces.df[show_par_i],  # .astype(dtype),
                                  self.orientations.df[show_par_f]],  # .astype(dtype)],
                                 keys=['interfaces', 'orientations'],
                                 sort=False)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]

        elif itype == 'formations':
            raw_data = self.formations
        elif itype == 'series':
            raw_data = self.series
        elif itype == 'faults':
            raw_data = self.faults
        elif itype == 'faults_relations_df' or itype == 'faults_relations':
            raw_data = self.faults.faults_relations_df
        elif itype == 'additional data' or itype == 'additional_data':
            raw_data = self.additional_data
        else:
            raise AttributeError('itype has to be \'data\', \'additional data\', \'interfaces\', \'orientations\','
                                 ' \'formations\',\'series\', \'faults\' or \'faults_relations_df\'')

        return raw_data

    # def get_theano_input(self):
    #     pass





























