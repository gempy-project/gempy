import os
import sys
from os import path

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import pandas as pn
pn.options.mode.chained_assignment = None
from .data import *
from gempy.utils.meta import _setdoc
from gempy.plot.plot import vtkPlot

@_setdoc([MetaData.__doc__, GridClass.__doc__])
class Model(object):
    """
    Container class of all objects that constitute a GemPy model. In addition the class provides the methods that
    act in more than one of this class.
    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        self.grid = GridClass()
        self.faults = Faults()
        self.series = Series(self.faults)
        self.formations = Formations(self.series)
        self.interfaces = Interfaces()
        self.orientations = Orientations()

        self.rescaling = RescaledData(self.interfaces, self.orientations, self.grid)
        self.additional_data = AdditionalData(self.interfaces, self.orientations, self.grid, self.faults,
                                              self.formations, self.rescaling)
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

    def get_data(self, itype='data', numeric=False, verbosity=0):
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
        if verbosity == 0:
            show_par_f = self.orientations._columns_o_1
            show_par_i = self.interfaces._columns_i_1
        else:
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

    def set_grid(self, grid: GridClass, update_model=True):
        self.grid = grid
        if update_model is True:
            self.additional_data.grid = grid
            self.rescaling.grid = grid
            self.rescaling.set_rescaled_grid()
            self.interpolator.grid = grid
            self.interpolator.set_theano_share_input()

            self.solutions.grid = grid

    def set_series(self):
        pass

    def set_formations(self):
        pass

    def set_faults(self):
        pass

    def set_interfaces(self, interfaces: Interfaces, update_model=True):
        self.interfaces = interfaces
        if update_model is True:
            self.additional_data.interfaces = interfaces
            self.update_structure()
            self.rescaling.interfaces = interfaces
            self.rescaling.set_rescaled_interfaces()
            self.interpolator.interfaces = interfaces

    def set_orientations(self):
        pass

    def set_interpolator(self, interpolator: Interpolator):
        self.interpolator = interpolator

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

    def modify_kriging_parameters(self, vtk_object: vtkPlot=None, **properties):
        d = pn.DataFrame(properties).T
        self.additional_data.kriging_data.loc[d.index, 'values'] = d
        self.update_plot(vtk_object)

    def add_interfaces(self, vtk_object: vtkPlot=None, **properties):

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

    def add_orientations(self, vtk_object: vtkPlot=None, **properties):

        d = pn.DataFrame(properties)
        d[['X_r', 'Y_r', 'Z_r']] = self.rescaling.rescale_data_point(d[['X', 'Y', 'Z']])
        try:
            self.map_data_df(d)
        except KeyError:
            pass

        for index, frame in d.iterrows():
            new_ind = self.orientations.df.last_valid_index() + 1
            self.orientations.df.loc[new_ind, d.columns] = frame

            if vtk_object is not None:
                vtk_object.render_add_orientations(new_ind)

        self.orientations.sort_table()
        _checker = 0

        if d.columns.isin(['G_x', "G_y", 'G_z']).sum() == 3:
            self.orientations.calculate_orientations()
            _checker += 1
        elif d.columns.isin(['dip', 'azimuth', 'polarity']).sum() == 3:
            self.orientations.calculate_gradient()
            _checker += 1
            if _checker == 2:
                raise AttributeError('add orientation only accept either orientation data [dip, azimuth, polarity] or'
                                     'gradient data [G_x, G_y, G_z]')
        else:
            raise AttributeError('Not enough angular data to calculate the gradients. Pass orientations or gradients')

        self.update_structure()

    def add_series(self, vtk_object: vtkPlot=None, **properties):
        pass

    def delete_interfaces(self, indices: Union[list, int], vtk_object: vtkPlot=None,):
        self.interfaces.df.drop(indices)

        if vtk_object is not None:
            vtk_object.render_delete_interfaes(indices)

        self.update_structure()

    def delete_orientations(self, indices: Union[list, int], vtk_object: vtkPlot=None,):
        self.orientations.df.drop(indices)

        if vtk_object is not None:
            vtk_object.render_delete_orientations(indices)

        self.update_structure()

    def modify_interfaces(self, indices: list, vtk_object: vtkPlot=None, **properties: list):
        indices = np.array(indices, ndmin=1)
        keys = list(properties.keys())
        xyz_check = ~np.isin(['X', 'Y', 'Z'], keys)
        d = pn.DataFrame(properties, columns=np.append(np.array(['X', 'Y', 'Z'])[xyz_check], keys), index=indices)
        is_formation = any(d.columns.isin(['formation']))

        if is_formation:
            self.map_data_df(d)

        # To be sure that we
        xyz_exist = np.array(['X', 'Y', 'Z'])
        xyz_res = np.array(['X_r', 'Y_r', 'Z_r'])
        d[xyz_res] = self.rescaling.rescale_data_point(d[xyz_exist])
        d.dropna(axis=1, inplace=True)

        assert indices.shape[0] == d.shape[0], 'The number of values passed in the properties does not match with the' \
                                               'length of indices.'

        self.interfaces.df.loc[indices, d.columns] = d.values
        if is_formation:
            self.interfaces.sort_table()
            self.update_structure()

        if vtk_object is not None:
            vtk_object.render_move_interfaces(indices)

    def modify_orientations(self, indices: list, vtk_object: vtkPlot=None, **properties: list):
        indices = np.array(indices, ndmin=1)
        keys = list(properties.keys())
        xyz_check = ~np.isin(['X', 'Y', 'Z'], keys)

        d = pn.DataFrame(properties, columns=np.append(np.array(['X', 'Y', 'Z'])[xyz_check], keys),  index=indices)
        is_formation = any(d.columns.isin(['formation']))

        if is_formation:
            self.map_data_df(d)

        _checker = 0
        if d.columns.isin(['G_x', "G_y", 'G_z']).sum() == 3:
            self.orientations.calculate_orientations()
            _checker += 1
        elif d.columns.isin(['dip', 'azimuth', 'polarity']).sum() == 3:
            self.orientations.calculate_gradient()
            _checker += 1
            if _checker == 2:
                raise AttributeError('add orientation only accept either orientation data [dip, azimuth, polarity] or'
                                     'gradient data [G_x, G_y, G_z]')

        # To be sure that we
        xyz_exist = np.array(['X', 'Y', 'Z'])
        xyz_res = np.array(['X_r', 'Y_r', 'Z_r'])
        d[xyz_res] = self.rescaling.rescale_data_point(d[xyz_exist])
        d.dropna(axis=1, inplace=True)

        assert indices.shape[0] == d.shape[0], 'The number of values passed in the properties does not match with the' \
                                               'length of indices.'

        self.orientations.df.loc[indices, d.columns] = d.values
        if is_formation:
            self.orientations.sort_table()
            self.update_structure()

        if vtk_object is not None:
            vtk_object.render_move_orientations(indices)



























