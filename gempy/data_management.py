"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
from os import path

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import copy
import numpy as np
import pandas as pn

import warnings

try:
    import qgrid
except ImportError:
    warnings.warn('qgrid package is not installed. No interactive dataframes available.')

pn.options.mode.chained_assignment = None


class InputData(object):
    """
    Class that contains all raw data of our models

    Args:
        extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        Resolution (Optional[list]): [nx, ny, nz]. Defaults to 50
        path_i (Optional[str]): Path to the data bases of interfaces. Default os.getcwd(),
        path_o (Optional[str]): Path to the data bases of orientations. Default os.getcwd()
        kwargs: key words passed to :class:`gempy.data_management.GridClass`

    Attributes:
        interfaces (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
            the interface points of the model
        orientations (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
            the orientations of the model
        formations (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the formations names and the value
            used for each voxel in the final model and the lithological order
        series (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and the formations contained
            on them
        faults (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and if they are faults or
            not (otherwise they are lithologies) and in case of being fault if is finite
        faults_relations (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the offsetting relations
            between each fault and the rest of the series (either other faults or lithologies)
        grid (:class:`gempy.data_management.GridClass`): grid object containing mainly the coordinates to interpolate
            the model
        extent(list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution (Optional[list]): [nx, ny, nz]

    """

    def __init__(self,
                 extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_o=None, path_f =None,
                 **kwargs):

        self._formation_values_set = False

        if path_f and path_o is None:
            warnings.warn('path_f is deprecated use instead path_o')
            path_o = path_f

        # Set extent and resolution
        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

        # Init number of faults
        self.n_faults = 0
        self.faults_relations = None
        self.formations = None
        self.series = None
        self.faults = None

        self._columns_i_all = ['X', 'Y', 'Z', 'formation', 'series', 'X_std', 'Y_std', 'Z_std', 'order_series', 'formation_number']
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'formation', 'series', 'X_std', 'Y_std', 'Z_std', 'dip_std', 'azimuth_std', 'order_series', 'formation_number']

        self._columns_i_1 = ['X', 'Y', 'Z', 'formation', 'series', 'formation_number', 'order_series', 'isFault']
        self._columns_o_1 = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity', 'formation', 'series', 'formation_number', 'order_series', 'isFault']

        self._columns_i_num = ['X', 'Y', 'Z']
        self._columns_o_num = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']


        # Create the pandas dataframes
        # if we dont read a csv we create an empty dataframe with the columns that have to be filled
        self.orientations = pn.DataFrame(columns=self._columns_o_1)
        self.orientations.itype = 'orientations'

        self.interfaces = pn.DataFrame(columns=self._columns_i_1)
        self.interfaces.itype = 'interfaces'

        if path_o or path_i:
            # TODO choose the default source of data. So far only csv
            self.import_data_csv(path_i=path_i, path_o=path_o)

        # Init all df
        self.update_df()
        self.set_basement()

        # Compute gradients given azimuth and dips to plot data
        self.calculate_gradient()

        # Create default grid object. TODO: (Is this necessary now?)
        self.grid = self.set_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)
        self.order_table()
        self.potential_at_interfaces = 0

        # Set dtypes
        self.interfaces['isFault'] = self.interfaces['isFault'].astype('bool')
        self.orientations['isFault'] = self.orientations['isFault'].astype('bool')

    def set_basement(self):

        try:
            self.interfaces['formation'].cat.add_categories('basement', inplace=True)
        except ValueError:
            pass

        try:
            n_series = self.interfaces['order_series'].unique().max()
        except ValueError:
            n_series = 0

        drop_basement = self.interfaces['formation'] == 'basement'
        original_frame = self.interfaces[~drop_basement]

        try:
            n_formation = original_frame['formation_number'].unique().max() + 1
        except ValueError:
            n_formation = 1
        l = len(self.interfaces)

        if not 'basement' in self.interfaces['formation'].values:

            try:
                columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation':'basement', 'order_series': n_series, 'formation_number': n_formation, 'series': self.series.columns[-1]}
            except AttributeError:
                columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation':'basement', 'order_series': n_series, 'formation_number': n_formation, 'series': 'Default series'}

            for key in columns:
                self.interfaces.at[l, str(key)] = columns[key]

            self.order_table()
           # sef.add_interface(formation='basement', order_series=n_series, formation_number = n_formation)
        else:
            self.modify_interface((drop_basement.index[drop_basement])[0], formation='basement', order_series=n_series, formation_number = n_formation)

        self.order_table()

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations

        Attributes:
            orientations: extra columns with components xyz of the unity vector.
        """

        self.orientations['G_x'] = np.sin(np.deg2rad(self.orientations["dip"].astype('float'))) * \
                                 np.sin(np.deg2rad(self.orientations["azimuth"].astype('float'))) * \
                                 self.orientations["polarity"].astype('float')+1e-12
        self.orientations['G_y'] = np.sin(np.deg2rad(self.orientations["dip"].astype('float'))) * \
                                   np.cos(np.deg2rad(self.orientations["azimuth"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')+1e-12
        self.orientations['G_z'] = np.cos(np.deg2rad(self.orientations["dip"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')+1e-12

    def calculate_orientations(self):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.
        """

        self.orientations["dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.orientations["G_z"] / self.orientations["polarity"])))

        # TODO if this way to compute azimuth breaks there is in rgeomod=kml_to_plane line 170 a good way to do it
        self.orientations["azimuth"] = np.rad2deg(np.nan_to_num(np.arctan(self.orientations["G_x"]/self.orientations["G_y"])))
                                                  # np.arcsin(self.orientations["G_x"]) /
                                                  # (np.sin(np.arccos(self.orientations["G_y"] /
                                                  # self.orientations["polarity"])))))

    def count_faults(self):
        """
        Read the string names of the formations to detect automatically the number of faults.
        """
        faults_series = []
        for i in self.interfaces['series'].unique():
            try:
                if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                    faults_series.append(i)
            except TypeError:
                pass
        return faults_series

    def create_orientation_from_interfaces(self, indices):

        selected_points = self.interfaces[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = get_orientation(normal)
        return np.array([*center, *orientation, *normal])

    def data_to_pickle(self, path=False):
        """
        Save InputData object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            None
        """

        if not path:
            path = './geo_data'
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def set_default_orientation(self):
        ori = pn.DataFrame([[(self.extent[1] - self.extent[0]) / 2,
                             (self.extent[3] - self.extent[2]) / 2,
                             (self.extent[4] - self.extent[5]) / 2,
                             0, 0, 1,
                             0, 0, 1,
                             'basement',
                             'Default series',
                             1, 1, False]], columns=self._columns_o_1)

        self.set_orientations(ori)

    def set_default_interface(self):
        if self.formations.index[0] is 'basement':
            formation = 'default'
            self.set_formations(formation_order=[formation])
        else:
            formation = self.formations.index[0]
            #self.set_formations(formation_order=[self.formations.index])

        self.set_interfaces(pn.DataFrame({'X': [(self.extent[1] - self.extent[0]) / 2],
                                          'Y': [(self.extent[3] - self.extent[2]) / 2],
                                          'Z': [(self.extent[4] - self.extent[5]) / 2],
                                          'formation':[formation], 'order_series':[0],
                                          'formation_number': [1], 'series': ['Default series'],
                                          'isFault': False}))

        self.set_basement()

    def get_formations(self):
        """
        Returns:
             pandas.core.frame.DataFrame: Returns a list of formations

        """
        return self.interfaces["formation"].unique()

    def get_data(self, itype='all', numeric=False, verbosity=0):
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
        #dtype = 'object'

        if verbosity == 0:
            show_par_f = self._columns_o_1
            show_par_i = self._columns_i_1
        else:
            show_par_f = self.orientations.columns
            show_par_i = self.interfaces.columns

        if numeric:
            show_par_f = self._columns_o_num
            show_par_i = self._columns_i_num
            dtype = 'float'


        if itype == 'orientations':
            raw_data = self.orientations[show_par_f]#.astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'interfaces':
            raw_data = self.interfaces[show_par_i]#.astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces[show_par_i],#.astype(dtype),
                                 self.orientations[show_par_f]],#.astype(dtype)],
                                 keys=['interfaces', 'orientations'])
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]

        elif itype is 'formations':
            raw_data = self.formations
        elif itype is 'series':
            raw_data = self.series
        elif itype is 'faults':
            raw_data = self.faults
        elif itype is 'faults_relations':
            raw_data = self.faults_relations
        else:
            raise AttributeError('itype has to be \'all\', \'interfaces\', \'orientations\', \'formations\', \
                                       \'serires\', \'faults\' or \'faults_relations\'')

        #else:
        #    raise AttributeError('itype has to be: \'orientations\', \'interfaces\', or \'all\'')

        return raw_data

    # def get_formation_number(self):
    #     """
    #         Get a dictionary with the key the name of the formation and the value their number
    #
    #         Returns:
    #             dict: key the name of the formation and the value their number
    #         """
    #     pn_series = self.interfaces.groupby('formation_number').formation.unique()
    #     ip_addresses = {}
    #     for e, i in enumerate(pn_series):
    #         ip_addresses[i[0]] = e + 1
    #  #   ip_addresses['DefaultBasement'] = 0
    #     return ip_addresses

    def interactive_df_open(self, itype='all',  numeric=False, verbosity=0):

        toolbar = True

        if itype is 'all':
            toolbar = False
        elif itype is 'formations':
            toolbar = False
        elif itype is 'faults':
            toolbar = False
        elif itype is 'faults_relations':
            toolbar = False

        if not toolbar:
            warnings.warn('for this itype Add Row does not work. If needed try using interfaces or orientations'
                          'instead')

        df_ = self.get_data(itype=itype, verbosity=verbosity)
        self.qgrid_widget = qgrid.QgridWidget(df=df_, show_toolbar=toolbar)

        return self.qgrid_widget

    def interactive_df_get_changed_df(self, only_selected=False):
        if only_selected is True:
            sol = self.qgrid_widget.get_selected_df()
        else:
            sol = self.qgrid_widget.get_changed_df()

        return sol

    def import_data_csv(self, path_i, path_o, **kwargs):
        """
        Method to import interfaces and orientations from csv. The format is the same as the export 3D model data of
        GeoModeller (check in the input_data data folder for an example).

        Args:
            path_i (str): path to the csv table
            path_o (str): path to the csv table
            **kwargs: kwargs of :func: `~pn.read_csv`

        Attributes:
            orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
            Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        """

        if path_o:
            orientations_read = self.load_data_csv(data_type="orientations", path=path_o, **kwargs)

            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(orientations_read.columns),\
                "One or more columns do not match with the expected values " + str(orientations_read.columns)

            self.orientations[orientations_read.columns] = orientations_read[orientations_read.columns]

        if path_i:
            interfaces_read = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(interfaces_read.columns), \
                "One or more columns do not match with the expected values " + str(interfaces_read.columns)

            c = np.array(self._columns_i_1)
            interfaces_read = interfaces_read.assign(**dict.fromkeys(c[~np.in1d(c, interfaces_read.columns)], False))
            self.set_interfaces(interfaces_read, append=True)
            #self.interfaces[interfaces_read.columns] = interfaces_read[interfaces_read.columns]
            #gagag
        self.update_df()

    def modify_interface(self, index, **kwargs):
        """
        Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

        Args:
            index: dataframe index of the orientation point
            **kwargs: X, Y, Z (int or float)

        Returns:
            None
        """
        for key in kwargs:
            self.interfaces.ix[index, str(key)] = kwargs[key]

    def add_interface(self, **kwargs):
        """
        Adds interface to dataframe.

        Args:
            **kwargs: X, Y, Z, formation, labels, order_series, series

        Returns:
            None

        """
        l = len(self.interfaces)
        for key in kwargs:
            self.interfaces.ix[l, str(key)] = kwargs[key]
        if not 'series' in kwargs:
            self.set_series()
        self.set_basement()
        self.order_table()

    def drop_interface(self, index):
        """
        Drops interface from dataframe identified by index

        Args:
            index: dataframe index

        Returns:
            None

        """
        self.interfaces.drop(index, inplace=True)

    def modify_orientation(self, index, recalculate_gradient=False, recalculate_orientations=False, **kwargs):
        """
        Allows modification of orientation data at specified dataframe index.

        Args:
            index: dataframe index of the orientation point
            **kwargs: G_x, G_y, G_z, X, Y, Z, azimuth, dip, formation, labels, order_series, polarity

        Returns:
            None
        """
        for key in kwargs:
            self.orientations.ix[index, str(key)] = kwargs[key]

        # TODO: EASY: make the condition check automatic regarding the keys that are modified
        if recalculate_gradient:
            self.calculate_gradient()
        if recalculate_orientations:
            self.calculate_orientations()

    @staticmethod
    def plane_fit(point_list):
        """
        Fit plane to points in PointSet
        Fit an d-dimensional plane to the points in a point set.
        adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

        Args:
            point_list (array_like): array of points XYZ

        Returns:
            Return a point, p, on the plane (the point-cloud centroid),
            and the normal, n.
        """

        import numpy as np

        #     points = np.empty((3, len(point_list)))
        #     for i, point in enumerate(point_list):
        #         points[0, i] = point.x
        #         points[1, i] = point.y
        #         points[2, i] = point.z
        points = point_list

        from numpy.linalg import svd
        points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                       points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)  # Could also use np.cov(x) here.

        # ctr = Point(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
        normal = svd(M)[0][:, -1]
        # return ctr, svd(M)[0][:, -1]
        if normal[2] < 0:
            normal = - normal

        return ctr, normal

    def add_orientation(self, **kwargs):
        """
        Adds orientation to dataframe.

        Args:
            **kwargs: G_x, G_y, G_z, X, Y, Z, azimuth, dip, formation, labels, order_series, polarity, series

        Returns: Nothing

        """
        l = len(self.orientations)
        try:
            for key in kwargs:
                self.orientations.ix[l, str(key)] = kwargs[key]
        except ValueError:
            self.orientations['formation'].cat.add_categories(kwargs['formation'], inplace=True)
            for key in kwargs:
                self.orientations.ix[l, str(key)] = kwargs[key]

        self.calculate_gradient()
        self.calculate_orientations()
        if not 'series' in kwargs:
            self.set_series()

        self.set_basement()
        self.order_table()

    def drop_orientations(self, index):
        """
        Drops orientation from dataframe identified by index

        Args:
            index: dataframe index

        Returns:
            None

        """
        self.orientations.drop(index, inplace=True)

    @staticmethod
    def load_data_csv(data_type, path=os.getcwd(), **kwargs):
        """
        Method to load either interface or orientations data csv files. Normally this is in which GeoModeller exports it

        Args:
            data_type (str): 'interfaces' or 'orientations'
            path (str): path to the files. Default os.getcwd()
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # TODO: in case that the columns have a different name specify in pandas which columns are interfaces /
        # coordinates, dips and so on.
        # TODO: use pandas to read any format file not only csv

        if data_type == "orientations":
            return pn.read_csv(path, **kwargs)
        elif data_type == 'interfaces':
            return pn.read_csv(path, **kwargs)
        else:
            raise NameError('Data type not understood. Try interfaces or orientations')

    def reset_indices(self):
        """
        Resets dataframe indices for orientations and interfaces.

        Returns:
            None
        """
        self.interfaces.reset_index(inplace=True, drop=True)
        self.orientations.reset_index(inplace=True, drop=True)
        self.set_annotations()

    def set_grid(self, custom_grid=None, extent=None, resolution=None, grid_type=None, **kwargs):
        """
        Method to initialize the class GridClass. You can pass either a custom set of points or create a regular grid

        Args:
            grid_type (str): regular_3D or None
            custom_grid(array_like): 2D array with XYZ columns. To exploit gempy functionality the indexing has to be ij
                (See Also numpy.meshgrid documentation)
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self.grid(gempy.GridClass): Object that contain different grids
        """
        self.grid = GridClass()
        if custom_grid is not None:
            assert custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3) where n is' \
                                                                        'the number of points of the grid'

            self.grid.create_custom_grid(custom_grid)
        if grid_type is 'regular_3D':
            if not extent:
                extent = self.extent
            if not resolution:
                resolution = self.resolution
            self.grid.create_regular_grid_3d(extent, resolution)

            return self.grid

    def set_interfaces(self, interf_Dataframe, append=False, order_table=True):
        """
        Method to change or append a Dataframe to interfaces in place. A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'formation'] has to be passed.

        Args:
            interf_Dataframe: pandas.core.frame.DataFrame with the data
            append: Bool: if you want to append the new data frame or substitute it
        """
        assert set(self._columns_i_1).issubset(interf_Dataframe.columns), \
            "One or more columns do not match with the expected values " + str(interf_Dataframe.columns)

        interf_Dataframe[self._columns_i_num] = interf_Dataframe[self._columns_i_num].astype(float, copy=True)
        interf_Dataframe[['formation_number', 'order_series']] = interf_Dataframe[['formation_number', 'order_series']].astype(int, copy=True)
        interf_Dataframe['formation'] = interf_Dataframe['formation'].astype('category', copy=True)
        interf_Dataframe['series'] = interf_Dataframe['series'].astype('category', copy=True)

        if append:
            self.interfaces = self.interfaces.append(interf_Dataframe)
        else:
            # self.interfaces[self._columns_i_1] = interf_Dataframe[self._columns_i_1]
            self.interfaces = interf_Dataframe[self._columns_i_1]

        self.interfaces = self.interfaces[~self.interfaces[['X', 'Y', 'Z']].isna().any(1)]

       # self.set_annotations()


        #self.set_annotations()
        if not self.interfaces.index.is_unique:
            self.interfaces.reset_index(drop=True, inplace=True)

        if order_table:

            self.set_series()
            self.order_table()

        # # We check if in the df we are setting there is a new formation. if yes we append it to to the cat
        # new_cat = interf_Dataframe['formation'].cat.categories[~np.in1d(interf_Dataframe['formation'].cat.categories,
        #                                                                self.formations)]
        # self.formations.index.insert(0, new_cat)
        self.set_series()
        self.set_formations()
        self.set_faults()
        self.interfaces.sort_index()

    def set_orientations(self, foliat_Dataframe, append=False, order_table=True):
        """
          Method to change or append a Dataframe to orientations in place.  A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(self._columns_o_1).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)

        foliat_Dataframe[self._columns_o_num] = foliat_Dataframe[self._columns_o_num].astype(float, copy=True)

        if append:
            self.orientations = self.orientations.append(foliat_Dataframe)
        else:
            #self.orientations[self._columns_o_1] = foliat_Dataframe[self._columns_o_1]
            self.orientations = foliat_Dataframe[self._columns_o_1]

        # self.calculate_orientations()
        self.calculate_gradient()

        self.orientations = self.orientations[~self.orientations[['X', 'Y', 'Z']].isna().any(1)]

        self.set_series()
        self.set_formations()
        self.set_faults()
       # self.set_annotations()

        if order_table:
          #  self.set_formation_number()
            self.set_series()
            self.order_table()

        self.orientations.sort_index()

    def set_new_df(self, new_df, append=False):

        new_df.apply(pn.to_numeric, errors='ignore')

        try:
            self.set_interfaces(new_df.xs('interfaces'), append=append, order_table=False)
            self.set_orientations(new_df.xs('orientations'), append=append, order_table=False)

        except KeyError:

            if set(self._columns_o_1).issubset(new_df.columns):  # if true is orientations

                self.set_orientations(new_df, append=append, order_table=False)

            elif set(self._columns_i_1).issubset(new_df.columns):
                self.set_interfaces(new_df, append=append, order_table=False)

            else:
                raise AttributeError('The given dataframe does not have the right formt')

    def set_series(self, series_distribution=None, order=None):
        """
        Method to define the different series of the project.

        Args:
            series_distribution (dict): with the name of the serie as key and the name of the formations as values.
            order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
                random. This is important to set the erosion relations between the different series

        Returns:
            self.series: A pandas DataFrame with the series and formations relations
            self.interfaces: one extra column with the given series
            self.orientations: one extra column with the given series
        """

        # if series_distribution is None:
        #     # set to default series
        #     # TODO see if some of the formations have already a series and not overwrite
        #     _series = {"Default serie": self.interfaces["formation"].unique()}
        #
        # else:
        #     assert type(series_distribution) is dict, "series_distribution must be a dictionary, " \
        #                                               "see Docstring for more information"
        #
        #     # TODO if self.series exist already maybe we should append instead of overwrite
        #     _series = series_distribution
        #
        # # The order of the series is very important since it dictates which one is on top of the stratigraphic pile
        # # If it is not given we take the dictionaries keys. NOTICE that until python 3.6 these keys are pretty much
        # # random
        # if order is None:
        #     order = _series.keys()
        #
        # # TODO assert len order is equal to len of the dictionary
        # # We create a dataframe with the links
        # _series = pn.DataFrame(dict([ (k,pn.Series(v)) for k,v in _series.items() ]), columns=order)

        if series_distribution is None:
            if self.series is None:
                self.series = pn.DataFrame({"Default series": self.interfaces["formation"].unique().astype(list)},
                                           dtype=str)
            #     # Check if there is already a df
            #     self.series
            # except AttributeError:
            #     # set to default series

        else:
            if type(series_distribution) is dict:
                if order is None:
                    order = series_distribution.keys()
                else:
                    assert all(np.in1d(order, list(series_distribution.keys()))), 'Order series must contain the same keys as' \
                                                                       'the passed dictionary ' + str(series_distribution.keys())
                self.series = pn.DataFrame(dict([(k, pn.Series(v)) for k, v in series_distribution.items()]),
                                           columns=order)

            elif type(series_distribution) is pn.core.frame.DataFrame:
                self.series = series_distribution

            else:
                raise AttributeError('series_distribution must be a dictionary, see Docstring for more information')

            # Addind the formations of the new series to the formations df
            new_formations = self.series.values.reshape(1, -1)
            # Dropping nans
            new_formations = new_formations[~pn.isna(new_formations)]
            self.set_formations(formation_order=new_formations)

        if 'basement' not in self.series.iloc[:, -1].values:
            self.series.loc[self.series.shape[0], self.series.columns[-1]] = 'basement'


        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        self.interfaces["series"] = [(i == self.series).sum().idxmax() for i in self.interfaces["formation"]]
        self.interfaces["series"] = self.interfaces["series"].astype('category')
        self.interfaces["order_series"] = [(i == self.series).sum().as_matrix().argmax().astype(int) + 1
                                           for i in self.interfaces["formation"]]
        self.orientations["series"] = [(i == self.series).sum().idxmax() for i in self.orientations["formation"]]
        self.orientations["series"] = self.orientations["series"].astype('category')
        self.orientations["order_series"] = [(i == self.series).sum().as_matrix().argmax().astype(int) + 1
                                           for i in self.orientations["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
       # if series_distribution is not None:
        self.interfaces.sort_values(by='order_series', inplace=True)
        self.orientations.sort_values(by='order_series', inplace=True)

        self.interfaces['series'].cat.set_categories(self.series.columns, inplace=True)
        self.orientations['series'].cat.set_categories(self.series.columns, inplace=True)

        # faults_series = self.count_faults()
        #
        # self.set_faults(faults_series)
        # self.reset_indices()
        #
        # self.set_formation_number()
        #
        # self.order_table()
        #
        # self.set_fault_relation_matrix(np.zeros((self.interfaces['series'].nunique(),
        #                                          self.interfaces['series'].nunique())))

        return self.series

    def update_df(self, series_distribution=None, order=None):

        self.interfaces['formation'] = self.interfaces['formation'].astype('category')
        self.orientations['formation'] = self.orientations['formation'].astype('category')

        self.set_series(series_distribution=series_distribution, order=order)
        self.set_basement()
        faults_series = self.count_faults()
        self.set_faults(faults_series)

       # self.reset_indices()

        self.set_formations()
        self.order_table()
        self.set_fault_relation()

        #self.set_annotations()

    def set_formations(self, formation_values = None, formation_order = None):


        self.interfaces['formation'] = self.interfaces['formation'].astype('category')
        self.orientations['formation'] = self.orientations['formation'].astype('category')

        if formation_order is None:
            if self.formations is None:
            #if self._formation_values_set is False:
                formation_order = self.interfaces['formation'].cat.categories
            else:
                # We check if in the df we are setting there is a new formation. if yes we append it to to the cat
                new_cat = self.interfaces['formation'].cat.categories[
                    ~np.in1d(self.interfaces['formation'].cat.categories,
                             self.formations.index)]
                if new_cat.empty:
                    formation_order = self.formations.index
                else:
                    formation_order = np.insert(self.formations.index.get_values(), 0, new_cat)
            # try:
            #     # Check if there is already a df
            #     formation_order = self.formations.index
            #
            # except AttributeError:
            #

        if 'basement' not in formation_order:
            formation_order = np.append(formation_order, 'basement')

        if formation_values is None:
            if self._formation_values_set:
                # Check if there is already a df
                formation_values = self.formations['value'].squeeze()
            else:
                formation_values = np.arange(1, formation_order.shape[0]+1)
        else:
            self._formation_values_set = True

        if np.atleast_1d(formation_values).shape[0] < np.atleast_1d(formation_order).shape[0]:
            formation_values = np.append(formation_values, formation_values.max()+1)

        self.formations = pn.DataFrame(index=formation_order,
                                       columns=['value', 'formation_number'])

        self.formations['value'] = formation_values
        self.formations['formation_number'] = np.arange(1, self.formations.shape[0]+1)

        self.interfaces['formation_number'] = self.interfaces['formation'].map(self.formations.iloc[:, 1])
        self.orientations['formation_number'] = self.orientations['formation'].map(self.formations.iloc[:, 1])

        self.interfaces['formation_value'] = self.interfaces['formation'].map(self.formations.iloc[:, 0])
        self.orientations['formation_value'] = self.orientations['formation'].map(self.formations.iloc[:, 0])

        self.interfaces['formation'].cat.set_categories(formation_order, inplace=True)
        self.orientations['formation'].cat.set_categories(formation_order, inplace=True)

    def _set_formation_number(self, formation_order=None):
        """
        Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
        to know it and also now the numbers must be set in the order of the series as well. Therefore this method
        has been moved to the interpolator class as preprocessing

        Returns:
            Column in the interfaces and orientations dataframes
        """
        #

        if formation_order is None:
            try:
                formation_order = self.formations.index
            except AttributeError:
                formation_order = self.interfaces['formation'].unique()
        if 'basement' not in formation_order:
            formation_order = np.append(formation_order, 'basement')

        self.interfaces['formation'] = self.interfaces['formation'].astype('category')
        self.orientations['formation'] = self.orientations['formation'].astype('category')

        # formation_order = self.interfaces["formation"].unique()
        self.interfaces['formation'].cat.reorder_categories(formation_order, inplace=True)

        self.orientations['formation'].cat.reorder_categories(
            formation_order[np.in1d(formation_order, self.orientations['formation'].cat.categories)],
            inplace=True)

        self.interfaces['formation_number'] = self.interfaces['formation'].cat.codes + 1
        self.orientations['formation_number'] = self.orientations['formation'].cat.codes + 1

        self.set_formation_values()

    def _set_formation_values(self, formation_values = None,):

        if formation_values is None:
        #     try:
        #         # Check if there is already a df
        #         self.formations
        #         print('I am changing formations1')
        #     except AttributeError:
            # set to default series
            self.formations = pn.DataFrame(index=self.interfaces['formation'].cat.categories, columns=[['value', 'formation_number']])
            self.formations['value'] = self.interfaces['formation_number'].unique()
            self.formations['formation_number'] = self.interfaces['formation_number'].unique()

        else:
            if type(formation_values) is dict:
                self.formations = pn.DataFrame(formation_values)
                self.formations['formation_number'] = self.interfaces['formation_number'].unique()
            elif type(formation_values) is pn.core.frame.DataFrame:
                self.formations = formation_values
                self.formations['formation_number'] = self.interfaces['formation_number'].unique()

            elif type(np.ndarray):
                self.formations['value'] = formation_values
            else:
                raise AttributeError('formation_values must be a dictionary, a Dataframe or array, see Docstring for more information')

    def set_faults(self, series_name=None):
        """
        Set a flag to the series that are faults.

        Args:
            series_name(list or array_like): Name of the series which are faults
        """

        try:
            # Check if there is already a df
            self.faults

            try:
                if any(self.faults.columns != self.series.columns):
                    series_name = self.count_faults()
                    self.faults = pn.DataFrame(index=self.series.columns, columns=['isFault'])
                    self.faults['isFault'] = self.faults.index.isin(series_name)
            except ValueError:
                series_name = self.count_faults()
                self.faults = pn.DataFrame(index=self.series.columns, columns=['isFault'])
                self.faults['isFault'] = self.faults.index.isin(series_name)

            if series_name:
                self.faults['isFault'] = self.faults.index.isin(series_name)

        except AttributeError:

            if not series_name:
                series_name = self.count_faults()
                self.faults = pn.DataFrame(index=self.series.columns, columns=['isFault'])
                self.faults['isFault'] = self.faults.index.isin(series_name)

        self.interfaces.loc[:, 'isFault'] = self.interfaces['series'].isin(self.faults.index[self.faults['isFault']])
        self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(self.faults.index[self.faults['isFault']])

        self.n_faults = self.faults['isFault'].sum()

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """
        point_num = self.interfaces.groupby('formation_number').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.interfaces['formation_number'])]

        orientation_num = self.orientations.groupby('formation_number').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(orientation_num, self.orientations['formation_number'])]

        self.interfaces['annotations'] = point_l
        self.orientations['annotations'] = foli_l

    def set_fault_relation(self, rel_matrix = None):
        """
        Method to set the faults that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        #TODO: Change the fault relation automatically every time we add a fault
        try:
            self.faults_relations
            if not rel_matrix:
               rel_matrix = np.zeros((self.series.columns.shape[0],
                                      self.series.columns.shape[0]))

            self.faults_relations = pn.DataFrame(rel_matrix, index=self.series.columns,
                                                 columns= self.series.columns, dtype='bool')
        except AttributeError:

            if rel_matrix is not None:
                self.faults_relations = pn.DataFrame(rel_matrix, index=self.series.columns,
                                                     columns=self.series.columns, dtype='bool')

    def order_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
        the formations. All inplace
        """

        # We order the pandas table by series
        self.interfaces.sort_values(by=['order_series'],
                                    ascending=True, kind='mergesort',
                                    inplace=True)

        self.orientations.sort_values(by=['order_series'],
                                      ascending=True, kind='mergesort',
                                      inplace=True)

        # Give formation_number
        if not 'formation_number' in self.interfaces.columns or not 'formation_number' in self.orientations.columns:

            self.set_formations()

        # We order the pandas table by formation (also by series in case something weird happened)
        self.interfaces.sort_values(by=['order_series', 'formation_number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        self.orientations.sort_values(by=['order_series', 'formation_number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
        # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
       # self.reset_indices()
        # DEP
        # self.interfaces.reset_index(drop=True, inplace=True)
        # self.orientations.reset_index(drop=True, inplace=True)

        # Update labels for anotations
        self.set_annotations()

    # # TODO think where this function should go
    def _read_vox(self, path):
        """
        read vox from geomodeller and transform it to gempy format
        Returns:
            numpy.array: block model
        """

        geo_res = pn.read_csv(path)

        geo_res = geo_res.iloc[9:]

        ip_dict = self.get_formation_number()

        geo_res_num = geo_res.iloc[:, 0].replace(ip_dict)
        block_geomodeller = np.ravel(geo_res_num.as_matrix().reshape(
                                        self.resolution[0], self.resolution[1], self.resolution[2], order='C').T)
        return block_geomodeller


def get_orientation(normal):
    """Get orientation (dip, azimuth, polarity ) for points in all point set"""
    #    if "normal" not in dir(self):
    #        self.plane_fit()

    # calculate dip
    dip = np.arccos(normal[2]) / np.pi * 180.

    # calculate dip direction
    # +/+
    if normal[0] >= 0 and normal[1] > 0:
        dip_direction = np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # border cases where arctan not defined:
    elif normal[0] > 0 and normal[1] == 0:
        dip_direction = 90
    elif normal[0] < 0 and normal[1] == 0:
        dip_direction = 270
    # +-/-
    elif normal[1] < 0:
        dip_direction = 180 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # -/-
    elif normal[0] < 0 and normal[1] >= 0:
        dip_direction = 360 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # if dip is just straight up vertical
    elif normal[0] == 0 and normal[1] == 0:
        dip_direction = 0

    if -90 < dip < 90:
        polarity = 1
    else:
        polarity = -1

    return dip, dip_direction, polarity


class GridClass(object):
    """
    Class to generate grids to pass later on to a InputData class.
    """

    def __init__(self):

        self.values = None

    def create_custom_grid(self, custom_grid):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'
        self.values = custom_grid
        return self.values

    def create_regular_grid_3d(self, extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """
        self.extent = extent
        self.resolution = resolution

        self.dx, self.dy, self.dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0],\
                                    (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(self.extent[0] + self.dx / 2, self.extent[1] - self.dx / 2, self.resolution[0], dtype="float32"),
            np.linspace(self.extent[2] + self.dy / 2, self.extent[3] - self.dy / 2, self.resolution[1], dtype="float32"),
            np.linspace(self.extent[4] + self.dz / 2, self.extent[5] - self.dz / 2, self.resolution[2], dtype="float32"), indexing="ij"
        )

        self.values = np.vstack(map(np.ravel, g)).T.astype("float32")
        return self.values
