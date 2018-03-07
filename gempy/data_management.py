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
from gempy import theano_graph
import theano
import warnings

pn.options.mode.chained_assignment = None  #


class InputData(object):
    """
    Class to import the raw data of the model and set data classifications into formations and series.
    This objects will contain the main information of the model.

    Args:
        extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_o: Path to the data bases of orientations. Default os.getcwd()

    Attributes:
        extent(list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution ((Optional[list])): [nx, ny, nz]
        orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
        Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        series(pandas.core.frame.DataFrame): Pandas data frame which contains every formation within each series
    """

    def __init__(self,
                 extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_o=None, path_f =None,
                 **kwargs):

        if path_f and path_o is None:
            warnings.warn('path_f is deprecated use instead path_o')
            path_o = path_f

        # Set extent and resolution
        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

        # Init number of faults
        self.n_faults = 0

        # TODO choose the default source of data. So far only csv
        # Create the pandas dataframes
        # if we dont read a csv we create an empty dataframe with the columns that have to be filled
        self.orientations = pn.DataFrame(columns=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                                                'formation', 'series', 'X_std', 'Y_std', 'Z_std',
                                                'dip_std', 'azimuth_std'])

        self.interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series',
                                                'X_std', 'Y_std', 'Z_std'])

        if path_o or path_i:
            self.import_data_csv(path_i=path_i, path_o=path_o)

        # If not provided set default series
        self.series = self.set_series()

        # Compute gradients given azimuth and dips to plot data
        self.calculate_gradient()

        # Create default grid object. TODO: (Is this necessary now?)
        self.grid = self.set_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)

        self.order_table()

        self.potential_at_interfaces = 0

        self.fault_relation = None

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations

        Attributes:
            orientations: extra columns with components xyz of the unity vector.
        """

        self.orientations['G_x'] = np.sin(np.deg2rad(self.orientations["dip"].astype('float'))) * \
                                 np.sin(np.deg2rad(self.orientations["azimuth"].astype('float'))) * \
                                 self.orientations["polarity"].astype('float')+1e-7
        self.orientations['G_y'] = np.sin(np.deg2rad(self.orientations["dip"].astype('float'))) * \
                                 np.cos(np.deg2rad(self.orientations["azimuth"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')+1e-7
        self.orientations['G_z'] = np.cos(np.deg2rad(self.orientations["dip"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')+1e-7

    def calculate_orientations(self):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.
        """
        self.orientations["dip"] = np.nan_to_num(np.arccos(self.orientations["G_z"] / self.orientations["polarity"]))

        # TODO if this way to compute azimuth breaks there is in rgeomod=kml_to_plane line 170 a good way to do it
        self.orientations["azimuth"] = np.nan_to_num(np.arcsin(self.orientations["G_x"]) / (np.sin(np.arccos(self.orientations["G_z"] / self.orientations["polarity"])) * self.orientations["polarity"]))

    def count_faults(self):
        """
        Read the string names of the formations to detect automatically the number of faults.
        """
        faults_series = []
        for i in self.interfaces['series'].unique():
            if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                faults_series.append(i)
        return faults_series

    def create_orientation_from_interfaces(self, indices):

        selected_points = self.interfaces[['X', 'Y', 'Z']].iloc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = get_orientation(normal)
        return [*center, *orientation, *normal]

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
        dtype = 'object'

        if verbosity == 0:
            show_par_f = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity', 'formation', 'series']
            show_par_i = ['X', 'Y', 'Z', 'formation', 'series']
        else:
            show_par_f = self.orientations.columns
            show_par_i = self.interfaces.columns

        if numeric:
            show_par_f = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
            show_par_i = ['X', 'Y', 'Z']
            dtype = 'float'
        if itype == 'orientations':
            raw_data = self.orientations[show_par_f].astype(dtype)
        elif itype == 'interfaces':
            raw_data = self.interfaces[show_par_i].astype(dtype)
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces[show_par_i].astype(dtype),
                                 self.orientations[show_par_f].astype(dtype)],
                                 keys=['interfaces', 'orientations'])
        else:
            raise AttributeError('itype has to be: \'orientations\', \'interfaces\', or \'all\'')

        # Be sure that the columns are in order when used for operations
        if numeric:
            raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        return raw_data

    def get_formation_number(self):
        """
            Get a dictionary with the key the name of the formation and the value their number

            Returns:
                dict: key the name of the formation and the value their number
            """
        pn_series = self.interfaces.groupby('formation number').formation.unique()
        ip_addresses = {}
        for e, i in enumerate(pn_series):
            ip_addresses[i[0]] = e + 1
        ip_addresses['DefaultBasement'] = 0
        return ip_addresses


    # DEP so far: Changing just a value from the dataframe gives too many problems
    # def i_open_set_data(self, itype="orientations"):
    #     """
    #     Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
    #      the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
    #      easier solution than calling two different methods.
    #     Args:
    #         itype: input_data data type, either 'orientations' or 'interfaces'
    #
    #     Returns:
    #         pandas.core.frame.DataFrame: Data frame with the changed data on real time
    #     """
    #     try:
    #         import qgrid
    #     except:
    #         raise ModuleNotFoundError('It is necessary to instal qgrid to have interactive tables')
    #
    #     # if the data frame is empty the interactive table is bugged. Therefore I create a default raw when the method
    #     # is called
    #     if self.orientations.empty:
    #         self.orientations = pn.DataFrame(
    #             np.array([0., 0., 0., 0., 0., 1., 'Default Formation', 'Default series']).reshape(1, 8),
    #             columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series']).\
    #             convert_objects(convert_numeric=True)
    #
    #     if self.interfaces.empty:
    #         self.interfaces = pn.DataFrame(
    #             np.array([0, 0, 0, 'Default Formation', 'Default series']).reshape(1, 5),
    #             columns=['X', 'Y', 'Z', 'formation', 'series']).convert_objects(convert_numeric=True)
    #
    #     # Setting some options
    #     qgrid.nbinstall(overwrite=True)
    #     qgrid.set_defaults(show_toolbar=True)
    #     assert itype is 'orientations' or itype is 'interfaces', 'itype must be either orientations or interfaces'
    #
    #     import warnings
    #     warnings.warn('Remember to call i_close_set_data after the editing.')
    #
    #     # We kind of set the show grid to a variable so we can close it afterwards
    #     self.pandas_frame = qgrid.show_grid(self.get_data(itype=itype))
    #
    # def i_close_set_data(self):
    #
    #     """
    #     Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
    #      the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
    #      easier solution than calling two different methods.
    #     Args:
    #         itype: input_data data type, either 'orientations' or 'interfaces'
    #
    #     Returns:
    #         pandas.core.frame.DataFrame: Data frame with the changed data on real time
    #     """
    #     # We close it to guarantee that after this method it is not possible further modifications
    #     self.pandas_frame.close()
    #
    #     # Set parameters
    #     self.series = self.set_series()
    #     self.calculate_gradient()
    #     self.order_table()

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

            self.interfaces[interfaces_read.columns] = interfaces_read[interfaces_read.columns]

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
        for key in kwargs:
            self.orientations.ix[l, str(key)] = kwargs[key]
        self.calculate_gradient()
        self.calculate_orientations()
        self.set_series()
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

    def set_interfaces(self, interf_Dataframe, append=False):
        """
        Method to change or append a Dataframe to interfaces in place. A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'formation'] has to be passed.

        Args:
            interf_Dataframe: pandas.core.frame.DataFrame with the data
            append: Bool: if you want to append the new data frame or substitute it
        """
        assert set(['X', 'Y', 'Z', 'formation']).issubset(interf_Dataframe.columns), \
            "One or more columns do not match with the expected values " + str(interf_Dataframe.columns)

        if append:
            self.interfaces = self.interfaces.append(interf_Dataframe)
        else:
            self.interfaces = interf_Dataframe

        self.set_series()
        self.order_table()

    def set_orientations(self, foliat_Dataframe, append=False):
        """
          Method to change or append a Dataframe to orientations in place.  A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)
        if append:
            self.orientations = self.orientations.append(foliat_Dataframe)
        else:
            self.orientations = foliat_Dataframe

        self.set_series()
        self.order_table()
        self.calculate_gradient()

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

        if series_distribution is None:
            # set to default series
            # TODO see if some of the formations have already a series and not overwrite
            _series = {"Default serie": self.interfaces["formation"].unique()}

        else:
            assert type(series_distribution) is dict, "series_distribution must be a dictionary, " \
                                                      "see Docstring for more information"

            # TODO if self.series exist already maybe we should append instead of overwrite
            _series = series_distribution

        # The order of the series is very important since it dictates which one is on top of the stratigraphic pile
        # If it is not given we take the dictionaries keys. NOTICE that until python 3.6 these keys are pretty much
        # random
        if order is None:
            order = _series.keys()

        # TODO assert len order is equal to len of the dictionary
        # We create a dataframe with the links
        _series = pn.DataFrame(dict([ (k,pn.Series(v)) for k,v in _series.items() ]), columns=order)

        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        self.interfaces["series"] = [(i == _series).sum().idxmax() for i in self.interfaces["formation"]]
        self.interfaces["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.interfaces["formation"]]
        self.orientations["series"] = [(i == _series).sum().idxmax() for i in self.orientations["formation"]]
        self.orientations["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.orientations["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
       # if series_distribution is not None:
        self.interfaces.sort_values(by='order_series', inplace=True)
        self.orientations.sort_values(by='order_series', inplace=True)

        # Save the dataframe in a property. This is used in the pile
        self.series = _series

        faults_series = self.count_faults()

        self.set_faults(faults_series)
        self.reset_indices()

        self.set_formation_number()

        self.order_table()

        self.set_fault_relation_matrix(np.zeros((self.interfaces['series'].nunique(),
                                                 self.interfaces['series'].nunique())))

        return _series

    def set_faults(self, series_name):
        """
        Set a flag to the series that are faults.

        Args:
            series_name(list or array_like): Name of the series which are faults
        """

        self.interfaces.loc[:, 'isFault'] = self.interfaces['series'].isin(series_name)
        self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(series_name)

        self.n_faults = len(series_name)

    def set_formation_number(self, formation_order=None):
        """
        Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
        to know it and also now the numbers must be set in the order of the series as well. Therefore this method
        has been moved to the interpolator class as preprocessing

        Returns:
            Column in the interfaces and orientations dataframes
        """

        if formation_order is None:
            formation_order = self.interfaces["formation"].unique()

        else:
            assert self.interfaces['formation'].isin(formation_order).all(), 'Some of the formations given are not in '\
                                                                             'the formations data frame. Check misspells'\
                                                                             'and that you include the name of the faults!'
        try:
            ip_addresses = formation_order
            ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses)+1)))
            self.interfaces.loc[:, 'formation number'] = self.interfaces['formation'].replace(ip_dict)
            self.orientations.loc[:, 'formation number'] = self.orientations['formation'].replace(ip_dict)
        except ValueError:
            pass

        self.order_table()

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """
        point_num = self.interfaces.groupby('formation number').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.interfaces['formation number'])]

        orientation_num = self.orientations.groupby('formation number').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(orientation_num, self.orientations['formation number'])]

        self.interfaces['annotations'] = point_l
        self.orientations['annotations'] = foli_l

    def set_fault_relation_matrix(self, rel_matrix):
        """
        Method to set the faults that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        #TODO: Change the fault relation automatically every time we add a fault
        self.fault_relation = rel_matrix

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

        # Give formation number
        if not 'formation number' in self.interfaces.columns or not 'formation number' in self.orientations.columns:

            self.set_formation_number()

        # We order the pandas table by formation (also by series in case something weird happened)
        self.interfaces.sort_values(by=['order_series', 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        self.orientations.sort_values(by=['order_series', 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
        # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
        self.interfaces.reset_index(drop=True, inplace=True)

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

    # # TODO Alex: Documentation
    # def set_triangle_orientations(self, verbose=False):
    #     # next we need to iterate over every unique triangle id to create a orientation from each triplet
    #     # of points and assign the same triange_id to it
    #     tri_ids = np.unique(self.interfaces["triangle_id"])
    #
    #     # check if column in orientations too, else create it
    #     if "triangle_id" not in self.orientations.columns:
    #         self.orientations["triangle_id"] = "NaN"
    #         if verbose:
    #             print("Setting triangle_id column in geo_data.orientations.")
    #
    #     # loop over all triangle_id's
    #     for tri_id in tri_ids[tri_ids != "NaN"]:
    #         # get the three points dataframe
    #         _filter = self.interfaces["triangle_id"] == tri_id
    #
    #         # check if triangle orientation value already exists
    #         if tri_id in np.unique(self.orientations["triangle_id"]):
    #             if verbose:
    #                 print("triangle_id already in geo_data.orientations - skipping it.")
    #             continue  # if yes, continue with the next iteration not not double append
    #
    #         if verbose:
    #             print("tri_id: "+tri_id)
    #         if len(self.interfaces[_filter]) == 3:
    #             # get points as [x,y,z]
    #             _points = []
    #             for i, interf in self.interfaces[_filter].iterrows():
    #                 _points.append([interf["X"], interf["Y"], interf["Z"]])
    #             if verbose:
    #                 print("3 points xyz:",_points)
    #
    #             # get plane normal from three points
    #             _normal = _get_plane_normal(_points[0], _points[1], _points[2], verbose=verbose)
    #             # get dip and azimuth
    #             _dip, _az = _get_dip(_normal)
    #             # now get centroid of three points
    #             _centroid = _get_centroid(_points[0], _points[1], _points[2])
    #             # set polarity according to overturned or not
    #             if -90 < _dip < 90:
    #                 _pol = 1
    #             else:
    #                 _pol = -1
    #
    #             _fmt = np.unique(self.interfaces[_filter]["formation"])[0]
    #             # _series = np.unique(self.interfaces[_filter]["series"])[0]
    #
    #             if verbose:
    #                 print("plane normal:", _normal)
    #                 print("dip", _dip)
    #                 print("az", _az)
    #                 print("centroid x,y,z:", _centroid)
    #
    #             _f = [_centroid[0], _centroid[1], _centroid[2], _dip, _az, _pol, _fmt, tri_id]
    #             _fs = pn.Series(_f, ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'triangle_id'])
    #             _df = _fs.to_frame().transpose()
    #             self.set_orientations(_df, append=True)
    #         elif len(self.interfaces[_filter]) > 3:
    #             print("More than three points share the same triangle-id: " + str(
    #                 tri_id) + ". Only exactly 3 points are supported.")
    #         elif len(self.interfaces[_filter]) < 3:
    #             print("Less than three points share the same triangle-id: " + str(
    #                 tri_id) + ". Only exactly 3 points are supported.")
    #


def get_orientation(normal):
    """Get orientation (dip, azimuth, polarity ) for points in all point set"""
    #    if "normal" not in dir(self):
    #        self.plane_fit()

    # calculate dip
    dip = np.arccos(normal[2]) / np.pi * 180.

    print(normal)

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
        self._grid_ext = extent
        self._grid_res = resolution

        self.dx, self.dy, self.dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0],\
                                    (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(self._grid_ext[0] + self.dx/2, self._grid_ext[1] - self.dx/2, self._grid_res[0], dtype="float32"),
            np.linspace(self._grid_ext[2] + self.dy/2, self._grid_ext[3] - self.dy/2, self._grid_res[1], dtype="float32"),
            np.linspace(self._grid_ext[4] + self.dz/2, self._grid_ext[5] - self.dz/2, self._grid_res[2], dtype="float32"), indexing="ij"
        )

        self.values = np.vstack(map(np.ravel, g)).T.astype("float32")
        return self.values



# DEP
# # TODO: @Alex documentation
# class FoliaitionsFromInterfaces:
#     def __init__(self, geo_data, group_id, mode, verbose=False):
#         """
#
#         Args:
#             geo_data: InputData object
#             group_id: (str) identifier for the data group
#             mode: (str), either 'interf_to_fol' or 'fol_to_interf'
#             verbose: (bool) adjusts verbosity, default False
#         """
#         self.geo_data = geo_data
#         self.group_id = group_id
#
#         if mode is "interf_to_fol":
#             # df bool filter
#             self._f = self.geo_data.interfaces["group_id"] == self.group_id
#             # get formation string
#             self.formation = self.geo_data.interfaces[self._f]["formation"].values[0]
#             # df indices
#             self.interf_i = self.geo_data.interfaces[self._f].index
#             # get point coordinates from df
#             self.interf_p = self._get_points()
#             # get point cloud centroid and normal vector of plane
#             self.centroid, self.normal = self._fit_plane_svd()
#             # get dip and azimuth of plane from normal vector
#             self.dip, self.azimuth, self.polarity = self._get_dip()
#
#         elif mode == "fol_to_interf":
#             self._f = self.geo_data.orientations["group_id"] == self.group_id
#             self.formation = self.geo_data.orientations[self._f]["formation"].values[0]
#
#             # get interface indices
#             self.interf_i = self.geo_data.interfaces[self.geo_data.interfaces["group_id"]==self.group_id].index
#             # get interface point coordinates from df
#             self.interf_p = self._get_points()
#             self.normal = [self.geo_data.orientations[self._f]["G_x"],
#                            self.geo_data.orientations[self._f]["G_y"],
#                            self.geo_data.orientations[self._f]["G_z"]]
#             self.centroid = [self.geo_data.orientations[self._f]["X"],
#                              self.geo_data.orientations[self._f]["Y"],
#                              self.geo_data.orientations[self._f]["Z"]]
#             # modify all Z of interface points belonging to group_id to fit plane
#             self._fol_to_p()
#
#         else:
#             print("Mode must be either 'interf_to_fol' or 'fol_to_interf'.")
#
#     def _fol_to_p(self):
#         a, b, c = self.normal
#         d = -a * self.centroid[0] - b * self.centroid[1] - c * self.centroid[2]
#         for i, row in self.geo_data.interfaces[self.geo_data.interfaces["group_id"] == self.group_id].iterrows():
#             # iterate over each point and recalculate Z, set Z
#             # x, y, z = row["X"], row["Y"], row["Z"]
#             Z = (a*row["X"] + b*row["Y"] + d)/-c
#             self.geo_data.interfaces.set_value(i, "Z", Z)
#
#     def _get_points(self):
#         """Returns n points from geo_data.interfaces matching group_id in np.array shape (n, 3)."""
#         # TODO: zip
#         x = []
#         y = []
#         z = []
#         for i, row in self.geo_data.interfaces[self.geo_data.interfaces["group_id"]==self.group_id].iterrows():
#             x.append(float(row["X"]))
#             y.append(float(row["Y"]))
#             z.append(float(row["Z"]))
#         return np.array([x, y, z])
#
#     def _fit_plane_svd(self):
#         """Fit plane to points using singular value decomposition (svd). Returns point cloud centroid [x,y,z] and
#         normal vector of plane [x,y,z]."""
#         from numpy.linalg import svd
#         # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
#         ctr = self.interf_p.mean(axis=1)  # calculate point cloud centroid [x,y,z]
#         x = self.interf_p - ctr[:, np.newaxis]
#         m = np.dot(x, x.T)  # np.cov(x)
#         return ctr, svd(m)[0][:, -1]
#
#     def _get_dip(self):
#         """Returns dip angle and azimuth of normal vector [x,y,z]."""
#         dip = np.arccos(self.normal[2] / np.linalg.norm(self.normal)) / np.pi * 180.
#
#         azimuth = None
#         if self.normal[0] >= 0 and self.normal[1] > 0:
#             azimuth = np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
#         # border cases where arctan not defined:
#         elif self.normal[0] > 0 and self.normal[1] == 0:
#             azimuth = 90
#         elif self.normal[0] < 0 and self.normal[1] == 0:
#             azimuth = 270
#         elif self.normal[1] < 0:
#             azimuth = 180 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
#         elif self.normal[1] >= 0 < self.normal[0]:
#             azimuth = 360 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
#
#         if -90 < dip < 90:
#             polarity = 1
#         else:
#             polarity = -1
#
#         return dip, azimuth, polarity
#
#     def set_fol(self):
#         """Appends orientation data point for group_id to geo_data.orientations."""
#         if "group_id" not in self.geo_data.orientations.columns:
#             self.geo_data.orientations["group_id"] = "NaN"
#         fol = [self.centroid[0], self.centroid[1], self.centroid[2],
#                self.dip, self.azimuth, self.polarity,
#                self.formation, self.group_id]
#         fol_series = pn.Series(fol, ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'group_id'])
#         fol_df = fol_series.to_frame().transpose()
#         self.geo_data.set_orientations(fol_df, append=True)
#
#     def _get_plane_normal(A, B, C, verbose=False):
#         """Returns normal vector of plane defined by points A,B,C as [x,y,z]."""
#         A = np.array(A)
#         B = np.array(B)
#         C = np.array(C)
#
#         v1 = C - A
#         v2 = B - A
#         if verbose:
#             print("vector C-A", v1)
#             print("vector B-A", v2)
#
#         return np.cross(v1, v2)
#
#     def _get_centroid(A, B, C):
#         """Returns centroid (x,y,z) of three points 3x[x,y,z]."""
#         X = (A[0] + B[0] + C[0]) / 3
#         Y = (A[1] + B[1] + C[1]) / 3
#         Z = (A[2] + B[2] + C[2]) / 3
#         return X, Y, Z