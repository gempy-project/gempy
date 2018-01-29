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
            import warnings
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
        self.orientations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity',
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
                                 self.orientations["polarity"].astype('float')
        self.orientations['G_y'] = np.sin(np.deg2rad(self.orientations["dip"].astype('float'))) * \
                                 np.cos(np.deg2rad(self.orientations["azimuth"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')
        self.orientations['G_z'] = np.cos(np.deg2rad(self.orientations["dip"].astype('float'))) *\
                                 self.orientations["polarity"].astype('float')

    def calculate_orientations(self):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.
        """
        self.orientations["dip"] = np.arccos(self.orientations["G_z"] / self.orientations["polarity"])
        self.orientations["azimuth"] = np.arcsin(self.orientations["G_x"]) / (np.sin(np.arccos(self.orientations["G_z"] / self.orientations["polarity"])) * self.orientations["polarity"])

    def count_faults(self):
        """
        Read the string names of the formations to detect automatically the number of faults.
        """
        faults_series = []
        for i in self.interfaces['series'].unique():
            if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                faults_series.append(i)
        return faults_series

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
            show_par_f = ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series']
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
            self.orientations = self.load_data_csv(data_type="orientations", path=path_o, **kwargs)
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.orientations.columns), \
                "One or more columns do not match with the expected values " + str(self.orientations.columns)
            self.orientations = self.orientations[['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']]

        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)

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
        self.interfaces["series"] = [(i == _series).sum().argmax() for i in self.interfaces["formation"]]
        self.interfaces["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.interfaces["formation"]]
        self.orientations["series"] = [(i == _series).sum().argmax() for i in self.orientations["formation"]]
        self.orientations["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.orientations["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
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

    # TODO Alex: Documentation
    def set_triangle_orientations(self, verbose=False):
        # next we need to iterate over every unique triangle id to create a orientation from each triplet
        # of points and assign the same triange_id to it
        tri_ids = np.unique(self.interfaces["triangle_id"])

        # check if column in orientations too, else create it
        if "triangle_id" not in self.orientations.columns:
            self.orientations["triangle_id"] = "NaN"
            if verbose:
                print("Setting triangle_id column in geo_data.orientations.")

        # loop over all triangle_id's
        for tri_id in tri_ids[tri_ids != "NaN"]:
            # get the three points dataframe
            _filter = self.interfaces["triangle_id"] == tri_id

            # check if triangle orientation value already exists
            if tri_id in np.unique(self.orientations["triangle_id"]):
                if verbose:
                    print("triangle_id already in geo_data.orientations - skipping it.")
                continue  # if yes, continue with the next iteration not not double append

            if verbose:
                print("tri_id: "+tri_id)
            if len(self.interfaces[_filter]) == 3:
                # get points as [x,y,z]
                _points = []
                for i, interf in self.interfaces[_filter].iterrows():
                    _points.append([interf["X"], interf["Y"], interf["Z"]])
                if verbose:
                    print("3 points xyz:",_points)

                # get plane normal from three points
                _normal = _get_plane_normal(_points[0], _points[1], _points[2], verbose=verbose)
                # get dip and azimuth
                _dip, _az = _get_dip(_normal)
                # now get centroid of three points
                _centroid = _get_centroid(_points[0], _points[1], _points[2])
                # set polarity according to overturned or not
                if -90 < _dip < 90:
                    _pol = 1
                else:
                    _pol = -1

                _fmt = np.unique(self.interfaces[_filter]["formation"])[0]
                # _series = np.unique(self.interfaces[_filter]["series"])[0]

                if verbose:
                    print("plane normal:", _normal)
                    print("dip", _dip)
                    print("az", _az)
                    print("centroid x,y,z:", _centroid)

                _f = [_centroid[0], _centroid[1], _centroid[2], _dip, _az, _pol, _fmt, tri_id]
                _fs = pn.Series(_f, ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'triangle_id'])
                _df = _fs.to_frame().transpose()
                self.set_orientations(_df, append=True)
            elif len(self.interfaces[_filter]) > 3:
                print("More than three points share the same triangle-id: " + str(
                    tri_id) + ". Only exactly 3 points are supported.")
            elif len(self.interfaces[_filter]) < 3:
                print("Less than three points share the same triangle-id: " + str(
                    tri_id) + ". Only exactly 3 points are supported.")


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


class InterpolatorData:
    """
    InterpolatorInput is a class that contains all the preprocessing operations to prepare the data to compute the model.
    Also is the object that has to be manipulated to vary the data without recompile the modeling function.

    Args:
        geo_data(gempy.data_management.InputData): All values of a DataManagement object
        geophysics(gempy.geophysics): Object with the corresponding geophysical precomputations
        compile_theano (bool): select if the theano function is compiled during the initialization. Default: True
        compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
         the block model of faults. If False only return the block model of lithologies. This may be important to speed
          up the computation. Default True
        u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
        be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
        depends on the number of points given as input_data to try to avoid singular matrix. NOTE: if during the computation
        of the model a singular matrix is returned try to reduce the u_grade of the series.
        rescaling_factor (float): rescaling factor of the input_data data to improve the stability when float32 is used. By
        defaut the rescaling factor is calculated to obtein values between 0 and 1.

    Keyword Args:
         dtype ('str'): Choosing if using float32 or float64. This is important if is intended to use the GPU
         See Also InterpolatorClass kwargs

    Attributes:
        geo_data: Original gempy.DataManagement.InputData object
        geo_data_res: Rescaled data. It has the same structure has gempy.InputData
        interpolator: Instance of the gempy.DataManagement.InterpolaorInput.InterpolatorClass. See Also
         gempy.DataManagement.InterpolaorInput.InterpolatorClass docs
         th_fn: Theano function which compute the interpolation
        dtype:  type of float

    """
    def __init__(self, geo_data, geophysics=None, output='geology', compile_theano=False, theano_optimizer='fast_compile',
                 u_grade=None, rescaling_factor=None, **kwargs):
        # TODO add all options before compilation in here. Basically this is n_faults, n_layers, verbose, dtype, and \
        # only block or all
        assert isinstance(geo_data, InputData), 'You need to pass a InputData object'

        # Store the original InputData object
        self._geo_data = geo_data

        # Here we can change the dtype for stability and GPU vs CPU
        self.dtype = kwargs.get('dtype', 'float32')

        # Set some parameters. TODO possibly this should go in kwargs
        self.u_grade = u_grade

        # This two properties get set calling rescale data
        self.rescaling_factor = None
        self.centers = None
        self.extent_rescaled = None

        # Rescaling
        self.geo_data_res = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)

        # Creating interpolator class with all the precompilation options
        self.interpolator = self.InterpolatorTheano(self, output=output, theano_optimizer=theano_optimizer, **kwargs)
        if compile_theano:
            self.th_fn = self.compile_th_fn(output)

        self.geophy = geophysics

    def compile_th_fn(self, output):
        """
        Compile the theano function given the input_data data.

        Args:
            compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
             the block model of faults. If False only return the block model of lithologies. This may be important to speed
              up the computation. Default True

        Returns:
            theano.function: Compiled function if C or CUDA which computes the interpolation given the input_data data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
        """


        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.interpolator.tg.input_parameters_list()

        print('Compiling theano function...')

        if output is 'geology':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_geological_model(self.geo_data_res.n_faults),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        if output is 'gravity':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_forward_gravity(self.geo_data_res.n_faults),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)
        print('Compilation Done!')
        print('Level of Optimization: ', theano.config.optimizer)
        print('Device: ', theano.config.device)
        print('Precision: ', self.dtype)
        return th_fn

    def rescale_data(self, geo_data, rescaling_factor=None):
        """
        Rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.

        Args:
            geo_data: Original gempy.DataManagement.InputData object
            rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

        Returns:
            gempy.data_management.InputData: Rescaled data

        """

        # Check which axis is the largest
        max_coord = pn.concat(
            [geo_data.orientations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
        min_coord = pn.concat(
            [geo_data.orientations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

        # Compute rescalin factor if not given
        if not rescaling_factor:
            rescaling_factor = 2 * np.max(max_coord - min_coord)

        # Get the centers of every axis
        centers = (max_coord + min_coord) / 2

        # Change the coordinates of interfaces
        new_coord_interfaces = (geo_data.interfaces[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        # Change the coordinates of orientations
        new_coord_orientations = (geo_data.orientations[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        # Rescaling the std in case of stochastic values
        try:
            geo_data.interfaces[['X_std', 'Y_std', 'Z_std']] = (geo_data.interfaces[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
            geo_data.orientations[['X_std', 'Y_std', 'Z_std']] = (geo_data.orientations[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
        except KeyError:
            pass

        # Updating properties
        new_coord_extent = (geo_data.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001

        geo_data_rescaled = copy.deepcopy(geo_data)
        geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
        geo_data_rescaled.orientations[['X', 'Y', 'Z']] = new_coord_orientations
        geo_data_rescaled.extent = new_coord_extent.as_matrix()

        geo_data_rescaled.grid.values = (geo_data.grid.values - centers.as_matrix()) / rescaling_factor + 0.5001

        # Saving useful values for later
        self.rescaling_factor = rescaling_factor
        geo_data_rescaled.rescaling_factor = rescaling_factor
        self.centers = centers

        return geo_data_rescaled

    def set_geo_data_rescaled(self, geo_data, rescaling_factor=None):
        """
        Set the rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.

        Args:
             geo_data: Original gempy.DataManagement.InputData object
             rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

        Returns:
             gempy.data_management.InputData: Rescaled data

         """
        self.geo_data_res = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)

    def update_interpolator(self, geo_data=None, **kwargs):
        """
        Method to update the constant parameters of the class interpolator (i.e. theano shared) WITHOUT recompiling.
        All the constant parameters for the interpolation can be passed
        as kwargs, otherwise they will take the default value (TODO: documentation of the dafault values)

        Args:
            geo_data: Rescaled gempy.DataManagement.InputData object. If None the stored geo_data_res will be used

        Keyword Args:
           range_var: Range of the variogram. Default None
           c_o: Covariance at 0. Default None
           nugget_effect: Nugget effect of the gradients. Default 0.01
           u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
        """

        if geo_data:
            geo_data_in = self.rescale_data(geo_data)
            self.geo_data_res = geo_data_in
        else:
            geo_data_in = self.geo_data_res

        # I update the interpolator data
        self.interpolator.geo_data_res = geo_data_in
        self.interpolator.order_table()
        self.interpolator.set_theano_shared_parameteres(**kwargs)
        self.interpolator.data_prep()

    def get_input_data(self, u_grade=None):
        """
        Get the theano variables that are input_data. This are necessary to compile the theano function
        or a theno op for pymc3
        Args:
             u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
            be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
            depends on the number of points given as input_data to try to avoid singular matrix. NOTE: if during the computation
            of the model a singular matrix is returned try to reduce the u_grade of the series.

        Returns:
            theano.variables: input_data nodes of the theano graph
        """

        if not u_grade:
            u_grade = self.u_grade
        return self.interpolator.data_prep(u_grade=u_grade)

    # =======
    # Gravity
    def create_geophysics_obj(self, ai_extent, ai_resolution, ai_z=None, range_max=None):
        from .geophysics import GravityPreprocessing
        self.geophy = GravityPreprocessing(self, ai_extent, ai_resolution, ai_z=ai_z, range_max=range_max)

    def set_gravity_precomputation(self, gravity_obj):
        """
        Set a gravity object to the interpolator to pass the values to the theano graph

        Args:
            gravity_obj (gempy.geophysics.GravityPreprocessing): GravityPreprocessing
             (See Also gempy.geophysics.GravityPreprocessing documentation)

        """
        # TODO assert that is a gravity object
        self.geophy = gravity_obj

    class InterpolatorTheano(object):
        """
         Class which contain all needed methods to perform potential field implicit modelling in theano.
         Here there are methods to modify the shared parameters of the theano graph as well as the final
         preparation of the data from DataFrames to numpy arrays. This class is intended to be hidden from the user
         leaving the most useful calls into the InterpolatorData class

        Args:
            interp_data (gempy.data_management.InterpolatorData): InterpolatorData: data rescaled plus geophysics and
            other additional data

        Keyword Args:
            range_var: Range of the variogram. Default None
            c_o: Covariance at 0. Default None
            nugget_effect: Nugget effect of the gradients. Default 0.01
            u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
            rescaling_factor: Magic factor that multiplies the covariances). Default 2
            verbose(list of str): Level of verbosity during the execution of the functions. List of the strings with
            the parameters to be printed during the theano execution. TODO Make the list
        """

        def __init__(self, interp_data, **kwargs):

            import importlib
            importlib.reload(theano_graph)

            # verbose is a list of strings. See theanograph
            verbose = kwargs.get('verbose', [0])

            # Here we can change the dtype for stability and GPU vs CPU
            self.dtype = kwargs.get('dtype', 'float32')
            # Here we change the graph type
            output = kwargs.get('output', 'geology')
            # Optimization flag
            theano_optimizer = kwargs.get('theano_optimizer', 'fast_compile')

            self.output = output

            if 'dtype' in verbose:
                print(self.dtype)

            # Drift grade
            u_grade = kwargs.get('u_grade', [3, 3])

            # We hide the scaled copy of DataManagement object from the user.
            self.geo_data_res = interp_data.geo_data_res

            # Importing the theano graph. The methods of this object generate different parts of graph.
            # See theanograf doc
            self.tg = theano_graph.TheanoGraph(output=output, optimizer=theano_optimizer, dtype=self.dtype, verbose=verbose, )

            # Avoid crashing my pc
            import theano
            if theano.config.optimizer != 'fast_run':
                assert self.tg.grid_val_T.get_value().shape[0] * \
                       np.math.factorial(len(self.tg.len_series_i.get_value())) < 2e7, \
                    'The grid is too big for the number of scalar fields. Reduce the grid or change the' \
                    'optimization flag to fast run'

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Setting theano parameters
            self.set_theano_shared_parameteres(**kwargs)

            # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            self.data_prep(**kwargs)

        def set_formation_number(self):
            """
            Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
            to know it and also now the numbers must be set in the order of the series as well. Therefore this method
            has been moved to the interpolator class as preprocessing

            Returns:
                Column in the interfaces and orientations dataframes
            """
            try:
                ip_addresses = self.geo_data_res.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self.geo_data_res.interfaces['formation number'] = self.geo_data_res.interfaces['formation'].replace(ip_dict)
                self.geo_data_res.orientations['formation number'] = self.geo_data_res.orientations['formation'].replace(ip_dict)
            except ValueError:
                pass

        def order_table(self):
            """
            First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
            the formations. All inplace
            """

            # We order the pandas table by series
            self.geo_data_res.interfaces.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self.geo_data_res.orientations.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Give formation number
            if not 'formation number' in self.geo_data_res.interfaces.columns:
                # print('I am here')
                self.set_formation_number()

            # We order the pandas table by formation (also by series in case something weird happened)
            self.geo_data_res.interfaces.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self.geo_data_res.orientations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self.geo_data_res.interfaces.reset_index(drop=True, inplace=True)

        def data_prep(self, **kwargs):
            """
            Ideally this method will only extract the data from the pandas dataframes to individual numpy arrays to be input_data
            of the theano function. However since some of the shared parameters are function of these arrays shape I also
            set them here

            Returns:
                idl (list): List of arrays which are the input_data for the theano function:

                    - numpy.array: dips_position
                    - numpy.array: dip_angles
                    - numpy.array: azimuth
                    - numpy.array: polarity
                    - numpy.array: ref_layer_points
                    - numpy.array: rest_layer_points
            """
            verbose = kwargs.get('verbose', [])
            u_grade = kwargs.get('u_grade', None)
            # ==================
            # Extracting lengths
            # ==================
            # Array containing the size of every formation. Interfaces
            len_interfaces = np.asarray(
                [np.sum(self.geo_data_res.interfaces['formation number'] == i)
                 for i in self.geo_data_res.interfaces['formation number'].unique()])

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form)

            # Position of the first point of every layer
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop the reference points using pandas indeces to get just the rest_layers array
            pandas_rest_layer_points = self.geo_data_res.interfaces.drop(ref_position)
            self.pandas_rest_layer_points = pandas_rest_layer_points

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum())

            # Array containing the size of every series. orientations.
            len_series_f = np.asarray(
                [np.sum(self.geo_data_res.orientations['order_series'] == i)
                 for i in self.geo_data_res.orientations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum())

            # =========================
            # Choosing Universal drifts
            # =========================
            if u_grade is None:
                u_grade = np.zeros_like(len_series_i)
                u_grade[(len_series_i > 4)] = 1

            else:
                u_grade = np.array(u_grade)

            if 'u_grade' in verbose:
                print(u_grade)

            n_universal_eq = np.zeros_like(len_series_i)
            n_universal_eq[u_grade == 0] = 0
            n_universal_eq[u_grade == 1] = 3
            n_universal_eq[u_grade == 2] = 9

            # it seems I have to pass list instead array_like that is weird
            self.tg.n_universal_eq_T.set_value(list(n_universal_eq))

            # ================
            # Prepare Matrices
            # ================
            # Rest layers matrix # PYTHON VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            self.pandas_ref_layer_points = self.geo_data_res.interfaces.iloc[ref_position]
            self.len_interfaces = len_interfaces

            pandas_ref_layer_points_rep = self.pandas_ref_layer_points.apply(lambda x: np.repeat(x, len_interfaces - 1))
            ref_layer_points = pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()

            self.ref_layer_points = ref_layer_points
            self.pandas_ref_layer_points_rep = pandas_ref_layer_points_rep
            # Check no reference points in rest points (at least in coor x)
            assert not any(ref_layer_points[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # orientations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self.geo_data_res.orientations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self.geo_data_res.orientations["dip"].as_matrix()
            azimuth = self.geo_data_res.orientations["azimuth"].as_matrix()
            polarity = self.geo_data_res.orientations["polarity"].as_matrix()

            # Set all in a list casting them in the chosen dtype
            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def set_theano_shared_parameteres(self, **kwargs):
            """
            Here we create most of the kriging parameters. The user can pass them as kwargs otherwise we pick the
            default values from the DataManagement info. The share variables are set in place. All the parameters here
            are independent of the input_data data so this function only has to be called if you change the extent or grid or
            if you want to change one the kriging parameters.

            Keyword Args:
                u_grade (int): Drift grade. Default to 2.
                range_var (float): Range of the variogram. Default 3D diagonal of the extent
                c_o (float): Covariance at lag 0. Default range_var ** 2 / 14 / 3. See my paper when I write it
                nugget_effect (flaot): Nugget effect of orientations. Default to 0.01
            """

            # Kwargs
            # --This is DEP because is a condition not a shared-- u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)

            # Default range
            if not range_var:
                range_var = np.sqrt((self.geo_data_res.extent[0] - self.geo_data_res.extent[1]) ** 2 +
                                    (self.geo_data_res.extent[2] - self.geo_data_res.extent[3]) ** 2 +
                                    (self.geo_data_res.extent[4] - self.geo_data_res.extent[5]) ** 2)

            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Creating the drift matrix.
            universal_matrix = np.vstack((self.geo_data_res.grid.values.T,
                                         (self.geo_data_res.grid.values ** 2).T,
                                          self.geo_data_res.grid.values[:, 0] * self.geo_data_res.grid.values[:, 1],
                                          self.geo_data_res.grid.values[:, 0] * self.geo_data_res.grid.values[:, 2],
                                          self.geo_data_res.grid.values[:, 1] * self.geo_data_res.grid.values[:, 2]))

            # Setting shared variables
            # Range
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))

            # Covariance at 0
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))

            # orientations nugget effect
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            # Just grid. I add a small number to avoid problems with the origin point
            self.tg.grid_val_T.set_value(np.cast[self.dtype](self.geo_data_res.grid.values + 10e-6))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](universal_matrix + 1e-10))

            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((1, self.geo_data_res.grid.values.shape[0]), dtype=self.dtype))

            # TODO DEP?
            # Initialization of the boolean array that represent the areas of the block model to be computed in the
            # following series
            #self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))

            # Unique number assigned to each lithology
            self.tg.n_formation.set_value(self.geo_data_res.interfaces['formation number'].unique().astype('int64'))

            # Number of formations per series. The function is not pretty but the result is quite clear
            self.tg.n_formations_per_serie.set_value(
                np.insert(self.geo_data_res.interfaces.groupby('order_series').formation.nunique().values.cumsum(), 0, 0))

            # Init the list to store the values at the interfaces. Here we init the shape for the given dataset
            self.tg.final_scalar_field_at_formations.set_value(np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                                                                        dtype=self.dtype))
            self.tg.final_scalar_field_at_faults.set_value(np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                                                                    dtype=self.dtype))

            # TODO: Push this condition to the geo_data
            # Set fault relation matrix
            if self.geo_data_res.fault_relation is not None:
                self.tg.fault_relation.set_value(self.geo_data_res.fault_relation.astype('int'))
            else:
                fault_rel = np.zeros((self.geo_data_res.interfaces['series'].nunique(),
                                      self.geo_data_res.interfaces['series'].nunique()))

                self.tg.fault_relation.set_value(fault_rel.astype('int'))

        # TODO change name to weights!
        def set_densities(self, densities):
            """
            WORKING IN PROGRESS -- Set the weight of each voxel given a density
            Args:
                densities:

            Returns:

            """
            resolution = [50,50,50]

            #
            dx, dy, dz = (self.geo_data_res.extent[1] - self.geo_data_res.extent[0]) / resolution[0], (self.geo_data_res.extent[3] - self.geo_data_res.extent[2]) / resolution[
                0], (self.geo_data_res.extent[5] - self.geo_data_res.extent[4]) / resolution[0]

            #dx, dy, dz = self.geo_data_res.grid.dx, self.geo_data_res.grid.dy, self.geo_data_res.grid.dz
            weight = (
                #(dx * dy * dz) *
                 np.array(densities))

            self.tg.densities.set_value(np.array(weight, dtype=self.dtype))

        def set_z_comp(self, tz, selected_cells):
            """
            Set z component precomputation for the gravity.
            Args:
                tz:
                selected_cells:

            Returns:

            """


            self.tg.tz.set_value(tz.astype(self.dtype))
            self.tg.select.set_value(selected_cells.astype(bool))

        def get_kriging_parameters(self, verbose=0):
            """
            Print the kringing parameters

            Args:
                verbose (int): if > 0 print all the shape values as well.

            Returns:
                None
            """
            # range
            print('range', self.tg.a_T.get_value(), self.tg.a_T.get_value() * self.geo_data_res.rescaling_factor)
            # Number of drift equations
            print('Number of drift equations', self.tg.n_universal_eq_T.get_value())
            # Covariance at 0
            print('Covariance at 0', self.tg.c_o_T.get_value())
            # orientations nugget effect
            print('orientations nugget effect', self.tg.nugget_effect_grad_T.get_value())

            if verbose > 0:
                # Input data shapes

                # Lenght of the interfaces series
                print('Length of the interfaces series', self.tg.len_series_i.get_value())
                # Length of the orientations series
                print('Length of the orientations series', self.tg.len_series_f.get_value())
                # Number of formation
                print('Number of formations', self.tg.n_formation.get_value())
                # Number of formations per series
                print('Number of formations per series', self.tg.n_formations_per_serie.get_value())
                # Number of points per formation
                print('Number of points per formation (rest)', self.tg.number_of_points_per_formation_T.get_value())

# TODO: @Alex documentation
class FoliaitionsFromInterfaces:
    def __init__(self, geo_data, group_id, mode, verbose=False):
        """

        Args:
            geo_data: InputData object
            group_id: (str) identifier for the data group
            mode: (str), either 'interf_to_fol' or 'fol_to_interf'
            verbose: (bool) adjusts verbosity, default False
        """
        self.geo_data = geo_data
        self.group_id = group_id

        if mode is "interf_to_fol":
            # df bool filter
            self._f = self.geo_data.interfaces["group_id"] == self.group_id
            # get formation string
            self.formation = self.geo_data.interfaces[self._f]["formation"].values[0]
            # df indices
            self.interf_i = self.geo_data.interfaces[self._f].index
            # get point coordinates from df
            self.interf_p = self._get_points()
            # get point cloud centroid and normal vector of plane
            self.centroid, self.normal = self._fit_plane_svd()
            # get dip and azimuth of plane from normal vector
            self.dip, self.azimuth, self.polarity = self._get_dip()

        elif mode == "fol_to_interf":
            self._f = self.geo_data.orientations["group_id"] == self.group_id
            self.formation = self.geo_data.orientations[self._f]["formation"].values[0]

            # get interface indices
            self.interf_i = self.geo_data.interfaces[self.geo_data.interfaces["group_id"]==self.group_id].index
            # get interface point coordinates from df
            self.interf_p = self._get_points()
            self.normal = [self.geo_data.orientations[self._f]["G_x"],
                           self.geo_data.orientations[self._f]["G_y"],
                           self.geo_data.orientations[self._f]["G_z"]]
            self.centroid = [self.geo_data.orientations[self._f]["X"],
                             self.geo_data.orientations[self._f]["Y"],
                             self.geo_data.orientations[self._f]["Z"]]
            # modify all Z of interface points belonging to group_id to fit plane
            self._fol_to_p()

        else:
            print("Mode must be either 'interf_to_fol' or 'fol_to_interf'.")

    def _fol_to_p(self):
        a, b, c = self.normal
        d = -a * self.centroid[0] - b * self.centroid[1] - c * self.centroid[2]
        for i, row in self.geo_data.interfaces[self.geo_data.interfaces["group_id"] == self.group_id].iterrows():
            # iterate over each point and recalculate Z, set Z
            # x, y, z = row["X"], row["Y"], row["Z"]
            Z = (a*row["X"] + b*row["Y"] + d)/-c
            self.geo_data.interfaces.set_value(i, "Z", Z)

    def _get_points(self):
        """Returns n points from geo_data.interfaces matching group_id in np.array shape (n, 3)."""
        # TODO: zip
        x = []
        y = []
        z = []
        for i, row in self.geo_data.interfaces[self.geo_data.interfaces["group_id"]==self.group_id].iterrows():
            x.append(float(row["X"]))
            y.append(float(row["Y"]))
            z.append(float(row["Z"]))
        return np.array([x, y, z])

    def _fit_plane_svd(self):
        """Fit plane to points using singular value decomposition (svd). Returns point cloud centroid [x,y,z] and
        normal vector of plane [x,y,z]."""
        from numpy.linalg import svd
        # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
        ctr = self.interf_p.mean(axis=1)  # calculate point cloud centroid [x,y,z]
        x = self.interf_p - ctr[:, np.newaxis]
        m = np.dot(x, x.T)  # np.cov(x)
        return ctr, svd(m)[0][:, -1]

    def _get_dip(self):
        """Returns dip angle and azimuth of normal vector [x,y,z]."""
        dip = np.arccos(self.normal[2] / np.linalg.norm(self.normal)) / np.pi * 180.

        azimuth = None
        if self.normal[0] >= 0 and self.normal[1] > 0:
            azimuth = np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
        # border cases where arctan not defined:
        elif self.normal[0] > 0 and self.normal[1] == 0:
            azimuth = 90
        elif self.normal[0] < 0 and self.normal[1] == 0:
            azimuth = 270
        elif self.normal[1] < 0:
            azimuth = 180 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
        elif self.normal[1] >= 0 < self.normal[0]:
            azimuth = 360 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.

        if -90 < dip < 90:
            polarity = 1
        else:
            polarity = -1

        return dip, azimuth, polarity

    def set_fol(self):
        """Appends orientation data point for group_id to geo_data.orientations."""
        if "group_id" not in self.geo_data.orientations.columns:
            self.geo_data.orientations["group_id"] = "NaN"
        fol = [self.centroid[0], self.centroid[1], self.centroid[2],
               self.dip, self.azimuth, self.polarity,
               self.formation, self.group_id]
        fol_series = pn.Series(fol, ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'group_id'])
        fol_df = fol_series.to_frame().transpose()
        self.geo_data.set_orientations(fol_df, append=True)

    def _get_plane_normal(A, B, C, verbose=False):
        """Returns normal vector of plane defined by points A,B,C as [x,y,z]."""
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        v1 = C - A
        v2 = B - A
        if verbose:
            print("vector C-A", v1)
            print("vector B-A", v2)

        return np.cross(v1, v2)

    def _get_centroid(A, B, C):
        """Returns centroid (x,y,z) of three points 3x[x,y,z]."""
        X = (A[0] + B[0] + C[0]) / 3
        Y = (A[1] + B[1] + C[1]) / 3
        Z = (A[2] + B[2] + C[2]) / 3
        return X, Y, Z