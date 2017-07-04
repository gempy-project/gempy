from __future__ import division

import os
from os import path
import sys

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import copy
import numpy as np
import pandas as pn
from gempy import theanograf
import theano

class InputData(object):
    """
    -DOCS NOT UPDATED- Class to import the raw data of the model and set data classifications into formations and series

    Args:
        extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_f: Path to the data bases of foliations. Default os.getcwd()

    Attributes:
        extent(list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution ((Optional[list])): [nx, ny, nz]
        Foliations(pandas.core.frame.DataFrame): Pandas data frame with the foliations data
        Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        formations(numpy.ndarray): Dictionary that contains the name of the formations
        series(pandas.core.frame.DataFrame): Pandas data frame which contains every formation within each series
    """

    # TODO: Data management using pandas, find an easy way to add values
    # TODO: Probably at some point I will have to make an static and dynamic data classes
    def __init__(self,
                 extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_f=None,
                 **kwargs):

        # Set extent and resolution
        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

        self.n_faults = 0

        # TODO choose the default source of data. So far only csv
        # Create the pandas dataframes

        # if we dont read a csv we create an empty dataframe with the columns that have to be filled
        self.foliations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity',
                                                'formation', 'series', 'X_std', 'Y_std', 'Z_std',
                                                'dip_std', 'azimuth_std'])

        self.interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series',
                                                'X_std', 'Y_std', 'Z_std'])

        if path_f or path_i:
            self.import_data(path_i=path_i, path_f=path_f)

        # DEP-
        self._set_formations()

        # If not provided set default series
        self.series = self.set_series()
        # DEP- self.set_formation_number()

        # Compute gradients given azimuth and dips to plot data
        self.calculate_gradient()

        # Create default grid object. TODO: (Is this necessary now?)
        self.grid = self.set_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)

    def import_data(self, path_i, path_f, **kwargs):
        """

        Args:
            path_i:
            path_f:
            **kwargs:

        Returns:

        """

        if path_f:
            self.foliations = self.load_data_csv(data_type="foliations", path=path_f, **kwargs)
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.foliations.columns), \
                "One or more columns do not match with the expected values " + str(self.foliations.columns)

        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)

    def _set_formations(self):
        """
        -DEPRECATED- Function to import the formations that will be used later on. By default all the formations in the tables are
        chosen.

        Returns:
             pandas.core.frame.DataFrame: Data frame with the raw data

        """

        try:
            # foliations may or may not be in all formations so we need to use interfaces
            self.formations = self.interfaces["formation"].unique()

            # TODO: Trying to make this more elegant?
            # for el in self.formations:
            #     for check in self.formations:
            #         assert (el not in check or el == check), "One of the formations name contains other" \
            #                                                  " string. Please rename." + str(el) + " in " + str(
            #             check)

                    # TODO: Add the possibility to change the name in pandas directly
                    # (adding just a 1 in the contained string)
        except AttributeError:
            pass

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the foliations

        Returns:
            self.foliations: extra columns with components xyz of the unity vector.
        """

        self.foliations['G_x'] = np.sin(np.deg2rad(self.foliations["dip"].astype('float'))) * \
                                 np.sin(np.deg2rad(self.foliations["azimuth"].astype('float'))) * \
                                 self.foliations["polarity"].astype('float')
        self.foliations['G_y'] = np.sin(np.deg2rad(self.foliations["dip"].astype('float'))) * \
                                 np.cos(np.deg2rad(self.foliations["azimuth"].astype('float'))) *\
                                 self.foliations["polarity"].astype('float')
        self.foliations['G_z'] = np.cos(np.deg2rad(self.foliations["dip"].astype('float'))) *\
                                 self.foliations["polarity"].astype('float')

    # DEP?
    def create_grid(self, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
        """
        Method to initialize the class grid. So far is really simple and only has the regular grid type

        Args:
            grid_type (str): regular_3D or regular_2D (I am not even sure if regular 2D still working)
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self.grid(GeMpy_core.grid): Object that contain different grids
        """

        if not extent:
            extent = self.extent
        if not resolution:
            resolution = self.resolution

        return self.GridClass(extent, resolution, grid_type=grid_type, **kwargs)

    def set_grid(self, new_grid=None, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
        """
        Method to initialize the class new_grid. So far is really simple and only has the regular new_grid type

        Args:
            grid_type (str): regular_3D or regular_2D (I am not even sure if regular 2D still working)
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self.new_grid(GeMpy_core.new_grid): Object that contain different grids
        """
        if new_grid is not None:
            assert new_grid.shape[1] is 3 and len(new_grid.shape) is 2, 'The shape of new grid must be (n,3) where n is' \
                                                                        'the number of points of the grid'
            self.grid.grid = new_grid
        else:
            if not extent:
                extent = self.extent
            if not resolution:
                resolution = self.resolution

            return self.GridClass(extent, resolution, grid_type=grid_type, **kwargs)

    def data_to_pickle(self, path=False):
        if not path:
            path = './geo_data'
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_raw_data(self, itype='all'):
        """
        Method that returns the interfaces and foliations pandas Dataframes. Can return both at the same time or only
        one of the two
        Args:
            itype: input data type, either 'foliations', 'interfaces' or 'all' for both.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        import pandas as pn
        if itype == 'foliations':
            raw_data = self.foliations
        elif itype == 'interfaces':
            raw_data = self.interfaces
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces, self.foliations], keys=['interfaces', 'foliations'])
        return raw_data

    def i_open_set_data(self, itype="foliations"):
        """
        Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
         the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
         easier solution than calling two different methods.
        Args:
            itype: input data type, either 'foliations' or 'interfaces'

        Returns:
            pandas.core.frame.DataFrame: Data frame with the changed data on real time
        """

        # if the data frame is empty the interactive table is bugged. Therefore I create a default raw when the method
        # is called
        if self.foliations.empty:
            self.foliations = pn.DataFrame(
                np.array([0., 0., 0., 0., 0., 1., 'Default Formation', 'Default series']).reshape(1, 8),
                columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series']).\
                convert_objects(convert_numeric=True)

        if self.interfaces.empty:
            self.interfaces = pn.DataFrame(
                np.array([0, 0, 0, 'Default Formation', 'Default series']).reshape(1, 5),
                columns=['X', 'Y', 'Z', 'formation', 'series']).convert_objects(convert_numeric=True)

        # TODO leave qgrid as a dependency since in the end I did not change the code of the package
        import qgrid

        # Setting some options
        qgrid.nbinstall(overwrite=True)
        qgrid.set_defaults(show_toolbar=True)
        assert itype is 'foliations' or itype is 'interfaces', 'itype must be either foliations or interfaces'

        import warnings
        warnings.warn('Remember to call i_close_set_data after the editing.')

        # We kind of set the show grid to a variable so we can close it afterwards
        self.pandas_frame = qgrid.show_grid(self.get_raw_data(itype=itype))

        # TODO set

    def i_close_set_data(self):

        """
        Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
         the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
         easier solution than calling two different methods.
        Args:
            itype: input data type, either 'foliations' or 'interfaces'

        Returns:
            pandas.core.frame.DataFrame: Data frame with the changed data on real time
        """
        # We close it to guarantee that after this method it is not possible further modifications
        self.pandas_frame.close()
        # -DEP- self._set_formations()
        # -DEP- self.set_formation_number()
        # Set parameters
        self.series = self.set_series()
        self.calculate_gradient()

    @staticmethod
    def load_data_csv(data_type, path=os.getcwd(), **kwargs):
        """
        Method to load either interface or foliations data csv files. Normally this is in which GeoModeller exports it

        Args:
            data_type (str): 'interfaces' or 'foliations'
            path (str): path to the files. Default os.getcwd()
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # TODO: in case that the columns have a different name specify in pandas which columns are interfaces /
        #  coordinates, dips and so on.
        # TODO: use pandas to read any format file not only csv

        if data_type == "foliations":
            return pn.read_csv(path, **kwargs)
        elif data_type == 'interfaces':
            return pn.read_csv(path, **kwargs)
        else:
            raise NameError('Data type not understood. Try interfaces or foliations')

        # TODO if we load different data the Interpolator parameters must be also updated. Prob call gradients and
        # series

    def set_interfaces(self, interf_Dataframe, append=False):
        """
        Method to change or append a Dataframe to interfaces in place.
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


        self._set_formations()
        self.set_series()
        #self.set_formation_number()
        self.interfaces.reset_index(drop=True, inplace=True)

    def set_foliations(self, foliat_Dataframe, append=False):
        """
          Method to change or append a Dataframe to foliations in place.
          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)
        if append:
            self.foliations = self.foliations.append(foliat_Dataframe)
        else:
            self.foliations = foliat_Dataframe


        self._set_formations()
        self.set_series()
        #self.set_formation_number()
        self.calculate_gradient()
        self.foliations.reset_index(drop=True, inplace=True)

    def set_series(self, series_distribution=None, order=None):
        """
        Method to define the different series of the project

        Args:
            series_distribution (dict): with the name of the serie as key and the name of the formations as values.
            order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
                random. This is important to set the erosion relations between the different series

        Returns:
            self.series: A pandas DataFrame with the series and formations relations
            self.interfaces: one extra column with the given series
            self.foliations: one extra column with the given series
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
        if not order:
            order = _series.keys()

        # TODO assert len order is equal to len of the dictionary

        # We create a dataframe with the links
        _series = pn.DataFrame(data=_series, columns=order)

        # Now we fill the column series in the interfaces and foliations tables with the correspondant series and
        # assigned number to the series
        self.interfaces["series"] = [(i == _series).sum().argmax() for i in self.interfaces["formation"]]
        self.interfaces["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.interfaces["formation"]]
        self.foliations["series"] = [(i == _series).sum().argmax() for i in self.foliations["formation"]]
        self.foliations["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.foliations["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
        self.interfaces.sort_values(by='order_series', inplace=True)
        self.foliations.sort_values(by='order_series', inplace=True)

        # Save the dataframe in a property
        self.series = _series

        # Set default faults
        faults_series = []
        for i in self.series.columns:
            if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                faults_series.append(i)

        self.set_faults(faults_series)
        self.reset_indices()

        return _series

    def set_faults(self, series_name):
        """

        Args:
            series_name(list or array_like):

        Returns:

        """
        if not len(series_name) == 0:
            self.interfaces['isFault'] = self.interfaces['series'].isin(series_name)
            self.foliations['isFault'] = self.foliations['series'].isin(series_name)
            self.n_faults = len(series_name)

    def set_formation_number(self, formation_order):
        """
        Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
        to know it and also now the numbers must be set in the order of the series as well. Therefore this method
        has been moved to the interpolator class as preprocessing

        Returns: Column in the interfaces and foliations dataframes
        """
        try:
            ip_addresses = formation_order
            ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses)+1)))
            self.interfaces['formation number'] = self.interfaces['formation'].replace(ip_dict)
            self.foliations['formation number'] = self.foliations['formation'].replace(ip_dict)
        except ValueError:
            pass

    def reset_indices(self):
        """
        Resets dataframe indices for foliations and interfaces.
        Returns: Nothing

        """
        self.interfaces.reset_index(inplace=True, drop=True)
        self.foliations.reset_index(inplace=True, drop=True)

    def interface_modify(self, index, **kwargs):
        """
        Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.
        Args:
            index: dataframe index of the foliation point
            **kwargs: X, Y, Z (int or float)

        Returns: Nothing

        """
        for key in kwargs:
            self.interfaces.ix[index, str(key)] = kwargs[key]

    def interface_add(self, **kwargs):
        """
        Adds interface to dataframe.
        Args:
            **kwargs: X, Y, Z, formation, labels, order_series, series

        Returns: Nothing

        """
        l = len(self.interfaces)
        for key in kwargs:
            self.interfaces.ix[l, str(key)] = kwargs[key]

    def interface_drop(self, index):
        """
        Drops interface from dataframe identified by index
        Args:
            index: dataframe index

        Returns: Nothing

        """
        self.interfaces.drop(index, inplace=True)

    def foliation_modify(self, index, **kwargs):
        """
        Allows modification of foliation data at specified dataframe index.
        Args:
            index: dataframe index of the foliation point
            **kwargs: G_x, G_y, G_z, X, Y, Z, azimuth, dip, formation, labels, order_series, polarity

        Returns: Nothing
        """
        for key in kwargs:
            self.foliations.ix[index, str(key)] = kwargs[key]

    def foliation_add(self, **kwargs):
        """
        Adds foliation to dataframe.
        Args:
            **kwargs: G_x, G_y, G_z, X, Y, Z, azimuth, dip, formation, labels, order_series, polarity, series

        Returns: Nothing

        """
        l = len(self.foliations)
        for key in kwargs:
            self.foliations.ix[l, str(key)] = kwargs[key]

    def foliations_drop(self, index):
        """
        Drops foliation from dataframe identified by index
        Args:
            index: dataframe index

        Returns: Nothing

        """
        self.foliations.drop(index, inplace=True)

    def get_formation_number(self):
        pn_series = self.interfaces.groupby('formation number').formation.unique()
        ip_addresses = {}
        for e, i in enumerate(pn_series):
            ip_addresses[i[0]] = e + 1
        ip_addresses['DefaultBasement'] = 0
        return ip_addresses

    # TODO think where this function should go
    def read_vox(self, path):
        """
        read vox from geomodeller and transform it to gempy format
        Returns:
            numpy.array: block model
        """

        geo_res = pn.read_csv(path)

        geo_res = geo_res.iloc[9:]

        #ip_addresses = geo_res['nx 50'].unique()  # geo_data.interfaces["formation"].unique()
        ip_dict = self.get_formation_number()

        geo_res_num = geo_res.iloc[:, 0].replace(ip_dict)
        block_geomodeller = np.ravel(geo_res_num.as_matrix().reshape(
                                        self.resolution[0], self.resolution[1], self.resolution[2], order='C').T)
        return block_geomodeller

    class GridClass(object):
        """
        -DOCS NOT UPDATED- Class with set of functions to generate grids

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].
            grid_type(str): Type of grid. So far only regular 3D is implemented
        """

        def __init__(self, extent, resolution, grid_type="regular_3D"):
            self._grid_ext = extent
            self._grid_res = resolution

            if grid_type == "regular_3D":
                self.grid = self.create_regular_grid_3d()
            elif grid_type == "regular_2D":
                self.grid = self.create_regular_grid_2d()
            else:
                print("Wrong type")

        def create_regular_grid_3d(self):
            """
            Method to create a 3D regular grid where is interpolated

            Returns:
                numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
            """

            g = np.meshgrid(
                np.linspace(self._grid_ext[0], self._grid_ext[1], self._grid_res[0], dtype="float32"),
                np.linspace(self._grid_ext[2], self._grid_ext[3], self._grid_res[1], dtype="float32"),
                np.linspace(self._grid_ext[4], self._grid_ext[5], self._grid_res[2], dtype="float32"), indexing="ij"
            )

          #  self.grid = np.vstack(map(np.ravel, g)).T.astype("float32")
            return np.vstack(map(np.ravel, g)).T.astype("float32")

    # DEP!
    class InterpolatorClass(object):
        """
        -DOCS NOT UPDATED- Class which contain all needed methods to perform potential field implicit modelling in theano

        Args:
            _data(GeMpy_core.DataManagement): All values of a DataManagement object
            _grid(GeMpy_core.grid): A grid object
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
        """

        def __init__(self, _data_scaled, _grid_scaled=None, *args, **kwargs):

            # verbose is a list of strings. See theanograph
            verbose = kwargs.get('verbose', [0])
            # -DEP-rescaling_factor = kwargs.get('rescaling_factor', None)

            # Here we can change the dtype for stability and GPU vs CPU
            dtype = kwargs.get('dtype', 'float32')
            self.dtype = dtype

            range_var = kwargs.get('range_var', None)

            # Drift grade
            u_grade = kwargs.get('u_grade', [2, 2])



            # We hide the scaled copy of DataManagement object from the user. The scaling happens in gempy what is a
            # bit weird. Maybe at some point I should bring the function to this module
            self._data_scaled = _data_scaled

            # In case someone wants to provide a grid otherwise we extract it from the DataManagement object.
            if not _grid_scaled:
                self._grid_scaled = _data_scaled.grid
            else:
                self._grid_scaled = _grid_scaled

            # Importing the theano graph. The methods of this object generate different parts of graph.
            # See theanograf doc
            self.tg = theanograf.TheanoGraph_pro(dtype=dtype, verbose=verbose,)

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Setting theano parameters
            self.set_theano_shared_parameteres(range_var=range_var)

            # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            self.data_prep(u_grade=u_grade)

            # Avoid crashing my pc
            import theano
            if theano.config.optimizer != 'fast_run':
                assert self.tg.grid_val_T.get_value().shape[0] * \
                       np.math.factorial(len(self.tg.len_series_i.get_value())) < 2e7, \
                       'The grid is too big for the number of potential fields. Reduce the grid or change the' \
                       'optimization flag to fast run'

        def set_formation_number(self):
            """
                    Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
                    to know it and also now the numbers must be set in the order of the series as well. Therefore this method
                    has been moved to the interpolator class as preprocessing

            Returns: Column in the interfaces and foliations dataframes
            """
            try:
                ip_addresses = self._data_scaled.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self._data_scaled.interfaces['formation number'] = self._data_scaled.interfaces['formation'].replace(ip_dict)
                self._data_scaled.foliations['formation number'] = self._data_scaled.foliations['formation'].replace(ip_dict)
            except ValueError:
                pass

        def order_table(self):
            """
            First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
            the formations. All inplace
            """

            # We order the pandas table by series
            self._data_scaled.interfaces.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Give formation number
            if not 'formation number' in self._data_scaled.interfaces.columns:
                print('I am here')
                self.set_formation_number()

            # We order the pandas table by formation (also by series in case something weird happened)
            self._data_scaled.interfaces.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self._data_scaled.interfaces.reset_index(drop=True, inplace=True)

        def data_prep(self, **kwargs):
            """
            Ideally this method will extract the data from the pandas dataframes to individual numpy arrays to be input
            of the theano function. However since some of the shared parameters are function of these arrays shape I also
            set them here
            Returns:
                idl (list): List of arrays which are the input for the theano function:
                    - numpy.array: dips_position
                    - numpy.array: dip_angles
                    - numpy.array: azimuth
                    - numpy.array: polarity
                    - numpy.array: ref_layer_points
                    - numpy.array: rest_layer_points
            """

            u_grade = kwargs.get('u_grade', None)
            # ==================
            # Extracting lengths
            # ==================
            # Array containing the size of every formation. Interfaces
            len_interfaces = np.asarray(
                [np.sum(self._data_scaled.interfaces['formation number'] == i)
                 for i in self._data_scaled.interfaces['formation number'].unique()])

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form)

            # Position of the first point of every layer
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop the reference points using pandas indeces to get just the rest_layers array
            pandas_rest_layer_points = self._data_scaled.interfaces.drop(ref_position)
            self.pandas_rest_layer_points = pandas_rest_layer_points
            # TODO: do I need this? PYTHON
            # DEP- because per series the foliations do not belong to a formation but to the whole series
            # len_foliations = np.asarray(
            #     [np.sum(self._data_scaled.foliations['formation number'] == i)
            #      for i in self._data_scaled.foliations['formation number'].unique()])

            # -DEP- I think this was just a kind of print to know what was going on
            #self.pandas_rest = pandas_rest_layer_points

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum())

            # Array containing the size of every series. Foliations.
            len_series_f = np.asarray(
                [np.sum(self._data_scaled.foliations['order_series'] == i)
                 for i in self._data_scaled.foliations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum())

            # =========================
            # Choosing Universal drifts
            # =========================

            if u_grade is None:
                u_grade = np.zeros_like(len_series_i)
                u_grade[len_series_i > 12] = 9
                u_grade[(len_series_i > 6) & (len_series_i < 12)] = 3
            print(u_grade)
            # it seems I have to pass list instead array_like that is weird
            self.tg.u_grade_T.set_value(list(u_grade))

            # ================
            # Prepare Matrices
            # ================
            # Rest layers matrix # PYTHON VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # TODO delete
            # -DEP- Again i was just a check point
            # self.rest_layer_points = rest_layer_points

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            aux_1 = self._data_scaled.interfaces.iloc[ref_position][['X', 'Y', 'Z']].as_matrix()

            # We initialize the matrix
            ref_layer_points = np.zeros((0, 3))

            # TODO I hate loop it has to be a better way
            # Tiling very reference points as many times as rest of the points we have
            for e, i in enumerate(len_interfaces):
                ref_layer_points = np.vstack((ref_layer_points, np.tile(aux_1[e], (i - 1, 1))))

            # -DEP- was just a check point
            #self.ref_layer_points = ref_layer_points

            # Check no reference points in rest points (at least in coor x)
            assert not any(aux_1[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # Foliations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self._data_scaled.foliations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data_scaled.foliations["dip"].as_matrix()
            azimuth = self._data_scaled.foliations["azimuth"].as_matrix()
            polarity = self._data_scaled.foliations["polarity"].as_matrix()

            # Set all in a list casting them in the chosen dtype
            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def set_theano_shared_parameteres(self, **kwargs):
            """
            Here we create most of the kriging parameters. The user can pass them as kwargs otherwise we pick the
            default values from the DataManagement info. The share variables are set in place. All the parameters here
            are independent of the input data so this function only has to be called if you change the extent or grid or
            if you want to change one the kriging parameters.
            Args:
                _data_rescaled: DataManagement object
                _grid_rescaled: Grid object
            Keyword Args:
                u_grade (int): Drift grade. Default to 2.
                range_var (float): Range of the variogram. Default 3D diagonal of the extent
                c_o (float): Covariance at lag 0. Default range_var ** 2 / 14 / 3. See my paper when I write it
                nugget_effect (flaot): Nugget effect of foliations. Default to 0.01
            """

            # Kwargs
            u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)

            # -DEP- Now I rescale the data so we do not need this
            # rescaling_factor = kwargs.get('rescaling_factor', None)

            # Default range
            if not range_var:
                range_var = np.sqrt((self._data_scaled.extent[0] - self._data_scaled.extent[1]) ** 2 +
                                    (self._data_scaled.extent[2] - self._data_scaled.extent[3]) ** 2 +
                                    (self._data_scaled.extent[4] - self._data_scaled.extent[5]) ** 2)


            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Asserting that the drift grade is in this range
           # assert (0 <= all(u_grade) <= 2)

            # Creating the drift matrix. TODO find the official name of this matrix?
            _universal_matrix = np.vstack((self._grid_scaled.grid.T,
                                           (self._grid_scaled.grid ** 2).T,
                                           self._grid_scaled.grid[:, 0] * self._grid_scaled.grid[:, 1],
                                           self._grid_scaled.grid[:, 0] * self._grid_scaled.grid[:, 2],
                                           self._grid_scaled.grid[:, 1] * self._grid_scaled.grid[:, 2]))

            # Setting shared variables
            # Range
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            # Covariance at 0
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            # Foliations nugget effect
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            # TODO change the drift to the same style I have the faults so I do not need to do this
            # # Drift grade
            # if u_grade == 0:
            #     self.tg.u_grade_T.set_value(u_grade)
            # else:
            #     self.tg.u_grade_T.set_value(u_grade)
                # TODO: To be sure what is the mathematical meaning of this -> It seems that nothing
                # TODO Deprecated
                # self.tg.c_resc.set_value(1)

            # Just grid. I add a small number to avoid problems with the origin point
            self.tg.grid_val_T.set_value(np.cast[self.dtype](self._grid_scaled.grid + 10e-6))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](_universal_matrix + 1e-10))

            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((1, self._grid_scaled.grid.shape[0]), dtype='float32'))

            # Initialization of the boolean array that represent the areas of the block model to be computed in the
            # following series
            #self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))

            # Unique number assigned to each lithology
            #self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
            #                                        0, 0)[::-1])

            self.tg.n_formation.set_value(self._data_scaled.interfaces['formation number'].unique())

            # Number of formations per series. The function is not pretty but the result is quite clear
            self.tg.n_formations_per_serie.set_value(
                np.insert(self._data_scaled.interfaces.groupby('order_series').formation.nunique().values.cumsum(), 0, 0))

        def get_kriging_parameters(self, verbose=0):
            # range
            print('range', self.tg.a_T.get_value(), self.tg.a_T.get_value() * self._data_scaled.rescaling_factor)
            # Number of drift equations
            print('Number of drift equations', self.tg.u_grade_T.get_value())
            # Covariance at 0
            print('Covariance at 0', self.tg.c_o_T.get_value())
            # Foliations nugget effect
            print('Foliations nugget effect', self.tg.nugget_effect_grad_T.get_value())

            if verbose > 0:
                # Input data shapes

                # Lenght of the interfaces series
                print('Length of the interfaces series', self.tg.len_series_i.get_value())
                # Length of the foliations series
                print('Length of the foliations series', self.tg.len_series_f.get_value())
                # Number of formation
                print('Number of formations', self.tg.n_formation.get_value())
                # Number of formations per series
                print('Number of formations per series', self.tg.n_formations_per_serie.get_value())
                # Number of points per formation
                print('Number of points per formation (rest)', self.tg.number_of_points_per_formation_T.get_value())


class InterpolatorInput:
    def __init__(self, geo_data, compile_theano=True, compute_all=True, u_grade=None, rescaling_factor=None, **kwargs):
        # TODO add all options before compilation in here. Basically this is n_faults, n_layers, verbose, dtype, and \
        # only block or all
        assert isinstance(geo_data, InputData), 'You need to pass a InputData object'
        # Here we can change the dtype for stability and GPU vs CPU
        self.dtype = kwargs.get('dtype', 'float32')


        #self.in_data = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)
        # Set some parameters. TODO posibly this should go in kwargs
        self.u_grade = u_grade

        # This two properties get set calling rescale data
        self.rescaling_factor = None
        self.centers = None
        self.extent_rescaled = None

        # Rescaling
        self.data = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)

        # Creating interpolator class with all the precompilation options
        self.interpolator = self.set_interpolator(**kwargs)

        if compile_theano:
            self.th_fn = self.compile_th_fn(compute_all=compute_all)

    # DEP all options since it goes in set_interpolator
    def compile_th_fn(self, compute_all=True, dtype=None, u_grade=None, **kwargs):
        """

        Args:
            geo_data:
            **kwargs:

        Returns:

        """

        # Choosing float precision for the computation

        if not dtype:
            if theano.config.device == 'gpu':
                dtype = 'float32'
            else:
                dtype = 'float64'

        # We make a rescaled version of geo_data for stability reasons
        #data_interp = self.set_interpolator(geo_data, dtype=dtype)

        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.interpolator.tg.input_parameters_list()

        # This prepares the user data to the theano function
        # input_data_P = data_interp.interpolator.data_prep(u_grade=u_grade)

        # then we compile we have to pass the number of formations that are faults!!
        th_fn = theano.function(input_data_T, self.interpolator.tg.whole_block_model(self.data.n_faults,
                                                                                     compute_all=compute_all),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)
        return th_fn

    def rescale_data(self, geo_data, rescaling_factor=None):
        """
        Rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.
        Args:
            geo_data: DataManagement object with the real scale data
            rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

        Returns:

        """
        # TODO split this function in compute rescaling factor and rescale z
        max_coord = pn.concat(
            [geo_data.foliations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
        min_coord = pn.concat(
            [geo_data.foliations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

        if not rescaling_factor:
            rescaling_factor = 2 * np.max(max_coord - min_coord)

        centers = (max_coord + min_coord) / 2

        new_coord_interfaces = (geo_data.interfaces[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        new_coord_foliations = (geo_data.foliations[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001
        try:
            geo_data.interfaces[['X_std', 'Y_std', 'Z_std']] = (geo_data.interfaces[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
            geo_data.foliations[['X_std', 'Y_std', 'Z_std']] = (geo_data.foliations[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
        except KeyError:
            pass

        new_coord_extent = (geo_data.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001

        geo_data_rescaled = copy.deepcopy(geo_data)
        geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
        geo_data_rescaled.foliations[['X', 'Y', 'Z']] = new_coord_foliations
        geo_data_rescaled.extent = new_coord_extent.as_matrix()

        geo_data_rescaled.grid.grid = (geo_data.grid.grid - centers.as_matrix()) / rescaling_factor + 0.5001

        self.rescaling_factor = rescaling_factor
        geo_data_rescaled.rescaling_factor = rescaling_factor
        self.centers = centers
        self.extent_rescaled = new_coord_extent

        return geo_data_rescaled

    # DEP?
    def set_airbore_plane(self, z, res_grav):

        # Rescale z
        z_res = (z-self.centers[2])/self.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(self.extent_rescaled.iloc[0],
                                     self.extent_rescaled.iloc[1], res_grav[0]),
                         np.linspace(self.extent_rescaled.iloc[2],
                                     self.extent_rescaled.iloc[3], res_grav[1]))
        z = np.ones(res_grav[0]*res_grav[1])*z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.dtype)

        return airborne_plane

    def set_interpolator(self, geo_data = None, *args, **kwargs):
        """
        Method to initialize the class interpolator. All the constant parameters for the interpolation can be passed
        as args, otherwise they will take the default value (TODO: documentation of the dafault values)

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            range_var: Range of the variogram. Default None
            c_o: Covariance at 0. Default None
            nugget_effect: Nugget effect of the gradients. Default 0.01
            u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
            rescaling_factor: Magic factor that multiplies the covariances). Default 2

        Returns:
            self.Interpolator (GeMpy_core.Interpolator): Object to perform the potential field method
            self.Plot(GeMpy_core.PlotData): Object to visualize data and results. It gets updated.
        """

        if 'u_grade' in kwargs:
            compile_theano = True

        range_var = kwargs.get('range_var', None)

        rescaling_factor = kwargs.get('rescaling_factor', None)


        #DEP?
        #if not getattr(geo_data, 'grid', None):
        #    set_grid(geo_data)

        if geo_data:
            geo_data_in = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)
            self.data = geo_data_in
        else:
            geo_data_in = self.data

        # First creation
        if not getattr(self, 'interpolator', None):
            print('I am in the setting')
            interpolator = self.InterpolatorClass(geo_data_in, geo_data_in.grid, *args, **kwargs)

        # Update
        else:
            print('I am in update')
            self.interpolator._data_scaled = geo_data_in
            self.interpolator._grid_scaled = geo_data_in.grid
            self.interpolator.order_table()
            self.interpolator.set_theano_shared_parameteres(range_var=range_var)
            interpolator = None

        return interpolator

    def update_interpolator(self, geo_data=None, *args, **kwargs):
        """
        Update variables without compiling the theano function
        Args:
            geo_data:
            *args:
            **kwargs:

        Returns:

        """
        if 'u_grade' in kwargs:
            compile_theano = True

        range_var = kwargs.get('range_var', None)

        rescaling_factor = kwargs.get('rescaling_factor', None)


        if geo_data:
            geo_data_in = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)
            self.data = geo_data_in
        else:
            geo_data_in = self.data

        print('I am in update')
        self.interpolator._data_scaled = geo_data_in
        self.interpolator._grid_scaled = geo_data_in.grid
        self.interpolator.order_table()
        self.interpolator.set_theano_shared_parameteres(range_var=range_var)

    def get_input_data(self, u_grade=None):
        if not u_grade:
            u_grade = self.u_grade
        return self.interpolator.data_prep(u_grade=u_grade)

    class InterpolatorClass(object):
        """
        -DOCS NOT UPDATED- Class which contain all needed methods to perform potential field implicit modelling in theano

        Args:
            _data(GeMpy_core.DataManagement): All values of a DataManagement object
            _grid(GeMpy_core.grid): A grid object
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
        """

        def __init__(self, _data_scaled, _grid_scaled=None, *args, **kwargs):

            # verbose is a list of strings. See theanograph
            verbose = kwargs.get('verbose', [0])
            # -DEP-rescaling_factor = kwargs.get('rescaling_factor', None)

            # Here we can change the dtype for stability and GPU vs CPU
            dtype = kwargs.get('dtype', 'float32')
            self.dtype = dtype
            print(self.dtype)
            range_var = kwargs.get('range_var', None)

            # Drift grade
            u_grade = kwargs.get('u_grade', [2, 2])



            # We hide the scaled copy of DataManagement object from the user. The scaling happens in gempy what is a
            # bit weird. Maybe at some point I should bring the function to this module
            self._data_scaled = _data_scaled

            # In case someone wants to provide a grid otherwise we extract it from the DataManagement object.
            if not _grid_scaled:
                self._grid_scaled = _data_scaled.grid
            else:
                self._grid_scaled = _grid_scaled

            # Importing the theano graph. The methods of this object generate different parts of graph.
            # See theanograf doc
            self.tg = theanograf.TheanoGraph_pro(dtype=dtype, verbose=verbose,)

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Setting theano parameters
            self.set_theano_shared_parameteres(range_var=range_var)

            # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            self.data_prep(u_grade=u_grade)

            # Avoid crashing my pc
            import theano
            if theano.config.optimizer != 'fast_run':
                assert self.tg.grid_val_T.get_value().shape[0] * \
                       np.math.factorial(len(self.tg.len_series_i.get_value())) < 2e7, \
                       'The grid is too big for the number of potential fields. Reduce the grid or change the' \
                       'optimization flag to fast run'

        def set_formation_number(self):
            """
                    Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
                    to know it and also now the numbers must be set in the order of the series as well. Therefore this method
                    has been moved to the interpolator class as preprocessing

            Returns: Column in the interfaces and foliations dataframes
            """
            try:
                ip_addresses = self._data_scaled.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self._data_scaled.interfaces['formation number'] = self._data_scaled.interfaces['formation'].replace(ip_dict)
                self._data_scaled.foliations['formation number'] = self._data_scaled.foliations['formation'].replace(ip_dict)
            except ValueError:
                pass

        def order_table(self):
            """
            First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
            the formations. All inplace
            """

            # We order the pandas table by series
            self._data_scaled.interfaces.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Give formation number
            if not 'formation number' in self._data_scaled.interfaces.columns:
                print('I am here')
                self.set_formation_number()

            # We order the pandas table by formation (also by series in case something weird happened)
            self._data_scaled.interfaces.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self._data_scaled.interfaces.reset_index(drop=True, inplace=True)

        def data_prep(self, **kwargs):
            """
            Ideally this method will extract the data from the pandas dataframes to individual numpy arrays to be input
            of the theano function. However since some of the shared parameters are function of these arrays shape I also
            set them here
            Returns:
                idl (list): List of arrays which are the input for the theano function:
                    - numpy.array: dips_position
                    - numpy.array: dip_angles
                    - numpy.array: azimuth
                    - numpy.array: polarity
                    - numpy.array: ref_layer_points
                    - numpy.array: rest_layer_points
            """

            u_grade = kwargs.get('u_grade', None)
            # ==================
            # Extracting lengths
            # ==================
            # Array containing the size of every formation. Interfaces
            len_interfaces = np.asarray(
                [np.sum(self._data_scaled.interfaces['formation number'] == i)
                 for i in self._data_scaled.interfaces['formation number'].unique()])

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form)

            # Position of the first point of every layer
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop the reference points using pandas indeces to get just the rest_layers array
            pandas_rest_layer_points = self._data_scaled.interfaces.drop(ref_position)
            self.pandas_rest_layer_points = pandas_rest_layer_points
            # TODO: do I need this? PYTHON
            # DEP- because per series the foliations do not belong to a formation but to the whole series
            # len_foliations = np.asarray(
            #     [np.sum(self._data_scaled.foliations['formation number'] == i)
            #      for i in self._data_scaled.foliations['formation number'].unique()])

            # -DEP- I think this was just a kind of print to know what was going on
            #self.pandas_rest = pandas_rest_layer_points

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum())

            # Array containing the size of every series. Foliations.
            len_series_f = np.asarray(
                [np.sum(self._data_scaled.foliations['order_series'] == i)
                 for i in self._data_scaled.foliations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum())

            # =========================
            # Choosing Universal drifts
            # =========================

            if u_grade is None:
                u_grade = np.zeros_like(len_series_i)
                u_grade[len_series_i > 12] = 9
                u_grade[(len_series_i > 6) & (len_series_i < 12)] = 3
            print(u_grade)
            # it seems I have to pass list instead array_like that is weird
            self.tg.u_grade_T.set_value(list(u_grade))

            # ================
            # Prepare Matrices
            # ================
            # Rest layers matrix # PYTHON VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # TODO delete
            # -DEP- Again i was just a check point
            # self.rest_layer_points = rest_layer_points

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            aux_1 = self._data_scaled.interfaces.iloc[ref_position][['X', 'Y', 'Z']].as_matrix()

            # We initialize the matrix
            ref_layer_points = np.zeros((0, 3))

            # TODO I hate loop it has to be a better way
            # Tiling very reference points as many times as rest of the points we have
            for e, i in enumerate(len_interfaces):
                ref_layer_points = np.vstack((ref_layer_points, np.tile(aux_1[e], (i - 1, 1))))

            # -DEP- was just a check point
            self.ref_layer_points = ref_layer_points

            # Check no reference points in rest points (at least in coor x)
            assert not any(aux_1[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # Foliations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self._data_scaled.foliations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data_scaled.foliations["dip"].as_matrix()
            azimuth = self._data_scaled.foliations["azimuth"].as_matrix()
            polarity = self._data_scaled.foliations["polarity"].as_matrix()

            # Set all in a list casting them in the chosen dtype
            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def set_theano_shared_parameteres(self, **kwargs):
            """
            Here we create most of the kriging parameters. The user can pass them as kwargs otherwise we pick the
            default values from the DataManagement info. The share variables are set in place. All the parameters here
            are independent of the input data so this function only has to be called if you change the extent or grid or
            if you want to change one the kriging parameters.
            Args:
                _data_rescaled: DataManagement object
                _grid_rescaled: Grid object
            Keyword Args:
                u_grade (int): Drift grade. Default to 2.
                range_var (float): Range of the variogram. Default 3D diagonal of the extent
                c_o (float): Covariance at lag 0. Default range_var ** 2 / 14 / 3. See my paper when I write it
                nugget_effect (flaot): Nugget effect of foliations. Default to 0.01
            """

            # Kwargs
            u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)
            # DEP
           # compute_all = kwargs.get('compute_all', True)

            # -DEP- Now I rescale the data so we do not need this
            # rescaling_factor = kwargs.get('rescaling_factor', None)

            # Default range
            if not range_var:
                range_var = np.sqrt((self._data_scaled.extent[0] - self._data_scaled.extent[1]) ** 2 +
                                    (self._data_scaled.extent[2] - self._data_scaled.extent[3]) ** 2 +
                                    (self._data_scaled.extent[4] - self._data_scaled.extent[5]) ** 2)


            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Asserting that the drift grade is in this range
           # assert (0 <= all(u_grade) <= 2)

            # Creating the drift matrix. TODO find the official name of this matrix?
            _universal_matrix = np.vstack((self._grid_scaled.grid.T,
                                           (self._grid_scaled.grid ** 2).T,
                                           self._grid_scaled.grid[:, 0] * self._grid_scaled.grid[:, 1],
                                           self._grid_scaled.grid[:, 0] * self._grid_scaled.grid[:, 2],
                                           self._grid_scaled.grid[:, 1] * self._grid_scaled.grid[:, 2]))

            # Setting shared variables
            # Range
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            # Covariance at 0
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            # Foliations nugget effect
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            # TODO change the drift to the same style I have the faults so I do not need to do this
            # # Drift grade
            # if u_grade == 0:
            #     self.tg.u_grade_T.set_value(u_grade)
            # else:
            #     self.tg.u_grade_T.set_value(u_grade)
                # TODO: To be sure what is the mathematical meaning of this -> It seems that nothing
                # TODO Deprecated
                # self.tg.c_resc.set_value(1)

            # Just grid. I add a small number to avoid problems with the origin point
            self.tg.grid_val_T.set_value(np.cast[self.dtype](self._grid_scaled.grid + 10e-6))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](_universal_matrix + 1e-10))

            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((1, self._grid_scaled.grid.shape[0]), dtype='float32'))

            # Initialization of the boolean array that represent the areas of the block model to be computed in the
            # following series
            #self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))

            # Unique number assigned to each lithology
            #self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
            #                                        0, 0)[::-1])

            self.tg.n_formation.set_value(self._data_scaled.interfaces['formation number'].unique())

            # Number of formations per series. The function is not pretty but the result is quite clear
            self.tg.n_formations_per_serie.set_value(
                np.insert(self._data_scaled.interfaces.groupby('order_series').formation.nunique().values.cumsum(), 0, 0))

        def get_kriging_parameters(self, verbose=0):
            # range
            print('range', self.tg.a_T.get_value(), self.tg.a_T.get_value() * self._data_scaled.rescaling_factor)
            # Number of drift equations
            print('Number of drift equations', self.tg.u_grade_T.get_value())
            # Covariance at 0
            print('Covariance at 0', self.tg.c_o_T.get_value())
            # Foliations nugget effect
            print('Foliations nugget effect', self.tg.nugget_effect_grad_T.get_value())

            if verbose > 0:
                # Input data shapes

                # Lenght of the interfaces series
                print('Length of the interfaces series', self.tg.len_series_i.get_value())
                # Length of the foliations series
                print('Length of the foliations series', self.tg.len_series_f.get_value())
                # Number of formation
                print('Number of formations', self.tg.n_formation.get_value())
                # Number of formations per series
                print('Number of formations per series', self.tg.n_formations_per_serie.get_value())
                # Number of points per formation
                print('Number of points per formation (rest)', self.tg.number_of_points_per_formation_T.get_value())

