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
    Class to import the raw data of the model and set data classifications into formations and series.
    This objects will contain the main information of the model.

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
        series(pandas.core.frame.DataFrame): Pandas data frame which contains every formation within each series
    """

    def __init__(self,
                 extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_f=None,
                 **kwargs):

        # Set extent and resolution
        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

        # Init number of faults
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
            self.import_data_csv(path_i=path_i, path_f=path_f)

        # DEP-
       # self._set_formations()

        # If not provided set default series
        self.series = self.set_series()
        # DEP- self.set_formation_number()

        # Compute gradients given azimuth and dips to plot data
        self.calculate_gradient()

        # Create default grid object. TODO: (Is this necessary now?)
        self.grid = self.set_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)

        self.order_table()
        # DEP
        #self.geo_data_type = 'InputData'

    def import_data_csv(self, path_i, path_f, **kwargs):
        """
        Method to import interfaces and foliations from csv. The format is the same as the export 3D model data of
        GeoModeller (check in the input data folder for an example).
        Args:
            path_i (str): path to the csv table
            path_f (str): path to the csv table
            **kwargs: kwargs of Pandas load_csv

        Attributes:
            Foliations(pandas.core.frame.DataFrame): Pandas data frame with the foliations data
            Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data

        """

        if path_f:
            self.foliations = self.load_data_csv(data_type="foliations", path=path_f, **kwargs)
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.foliations.columns), \
                "One or more columns do not match with the expected values " + str(self.foliations.columns)

        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)

    def get_formations(self):
        """
        Returns:
             pandas.core.frame.DataFrame: Returns a list of formations

        """
        return self.interfaces["formation"].unique()

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the foliations

        Attributes:
            foliations: extra columns with components xyz of the unity vector.
        """

        self.foliations['G_x'] = np.sin(np.deg2rad(self.foliations["dip"].astype('float'))) * \
                                 np.sin(np.deg2rad(self.foliations["azimuth"].astype('float'))) * \
                                 self.foliations["polarity"].astype('float')
        self.foliations['G_y'] = np.sin(np.deg2rad(self.foliations["dip"].astype('float'))) * \
                                 np.cos(np.deg2rad(self.foliations["azimuth"].astype('float'))) *\
                                 self.foliations["polarity"].astype('float')
        self.foliations['G_z'] = np.cos(np.deg2rad(self.foliations["dip"].astype('float'))) *\
                                 self.foliations["polarity"].astype('float')

    # # DEP?
    # def create_grid(self, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
    #     """
    #     Method to initialize the class grid. So far is really simple and only has the regular grid type
    #
    #     Args:
    #         grid_type (str): regular_3D or regular_2D (I am not even sure if regular 2D still working)
    #         **kwargs: Arbitrary keyword arguments.
    #
    #     Returns:
    #         self.grid(GeMpy_core.grid): Object that contain different grids
    #     """
    #
    #     if not extent:
    #         extent = self.extent
    #     if not resolution:
    #         resolution = self.resolution
    #
    #   return self.GridClass(extent, resolution, grid_type=grid_type, **kwargs)

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
            assert new_grid.shape[1] is 3, 'The shape of new grid must be (n,3) where n is' \
                                                                        'the number of points of the grid'
            self.grid.grid = new_grid
        else:
            if not extent:
                extent = self.extent
            if not resolution:
                resolution = self.resolution

            return GridClass(extent, resolution, grid_type=grid_type, **kwargs)

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

    def get_raw_data(self, itype='all', verbosity=0):
        """
        Method that returns the interfaces and foliations pandas Dataframes. Can return both at the same time or only
        one of the two
        Args:
            itype: input data type, either 'foliations', 'interfaces' or 'all' for both.
            verbosity (int): Number of properties shown
        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        import pandas as pn
        if verbosity == 0:
            show_par_f = ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity','formation', 'series', ]
            show_par_i = ['X', 'Y', 'Z', 'formation', 'series']
        else:
            show_par_f = self.foliations.columns
            show_par_i = self.interfaces.columns

        if itype == 'foliations':
            raw_data = self.foliations[show_par_f]
        elif itype == 'interfaces':
            raw_data = self.interfaces[show_par_i]
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces, self.foliations], keys=['interfaces', 'foliations'])
        else:
            raise AttributeError('itype has to be: \'foliations\', \'interfaces\', or \'all\'')
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
        try:
            import qgrid
        except:
            raise ModuleNotFoundError('It is necessary to instal qgrid to have interactive tables')

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

        # Setting some options
        qgrid.nbinstall(overwrite=True)
        qgrid.set_defaults(show_toolbar=True)
        assert itype is 'foliations' or itype is 'interfaces', 'itype must be either foliations or interfaces'

        import warnings
        warnings.warn('Remember to call i_close_set_data after the editing.')

        # We kind of set the show grid to a variable so we can close it afterwards
        self.pandas_frame = qgrid.show_grid(self.get_raw_data(itype=itype))

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

        # Set parameters
        self.series = self.set_series()
        self.calculate_gradient()
        self.order_table()

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
       # self.interfaces.reset_index(drop=True, inplace=True)

    def set_foliations(self, foliat_Dataframe, append=False):
        """
          Method to change or append a Dataframe to foliations in place.  A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.
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

        self.set_series()
        self.order_table()
        self.calculate_gradient()
      #  self.foliations.reset_index(drop=True, inplace=True)

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
        Set a flag to the series that are faults.
        Args:
            series_name(list or array_like): Name of the series which are faults
        """
        if not len(series_name) == 0:
            self.interfaces['isFault'] = self.interfaces['series'].isin(series_name)
            self.foliations['isFault'] = self.foliations['series'].isin(series_name)
            self.n_faults = len(series_name)

    def order_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
        the formations. All inplace
        """

        # We order the pandas table by series
        self.interfaces.sort_values(by=['order_series'],  # , 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        self.foliations.sort_values(by=['order_series'],  # , 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        # Give formation number
        if not 'formation number' in self.interfaces.columns:
            # print('I am here')
            self.set_formation_number()

        # We order the pandas table by formation (also by series in case something weird happened)
        self.interfaces.sort_values(by=['order_series', 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        self.foliations.sort_values(by=['order_series', 'formation number'],
                                                 ascending=True, kind='mergesort',
                                                 inplace=True)

        # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
        # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
        self.interfaces.reset_index(drop=True, inplace=True)

    def set_formation_number(self, formation_order=None):
        """
        Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
        to know it and also now the numbers must be set in the order of the series as well. Therefore this method
        has been moved to the interpolator class as preprocessing

        Returns:
            Column in the interfaces and foliations dataframes
        """
        if not formation_order:
            formation_order = self.interfaces["formation"].unique()
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

        Returns:
            None
        """
        self.interfaces.reset_index(inplace=True, drop=True)
        self.foliations.reset_index(inplace=True, drop=True)

    def interface_modify(self, index, **kwargs):
        """
        Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.
        Args:
            index: dataframe index of the foliation point
            **kwargs: X, Y, Z (int or float)

        Returns:
            None
        """
        for key in kwargs:
            self.interfaces.ix[index, str(key)] = kwargs[key]

    def interface_add(self, **kwargs):
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
        self.set_series()
        self.order_table()

    def interface_drop(self, index):
        """
        Drops interface from dataframe identified by index
        Args:
            index: dataframe index

        Returns:
            None

        """
        self.interfaces.drop(index, inplace=True)

    def foliation_modify(self, index, **kwargs):
        """
        Allows modification of foliation data at specified dataframe index.
        Args:
            index: dataframe index of the foliation point
            **kwargs: G_x, G_y, G_z, X, Y, Z, azimuth, dip, formation, labels, order_series, polarity

        Returns:
            None
        """
        for key in kwargs:
            self.foliations.ix[index, str(key)] = kwargs[key]

        self.calculate_gradient()

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
        self.calculate_gradient()
        self.set_series()
        self.order_table()

    def foliations_drop(self, index):
        """
        Drops foliation from dataframe identified by index
        Args:
            index: dataframe index

        Returns:
            None

        """
        self.foliations.drop(index, inplace=True)

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

    # # TODO think where this function should go
    # def read_vox(self, path):
    #     """
    #     read vox from geomodeller and transform it to gempy format
    #     Returns:
    #         numpy.array: block model
    #     """
    #
    #     geo_res = pn.read_csv(path)
    #
    #     geo_res = geo_res.iloc[9:]
    #
    #     #ip_addresses = geo_res['nx 50'].unique()  # geo_data.interfaces["formation"].unique()
    #     ip_dict = self.get_formation_number()
    #
    #     geo_res_num = geo_res.iloc[:, 0].replace(ip_dict)
    #     block_geomodeller = np.ravel(geo_res_num.as_matrix().reshape(
    #                                     self.resolution[0], self.resolution[1], self.resolution[2], order='C').T)
    #     return block_geomodeller


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

        return np.vstack(map(np.ravel, g)).T.astype("float32")


class InterpolatorInput:
    """
    InterpolatorInput is a class that contains all the preprocessing operations to prepare the data to compute the model.
    Also is the object that has to be manipulated to vary the data without recompile the modeling function.

    Args:
        geo_data(gempy.DataManagement.InputData): All values of a DataManagement object
        compile_theano (bool): select if the theano function is compiled during the initialization. Default: True
        compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
         the block model of faults. If False only return the block model of lithologies. This may be important to speed
          up the computation. Default True
        u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
        be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
        depends on the number of points given as input to try to avoid singular matrix. NOTE: if during the computation
        of the model a singular matrix is returned try to reduce the u_grade of the series.
        rescaling_factor (float): rescaling factor of the input data to improve the stability when float32 is used. By
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
    def __init__(self, geo_data, compile_theano=True, compute_all=True, u_grade=None, rescaling_factor=None, **kwargs):
        # TODO add all options before compilation in here. Basically this is n_faults, n_layers, verbose, dtype, and \
        # only block or all
        assert isinstance(geo_data, InputData), 'You need to pass a InputData object'

        # Store the original InputData object
        self.geo_data = geo_data

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
        self.geo_data_res = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)

        # # This are necessary parameters for the visualization package
        #self.resolution = self.geo_data.resolution
        #self.extent = self.extent_rescaled.as_matrix()

        # Creating interpolator class with all the precompilation options
        # --DEP-- self.interpolator = self.set_interpolator(**kwargs)
        self.interpolator = self.InterpolatorClass(self.geo_data_res, self.geo_data_res.grid, **kwargs)
        if compile_theano:
            self.th_fn = self.compile_th_fn(compute_all=compute_all)

    def compile_th_fn(self, compute_all=True):
        """
        Compile the theano function given the input data.
        Args:
            compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
             the block model of faults. If False only return the block model of lithologies. This may be important to speed
              up the computation. Default True

        Returns:
            theano.function: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
        """

        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.interpolator.tg.input_parameters_list()

        # then we compile we have to pass the number of formations that are faults!!
        th_fn = theano.function(input_data_T, self.interpolator.tg.whole_block_model(self.geo_data_res.n_faults,
                                                                                     compute_all=compute_all),
                                on_unused_input='ignore',
                                allow_input_downcast=False,
                                profile=False)
        return th_fn

    def rescale_data(self, geo_data, rescaling_factor=None):
        """
        Rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.
        Args:
            geo_data: Original gempy.DataManagement.InputData object
            rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

        Returns:
            gempy.DataManagement.InputData: Rescaled data

        """
        # TODO split this function in compute rescaling factor and rescale z

        # Check which axis is the largest
        max_coord = pn.concat(
            [geo_data.foliations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
        min_coord = pn.concat(
            [geo_data.foliations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

        # Compute rescalin factor if not given
        if not rescaling_factor:
            rescaling_factor = 2 * np.max(max_coord - min_coord)

        # Get the centers of every axis
        centers = (max_coord + min_coord) / 2

        # Change the coordinates of interfaces
        new_coord_interfaces = (geo_data.interfaces[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        # Change the coordinates of foliations
        new_coord_foliations = (geo_data.foliations[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        # Rescaling the std in case of stochastic values
        try:
            geo_data.interfaces[['X_std', 'Y_std', 'Z_std']] = (geo_data.interfaces[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
            geo_data.foliations[['X_std', 'Y_std', 'Z_std']] = (geo_data.foliations[
                                                                    ['X_std', 'Y_std', 'Z_std']]) / rescaling_factor
        except KeyError:
            pass

        # Updating properties
        new_coord_extent = (geo_data.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001

        geo_data_rescaled = copy.deepcopy(geo_data)
        geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
        geo_data_rescaled.foliations[['X', 'Y', 'Z']] = new_coord_foliations
        geo_data_rescaled.extent = new_coord_extent.as_matrix()

        geo_data_rescaled.grid.grid = (geo_data.grid.grid - centers.as_matrix()) / rescaling_factor + 0.5001

        # Saving useful values for later
        self.rescaling_factor = rescaling_factor
        geo_data_rescaled.rescaling_factor = rescaling_factor
        self.centers = centers
        self.extent_rescaled = new_coord_extent

        return geo_data_rescaled

    def get_formation_number(self):
        """
        Get a dictionary with the key the name of the formation and the value their number

        Returns:
            dict: key the name of the formation and the value their number
        """
        pn_series = self.geo_data_res.interfaces.groupby('formation number').formation.unique()
        ip_addresses = {}
        for e, i in enumerate(pn_series):
            ip_addresses[i[0]] = e + 1
        ip_addresses['DefaultBasement'] = 0
        return ip_addresses

    # --DEP--
    # def set_interpolator(self, geo_data=None, **kwargs):
    #     """
    #     Method to initialize the class interpolator. All the constant parameters for the interpolation can be passed
    #     as args, otherwise they will take the default value (TODO: documentation of the dafault values)
    #
    #     Args:
    #         geo_data: Original gempy.DataManagement.InputData object. If given it rescales it again Default takes the property
    #
    #     Keyword Args:
    #         range_var: Range of the variogram. Default None
    #         c_o: Covariance at 0. Default None
    #         nugget_effect: Nugget effect of the gradients. Default 0.01
    #         u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
    #         rescaling_factor: Magic factor that multiplies the covariances). Default 2
    #
    #     Returns:
    #         self.Interpolator (GeMpy_core.Interpolator): Object to perform the potential field method
    #     """
    #
    #     if 'u_grade' in kwargs:
    #         compile_theano = True
    #
    #     range_var = kwargs.get('range_var', None)
    #
    #     rescaling_factor = kwargs.get('rescaling_factor', None)
    #
    #     if geo_data:
    #         geo_data_in = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)
    #         self.geo_data_res = geo_data_in
    #     else:
    #         geo_data_in = self.geo_data_res
    #
    #     # First creation
    #     if not getattr(self, 'interpolator', None):
    #         # print('I am in the setting')
    #         interpolator = self.InterpolatorClass(geo_data_in, geo_data_in.grid, **kwargs)
    #
    #     # Update
    #     else:
    #         print('I am in update')
    #         # I update the data
    #         self.interpolator._data_scaled = geo_data_in
    #         self.interpolator._grid_scaled = geo_data_in.grid
    #         # I order it again just in case. TODO if this still necessary
    #         self.interpolator.order_table()
    #
    #         # Refresh all shared parameters of
    #         self.interpolator.set_theano_shared_parameteres(range_var=range_var)
    #         interpolator = None
    #
    #     return interpolator

    def update_interpolator(self, geo_data_res=None, **kwargs):
        """
        Method to update the constant parameters of the class interpolator (i.e. theano shared).
         All the constant parameters for the interpolation can be passed
        as kwargs, otherwise they will take the default value (TODO: documentation of the dafault values)

        Args:
            geo_data_res: Rescaled gempy.DataManagement.InputData object.

        Keyword Args:
           range_var: Range of the variogram. Default None
           c_o: Covariance at 0. Default None
           nugget_effect: Nugget effect of the gradients. Default 0.01
           u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
        """

        if geo_data_res:
            geo_data_in = geo_data_res
            self.geo_data_res = geo_data_in
        else:
            geo_data_in = self.geo_data_res

        print('I am in update')
        # I update the data
        self.interpolator.geo_data_res = geo_data_in
        self.interpolator.grid_res = geo_data_in.grid
        # I order it again just in case. TODO if this still necessary
        self.interpolator.order_table()
        self.interpolator.set_theano_shared_parameteres(**kwargs)

    def get_input_data(self, u_grade=None):
        """
        Get the theano variables that are input. This are necessary to compile the theano function
        or a theno op for pymc3
        Args:
             u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
            be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
            depends on the number of points given as input to try to avoid singular matrix. NOTE: if during the computation
            of the model a singular matrix is returned try to reduce the u_grade of the series.

        Returns:
            theano.variables: input nodes of the theano graph
        """

        if not u_grade:
            u_grade = self.u_grade
        return self.interpolator.data_prep(u_grade=u_grade)

    class InterpolatorClass(object):
        """
        -DOCS NOT UPDATED-
         Class which contain all needed methods to perform potential field implicit modelling in theano.
         Here there are methods to modify the shared parameters of the theano graph as well as the final
         preparation of the data from DataFrames to numpy arrays

        Args:
             geo_data_res (gempy.DataManagement.InterpolatorInput): Rescaled data. It has the same structure has gempy.InputData
            grid(gempy.DataManagement.grid): A grid object rescaled. Default takes it from the InterpolatorInput object.

        Keyword Args:
            range_var: Range of the variogram. Default None
            c_o: Covariance at 0. Default None
            nugget_effect: Nugget effect of the gradients. Default 0.01
            u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
            rescaling_factor: Magic factor that multiplies the covariances). Default 2
            verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
        """

        def __init__(self, geo_data_res, grid_res=None, **kwargs):

            # verbose is a list of strings. See theanograph
            verbose = kwargs.get('verbose', [0])
            # -DEP-rescaling_factor = kwargs.get('rescaling_factor', None)

            # Here we can change the dtype for stability and GPU vs CPU
            dtype = kwargs.get('dtype', 'float32')
            self.dtype = dtype

            if dtype in verbose:
                print(self.dtype)

            range_var = kwargs.get('range_var', None)

            # Drift grade
            u_grade = kwargs.get('u_grade', [2, 2])

            # We hide the scaled copy of DataManagement object from the user. The scaling happens in gempy what is a
            # bit weird. Maybe at some point I should bring the function to this module
            self.geo_data_res = geo_data_res

            # In case someone wants to provide a grid otherwise we extract it from the DataManagement object.
            if not grid_res:
                self.grid_res = geo_data_res.grid
            else:
                self.grid_res = grid_res

            # Importing the theano graph. The methods of this object generate different parts of graph.
            # See theanograf doc
            self.tg = theanograf.TheanoGraph_pro(dtype=dtype, verbose=verbose,)

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Setting theano parameters
            self.set_theano_shared_parameteres(**kwargs)

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

            Returns:
                Column in the interfaces and foliations dataframes
            """
            try:
                ip_addresses = self.geo_data_res.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self.geo_data_res.interfaces['formation number'] = self.geo_data_res.interfaces['formation'].replace(ip_dict)
                self.geo_data_res.foliations['formation number'] = self.geo_data_res.foliations['formation'].replace(ip_dict)
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

            self.geo_data_res.foliations.sort_values(by=['order_series'],  # , 'formation number'],
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

            self.geo_data_res.foliations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self.geo_data_res.interfaces.reset_index(drop=True, inplace=True)

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
                [np.sum(self.geo_data_res.foliations['order_series'] == i)
                 for i in self.geo_data_res.foliations['order_series'].unique()])

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

            # TODO: rethink how to do it for pymc
            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            self.pandas_ref_layer_points = self.geo_data_res.interfaces.iloc[ref_position]#.apply(
              #  lambda x: np.repeat(x, len_interfaces - 1))
            self.len_interfaces = len_interfaces


            pandas_ref_layer_points_rep = self.pandas_ref_layer_points.apply(lambda x: np.repeat(x, len_interfaces - 1))
            ref_layer_points = pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()

            # -DEP- was just a check point
            # self.ref_layer_points = ref_layer_points

            # Check no reference points in rest points (at least in coor x)
            assert not any(ref_layer_points[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # Foliations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self.geo_data_res.foliations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self.geo_data_res.foliations["dip"].as_matrix()
            azimuth = self.geo_data_res.foliations["azimuth"].as_matrix()
            polarity = self.geo_data_res.foliations["polarity"].as_matrix()

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

            Keyword Args:
                u_grade (int): Drift grade. Default to 2.
                range_var (float): Range of the variogram. Default 3D diagonal of the extent
                c_o (float): Covariance at lag 0. Default range_var ** 2 / 14 / 3. See my paper when I write it
                nugget_effect (flaot): Nugget effect of foliations. Default to 0.01
            """

            # Kwargs
            # --This is DEP because is a condition not a shared-- u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)
            # DEP
           # compute_all = kwargs.get('compute_all', True)

            # -DEP- Now I rescale the data so we do not need this
            # rescaling_factor = kwargs.get('rescaling_factor', None)

            # Default range
            if not range_var:
                range_var = np.sqrt((self.geo_data_res.extent[0] - self.geo_data_res.extent[1]) ** 2 +
                                    (self.geo_data_res.extent[2] - self.geo_data_res.extent[3]) ** 2 +
                                    (self.geo_data_res.extent[4] - self.geo_data_res.extent[5]) ** 2)


            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Asserting that the drift grade is in this range
           # assert (0 <= all(u_grade) <= 2)

            # Creating the drift matrix. TODO find the official name of this matrix?
            _universal_matrix = np.vstack((self.grid_res.grid.T,
                                           (self.grid_res.grid ** 2).T,
                                           self.grid_res.grid[:, 0] * self.grid_res.grid[:, 1],
                                           self.grid_res.grid[:, 0] * self.grid_res.grid[:, 2],
                                           self.grid_res.grid[:, 1] * self.grid_res.grid[:, 2]))

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
            self.tg.grid_val_T.set_value(np.cast[self.dtype](self.grid_res.grid + 10e-6))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](_universal_matrix + 1e-10))

            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((1, self.grid_res.grid.shape[0]), dtype='float32'))

            # Initialization of the boolean array that represent the areas of the block model to be computed in the
            # following series
            #self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))

            # Unique number assigned to each lithology
            #self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
            #                                        0, 0)[::-1])

            self.tg.n_formation.set_value(self.geo_data_res.interfaces['formation number'].unique())

            # Number of formations per series. The function is not pretty but the result is quite clear
            self.tg.n_formations_per_serie.set_value(
                np.insert(self.geo_data_res.interfaces.groupby('order_series').formation.nunique().values.cumsum(), 0, 0))

            self.tg.final_potential_field_at_formations.set_value(np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                                                                           dtype=self.dtype))
            self.tg.final_potential_field_at_faults.set_value(
                np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                         dtype=self.dtype))

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

