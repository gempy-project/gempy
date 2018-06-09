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
from .data_management import InputData

pn.options.mode.chained_assignment = None  #


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

        if geo_data.interfaces.shape[0] < 2:
            geo_data.set_default_interface()
            warnings.warn('Setting default interface')

        if geo_data.orientations.shape[0] < 1:
            geo_data.set_default_orientation()
            warnings.warn('Setting default interface')

        # Store the original InputData object
        self._geo_data = geo_data

        # Here we can change the dtype for stability and GPU vs CPU
        self.dtype = kwargs.get('dtype', 'float32')

        # Set some parameters. TODO possibly this should go in kwargs
        self.u_grade = u_grade

        # This two properties get set calling rescale data
        self.rescaling_factor = None
        self.centers = None
        #self.extent_original = copy.copy(geo_data.extent)

        # Rescaling
        self.geo_data_res = self.rescale_data(geo_data, rescaling_factor=rescaling_factor)

        # Creating interpolator class with all the precompilation options
        self.interpolator = self.InterpolatorTheano(self, output=output, theano_optimizer=theano_optimizer, **kwargs)
        if compile_theano:
            self.th_fn = self.compile_th_fn(output, **kwargs)

        self.geophy = geophysics

    def compile_th_fn(self, output, **kwargs):
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

        from theano.compile.nanguardmode import NanGuardMode

        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.interpolator.tg.input_parameters_list()

        print('Compiling theano function...')

        if output is 'geology':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_geological_model(),
                                  # mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gravity':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_forward_gravity(),
                                  #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gradients':

            gradients = kwargs.get('gradients', ['Gx', 'Gy', 'Gz'])
            self.interpolator.tg.gradients = gradients

            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_geological_model_gradient(self.geo_data_res.n_faults),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        else:
            raise SyntaxError('The output given does not exist. Please use geology, gradients or gravity ')

        print('Compilation Done!')
        print('Level of Optimization: ', theano.config.optimizer)
        print('Device: ', theano.config.device)
        print('Precision: ', self.dtype)
        print('Number of faults: ', self.geo_data_res.n_faults)
        return th_fn

    def data_to_pickle(self, path=False, recursionlimit=3000):
        """
        Save InterpData object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems.

        Args:
            path (str): path where save the pickle
            recursionlimit(int): limit passed to sys.setrecursionlimit()

        Returns:
            None
        """

        if not path:
            path = './interp_data'
        import pickle
        import sys
        sys.setrecursionlimit(recursionlimit)
        warnings.warn('Pickling an InterpolatorData object is EXPERIMENTAL and on your own risk. Try to adjust the recursionlimit.')
        with open(path+'.pickle', 'wb') as f:
            # Pickle the dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

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
            rescaling_factor = (2 * np.max(max_coord - min_coord))

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float)

        # Change the coordinates of interfaces
        new_coord_interfaces = (geo_data.interfaces[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        # Change the coordinates of orientations
        new_coord_orientations = (geo_data.orientations[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001


        # Updating properties
        new_coord_extent = (geo_data.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001

        geo_data_rescaled = copy.copy(geo_data)
        geo_data_rescaled.interfaces = copy.copy(geo_data.interfaces)
        geo_data_rescaled.orientations = copy.copy(geo_data.orientations)
        geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
        geo_data_rescaled.orientations[['X', 'Y', 'Z']] = new_coord_orientations

        try:
            # Rescaling the std in case of stochastic values
            geo_data_rescaled.interfaces[['X_std', 'Y_std', 'Z_std']] = geo_data.interfaces[['X_std', 'Y_std', 'Z_std']] / rescaling_factor
            geo_data_rescaled.orientations[['X_std', 'Y_std', 'Z_std']] = geo_data.orientations[['X_std', 'Y_std', 'Z_std']] / rescaling_factor

        except KeyError:
            pass

        geo_data_rescaled.extent = copy.copy(new_coord_extent.as_matrix())

        geo_data_rescaled.x_to_interp_given = copy.copy(geo_data.grid.values)
        geo_data_rescaled.x_to_interp_given = (geo_data.grid.values - centers.as_matrix()) / rescaling_factor + 0.5001

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
            geo_data: Rescaled gempy.DataManagement.InputData object. If None the stored geo_data_res_no_basement will be used

        Keyword Args:
           range_var: Range of the variogram. Default None
           c_o: Covariance at 0. Default None
           nugget_effect: Nugget effect of the gradients. Default 0.01
           u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
        """

        if geo_data:
            # Checking is geodata is already rescaled
            try:
                getattr(geo_data, 'rescaling_factor')
                # warnings.warn('You are passing a rescaled geo_data')
                geo_data_in = self.geo_data_res
            except AttributeError:
                geo_data_in = self.rescale_data(geo_data)
                self.geo_data_res = geo_data_in
        else:
            geo_data_in = self.geo_data_res

        import theano
        if theano.config.optimizer != 'fast_run':
            assert self.interpolator.tg.grid_val_T.get_value().shape[0] * \
                   len(self.interpolator.tg.len_series_i.get_value()) < 2e7, \
                'The grid is too big for the number of scalar fields. Reduce the grid or change the' \
                'optimization flag to fast run'

        # I update the interpolator data
        # self.interpolator.geo_data_res_no_basement = geo_data_in
        # self.interpolator.order_table()
        # self.interpolator.data_prep()
        # self.interpolator.set_theano_shared_parameteres(**kwargs)
        self.interpolator.prepare_data_frame(geo_data_in, **kwargs)

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
        return self.interpolator.input_data_python()

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

            # We hide the scaled copy of DataManagement object from the user.
            self.geo_data_res_no_basement = interp_data.geo_data_res
            self.geo_data_res_no_basement.interfaces = interp_data.geo_data_res.interfaces[
                ~(interp_data.geo_data_res.interfaces['formation'].values == 'basement')]  #

            # verbose is a list of strings. See theanograph
            self.verbose = kwargs.get('verbose', [0])

            # Here we can change the dtype for stability and GPU vs CPU
            self.dtype = kwargs.get('dtype', 'float32')
            # Here we change the graph type
            self.output = kwargs.get('output', 'geology')
            # Optimization flag
            self.theano_optimizer = kwargs.get('theano_optimizer', 'fast_compile')

            self.output = self.output

            if 'dtype' in self.verbose:
                print(self.dtype)

            # Drift grade
            u_grade = kwargs.get('u_grade', [3, 3])



            self.create_theano_graph(**kwargs)
            self.prepare_data_frame(interp_data.geo_data_res, **kwargs)

            # # We hide the scaled copy of DataManagement object from the user.
            # self.geo_data_res_no_basement = interp_data.geo_data_res
            # self.geo_data_res_no_basement.interfaces = self.geo_data_res_no_basement.interfaces[:-1]
            #
            # # Importing the theano graph. The methods of this object generate different parts of graph.
            # # See theanograf doc
            # self.tg = theano_graph.TheanoGraph(output=self.output, optimizer=self.theano_optimizer, dtype=self.dtype, verbose=self.verbose, )
            #
            # # Avoid crashing my pc
            # import theano
            # if theano.config.optimizer != 'fast_run':
            #     assert self.tg.grid_val_T.get_value().shape[0] * \
            #            len(self.tg.len_series_i.get_value()) < 2e7, \
            #         'The grid is too big for the number of scalar fields. Reduce the grid or change the' \
            #         'optimization flag to fast run'
            #
            # # Sorting data in case the user provides it unordered
            # self.order_table()
            #
            # # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            # self.data_prep(**kwargs)
            #
            # # Setting theano parameters
            # self.set_theano_shared_parameteres(**kwargs)
            #
        def prepare_data_frame(self, geo_data_res, **kwargs):



            # We hide the scaled copy of DataManagement object from the user.
            self.geo_data_res_no_basement = geo_data_res
            self.geo_data_res_no_basement.interfaces = geo_data_res.interfaces[~(geo_data_res.interfaces['formation'].values == 'basement')]#self.geo_data_res_no_basement.interfaces[:-1]

            self.formation_number = np.arange(1, self.geo_data_res_no_basement.interfaces[
                                              'formation_number'].nunique() + 2, dtype='int32')  # geo_data_res.interfaces['formation_number'].unique().astype('int32')

            self.formation_value = geo_data_res.formations['value'].values.squeeze().astype(self.dtype)

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            self.data_prep(**kwargs)

            # Setting theano parameters
            self.set_theano_shared_parameteres(**kwargs)

        def create_theano_graph(self, **kwargs):

            import importlib
            importlib.reload(theano_graph)

            # # verbose is a list of strings. See theanograph
            # verbose = kwargs.get('verbose', [0])
            #
            # # Here we can change the dtype for stability and GPU vs CPU
            # self.dtype = kwargs.get('dtype', 'float32')
            # # Here we change the graph type
            # output = kwargs.get('output', 'geology')
            # # Optimization flag
            # theano_optimizer = kwargs.get('theano_optimizer', 'fast_compile')
            #
            # self.output = output

            if 'dtype' in self.verbose:
                print(self.dtype)

            # Drift grade
            u_grade = kwargs.get('u_grade', [3, 3])

            is_lith = kwargs.get('is_lith', False)
            is_fault = kwargs.get('is_fault', False)

            if self.geo_data_res_no_basement.n_faults != 0:
                is_fault = True

            if self.geo_data_res_no_basement.orientations['formation'].nunique() - 1 > self.geo_data_res_no_basement.n_faults:
                is_lith = True

            # See theanograf doc
            self.tg = theano_graph.TheanoGraph(output=self.output, optimizer=self.theano_optimizer, dtype=self.dtype,
                                               verbose=self.verbose, is_lith=is_lith, is_fault=is_fault)

            # Avoid crashing my pc
            import theano
            if theano.config.optimizer != 'fast_run':
                assert self.tg.grid_val_T.get_value().shape[0] * \
                       len(self.tg.len_series_i.get_value()) < 2e7, \
                    'The grid is too big for the number of scalar fields. Reduce the grid or change the' \
                    'optimization flag to fast run'

        def set_formation_number(self):
            """
            Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
            to know it and also now the numbers must be set in the order of the series as well. Therefore this method
            has been moved to the interpolator class as preprocessing

            Returns:
                Column in the interfaces and orientations dataframes
            """
            try:
                ip_addresses = self.geo_data_res_no_basement.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self.geo_data_res_no_basement.interfaces['formation_number'] = self.geo_data_res_no_basement.interfaces['formation'].replace(ip_dict)
                self.geo_data_res_no_basement.orientations['formation_number'] = self.geo_data_res_no_basement.orientations['formation'].replace(ip_dict)
            except ValueError:
                pass

        def order_table(self):
            """
            First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
            the formations. All inplace
            """

            # We order the pandas table by series
            self.geo_data_res_no_basement.interfaces.sort_values(by=['order_series'],  # , 'formation_number'],
                                                                 ascending=True, kind='mergesort',
                                                                 inplace=True)

            self.geo_data_res_no_basement.orientations.sort_values(by=['order_series'],  # , 'formation_number'],
                                                                   ascending=True, kind='mergesort',
                                                                   inplace=True)

            # Give formation_number
            if not 'formation_number' in self.geo_data_res_no_basement.interfaces.columns:
                # print('I am here')
                self.set_formation_number()

            # We order the pandas table by formation (also by series in case something weird happened)
            self.geo_data_res_no_basement.interfaces.sort_values(by=['order_series', 'formation_number'],
                                                                 ascending=True, kind='mergesort',
                                                                 inplace=True)

            self.geo_data_res_no_basement.orientations.sort_values(by=['order_series', 'formation_number'],
                                                                   ascending=True, kind='mergesort',
                                                                   inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self.geo_data_res_no_basement.interfaces.reset_index(drop=True, inplace=True)

        def set_u_grade(self, **kwargs):

            u_grade = kwargs.get('u_grade', None)

            # =========================
            # Choosing Universal drifts
            # =========================
            if u_grade is None:
                u_grade = np.zeros_like(self.len_series_i)
                u_grade[(self.len_series_i > 1)] = 1

            else:
                u_grade = np.array(u_grade)

            n_universal_eq = np.zeros_like(self.len_series_i)
            n_universal_eq[u_grade == 0] = 0
            n_universal_eq[u_grade == 1] = 3
            n_universal_eq[u_grade == 2] = 9

            self.n_universal_eq = n_universal_eq
            # it seems I have to pass list instead array_like that is weird
            #self.tg.n_universal_eq_T.set_value(list(n_universal_eq.astype('int32')))

        def set_length_interface(self):
            # ==================
            # Extracting lengths
            # ==================
            # Array containing the size of every formation. Interfaces
            self.len_interfaces = np.asarray(
                [np.sum(self.geo_data_res_no_basement.interfaces['formation_number'] == i)
                 for i in self.geo_data_res_no_basement.interfaces['formation_number'].unique()])

        def set_length_orientations(self):

            # Array containing the size of every series. orientations.
            len_series_o = np.asarray(
                [np.sum(self.geo_data_res_no_basement.orientations['order_series'] == i)
                 for i in self.geo_data_res_no_basement.orientations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED

            assert len_series_o.shape[0] is self.len_series_i.shape[0], 'You need at least one orientation and two interfaces' \
                                                                        'per series'
            self.len_series_o = len_series_o

        # def set_pandas_rest_layers(self):
        #
        #     # Position of the first point of every layer
        #     ref_position = np.insert(self.len_interfaces[:-1], 0, 0).cumsum()
        #
        #     # Drop the reference points using pandas indeces to get just the rest_layers array
        #     pandas_rest_layer_points = self.geo_data_res_no_basement.interfaces.drop(ref_position)
        #     self.pandas_rest_layer_points = pandas_rest_layer_points

        def set_length_series(self):

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(self.pandas_rest_layer_points['order_series'] == i)
                 for i in self.pandas_rest_layer_points['order_series'].unique()])

            if len_series_i.shape[0] is 0:
                len_series_i = np.insert(len_series_i, 0, 0)

            self.len_series_i = len_series_i

        def set_ref_position(self):
            # Position of the first point of every layer
            self.ref_position = np.insert(self.len_interfaces[:-1], 0, 0).cumsum()

        def set_layers_rest(self):

            # Drop the reference points using pandas indeces to get just the rest_layers array
            try:
                self.pandas_rest_layer_points = self.geo_data_res_no_basement.interfaces.drop(self.ref_position)
            except ValueError:
                raise ValueError('It is necessary at least one point per layer to be able to compile')

            self.set_length_series()

        def set_layers_ref(self):

            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            self.pandas_ref_layer_points = self.geo_data_res_no_basement.interfaces.iloc[self.ref_position]
           # self.len_interfaces = self.len_interfaces

            pandas_ref_layer_points_rep = self.pandas_ref_layer_points.apply(lambda x: np.repeat(x, self.len_interfaces - 1))
           # ref_layer_points = pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()

            #self.ref_layer_points = ref_layer_points
            self.pandas_ref_layer_points_rep = pandas_ref_layer_points_rep
            # Check no reference points in rest points (at least in coor x)
            # assert not any(self.pandas_ref_layer_points_rep.iloc[:, 0]) in self.pandas_rest_layer_points.iloc[:, 0], \
            #     'A reference point is in the rest list point. Check you do ' \
            #     'not have duplicated values in your dataframes'

        def data_prep(self, **kwargs):
            # This logic is highly interdependent


            self.set_length_interface()
            self.set_ref_position()
            self.set_layers_rest()
            self.set_layers_ref()
            self.set_length_orientations()
            self.set_u_grade(**kwargs)

        def _data_prep(self, **kwargs):
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
                [np.sum(self.geo_data_res_no_basement.interfaces['formation_number'] == i)
                 for i in self.geo_data_res_no_basement.interfaces['formation_number'].unique()])

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form.astype('int32'))
            self.tg.npf.set_value(np.cumsum(np.concatenate(([0], len_rest_form))).astype('int32')) # Last value is useless and breaks the basement

            # Position of the first point of every layer
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop the reference points using pandas indeces to get just the rest_layers array
            pandas_rest_layer_points = self.geo_data_res_no_basement.interfaces.drop(ref_position)
            self.pandas_rest_layer_points = pandas_rest_layer_points

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])
            if len_series_i.shape[0] is 0:
                len_series_i = np.insert(len_series_i, 0, 0)
            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum().astype('int32'))

            # Array containing the size of every series. orientations.
            len_series_f = np.asarray(
                [np.sum(self.geo_data_res_no_basement.orientations['order_series'] == i)
                 for i in self.geo_data_res_no_basement.orientations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED

            assert len_series_f.shape[0] is len_series_i.shape[0], 'You need at least one orientation and two interfaces' \
                                                                   'per series'
            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum().astype('int32'))

            # =========================
            # Choosing Universal drifts
            # =========================
            if u_grade is None:
                u_grade = np.zeros_like(len_series_i)
                u_grade[(len_series_i > 1)] = 1

            else:
                u_grade = np.array(u_grade)

            if 'u_grade' in verbose:
                print(u_grade)

            n_universal_eq = np.zeros_like(len_series_i)
            n_universal_eq[u_grade == 0] = 0
            n_universal_eq[u_grade == 1] = 3
            n_universal_eq[u_grade == 2] = 9

            # it seems I have to pass list instead array_like that is weird
            self.tg.n_universal_eq_T.set_value(list(n_universal_eq.astype('int32')))

            # ================
            # Prepare Matrices
            # ================
            # Rest layers matrix # PYTHON VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            self.pandas_ref_layer_points = self.geo_data_res_no_basement.interfaces.iloc[ref_position]
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
            dips_position = self.geo_data_res_no_basement.orientations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self.geo_data_res_no_basement.orientations["dip"].as_matrix()
            azimuth = self.geo_data_res_no_basement.orientations["azimuth"].as_matrix()
            polarity = self.geo_data_res_no_basement.orientations["polarity"].as_matrix()

            # Set all in a list casting them in the chosen dtype
            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def compute_x_0(self):
            x_0 = np.vstack((self.geo_data_res_no_basement.x_to_interp_given,
                                          self.pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix(),
                                          self.pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()))

            return x_0

        def compute_universal_matrix(self, x_0):

            # Creating the drift matrix.
            universal_matrix = np.vstack((x_0.T,
                                          (x_0 ** 2).T,
                                          x_0[:, 0] * x_0[:, 1],
                                          x_0[:, 0] * x_0[:, 2],
                                          x_0[:, 1] * x_0[:, 2],
                                          ))
            return universal_matrix

        def input_data_python(self):
            # orientations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self.geo_data_res_no_basement.orientations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self.geo_data_res_no_basement.orientations["dip"].as_matrix()
            azimuth = self.geo_data_res_no_basement.orientations["azimuth"].as_matrix()
            polarity = self.geo_data_res_no_basement.orientations["polarity"].as_matrix()
            ref_layer_points = self.pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()
            rest_layer_points = self.pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            assert not any(ref_layer_points[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

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
                nugget_effect_gradient (flaot): Nugget effect of orientations. Default to 0.01
            """

            # Kwargs
            # --This is DEP because is a condition not a shared-- u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect_gradient = kwargs.get('nugget_effect_gradient', 0.01)
            nugget_effect_scalar = kwargs.get('nugget_effect_scalar', 1e-6)
            # Default range
            if not range_var:
                range_var = np.sqrt((self.geo_data_res_no_basement.extent[0] - self.geo_data_res_no_basement.extent[1]) ** 2 +
                                    (self.geo_data_res_no_basement.extent[2] - self.geo_data_res_no_basement.extent[3]) ** 2 +
                                    (self.geo_data_res_no_basement.extent[4] - self.geo_data_res_no_basement.extent[5]) ** 2)

            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # x_to_interpolate = np.vstack((self.geo_data_res_no_basement.x_to_interp_given,
            #                               self.pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix(),
            #                               self.pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()))
            #
            #
            # # Creating the drift matrix.
            # universal_matrix = np.vstack((x_to_interpolate.T,
            #                              (x_to_interpolate ** 2).T,
            #                               x_to_interpolate[:, 0] * x_to_interpolate[:, 1],
            #                               x_to_interpolate[:, 0] * x_to_interpolate[:, 2],
            #                               x_to_interpolate[:, 1] * x_to_interpolate[:, 2],
            #                               ))

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (self.len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form.astype('int32'))
            self.tg.npf.set_value(np.cumsum(np.concatenate(([0], len_rest_form))).astype('int32'))  # Last value is useless
                                                                                                    # and breaks the basement
            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(self.len_series_i, 0, 0).cumsum().astype('int32'))
            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_f.set_value(np.insert(self.len_series_o, 0, 0).cumsum().astype('int32'))
            # Setting shared variables
            # Range
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            # Covariance at 0
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            # universal grades
            self.tg.n_universal_eq_T.set_value(list(self.n_universal_eq.astype('int32')))
            # nugget effect
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect_gradient))
            self.tg.nugget_effect_scalar_T.set_value(np.cast[self.dtype](nugget_effect_scalar))
            # Just grid. I add a small number to avoid problems with the origin point
            x_0 = self.compute_x_0()
            self.tg.grid_val_T.set_value(np.cast[self.dtype](x_0 + 10e-9))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](self.compute_universal_matrix(x_0) + 1e-10))
            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((1, self.geo_data_res_no_basement.x_to_interp_given.shape[0]),
                                                   dtype=self.dtype))
            # Unique number assigned to each lithology
            self.tg.n_formation.set_value(self.formation_number)
            # Final values the lith block takes
            self.tg.formation_values.set_value(self.formation_value)
            # Number of formations per series. The function is not pretty but the result is quite clear
            n_formations_per_serie = np.insert(self.geo_data_res_no_basement.interfaces.groupby('order_series').
                                               formation.nunique().values.cumsum(), 0, 0).astype('int32')
            self.tg.n_formations_per_serie.set_value(n_formations_per_serie)
            # Init the list to store the values at the interfaces. Here we init the shape for the given dataset
            self.tg.final_scalar_field_at_formations.set_value(np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                                                                        dtype=self.dtype))
            self.tg.final_scalar_field_at_faults.set_value(np.zeros(self.tg.n_formations_per_serie.get_value()[-1],
                                                                    dtype=self.dtype))

            self.tg.n_faults.set_value(self.geo_data_res_no_basement.n_faults)
            # Set fault relation matrix
            self.check_fault_ralation()
            self.tg.fault_relation.set_value(self.fault_rel.astype('int32'))
            # if self.geo_data_res_no_basement.faults_relations is not None:
            #     self.tg.faults_relations.set_value(self.geo_data_res_no_basement.faults_relations.values.astype('int32'))
            # else:
            #     fault_rel = np.zeros((self.geo_data_res_no_basement.interfaces['series'].nunique(),
            #                           self.geo_data_res_no_basement.interfaces['series'].nunique()))
            #
            #     self.tg.faults_relations.set_value(fault_rel.astype('int32'))

        def check_fault_ralation(self):
            # Set fault relation matrix
            if self.geo_data_res_no_basement.faults_relations is None:
                self.fault_rel = np.zeros((self.geo_data_res_no_basement.interfaces['series'].nunique(),
                                      self.geo_data_res_no_basement.interfaces['series'].nunique()))
            else:
                self.fault_rel = self.geo_data_res_no_basement.faults_relations.values.astype('int32')

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
            dx, dy, dz = (self.geo_data_res_no_basement.extent[1] - self.geo_data_res_no_basement.extent[0]) / resolution[0], (self.geo_data_res_no_basement.extent[3] - self.geo_data_res_no_basement.extent[2]) / resolution[
                0], (self.geo_data_res_no_basement.extent[5] - self.geo_data_res_no_basement.extent[4]) / resolution[0]

            #dx, dy, dz = self.geo_data_res_no_basement.grid.dx, self.geo_data_res_no_basement.grid.dy, self.geo_data_res_no_basement.grid.dz
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
            print('range', self.tg.a_T.get_value(), self.tg.a_T.get_value() * self.geo_data_res_no_basement.rescaling_factor)
            # Number of drift equations
            print('Number of drift equations', self.tg.n_universal_eq_T.get_value())
            # Covariance at 0
            print('Covariance at 0', self.tg.c_o_T.get_value())
            # orientations nugget effect
            print('orientations nugget effect', self.tg.nugget_effect_grad_T.get_value())

            print('scalar nugget effect', self.tg.nugget_effect_scalar_T.get_value())
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
