"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/10 /2016

    @author: Miguel de la Varga
"""

from os import path
import sys

# This is for sphenix to find the packages
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as _np
from numpy import ndarray
import pandas as _pn
from pandas import DataFrame

import copy
import warnings

from gempy.plotting.visualization import PlotData2D, vtkVisualization
from gempy.data_management import InputData, GridClass
from gempy.interpolator import InterpolatorData
from gempy.plotting.sequential_pile import StratigraphicPile
from gempy.topology import topology_analyze as _topology_analyze
from gempy.utils.geomodeller_integration import ReadGeoModellerXML as _ReadGeoModellerXML
import gempy.posterior_analysis as pa # So far we use this type of import because the other one makes a copy and blows up some asserts


def compute_model(interp_data, output='geology', u_grade=None, get_potential_at_interfaces=False):
    """
    Computes the geological model and any extra output given.

    Args:

        interp_data (:class:`gempy.interpolator.InterpolatorData`)
        output ({'geology', 'gravity', 'gradients'}): Only if theano functions has not been compiled yet
        u_grade (array-like of {0, 1, 2}): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
            be either 0, 1 or 2  and the length has to be the number of series. By default the value
            depends on the number of points given as input to try to avoid singular matrix. NOTE: if during the computation
            of the model a singular matrix is returned try to reduce the u_grade of the series.
        get_potential_at_interfaces (bool): Get potential at interfaces

    Returns:
        list of :class:`_np.array`: depending on the chosen out returns different number of solutions:
            if output is geology:
                1) Lithologies: block and scalar field
                2) Faults: block and scalar field for each faulting plane
            if output is 'gravity':
                1) Weights: block and scalar field
                2) Faults: block and scalar field for each faulting plane
                3) Forward gravity
            if output is gradients:
                1) Lithologies: block and scalar field
                2) Faults: block and scalar field for each faulting plane
                3) Gradients of scalar field x
                4) Gradients of scalar field y
                5) Gradients of scalar field z
        In addition if get_potential_at_interfaces is True, the value of the potential field at each of
        the interfaces is given as well
    """
    if not getattr(interp_data, 'th_fn', None):
        interp_data.th_fn = interp_data.compile_th_fn(output=output)

    i = interp_data.get_input_data(u_grade=u_grade)

    assert interp_data.interpolator.len_interfaces.min() > 1,  \
        'To compute the model is necessary at least 2 interface points per layer'

    sol = interp_data.th_fn(*i)
    interp_data.potential_at_interfaces = sol[-1]

    if get_potential_at_interfaces:
        return sol
    else:
        return sol[:-1]


def compute_model_at(new_grid_array, interp_data, output='geology', u_grade=None, get_potential_at_interfaces=False):
    """
    This function does the same as :func:`~gempy.compute_model` plus the addion functionallity of passing a given
    array of point where evaluate the model instead of using the :class:`gempy.data_management.InputData` grid.

    Args:
        new_grid_array (:class:`_np.array`): 2D array with XYZ (columns) coorinates

    Returns:
        list of :class:`_np.array`: depending on the chosen out returns different number of solutions:
            if output is geology:
                1) Lithologies: block and scalar field
                2) Faults: block and scalar field for each faulting plane
            if output is 'gravity':
                1) Weights: block and scalar field
                2) Faults: block and scalar field for each faulting plane
                3) Forward gravity
            if output is gradients:
                1) Lithologies: block and scalar field
                2) Faults: block and scalar field for each faulting plane
                3) Gradients of scalar field x
                4) Gradients of scalar field y
                5) Gradients of scalar field z
        In addition if get_potential_at_interfaces is True, the value of the potential field at each of
        the interfaces is given as well
    """

    # First Create a new custom grid using the GridClass
    new_grid = GridClass()

    # Here we can pass the new coordinates as a 2D numpy array XYZ
    new_grid.create_custom_grid(new_grid_array)

    # Next we rescale the data. For this the main parameters are already stored in interp_data
    new_grid_res = (new_grid.values - interp_data.centers.as_matrix()) / interp_data.rescaling_factor + 0.5001

    # We stack the input data
    x_to_interpolate = _np.vstack((new_grid_res,
                                  interp_data.interpolator.pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix(),
                                  interp_data.interpolator.pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()))

    # And create the drift function matrix. (This step could be done within theano to speed up a bit)
    universal_matrix = _np.vstack((x_to_interpolate.T,
                                  (x_to_interpolate ** 2).T,
                                  x_to_interpolate[:, 0] * x_to_interpolate[:, 1],
                                  x_to_interpolate[:, 0] * x_to_interpolate[:, 2],
                                  x_to_interpolate[:, 1] * x_to_interpolate[:, 2],
                                  ))

    # Last step is to change the variables in the theano graph
    interp_data.interpolator.tg.grid_val_T.set_value(_np.cast[interp_data.interpolator.dtype](x_to_interpolate + 10e-9))
    interp_data.interpolator.tg.universal_grid_matrix_T.set_value(
        _np.cast[interp_data.interpolator.dtype](universal_matrix + 1e-10))

    # Now we are good to compute the model agai only in the new point
    sol = compute_model(interp_data, output=output, u_grade=u_grade, get_potential_at_interfaces=get_potential_at_interfaces)
    return sol


def create_data(extent, resolution=(50, 50, 50), **kwargs):
    """
    Method to create a :class:`gempy.data_management.InputData` object. It is analogous to gempy.InputData()

    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.

    Keyword Args:
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_f: Path to the data bases of orientations. Default os.getcwd()

    Returns:
        :class:`gempy.data_management.InputData`

    """

    return InputData(extent, resolution, **kwargs)


def create_from_geomodeller_xml(fp, resolution=(50, 50, 50), return_xml=False, **kwargs):
    """
    Creates InputData object from a GeoModeller xml file. Automatically extracts and sets model extent, interface
    and orientation data as well as the stratigraphic pile.

    Args:
        fp (str): Filepath for the GeoModeller xml file to be read.
        resolution (tuple, optional): Tuple containing the number of voxels in dimensions (x,y,z). Defaults to 50.
        return_xml (bool, optional): Toggles returning the ReadGeoModellerXML instance to leverage further info from the
            xml file (e.g. for stratigraphic pile ordering). Defaults to True.
        **kwargs: Keyword arguments for create_data function.

    Returns:
        gp.data_management.InputData
    """
    gmx = _ReadGeoModellerXML(fp)  # instantiate parser class with filepath of xml

    # instantiate InputData object with extent and resolution
    geo_data = create_data(gmx.extent, resolution, **kwargs)

    # set interface and orientation dataframes
    geo_data.interfaces = gmx.interfaces
    geo_data.orientations = gmx.orientations

    if return_xml:
        return geo_data, gmx
    else:
        return geo_data


def data_to_pickle(geo_data, path=False):
    """
     Save InputData object to a python pickle (serialization of python). Be aware that if the dependencies
     versions used to export and import the pickle differ it may give problems

     Args:
         geo_data (:class:`gempy.data_management.InputData`)
         path (str): path where save the pickle (without .pickle)

     Returns:
         None
     """
    geo_data.data_to_pickle(path)


def get_series(geo_data):
    """
    Args:
         geo_data (:class:`gempy.data_management.InputData`)

    Returns:
        :class:`DataFrame`: Return series and formations relations
    """
    return geo_data.series


def get_grid(geo_data):
    """
    Coordinates can be found in :class:`gempy.data_management.GridClass.values`

     Args:
          geo_data (:class:`gempy.interpolator.InterpolatorData`)

     Returns:
        :class:`gempy.data_management.GridClass`
    """
    return geo_data.grid


def get_resolution(geo_data):
    return geo_data.resolution


def get_extent(geo_data):
    return geo_data.extent


def get_data(geo_data, itype='all', numeric=False, verbosity=0):
    """
    Method to return the data stored in :class:`DataFrame` within a :class:`gempy.interpolator.InterpolatorData`
    object.

    Args:
        geo_data (:class:`gempy.interpolator.InterpolatorData`)
        itype(str {'all', 'interfaces', 'orientaions', 'formations', 'series', 'faults', 'fautls_relations'}): input
            data type to be retrieved.
        numeric (bool): if True it only returns numberical properties. This may be useful due to memory issues
        verbosity (int): Number of properties shown

    Returns:
        pandas.core.frame.DataFrame

    """
    return geo_data.get_data(itype=itype, numeric=numeric, verbosity=verbosity)


def get_sequential_pile(geo_data):
    """
    Visualize an interactive stratigraphic pile to move around the formations and the series. IMPORTANT NOTE:
    To have the interactive properties it is necessary the use of an interactive backend. (In notebook use:
    %matplotlib qt5 or notebook)

    Args:
        geo_data (:class:`gempy.interpolator.InterpolatorData`)

    Returns:
        :class:`matplotlib.pyplot.Figure`
    """
    return StratigraphicPile(geo_data)


# =====================================
# Functions for the InterpolatorData
# =====================================

def get_kriging_parameters(interp_data, verbose=0):
    """
    Print the kringing parameters

    Args:
        interp_data (:class:`gempy.data_management.InputData`)
        verbose (int): if > 0 print all the shape values as well.

    Returns:
        None
    """
    return interp_data.interpolator.get_kriging_parameters(verbose=verbose)

# TODO check that is a interp_data object and if not try to create within the function one from the geo_data


def get_th_fn(interp_data):
    """
    Get the compiled theano function

    Args:
        interp_data (:class:`gempy.data_management.InputData`)

    Returns:
        :class:`theano.compile.function_module.Function`: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
    """
    assert getattr(interp_data, 'th_fn', False), 'Theano has not been compiled yet'

    return interp_data.compile_th_fn()


def get_surfaces(interp_data, potential_lith=None, potential_fault=None, n_formation='all',
                 step_size=1, original_scale=True):
    """
    Compute vertices and simplices of the interfaces for its vtk visualization and further
    analysis

    Args:
        interp_data (:class:`gempy.data_management.InputData`)
        potential_lith (ndarray): 1D numpy array with the solution of the computation of the model
         containing the scalar field of potentials (second row of lith solution)
        potential_fault (ndarray): 1D numpy array with the solution of the computation of the model
         containing the scalar field of the faults (every second row of fault solution)
        n_formation (int or 'all'): Positive integer with the number of the formation of which the surface is returned.
         use method get_formation_number() to get a dictionary back with the values
        step_size (int): resolution of the method. This is every how many voxels the marching cube method is applied
        original_scale (bool): choosing if the coordinates of the vertices are given in the original or the rescaled
         coordinates

    Returns:
        vertices, simpleces
    """
    try:
        getattr(interp_data, 'potential_at_interfaces')
    except AttributeError:
        raise AttributeError('You need to compute the model first')

    def get_surface(potential_block, interp_data, pot_int, n_formation, step_size, original_scale):
        """
        Get an individual surface

        """

        assert n_formation >= 0, 'Number of the formation has to be positive'

        # In case the values are separated by series I put all in a vector
        pot_int = interp_data.potential_at_interfaces.sum(axis=0)

        from skimage import measure

        if not potential_block.max() > pot_int[n_formation]:
            pot_int[n_formation] = potential_block.max()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        if not potential_block.min() < pot_int[n_formation]:
            pot_int[n_formation] = potential_block.min()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        vertices_p, simplices_p, normals, values = measure.marching_cubes_lewiner(
            potential_block.reshape(interp_data.geo_data_res.resolution[0],
                                    interp_data.geo_data_res.resolution[1],
                                    interp_data.geo_data_res.resolution[2]),
            pot_int[n_formation],
            step_size=step_size,
            spacing=((interp_data.geo_data_res.extent[1] - interp_data.geo_data_res.extent[0]) / interp_data.geo_data_res.resolution[0],
                     (interp_data.geo_data_res.extent[3] - interp_data.geo_data_res.extent[2]) / interp_data.geo_data_res.resolution[1],
                     (interp_data.geo_data_res.extent[5] - interp_data.geo_data_res.extent[4]) / interp_data.geo_data_res.resolution[2]))

        if original_scale:
            vertices_p = interp_data.rescaling_factor * vertices_p + _np.array([interp_data._geo_data.extent[0],
                                                                            interp_data._geo_data.extent[2],
                                                                            interp_data._geo_data.extent[4]]).reshape(1, 3)
        else:
            vertices_p += _np.array([interp_data.geo_data_res.extent[0],
                                   interp_data.geo_data_res.extent[2],
                                   interp_data.geo_data_res.extent[4]]).reshape(1, 3)
        return vertices_p, simplices_p

    vertices = []
    simplices = []

    n_formations = _np.arange(interp_data.geo_data_res.interfaces['formation'].nunique())

    # Looping the scalar fields of the faults
    if potential_fault is not None:

        assert len(_np.atleast_2d(potential_fault)) == interp_data.geo_data_res.n_faults, 'You need to pass a potential field per fault'

        pot_int = interp_data.potential_at_interfaces[:interp_data.geo_data_res.n_faults + 1]
        for n in n_formations[:interp_data.geo_data_res.n_faults]:
            v, s = get_surface(_np.atleast_2d(potential_fault)[n], interp_data, pot_int, n,
                               step_size=step_size, original_scale=original_scale)
            vertices.append(v)
            simplices.append(s)

    # Looping the scalar fields of the lithologies
    if potential_lith is not None:
        pot_int = interp_data.potential_at_interfaces[interp_data.geo_data_res.n_faults:]

        # Compute the vertices of the lithologies
        if n_formation == 'all':

            for n in n_formations[interp_data.geo_data_res.n_faults:]: #interp_data.geo_data_res.interfaces['formation_number'][~interp_data.geo_data_res.interfaces['isFault']].unique():
                # if n == 0:
                #     continue
                #else:
                    v, s = get_surface(potential_lith, interp_data, pot_int, n,
                                       step_size=step_size, original_scale=original_scale)
                    vertices.append(v)
                    simplices.append(s)
        else:
            vertices, simplices = get_surface(potential_lith, interp_data, pot_int, n_formation,
                                              step_size=step_size, original_scale=original_scale)

    return vertices, simplices


def interactive_df_open(geo_data, itype):
    """
    Open the qgrid interactive DataFrame (http://qgrid.readthedocs.io/en/latest/).
    To seve the changes see: :func:`~gempy.gempy_front.interactive_df_change_df`


    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        itype(str {'all', 'interfaces', 'orientaions', 'formations', 'series', 'faults', 'fautls_relations'}): input
            data type to be retrieved.

    Returns:
        :class:`DataFrame`: Interactive DF
    """
    return geo_data.interactive_df_open(itype)


def interactive_df_change_df(geo_data, only_selected=False):
    """
    Confirm and return the changes made to a dataframe using qgrid interactively. To update the
    :class:`gempy.data_management.InputData` with the modify df use the correspondant set function.

    Args:
        geo_data (:class:`gempy.data_management.InputData`): the same :class:`gempy.data_management.InputData`
            object used to call :func:`~gempy.gempy_front.interactive_df_open`
        only_selected (bool) if True only returns the selected rows

    Returns:
        :class:`DataFrame`
    """
    return geo_data.interactive_df_get_changed_df(only_selected=only_selected)


def set_orientation_from_interfaces(geo_data, indices_array):
    """
    Create and set orientations from at least 3 points of the :attr:`gempy.data_management.InputData.interfaces`
     Dataframe
    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        indices_array (array-like): 1D or 2D array with the pandas indices of the
          :attr:`gempy.data_management.InputData.interfaces`. If 2D every row of the 2D matrix will be used to create an
          orientation
        verbose:

    Returns:
        :attr:`gempy.data_management.InputData.orientations`: Already updated inplace
    """

    if _np.ndim(indices_array) is 1:
        indices = indices_array
        form = geo_data.interfaces['formation'].loc[indices].unique()
        assert form.shape[0] is 1, 'The interface points must belong to the same formation'
        form = form[0]
        print()
        ori_parameters = geo_data.create_orientation_from_interfaces(indices)
        geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                 dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                 G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                 formation=form)
    elif _np.ndim(indices_array) is 2:
        for indices in indices_array:
            form = geo_data.interfaces['formation'].loc[indices].unique()[0]
            assert form.shape[0] is 1, 'The interface points must belong to the same formation'
            form = form[0]
            ori_parameters = geo_data.create_orientation_from_interfaces(indices)
            geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                     dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                     G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                     formation=form[0])

    geo_data.update_df()
    return geo_data.orientations


def precomputations_gravity(interp_data, n_chunck=25, densities=None):
    try:
        getattr(interp_data, 'geophy')
    except:
        raise AttributeError('You need to set a geophysical object first. See set_geophysics_obj')

    tz, select = interp_data.geophy.compute_gravity(n_chunck)
    interp_data.interpolator.set_z_comp(tz, select)

    if densities is not None:
        set_densities(interp_data, densities)
    return tz, select


def read_pickle(path):
    """
    Read InputData object from python pickle.

    Args:
       path (str): path where save the pickle

    Returns:
        :class:`gempy.data_management.InputData`

    """
    import pickle
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        return data


def rescale_factor_default(geo_data):
    """
    Returns the default rescaling factor for a given geo_data

    Args:
        geo_data(:class:`gempy.data_management.InputData`)

    Returns:
        float: rescaling factor
    """
    # Check which axis is the largest
    max_coord = _pn.concat(
        [geo_data.orientations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
    min_coord = _pn.concat(
        [geo_data.orientations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

    # Compute rescaling factor if not given
    rescaling_factor = 2 * _np.max(max_coord - min_coord)
    return rescaling_factor


def rescale_data(geo_data, rescaling_factor=None):
    """
    Rescale the data of a :class:`gempy.data_management.InputData`
    object between 0 and 1 due to stability problem of the float32.

    Args:
        geo_data(:class:`gempy.data_management.InputData`)
        rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

    Returns:
        gempy.data_management.InputData: Rescaled data

    """
    # TODO split this function in compute rescaling factor and rescale z

    # Check which axis is the largest
    max_coord = _pn.concat(
        [geo_data.orientations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
    min_coord = _pn.concat(
        [geo_data.orientations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

    # Compute rescalin factor if not given
    if not rescaling_factor:
        rescaling_factor = 2 * _np.max(max_coord - min_coord)

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
    new_coord_extent = (geo_data.extent - _np.repeat(centers, 2)) / rescaling_factor + 0.5001

    geo_data_rescaled = copy.deepcopy(geo_data)
    geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
    geo_data_rescaled.orientations[['X', 'Y', 'Z']] = new_coord_orientations
    geo_data_rescaled.extent = new_coord_extent.as_matrix()

    geo_data_rescaled.grid.values = (geo_data.grid.values - centers.as_matrix()) / rescaling_factor + 0.5001

    # Saving useful values for later
    geo_data_rescaled.rescaling_factor = rescaling_factor

    return geo_data_rescaled


def select_series(geo_data, series):
    """
    Return the formations of a given serie in string

    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        series(list of int or list of str): Subset of series to be selected

    Returns:
         :class:`gempy.data_management.InputData`: New object only containing the selected series
    """
    new_geo_data = copy.deepcopy(geo_data)

    if type(series) == int or type(series[0]) == int:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['order_series'].isin(series)]
        new_geo_data.orientations = geo_data.orientations[geo_data.orientations['order_series'].isin(series)]
    elif type(series[0]) == str:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['series'].isin(series)]
        new_geo_data.orientations = geo_data.orientations[geo_data.orientations['series'].isin(series)]

    # Count faults
    new_geo_data.set_faults(new_geo_data.count_faults())

    # Change the dataframe with the series
    new_geo_data.series = new_geo_data.series[new_geo_data.interfaces['series'].unique().
        remove_unused_categories().categories].dropna(how='all')
    new_geo_data.formations = new_geo_data.formations.loc[new_geo_data.interfaces['formation'].unique().
        remove_unused_categories().categories]
    new_geo_data.update_df()
    return new_geo_data


def set_series(geo_data, series_distribution=None, order_series=None, order_formations=None,
               verbose=0):
    """
    Function to set in place the different series of the project with their correspondent formations

    Args:
        geo_data (:class:`gempy.data_management.InputData`)
        series_distribution (dict or :class:`DataFrame`): with the name of the series as key and the name of the
          formations as values.
        order_series(Optional[list]): only necessary if passed a dict (python < 3.6)order of the series by default takes the
             dictionary keys which until python 3.6 are random. This is important to set the erosion relations between the different series
        order_formations(Optional[list]): only necessary if passed a dict (python < 3.6)order of the series by default takes the
            dictionary keys which until python 3.6 are random. This is important to set the erosion relations between the different series
        verbose(int): if verbose is True plot hte sequential pile

    Notes:
        The following dataframes will be modified in place
            1) geo_data.series: A pandas DataFrame with the series and formations relations
            2) geo_data.interfaces: one extra column with the given series
            3) geo_data.orientations: one extra column with the given series
    """
    geo_data.update_df(series_distribution=series_distribution, order=order_series)
    if order_formations is not None:
        geo_data.set_formations(formation_order=order_formations)
        geo_data.order_table()

    if verbose > 0:
         return get_sequential_pile(geo_data)
    else:
        return None


def set_order_formations(geo_data, order_formations):
    warnings.warn("set_order_formations will be removed in version 1.2, "
                  "use gempy.set_formations function instead", FutureWarning)
    geo_data.set_formations(formation_order=order_formations)


def set_formations(geo_data, formations=None, formations_order=None, formations_values=None):
    """
    Function to order and change the value of the model formations. The values of the formations will be the final
    numerical value that each formation will take in the interpolated geological model (lithology block)
    Args:
        geo_data (:class:`gempy.data_management.InputData`):
        formations_order (list of str): List with a given order of the formations. Due to the interpolation algorithm
            this order is only relevant to keep consistent the colors of layers and input data. The order ultimately is
            determined by the geometric sedimentary order
        formations (list of str): same as formations order. you can use any
        formations_values (list of floats or int):  values of the formations will be the final
    numerical value that each formation will take in the interpolated geological model (lithology block)

    Returns:
        :class:`DataFrame`: formations dataframe already updated in place

    """
    if formations and not formations_order:
        formations_order = formations

    geo_data.set_formations(formation_order=formations_order, formation_values=formations_values)
    return  geo_data.formations


def set_interfaces(geo_data, interf_dataframe, append=False):
    """
     Method to change or append a Dataframe to interfaces in place.

     Args:
         geo_data(:class:`gempy.data_management.InputData`)
         interf_dataframe (:class:`DataFrame`)
         append (Bool): if you want to append the new data frame or substitute it
     """
    geo_data.set_interfaces(interf_dataframe, append=append)


def set_orientations(geo_data, orient_dataframe, append=False):
    """
    Method to change or append a dataframe to orientations in place.  A equivalent Pandas Dataframe with
    ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

    Args:
         geo_data(:class:`gempy.data_management.InputData`)
         interf_dataframe (:class:`DataFrame`)
         append (Bool): if you want to append the new data frame or substitute it
    """

    geo_data.set_orientations(orient_dataframe, append=append)


def set_grid(geo_data, grid):
    """
    Set a new :class:`gempy.data_management.GridClass` object into a :class:`gempy.data_management.InputData` object.

    Args:
        geo_data (:class:`gempy.data_management.InputData`):
        grid (:class:`gempy.data_management.GridClass`):

    """
    assert isinstance(grid, GridClass)
    geo_data.grid = grid
    geo_data.extent = grid.extent
    geo_data.resolution = grid.resolution


def set_interpolation_data(geo_data, **kwargs):
    """
    Create a :class:`gempy.interpolator.InterpolatorData`. InterpolatorData is a class that contains all the
     preprocessing operations to prepare the data to compute the model.
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
    in_data = InterpolatorData(geo_data, **kwargs)
    return in_data


def set_geophysics_obj(interp_data, ai_extent, ai_resolution, ai_z=None, range_max=None):

    assert isinstance(interp_data, InterpolatorData), 'The object has to be instance of the InterpolatorInput'
    interp_data.create_geophysics_obj(ai_extent, ai_resolution, ai_z=ai_z, range_max=range_max)
    return interp_data.geophy


def set_densities(interp_data, densities):

    interp_data.interpolator.set_densities(densities)


# =====================================
# Functions for Geophysics
# =====================================
def topology_compute(geo_data, lith_block, fault_block,
                     cell_number=None, direction=None,
                     compute_areas=False, return_label_block=False):
    """
    Computes model topology and returns graph, centroids and look-up-tables.

    Args:
        geo_data (gempy.data_management.InputData): GemPy's data object for the model.
        lith_block (ndarray): Lithology block model.
        fault_block (ndarray): Fault block model.
        cell_number (int): Cell number for 2-D slice topology analysis. Default None.
        direction (str): "x", "y" or "z" specifying the slice direction for 2-D topology analysis. Default None.
        compute_areas (bool): If True computes adjacency areas for connected nodes in voxel number. Default False.
        return_label_block (bool): If True additionally returns the uniquely labeled block model as np.ndarray. Default False.

    Returns:
        tuple:
            G: Region adjacency graph object (skimage.future.graph.rag.RAG) containing the adjacency topology graph
                (G.adj).
            centroids (dict): Centroid node coordinates as a dictionary with node id's (int) as keys and (x,y,z) coordinates
                as values. {node id (int): tuple(x,y,z)}
            labels_unique (np.array): List of all labels used.
            lith_to_labels_lot (dict): Dictionary look-up-table to go from lithology id to node id.
            labels_to_lith_lot (dict): Dictionary look-up-table to go from node id to lithology id.

    """
    fault_block = _np.atleast_2d(fault_block)[::2].sum(axis=0)

    if cell_number is None or direction is None:  # topology of entire block
        lb = lith_block.reshape(geo_data.resolution)
        fb = fault_block.reshape(geo_data.resolution)
    elif direction == "x":
        lb = lith_block.reshape(geo_data.resolution)[cell_number, :, :]
        fb = fault_block.reshape(geo_data.resolution)[cell_number, :, :]
    elif direction == "y":
        lb = lith_block.reshape(geo_data.resolution)[:, cell_number, :]
        fb = fault_block.reshape(geo_data.resolution)[:, cell_number, :]
    elif direction == "z":
        lb = lith_block.reshape(geo_data.resolution)[:, :, cell_number]
        fb = fault_block.reshape(geo_data.resolution)[:, :, cell_number]

    return _topology_analyze(lb, fb, geo_data.n_faults, areas_bool=compute_areas, return_block=return_label_block)


# +++++++
# DEP visualization
#
def export_to_vtk(geo_data, path=None, name=None, lith_block=None, vertices=None, simplices=None):
    """
      Export data to a vtk file for posterior visualizations

      Args:
          geo_data(gempy.InputData): All values of a DataManagement object
          block(numpy.array): 3D array containing the lithology block
          path (str): path to the location of the vtk

      Returns:
          None
      """

    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    if lith_block is not None:
        vtkVisualization.export_vtk_lith_block(geo_data, lith_block, path=path)
    if vertices is not None and simplices is not None:
        vtkVisualization.export_vtk_surfaces(vertices, simplices, path=path, name=name)


def plot_surfaces_3D(geo_data, vertices_l, simplices_l,
                     #formations_names_l, formation_numbers_l,
                     alpha=1, plot_data=True,
                     size=(1920, 1080), fullscreen=False, bg_color=None):
    """
    Plot in vtk the surfaces. For getting vertices and simplices See gempy.get_surfaces

    Args:
        vertices_l (numpy.array): 2D array (XYZ) with the coordinates of the points
        simplices_l (numpy.array): 2D array with the value of the vertices that form every single triangle
        formations_names_l (list): Name of the formation of the surfaces
        formation_numbers_l (list): formation_numbers (int)
        alpha (float): Opacity
        plot_data (bool): Default True
        size (tuple): Resolution of the window
        fullscreen (bool): Launch window in full screen or not

    Returns:
        None
    """

    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)

    w = vtkVisualization(geo_data, bg_color=bg_color)
    w.set_surfaces(vertices_l, simplices_l,
                   #formations_names_l, formation_numbers_l,
                    alpha)

    if plot_data:
        w.set_interfaces()
        w.set_orientations()
    w.render_model(size=size, fullscreen=fullscreen)
    return w


def plot_data(geo_data, direction="y", data_type='all', series="all", legend_font_size=6, **kwargs):
    """
    Plot the projection of the raw data (interfaces and orientations) in 2D following a
    specific directions

    Args:
        direction(str): xyz. Caartesian direction to be plotted
        series(str): series to plot
        **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

    Returns:
        None
    """

    warnings.warn("gempy plotting functionality will be moved in version 1.2, use gempy.plotting module instead", FutureWarning)

    plot = PlotData2D(geo_data)

    # TODO saving options
    return plot.plot_data(direction=direction, data_type=data_type, series=series, legend_font_size=legend_font_size, **kwargs)


def plot_section(geo_data, block, cell_number, direction="y", **kwargs):
    """
    Plot a section of the block model

    Args:
        cell_number(int): position of the array to plot
        direction(str): xyz. Caartesian direction to be plotted
        interpolation(str): Type of interpolation of plt.imshow. Default 'none'.  Acceptable values are 'none'
        ,'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
        'lanczos'
       **kwargs: imshow keywargs

    Returns:
        None
    """
    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    plot = PlotData2D(geo_data)
    sec_plot = plot.plot_block_section(cell_number, block=block, direction=direction, **kwargs)
    # TODO saving options

def plot_scalar_field(geo_data, potential_field, cell_number, N=20,
                      direction="y", plot_data=True, series="all", *args, **kwargs):
    """
    Plot a potential field in a given direction.

    Args:
        cell_number(int): position of the array to plot
        potential_field(str): name of the potential field (or series) to plot
        n_pf(int): number of the  potential field (or series) to plot
        direction(str): xyz. Caartesian direction to be plotted
        serie: *Deprecated*
        **kwargs: plt.contour kwargs

    Returns:
        None
    """
    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    plot = PlotData2D(geo_data)
    plot.plot_scalar_field(potential_field, cell_number, N=N,
                              direction=direction,  plot_data=plot_data, series=series,
                              *args, **kwargs)

def plot_gradient(geo_data, scalar_field, gx, gy, gz, cell_number, q_stepsize=5,
                      direction="y", plot_scalar=True, **kwargs):
    """
        Plot the gradient of the scalar field in a given direction.

        Args:
            geo_data (gempy.DataManagement.InputData): Input data of the model
            scalar_field(numpy.array): scalar field to plot with the gradient
            gx(numpy.array): gradient in x-direction
            gy(numpy.array): gradient in y-direction
            gz(numpy.array): gradient in z-direction
            cell_number(int): position of the array to plot
            q_stepsize(int): step size between arrows to indicate gradient
            direction(str): xyz. Caartesian direction to be plotted
            plot_scalar(bool): boolean to plot scalar field
            **kwargs: plt.contour kwargs

        Returns:
            None
    """
    plot = PlotData2D(geo_data)
    plot.plot_gradient(scalar_field, gx, gy, gz, cell_number, q_stepsize=q_stepsize,
                           direction=direction, plot_scalar=plot_scalar,
                           **kwargs)

def plot_data_3D(geo_data, **kwargs):
    """
    Plot in vtk all the input data of a model
    Args:
        geo_data (gempy.DataManagement.InputData): Input data of the model

    Returns:
        None
    """
    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    vv = vtkVisualization(geo_data)
    vv.set_interfaces()
    vv.set_orientations()
    vv.render_model(**kwargs)
    return vv


def plot_surfaces_3D_real_time(interp_data, vertices_l, simplices_l,
                     #formations_names_l, formation_numbers_l,
                     alpha=1, plot_data=True, posterior=None, samples=None,
                     size=(1920, 1080), fullscreen=False):
    """
    Plot in vtk the surfaces in real time. Moving the input data will affect the surfaces.
    IMPORTANT NOTE it is highly recommended to have the flag fast_run in the theano optimization. Also note that the
    time needed to compute each model increases linearly with every potential field (i.e. fault or discontinuity). It
    may be better to just modify each potential field individually to increase the speed (See gempy.select_series).

    Args:
        vertices_l (numpy.array): 2D array (XYZ) with the coordinates of the points
        simplices_l (numpy.array): 2D array with the value of the vertices that form every single triangle
        formations_names_l (list): Name of the formation of the surfaces
        formation_numbers_l (list): formation_numbers (int)
        alpha (float): Opacity
        plot_data (bool): Default True
        size (tuple): Resolution of the window
        fullscreen (bool): Launch window in full screen or not

    Returns:
        None
    """
    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    assert isinstance(interp_data, InterpolatorData), 'The object has to be instance of the InterpolatorInput'
    w = vtkVisualization(interp_data.geo_data_res, real_time=True)
    w.set_surfaces(vertices_l, simplices_l,
                   #formations_names_l, formation_numbers_l,
                    alpha)

    if posterior is not None:
        assert isinstance(posterior, pa.Posterior), 'The object has to be instance of the Posterior class'
        w.post = posterior
        if samples is not None:
            samp_i = samples[0]
            samp_f = samples[1]
        else:
            samp_i = 0
            samp_f = posterior.n_iter

        w.create_slider_rep(samp_i, samp_f, samp_f)

    w.interp_data = interp_data
    if plot_data:
        w.set_interfaces()
        w.set_orientations()
    w.render_model(size=size, fullscreen=fullscreen)


def plot_topology(geo_data, G, centroids, direction="y"):
    """
    Plot the topology adjacency graph in 2-D.

    Args:
        geo_data (gempy.data_management.InputData):
        G (skimage.future.graph.rag.RAG):
        centroids (dict): Centroid node coordinates as a dictionary with node id's (int) as keys and (x,y,z) coordinates
                as values.
    Keyword Args
        direction (str): "x", "y" or "z" specifying the slice direction for 2-D topology analysis. Default None.

    Returns:
        Nothing, it just plots.
    """
    warnings.warn("gempy plotting functionality will be moved in version 1.2, "
                  "use gempy.plotting module instead", FutureWarning)
    PlotData2D.plot_topo_g(geo_data, G, centroids, direction=direction)
