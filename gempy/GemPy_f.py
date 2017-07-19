"""
Module with classes and methods to perform implicit regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 10/10 /2016

@author: Miguel de la Varga

"""
from __future__ import division
import os
from os import path
import sys

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

# --DEP
# import theano
# import theano.tensor as T
import numpy as _np

# --DEP-- import pandas as _pn
import copy
from gempy.Visualization import PlotData2D, steano3D, vtkVisualization
#from gempy.visualization_vtk import vtkVisualization

from gempy.DataManagement import InputData, InterpolatorInput
from IPython.core.debugger import Tracer
# from .Topology import Topology


def data_to_pickle(geo_data, path=False):

    geo_data.data_to_pickle(path)


def read_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        return data


def get_series(geo_gata):
    """

    Args:
        geo_gata:

    Returns:

    """
    return geo_gata.series


def get_grid(geo_data):
    return geo_data.grid.grid


def get_resolution(geo_data):
    return geo_data.resolution


def get_extent(geo_data):
    return geo_data.extent

def get_raw_data(geo_data, dtype='all'):
    return geo_data.get_raw_data(itype=dtype)


def create_data(extent, resolution=[50, 50, 50], **kwargs):
    """
    Method to initialize the class data. Calling this function some of the data has to be provided (TODO give to
    everything a default).

    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_f: Path to the data bases of foliations. Default os.getcwd()

    Returns:
        GeMpy.DataManagement: Object that encapsulate all raw data of the project


        dep: self.Plot(GeMpy_core.PlotData): Object to visualize data and results
    """

    return InputData(extent, resolution, **kwargs)


def i_set_data(geo_data, dtype="foliations", action="Open"):

    if action == 'Close':
        geo_data.i_close_set_data()

    if action == 'Open':
        geo_data.i_open_set_data(itype=dtype)


def select_series(geo_data, series):
    """
    Return the formations of a given serie in string
    :param series: list of int or list of str
    :return: formations of a given serie in string separeted by |
    """
    new_geo_data = copy.deepcopy(geo_data)

    if type(series) == int or type(series[0]) == int:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['order_series'].isin(series)]
        new_geo_data.foliations = geo_data.foliations[geo_data.foliations['order_series'].isin(series)]
    elif type(series[0]) == str:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['series'].isin(series)]
        new_geo_data.foliations = geo_data.foliations[geo_data.foliations['series'].isin(series)]
    return new_geo_data


def set_data_series(geo_data, series_distribution=None, order_series=None,
                        update_p_field=True, verbose=0):

    geo_data.set_series(series_distribution=series_distribution, order=order_series)
    try:
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass

    if verbose > 0:
        return get_series(geo_data)


def set_interfaces(geo_data, interf_Dataframe, append=False, update_p_field=True):
    """
     Method to change or append a Dataframe to interfaces in place.
     Args:
         interf_Dataframe: pandas.core.frame.DataFrame with the data
         append: Bool: if you want to append the new data frame or substitute it
     """
    geo_data.set_interfaces(interf_Dataframe, append=append)
    # To update the interpolator parameters without calling a new object
    try:
        geo_data.interpolator._data = geo_data
        geo_data.interpolator._grid = geo_data.grid
       # geo_data.interpolator._set_constant_parameteres(geo_data, geo_data.interpolator._grid)
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass


def set_foliations(geo_data, foliat_Dataframe, append=False, update_p_field=True):
    geo_data.set_foliations(foliat_Dataframe, append=append)
    # To update the interpolator parameters without calling a new object
    try:
        geo_data.interpolator._data = geo_data
        geo_data.interpolator._grid = geo_data.grid
      #  geo_data.interpolator._set_constant_parameteres(geo_data, geo_data.interpolator._grid)
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass

#DEP?
def set_grid(geo_data, new_grid=None, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
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
        geo_data.grid.grid = new_grid
    else:
        if not extent:
            extent = geo_data.extent
        if not resolution:
            resolution = geo_data.resolution

        geo_data.grid = geo_data.GridClass(extent, resolution, grid_type=grid_type, **kwargs)


def plot_data(geo_data, direction="y", series="all", **kwargs):
    plot = PlotData2D(geo_data)
    plot.plot_data(direction=direction, series=series, **kwargs)
    # TODO saving options
    return plot


def plot_section(geo_data, block, cell_number, direction="y", **kwargs):
    plot = PlotData2D(geo_data)
    plot.plot_block_section(cell_number, block=block, direction=direction, **kwargs)
    # TODO saving options
    return plot


def plot_potential_field(geo_data, potential_field, cell_number, n_pf=0,
                         direction="y", plot_data=True, series="all", *args, **kwargs):

    plot = PlotData2D(geo_data)
    plot.plot_potential_field(potential_field, cell_number, n_pf=n_pf,
                              direction=direction,  plot_data=plot_data, series=series,
                              *args, **kwargs)

def plot_data_3D(geo_data):
    r, i = visualize(geo_data)
    del r, i
    return None


def set_interpolation_data(geo_data, **kwargs):
    in_data = InterpolatorInput(geo_data, **kwargs)
    return in_data

# =====================================
# Functions for the InterpolatorData
# =====================================
# TODO check that is a interp_data object and if not try to create within the function one from the geo_data


def get_kriging_parameters(interp_data, verbose=0):
    return interp_data.interpolator.get_kriging_parameters(verbose=verbose)


def get_th_fn(interp_data, dtype=None, u_grade=None, **kwargs):
    """

    Args:
        geo_data:
        **kwargs:

    Returns:

    """


    # DEP?
    # Choosing float precision for the computation

    # if not dtype:
    #     if theano.config.device == 'gpu':
    #         dtype = 'float32'
    #     else:
    #         print('making float 64')
    #         dtype = 'float64'
    #
    # # We make a rescaled version of geo_data for stability reasons
    # data_interp = set_interpolator(geo_data, dtype=dtype)
    #
    # # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
    # input_data_T = data_interp.interpolator.tg.input_parameters_list()
    #
    # # This prepares the user data to the theano function
    # #input_data_P = data_interp.interpolator.data_prep(u_grade=u_grade)
    #
    # # then we compile we have to pass the number of formations that are faults!!
    # th_fn = theano.function(input_data_T, data_interp.interpolator.tg.whole_block_model(data_interp.n_faults),
    #                         on_unused_input='ignore',
    #                         allow_input_downcast=True,
    #                         profile=False)
    return interp_data.compile_th_fn(u_grade=u_grade, dtype=dtype, **kwargs)


def compute_model(interp_data, u_grade=None, get_potential_at_interfaces=False):
    if not getattr(interp_data, 'th_fn', None):
        interp_data.compile_th_fn()

    i = interp_data.get_input_data(u_grade=u_grade)
    sol, interp_data.potential_at_interfaces = interp_data.th_fn(*i)
    if get_potential_at_interfaces:
        return _np.squeeze(sol), interp_data.potential_at_interfaces
    else:
        return _np.squeeze(sol)


def get_surface(potential_block, interp_data, n_formation, step_size=1, original_scale=True):
    assert getattr(interp_data, 'potential_at_interfaces', None).any(), 'You need to compute the model first'
    assert n_formation > 0, 'Number of the formation has tobe positive'
    # In case the values are separated by series I put all in a vector
    pot_int = interp_data.potential_at_interfaces.sum(axis=0)

    from skimage import measure

    vertices, simplices, normals, values = measure.marching_cubes_lewiner(
        potential_block.reshape(interp_data.resolution[0],
                                interp_data.resolution[1],
                                interp_data.resolution[2]),
        pot_int[n_formation-1],
        step_size=step_size,
        spacing=((interp_data.extent[1] - interp_data.extent[0]) / interp_data.resolution[0],
                 (interp_data.extent[3] - interp_data.extent[2]) / interp_data.resolution[1],
                 (interp_data.extent[5] - interp_data.extent[4]) / interp_data.resolution[2]))

    if original_scale:
        vertices = interp_data.rescaling_factor * vertices + _np.array([0,
                                                                         0,
                                                                         -2000]).reshape(1, 3)

    return vertices, simplices
