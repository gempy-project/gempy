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
Tested on Ubuntu 14

Created on 10/10 /2016

@author: Miguel de la Varga

"""
import os
from os import path
import sys

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as _np
import pandas as _pn

import copy
from gempy.visualization import PlotData2D, steno3D, vtkVisualization
from gempy.data_management import InputData, GridClass
from gempy.interpolator import InterpolatorData
from gempy.sequential_pile import StratigraphicPile
from gempy.topology import topology_analyze as _topology_analyze
from gempy.utils.geomodeller_integration import ReadGeoModellerXML as _ReadGeoModellerXML
import gempy.posterior_analysis as pa # So far we use this type of import because the other one makes a copy and blows up some asserts


def compute_model(interp_data, output='geology', u_grade=None, get_potential_at_interfaces=False):
    """
    Computes the geological model.

    Args:
        interp_data (gempy.DataManagement.InterpolatorInput): Rescaled data.
        u_grade (list): grade of the polynomial for the universal part of the Kriging interpolations. The value has to
        be either 0, 3 or 9 (number of equations) and the length has to be the number of series. By default the value
        depends on the number of points given as input to try to avoid singular matrix. NOTE: if during the computation
        of the model a singular matrix is returned try to reduce the u_grade of the series.
        get_potential_at_interfaces (bool): Get potential at interfaces

    Returns:
        numpy.array: if compute_all was chosen in gempy.DataManagement.InterpolatorInput, the first
        row will be the lithology block model, the second the potential field and the third the fault
        network block. if compute_all was False only the lithology block model will be computed. In
        addition if get_potential_at_interfaces is True, the value of the potential field at each of
        the interfaces is given as well
    """
    if not getattr(interp_data, 'th_fn', None):
        interp_data.th_fn = interp_data.compile_th_fn(output=output)

    i = interp_data.get_input_data(u_grade=u_grade)

    if output is 'geology':
        lith_matrix, fault_matrix, potential_at_interfaces = interp_data.th_fn(*i)
        # TODO check if this is necessary yet
        if len(lith_matrix.shape) < 3:
            _np.expand_dims(lith_matrix, 0)
            _np.expand_dims(fault_matrix, 0)

        interp_data.potential_at_interfaces = potential_at_interfaces

        if get_potential_at_interfaces:
            return lith_matrix, fault_matrix, interp_data.potential_at_interfaces
        else:
            return lith_matrix, fault_matrix

    # TODO this should be a flag read from the compilation I guess
    if output is 'gravity':
        # TODO make asserts
        lith_matrix, fault_matrix, potential_at_interfaces, grav = interp_data.th_fn(*i)
        if len(lith_matrix.shape) < 3:
            _np.expand_dims(lith_matrix, 0)
            _np.expand_dims(fault_matrix, 0)

        interp_data.potential_at_interfaces = potential_at_interfaces

        if get_potential_at_interfaces:
            return lith_matrix, fault_matrix, grav, interp_data.potential_at_interfaces
        else:
            return lith_matrix, fault_matrix, grav


def create_data(extent, resolution=(50, 50, 50), **kwargs):
    """
    Method to create a InputData object. It is analogous to gempy.InputData()

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
        GeMpy.DataManagement: Object that encapsulate all raw data of the project


        dep: self.Plot(GeMpy_core.PlotData): Object to visualize data and results
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

    # interf = gmx.get_interfaces_df()
    # orient = gmx.get_orientation_df()
    # if interf is None or orient is None:
    #     raise ValueError("No 3D data stored in given XML file, can't extract interfaces or orientations.")
    # else:
    #     geo_data.interfaces = interf
    #     geo_data.orientations = orient
    #
    # # this seems to fuck things up so far:
    # for f in gmx.faults:
    #     gmx.series_distribution[f] = f
    #
    # try:  # try to set series ordering and stuff, if fails (because the xml files are fucked up) tell user to do it manually
    #     set_series(geo_data, gmx.series_distribution,
    #                order_series=list(gmx.faults + gmx.stratigraphic_column),
    #                order_formations=gmx.get_order_formations())
    # except AssertionError:
    #     print("Some of the formations given are not in the formations data frame. Therefore, you have to set the series"
    #           " order manually.")

    if return_xml:
        return geo_data, gmx
    else:
        return geo_data


def data_to_pickle(geo_data, path=False):
    """
     Save InputData object to a python pickle (serialization of python). Be aware that if the dependencies
     versions used to export and import the pickle differ it may give problems

     Args:
         path (str): path where save the pickle

     Returns:
         None
     """
    geo_data.data_to_pickle(path)


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
    if lith_block is not None:
        vtkVisualization.export_vtk_lith_block(geo_data, lith_block, path=path+str('v'))
    if vertices is not None and simplices is not None:
        vtkVisualization.export_vtk_surfaces(vertices, simplices, path=path+str('s'), name=name)


def get_series(geo_gata):
    """
    Args:
         geo_data (gempy.DataManagement.InputData object)

    Returns:
        Pandas.DataFrame: Return series and formations relations
    """
    return geo_gata.series


def get_grid(geo_data):
    """
     Args:
          geo_data (gempy.DataManagement.InputData object)

     Returns:
         numpy.array: Return series and formations relations
     """
    return geo_data.grid.values


def get_resolution(geo_data):
    return geo_data.resolution


def get_extent(geo_data):
    return geo_data.extent


def get_data(geo_data, dtype='all', numeric=False, verbosity=0):
    """
    Method that returns the interfaces and orientations pandas Dataframes. Can return both at the same time or only
    one of the two

    Args:
        dtype(str): input data type, either 'orientations', 'interfaces' or 'all' for both.
        verbosity (int): Number of properties shown

    Returns:
        pandas.core.frame.DataFrame: Data frame with the raw data

    """
    return geo_data.get_data(itype=dtype, numeric=numeric, verbosity=verbosity)


def get_sequential_pile(geo_data):
    """
    Visualize an interactive stratigraphic pile to move around the formations and the series. IMPORTANT NOTE:
    To have the interactive properties it is necessary the use of qt as interactive backend. (In notebook use:
    %matplotlib qt5)

    Args:
        geo_data: gempy.DataManagement.InputData object

    Returns:
        interactive Matplotlib figure
    """
    return StratigraphicPile(geo_data)


def get_kriging_parameters(interp_data, verbose=0):
    """
    Print the kringing parameters

    Args:
        interp_data (gempy.DataManagement.InterpolatorInput)
        verbose (int): if > 0 print all the shape values as well.

    Returns:
        None
    """
    return interp_data.interpolator.get_kriging_parameters(verbose=verbose)

# =====================================
# Functions for the InterpolatorData
# =====================================
# TODO check that is a interp_data object and if not try to create within the function one from the geo_data


def get_th_fn(interp_data, compute_all=True):
    """
    Get theano function

    Args:
        interp_data (gempy.DataManagement.InterpolatorInput): Rescaled data.
         compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
         the block model of faults. If False only return the block model of lithologies. This may be important to speed
          up the computation. Default True

    Returns:
        theano.function: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
    """

    return interp_data.compile_th_fn(compute_all=compute_all)


def get_surfaces(interp_data, potential_lith=None, potential_fault=None, n_formation='all', step_size=1, original_scale=True):
    """
    compute vertices and simplices of the interfaces for its vtk visualization or further
    analysis

    Args:
        potential_lith (numpy.array): 1D numpy array with the solution of the computation of the model
         containing the scalar field of potentials (second row of solution)
        interp_data (gempy.DataManagement.InterpolatorInput): Interpolator object.
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
        assert n_formation > 0, 'Number of the formation has to be positive'

        # In case the values are separated by series I put all in a vector
        pot_int = interp_data.potential_at_interfaces.sum(axis=0)

        from skimage import measure

        if not potential_block.max() > pot_int[n_formation-1]:
            pot_int[n_formation - 1] = potential_block.max()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        if not potential_block.min() < pot_int[n_formation - 1]:
            pot_int[n_formation - 1] = potential_block.min()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        vertices_p, simplices_p, normals, values = measure.marching_cubes_lewiner(
            potential_block.reshape(interp_data.geo_data_res.resolution[0],
                                    interp_data.geo_data_res.resolution[1],
                                    interp_data.geo_data_res.resolution[2]),
            pot_int[n_formation-1],
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

    if potential_fault is not None:

        assert len(_np.atleast_2d(potential_fault)) is interp_data.geo_data_res.n_faults, 'You need to pass a potential field per fault'

        pot_int = interp_data.potential_at_interfaces[:interp_data.geo_data_res.n_faults + 1]
        for n in interp_data.geo_data_res.interfaces['formation number'][
            interp_data.geo_data_res.interfaces['isFault']].unique():
            if n == 0:
                continue
            else:
                v, s = get_surface(_np.atleast_2d(potential_fault)[n-1], interp_data, pot_int, n,
                                   step_size=step_size, original_scale=original_scale)
                vertices.append(v)
                simplices.append(s)

    if potential_lith is not None:
        pot_int = interp_data.potential_at_interfaces[interp_data.geo_data_res.n_faults:]

        # Compute the vertices of the lithologies
        if n_formation == 'all':

            for n in interp_data.geo_data_res.interfaces['formation number'][~interp_data.geo_data_res.interfaces['isFault']].unique():
                if n == 0:
                    continue
                else:
                    v, s = get_surface(potential_lith, interp_data, pot_int, n,
                                       step_size=step_size, original_scale=original_scale)
                    vertices.append(v)
                    simplices.append(s)
        else:
            vertices, simplices = get_surface(potential_lith, interp_data, pot_int, n_formation,
                                              step_size=step_size, original_scale=original_scale)

    return vertices, simplices


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
        formation_numbers_l (list): Formation numbers (int)
        alpha (float): Opacity
        plot_data (bool): Default True
        size (tuple): Resolution of the window
        fullscreen (bool): Launch window in full screen or not

    Returns:
        None
    """
    w = vtkVisualization(geo_data, bg_color=bg_color)
    w.set_surfaces(vertices_l, simplices_l,
                   #formations_names_l, formation_numbers_l,
                    alpha)

    if plot_data:
        w.set_interfaces()
        w.set_orientations()
    w.render_model(size=size, fullscreen=fullscreen)
    return w


def set_orientation_from_interfaces(geo_data, indices_array, verbose=0):

    if _np.ndim(indices_array) is 1:
        indices = indices_array
        form = geo_data.interfaces['formation'].iloc[indices].unique()
        assert form.shape[0] is 1, 'The interface points must belong to the same formation'

        ori_parameters = geo_data.create_orientation_from_interfaces(indices)
        geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                 dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                 G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                 formation=form)
    elif _np.ndim(indices_array) is 2:
        for indices in indices_array:
            form = geo_data.interfaces['formation'].iloc[indices].unique()
            assert form.shape[0] is 1, 'The interface points must belong to the same formation'

            ori_parameters = geo_data.create_orientation_from_interfaces(indices)
            geo_data.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                     dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                     G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                     formation=form[0])
    if verbose:
        get_data()


def plot_data(geo_data, direction="y", data_type = 'all', series="all", legend_font_size=6, **kwargs):
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
    plot = PlotData2D(geo_data)
    plot.plot_data(direction=direction, data_type=data_type, series=series, legend_font_size=legend_font_size, **kwargs)
    # TODO saving options


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
    plot = PlotData2D(geo_data)
    plot.plot_block_section(cell_number, block=block, direction=direction, **kwargs)
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
    plot = PlotData2D(geo_data)
    plot.plot_scalar_field(potential_field, cell_number, N=N,
                              direction=direction,  plot_data=plot_data, series=series,
                              *args, **kwargs)

def plot_data_3D(geo_data):
    """
    Plot in vtk all the input data of a model
    Args:
        geo_data (gempy.DataManagement.InputData): Input data of the model

    Returns:
        None
    """
    vv = vtkVisualization(geo_data)
    vv.set_interfaces()
    vv.set_orientations()
    vv.render_model()
    return None


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
        formation_numbers_l (list): Formation numbers (int)
        alpha (float): Opacity
        plot_data (bool): Default True
        size (tuple): Resolution of the window
        fullscreen (bool): Launch window in full screen or not

    Returns:
        None
    """
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
        gempy.DataManagement.InputData: Input Data object
    """
    import pickle
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        return data


def rescale_factor_default(geo_data):
    """
    Gives the default rescaling factor for a given geo_data

    Args:
        geo_data: Original gempy.DataManagement.InputData object

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
    Rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.

    Args:
        geo_data: Original gempy.DataManagement.InputData object
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
        series: list of int or list of str

    Returns:
         formations of a given serie in string separeted by |
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
    new_geo_data.series = new_geo_data.series[new_geo_data.interfaces['series'].unique()]
    new_geo_data.set_formation_number()
    return new_geo_data


def set_series(geo_data, series_distribution=None, order_series=None, order_formations=None,
               verbose=0):
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
    geo_data.set_series(series_distribution=series_distribution, order=order_series)
    geo_data.order_table()
    if order_formations is not None:
        geo_data.set_formation_number(order_formations)

    if verbose > 0:
         return get_sequential_pile(geo_data)
    else:
        return None


def set_order_formations(geo_data, order_formations):
    geo_data.set_formation_number(order_formations)


def set_interfaces(geo_data, interf_Dataframe, append=False):
    """
     Method to change or append a Dataframe to interfaces in place.

     Args:
         interf_Dataframe: pandas.core.frame.DataFrame with the data
         append: Bool: if you want to append the new data frame or substitute it
     """
    geo_data.set_interfaces(interf_Dataframe, append=append)


def set_orientations(geo_data, orient_Dataframe, append=False):
    """
    Method to change or append a Dataframe to orientations in place.  A equivalent Pandas Dataframe with
    ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

    Args:
        interf_Dataframe: pandas.core.frame.DataFrame with the data
        append: Bool: if you want to append the new data frame or substitute it
    """

    geo_data.set_orientations(orient_Dataframe, append=append)


# TODO:
def set_grid(geo_data, grid):
    assert isinstance(grid, GridClass)
    geo_data.grid = grid
    geo_data.extent = grid._grid_ext
    geo_data.resolution = grid._grid_res


def set_interpolation_data(geo_data, **kwargs):
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
    in_data = InterpolatorData(geo_data, **kwargs)
    return in_data


def set_geophysics_obj(interp_data, ai_extent, ai_resolution, ai_z=None, range_max=None):

    assert isinstance(interp_data, InterpolatorData), 'The object has to be instance of the InterpolatorInput'
    interp_data.create_geophysics_obj(ai_extent, ai_resolution, ai_z=ai_z, range_max=range_max)
    return interp_data.geophy


# =====================================
# Functions for Geophysics
# =====================================
def set_densities(interp_data, densities):

    interp_data.interpolator.set_densities(densities)


def topology_compute(geo_data, lith_block, fault_block,
                     cell_number=None, direction=None,
                     compute_areas=False, return_label_block=False):
    """
    Computes model topology and returns graph, centroids and look-up-tables.

    Args:
        geo_data (gempy.data_management.InputData): GemPy's data object for the model.
        lith_block (np.ndarray): Lithology block model.
        fault_block (np.ndarray): Fault block model.
    Keyword Args:
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
    PlotData2D.plot_topo_g(geo_data, G, centroids, direction=direction)
