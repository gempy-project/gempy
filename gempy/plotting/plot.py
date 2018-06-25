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

    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/04/2018

    @author: Miguel de la Varga
"""

from os import path
import sys

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as _np
import pandas as _pn

import copy
from .visualization import PlotData2D, steno3D, vtkVisualization
import gempy as _gempy


class vtkPlot():
    def __init__(self, geo_data, alpha=1,
                     size=(1920, 1080), fullscreen=False, bg_color=None, verbose=0):

        self.geo_data = geo_data

        self.verbose = verbose
        self.alpha = alpha
        self.size = size
        self.fullscreen = fullscreen
        self.bg_color = bg_color

        self.vv = vtkVisualization(self.geo_data, bg_color=self.bg_color)

    #TODO
    def get_original_geo_data(self):
        return self._original_df

    def resume(self):
        self.vv.interactor.Start()

    def close(self):
        self.vv.close_window()

    def restart(self):
        try:
            self.vv.close_window()
        except AttributeError:
            pass

        self.vv = vtkVisualization(self.geo_data, bg_color=self.bg_color)

    def set_geo_data(self, geo_data):
        self.geo_data = geo_data

    def plot_surfaces_3D(self, vertices_l, simplices_l,
                     #formations_names_l, formation_numbers_l,
                      plot_data=True,
                     **kwargs):
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
        self.restart()

        self.vv.set_surfaces(vertices_l, simplices_l,
                   #formations_names_l, formation_numbers_l,
                    self.alpha)

        if plot_data:
            self.vv.set_interfaces()
            self.vv.set_orientations()

        self.vv.render_model(**kwargs)

    def plot_data_3D(self, **kwargs):
        """
        Plot in vtk all the input data of a model
        Args:
            geo_data (gempy.DataManagement.InputData): Input data of the model

        Returns:
            None
        """
        self.restart()
        self.vv.set_interfaces()
        self.vv.set_orientations()
        self.vv.render_model(**kwargs)

    def set_real_time_on(self, interp_data):

        #self.geo_data = interp_data.geo_data_res
        self.restart()
        #self.vv.geo_data = interp_data.geo_data_res
        self.vv.interp_data = interp_data
        self.vv.real_time = True

    def plot_surfaces_3D_real_time(self, interp_data, vertices_l=None, simplices_l=None,
                                   # formations_names_l, formation_numbers_l,
                                    plot_data=True, posterior=None, samples=None,
                                   **kwargs):
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
        assert isinstance(interp_data, _gempy.InterpolatorData), 'The object has to be instance of the InterpolatorInput'
      #  self.set_real_time_on(interp_data)

       # assert _np.max(vertices_l[0]) < 1.5, 'Real time plot only works with rescaled data. Change the original scale flag' \
       #                                      'in get surfaces to False'

        # self.interp_data = interp_data
        # self.geo_data = interp_data.geo_data_res
        # self.real_time = True
        self.set_real_time_on(interp_data)

        if vertices_l and simplices_l:
            self.vv.set_surfaces(vertices_l, simplices_l,
                           # formations_names_l, formation_numbers_l,
                           self.alpha)

        if posterior is not None:
            assert isinstance(posterior, _gempy.posterior_analysis.Posterior), 'The object has to be instance of the Posterior class'
            self.vv.post = posterior
            if samples is not None:
                samp_i = samples[0]
                samp_f = samples[1]
            else:
                samp_i = 0
                samp_f = posterior.n_iter

            self.vv.create_slider_rep(samp_i, samp_f, samp_f)

        if plot_data:
            self.vv.set_interfaces()
            self.vv.set_orientations()

        self.vv.render_model(**kwargs)

    def _move_interface(self, new_df):
        if self.verbose > 0:
            print(self.geo_data._columns_i_1, new_df.columns)

        # Check rows tht have changed
        b_i = (new_df[self.geo_data._columns_i_1].sort_index() != _gempy.get_data(
            self.geo_data, itype='interfaces')[self.geo_data._columns_i_1].sort_index()).any(1)

        # Get indices of changed rows
        ind_i = new_df.index[b_i].tolist()
        if self.verbose > 0:
            print('I am in modifing', ind_i)

        # Modify df
        self.geo_data.set_new_df(new_df)

        # Move sphere widget to new position
        self.vv.SphereCallbak_move_changes(ind_i)

    def _move_orientation(self, new_df):

        # Check rows tht have changed
        b_o = (new_df[self.geo_data._columns_o_1].sort_index() != _gempy.get_data(
                self.geo_data, itype='orientations')[self.geo_data._columns_o_1].sort_index()).any(1)

        # Get indices of changed rows
        ind_o = new_df.index[b_o].tolist()

        # Modify df
        self.geo_data.set_new_df(new_df)

        # Move widgets
        self.vv.planesCallback_move_changes(ind_o)

    def _move_interface_orientation(self, new_df):

        # Check rows tht have changed
        b_i = (new_df.xs('interfaces')[self.geo_data._columns_i_1].sort_index() != _gempy.get_data(
            self.geo_data, itype='interfaces')[self.geo_data._columns_i_1].sort_index()).any(1)

        # Get indices of changed rows
        ind_i = new_df.xs('interfaces').index[b_i].tolist()

        # Check rows tht have changed
        b_o = (new_df.xs('orientations')[self.geo_data._columns_o_1].sort_index() != _gempy.get_data(
            self.geo_data, itype='orientations')[self.geo_data._columns_o_1].sort_index()).any(1)

        # Get indices of changed rows
        ind_o = new_df.xs('orientations').index[b_o].tolist()

        # Modify df
        self.geo_data.set_new_df(new_df)

        # Move widgets
        self.vv.SphereCallbak_move_changes(ind_i)
        self.vv.planesCallback_move_changes(ind_o)

    def _delete_interface(self, new_df):

        # Finding deleted indeces
        ind_i = self.geo_data.interfaces.index.values[~_np.in1d(self.geo_data.interfaces.index.values,
                                                                new_df.index.values,
                                                                assume_unique=True)]

        # Deactivating widget
        for i in ind_i:
            self.vv.s_rend_1.loc[i, 'val'].Off()
            self.vv.s_rend_2.loc[i, 'val'].Off()
            self.vv.s_rend_3.loc[i, 'val'].Off()
            self.vv.s_rend_4.loc[i, 'val'].Off()
            self.vv.s_rend_1.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[0])
            self.vv.s_rend_2.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[1])
            self.vv.s_rend_3.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[2])
            self.vv.s_rend_4.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[3])
        # Modify fg
        self.geo_data.set_new_df(new_df)
        if self.verbose > 0:
            print('I am in deleting', ind_i)

    def _delete_orientation(self, new_df):

        # Finding deleted indeces
        ind_o = self.geo_data.orientations.index.values[~_np.in1d(self.geo_data.orientations.index.values,
                                                                new_df.index.values,
                                                                assume_unique=True)]

        # Deactivating widget
        for i in ind_o:
            self.vv.o_rend_1.loc[i, 'val'].Off()
            self.vv.o_rend_2.loc[i, 'val'].Off()
            self.vv.o_rend_3.loc[i, 'val'].Off()
            self.vv.o_rend_4.loc[i, 'val'].Off()
            self.vv.o_rend_1.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[0])
            self.vv.o_rend_2.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[1])
            self.vv.o_rend_3.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[2])
            self.vv.o_rend_4.loc[i, 'val'].SetCurrentRenderer(self.vv.ren_list[3])
        # Modify fg
        self.geo_data.set_new_df(new_df)
        if self.verbose > 0:
            print('I am in deleting o', ind_o)

    def _add_interface_restore(self, new_df):

        # Finding deleted indeces to restore
        ind_i = new_df.index.values[~_np.in1d(new_df.index.values,
                                              self.geo_data.interfaces.index.values,
                                              assume_unique=True)]
        # Reactivating widget
        for i in ind_i:
            self.vv.s_rend_1.loc[i, 'val'].On()
            self.vv.s_rend_2.loc[i, 'val'].On()
            self.vv.s_rend_3.loc[i, 'val'].On()
            self.vv.s_rend_4.loc[i, 'val'].On()


        self.geo_data.set_new_df(new_df.loc[ind_i], append=True)
        if self.verbose > 0:
            print('I am getting back', ind_i)

    def _add_orientation_restore(self, new_df):

        # Finding deleted indeces to restore
        ind_o = new_df.index.values[~_np.in1d(new_df.index.values,
                                              self.geo_data.orientations.index.values,
                                              assume_unique=True)]
        # Reactivating widget
        for i in ind_o:
            self.vv.o_rend_1.loc[i, 'val'].On()
            self.vv.o_rend_2.loc[i, 'val'].On()
            self.vv.o_rend_3.loc[i, 'val'].On()
            self.vv.o_rend_4.loc[i, 'val'].On()

        self.geo_data.set_new_df(new_df.loc[ind_o], append=True)
        if self.verbose > 0:
            print('I am getting back', ind_o)

    def _add_interface_new(self, new_df):

        # Finding the new indices added
        ind_i = new_df.index.values[~_np.in1d(new_df.index.values,
                                              self.geo_data.interfaces.index.values,
                                              assume_unique=True)]

        # Modifing df
        self.geo_data.set_new_df(new_df.loc[ind_i], append=True)

        # Creating new widget
        for i in ind_i:
            self.vv.set_interfaces(indices=i)
        if self.verbose > 0:
            print('I am in adding', ind_i)

    def _add_orientation_new(self, new_df):

        # Finding the new indices added
        ind_o = new_df.index.values[~_np.in1d(new_df.index.values,
                                              self.geo_data.orientations.index.values,
                                              assume_unique=True)]

        # Modifing df
        self.geo_data.set_new_df(new_df.loc[ind_o], append=True)
        if self.verbose > 0:
            print(ind_o)

        # Creating new widget
        for i in ind_o:
            self.vv.set_orientations(indices=i)
        if self.verbose > 0:
            print('I am in adding', ind_o)

    def qgrid_callBack(self, change):
        #  First we remove a column that is added by qgrid with the unfiltered indeces
        new_df = change['new'][change['new'].columns[1:]]

        # Check if we are modifing interfaces and orientations at the same time
        try:
            # Modify mode
            self._move_interface_orientation(new_df)
        except KeyError:
            # Check the itype data
            # ------------
            # Orientations
            if set(self.geo_data._columns_o_1).issubset(new_df.columns):
                # Checking the mode
                # ++++++++++
                # Delete mode
                if new_df.index.shape[0] < self.geo_data.orientations.index.shape[0]:
                    self._delete_orientation(new_df)
                # +++++++++++
                # Adding mode
                elif new_df.index.shape[0] > self.geo_data.orientations.index.shape[0]:
                    # Checking if is new point or a filter
                    # ===========
                    # Filter mode
                    if set(new_df.index).issubset(self._original_df.index):

                        self._add_orientation_restore(new_df)

                    # Adding new data mode
                    else:
                        self._add_orientation_new(new_df)
                # +++++++++++
                # Modify mode
                elif new_df.index.shape[0] == self.geo_data.orientations.index.shape[0]:  # Modify
                    self._move_orientation(new_df)

                else:
                    print('something went wrong')

                    # ----------
            # Interfaces
            elif set(self.geo_data._columns_i_1).issubset(new_df.columns):
                if self.verbose > 0:
                    print(new_df.index.shape[0])

                # Checking the mode
                # ++++++++++
                # Delete mode
                if new_df.index.shape[0] < self.geo_data.interfaces.index.shape[0]:
                    self._delete_interface(new_df)

                # +++++++++++
                # Adding mode
                elif new_df.index.shape[0] > self.geo_data.interfaces.index.shape[0]:  # Add mode

                    # print(set(new_df.index).issubset(self._original_df.index))

                    # Checking if is new point or a filter
                    # ===========
                    # Filter mode
                    if set(new_df.index).issubset(self._original_df.index):

                        self._add_interface_restore(new_df)

                    # Adding new data mode
                    else:
                        self._add_interface_new(new_df)

                # +++++++++++
                # Modify mode
                elif new_df.index.shape[0] == self.geo_data.interfaces.index.shape[0]:  # Modify
                    self._move_interface(new_df)

                else:
                    print('something went wrong')

        if self.vv.real_time:
            try:
                for surf in self.vv.surf_rend_1:
                    self.vv.ren_list[0].RemoveActor(surf)
                    self.vv.ren_list[1].RemoveActor(surf)
                    self.vv.ren_list[2].RemoveActor(surf)
                    self.vv.ren_list[3].RemoveActor(surf)
            except AttributeError:
                pass

            try:
                self.vv.geo_data.order_table()
                vertices, simpleces = self.vv.update_surfaces_real_time(self.vv.geo_data)
                self.vv.set_surfaces(vertices, simpleces)
            except AssertionError:
                print('Not enough data to compute the model')
            except NotImplementedError:
                print('If the theano graph expects faults and/or lithologies you need to pass at least one'
                      ' interface for each of them')
        #self.vv.interp_data.update_interpolator(self.geo_data)



        self.vv.interactor.Render()

    def qgrid_callBack_fr(self, change):
        new_df = change['new'][change['new'].columns[1:]]
        self.geo_data.faults_relations = new_df

        if self.vv.real_time:
            try:
                for surf in self.vv.surf_rend_1:
                    self.vv.ren_list[0].RemoveActor(surf)
                    self.vv.ren_list[1].RemoveActor(surf)
                    self.vv.ren_list[2].RemoveActor(surf)
                    self.vv.ren_list[3].RemoveActor(surf)
            except AttributeError:
                pass

            try:
                vertices, simpleces = self.vv.update_surfaces_real_time(self.vv.geo_data)
                self.vv.set_surfaces(vertices, simpleces)
            except AssertionError:
                print('Not enough data to compute the model')
            except NotImplementedError:
                print('If the theano graph expects faults and/or lithologies you need to pass at least one'
                      ' interface for each of them')

        self.vv.interactor.Render()

    def observe_df(self, geo_data = None, itype='all'):
        if not geo_data:
            geo_data = self.geo_data

        self._original_df = copy.deepcopy(_gempy.get_data(geo_data, itype=itype))
        qgrid_widget = geo_data.interactive_df_open(itype=itype)
        if itype is 'faults_relations':
            qgrid_widget.observe(self.qgrid_callBack_fr, names=['_df'])

        else:
            qgrid_widget.observe(self.qgrid_callBack, names=['_df'])

        return qgrid_widget


def plot_data_3D(geo_data, **kwargs):
    """
    Plot in vtk all the input data of a model
    Args:
        geo_data (gempy.DataManagement.InputData): Input data of the model

    Returns:
        None
    """
    vv = vtkPlot(geo_data, **kwargs)
    vv.plot_data_3D(**kwargs)
    # vv = vtkVisualization(geo_data)
    # vv.set_interfaces()
    # vv.set_orientations()
    # vv.render_model(**kwargs)
    #
    # vv.close_window()

    return vv


def plot_surfaces_3D_real_time(geo_data, interp_data, vertices_l, simplices_l,
                              plot_data=True, posterior=None, samples=None, **kwargs):
                     #formations_names_l, formation_numbers_l,
                   #  ,
                   #  size=(1920, 1080), fullscreen=False):
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

    assert isinstance(interp_data, _gempy.InterpolatorData), 'The object has to be instance of the InterpolatorInput'
    vv = vtkPlot(geo_data, **kwargs)
    vv.plot_surfaces_3D_real_time(interp_data, vertices_l, simplices_l, plot_data=plot_data, posterior=posterior,
                                  samples=samples, **kwargs)

    return vv


def plot_surfaces_3D(geo_data, vertices_l, simplices_l,
                     # formations_names_l, formation_numbers_l,
                     alpha=1, plot_data=True,
                     #size=(1920, 1080), fullscreen=False, bg_color=None
                     **kwargs):
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
    vv =  vtkPlot(geo_data, **kwargs)

    vv.plot_surfaces_3D( vertices_l, simplices_l,
                     #formations_names_l, formation_numbers_l,
                      plot_data=plot_data)
    # w = vtkVisualization(geo_data, bg_color=bg_color)
    # w.set_surfaces(vertices_l, simplices_l,
    #                # formations_names_l, formation_numbers_l,
    #                alpha)
    #
    # if plot_data:
    #     w.set_interfaces()
    #     w.set_orientations()
    # w.render_model(size=size, fullscreen=fullscreen)
    #
    # w.close_window()

    return vv


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
        vtkVisualization.export_vtk_lith_block(geo_data, lith_block, path=path)
    if vertices is not None and simplices is not None:
        vtkVisualization.export_vtk_surfaces(vertices, simplices, path=path, name=name)


def plot_data(geo_data, direction="y", data_type = 'all', series="all", legend_font_size=6, **kwargs):
    """
    Plot the projection of the raw data (interfaces and orientations) in 2D following a
    specific directions

    Args:
        direction(str): xyz. Caartesian direction to be plotted
        series(str): series to plot
        ve(float): Vertical exageration
        **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

    Returns:
        None
    """
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
        ve(float): Vertical exageration
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

