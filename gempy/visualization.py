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


Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016

@author: Miguel de la Varga
"""

import warnings
try:
    import vtk
except ImportError:
    warnings.warn('Vtk package is not installed. No vtk visualization available.')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from os import path
import sys
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from IPython.core.debugger import Pdb
from gempy.colors import color_lot, cmap, norm
import gempy as gp
import copy

sns.set_context('talk')
plt.style.use(['seaborn-white', 'seaborn-talk'])


class PlotData2D(object):
    """
    Class to make the different plot related with gempy

    Args:
        geo_data(gempy.InputData): All values of a DataManagement object
        block(numpy.array): 3D array containing the lithology block
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        scalar_field(numpy.ndarray): 3D array containing a individual potential field
        verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
    """

    def __init__(self, geo_data, color_lot=color_lot, cmap=cmap, norm=norm, **kwargs):

        self._data = geo_data
        self._color_lot = color_lot
        self._cmap = cmap
        self._norm = norm
        self.formation_names = self._data.interfaces['formation'].unique()
        self.formation_numbers = self._data.interfaces['formation number'].unique()

        # DEP?
        # if 'scalar_field' in kwargs:
        #     self._scalar_field_p = kwargs['scalar_field']

        self._set_style()

    def _set_style(self):
        """
        Private function to set some plotting options

        """

        plt.style.use(['seaborn-white', 'seaborn-talk'])
        sns.set_context("talk")

    def plot_data(self, direction="y", data_type='all', series="all", legend_font_size=10, **kwargs):
        """
        Plot the projecton of the raw data (interfaces and orientations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            data_type (str): type of data to plot. 'all', 'interfaces' or 'orientations'
            series(str): series to plot
            **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

        Returns:
            Data plot

        """
        if 'scatter_kws' not in kwargs:
            kwargs['scatter_kws'] = {"marker": "D",
                                     "s": 100,
                                     "edgecolors": "black",
                                     "linewidths": 1}

        x, y, Gx, Gy = self._slice(direction)[4:]
        extent = self._slice(direction)[3]
        aspect = (extent[1] - extent[0])/(extent[3] - extent[2])
        if aspect < 1:
            min_axis = 'width'
        else:
            min_axis = 'height'


        if series == "all":
            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"].
                isin(self._data.series.columns.values)]
            series_to_plot_f = self._data.orientations[self._data.orientations["series"].
                isin(self._data.series.columns.values)]

        else:

            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"] == series]
            series_to_plot_f = self._data.orientations[self._data.orientations["series"] == series]

        # Change dictionary keys numbers for formation names
        for i in zip(self.formation_names, self.formation_numbers):
            self._color_lot[i[0]] = self._color_lot[i[1]]

        if data_type == 'all':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="formation",
                           #scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette=self._color_lot,
                           **kwargs)

            # Plotting orientations
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=10)

        if data_type == 'interfaces':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="formation",
                           #scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette=self._color_lot,
                           **kwargs)

        if data_type == 'orientations':
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=15)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # plt.xlim(extent[0] - extent[0]*0.05, extent[1] + extent[1]*0.05)
        # plt.ylim(extent[2] - extent[2]*0.05, extent[3] + extent[3]*0.05)
        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plotting functions

        """
        _a, _b, _c = (slice(0, self._data.resolution[0]),
                      slice(0, self._data.resolution[1]),
                      slice(0, self._data.resolution[2]))
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self._data.extent[2], self._data.extent[3], self._data.extent[4], self._data.extent[5]
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self._data.extent[0], self._data.extent[1], self._data.extent[4], self._data.extent[5]
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self._data.extent[0], self._data.extent[1], self._data.extent[2], self._data.extent[3]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def plot_block_section(self, cell_number=13, block=None, direction="y", interpolation='none',
                           plot_data=False, **kwargs):
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
            Block plot
        """
        if block is not None:
            import theano
            import numpy
            assert (type(block) is theano.tensor.sharedvar.TensorSharedVariable or
                    type(block) is numpy.ndarray), \
                'Block has to be a theano shared object or numpy array.'
            if type(block) is numpy.ndarray:
                _block = block
            else:
                _block = block.get_value()
        else:
            try:
                _block = self._data.interpolator.tg.final_block.get_value()
            except AttributeError:
                raise AttributeError('There is no block to plot')

        plot_block = _block.reshape(self._data.resolution[0], self._data.resolution[1], self._data.resolution[2])
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        if plot_data:
            self.plot_data(direction, 'all')
        # TODO: plot_topo option - need fault_block for that

        if 'cmap' not in kwargs:
            kwargs['cmap'] = self._cmap
        if 'norm' not in kwargs:
            kwargs['norm'] = self._norm

        im = plt.imshow(plot_block[_a, _b, _c].T, origin="bottom",
                   extent=extent_val,
                   interpolation=interpolation, **kwargs)

        import matplotlib.patches as mpatches
        colors = [im.cmap(im.norm(value)) for value in self.formation_numbers]
        patches = [ mpatches.Patch(color=colors[i], label=self.formation_names[i]) for i in range(len(self.formation_names))]
        if not plot_data:
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel(x)
        plt.ylabel(y)

    def plot_scalar_field(self, scalar_field, cell_number, N=20,
                             direction="y", plot_data=True, series="all", *args, **kwargs):
        """
        Plot a scalar field in a given direction.

        Args:
            cell_number(int): position of the array to plot
            scalar_field(str): name of the scalar field (or series) to plot
            n_pf(int): number of the  scalar field (or series) to plot
            direction(str): xyz. Caartesian direction to be plotted
            serie: *Deprecated*
            **kwargs: plt.contour kwargs

        Returns:
            scalar field plot
        """

        if plot_data:
            self.plot_data(direction, 'all')

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
        plt.contour(scalar_field.reshape(
            self._data.resolution[0], self._data.resolution[1], self._data.resolution[2])[_a, _b, _c].T,
                    N,
                    extent=extent_val, *args,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.xlabel(x)
        plt.ylabel(y)

    def plot_topo_g(geo_data, G, centroids, direction="y"):
        if direction == "y":
            c1, c2 = (0, 2)
            e1 = geo_data.extent[1] - geo_data.extent[0]
            e2 = geo_data.extent[5] - geo_data.extent[4]
            d1 = geo_data.extent[0]
            d2 = geo_data.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = geo_data.resolution[0]
            r2 = geo_data.resolution[2]
        elif direction == "x":
            c1, c2 = (1, 2)
            e1 = geo_data.extent[3] - geo_data.extent[2]
            e2 = geo_data.extent[5] - geo_data.extent[4]
            d1 = geo_data.extent[2]
            d2 = geo_data.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = geo_data.resolution[1]
            r2 = geo_data.resolution[2]
        elif direction == "z":
            c1, c2 = (0, 1)
            e1 = geo_data.extent[1] - geo_data.extent[0]
            e2 = geo_data.extent[3] - geo_data.extent[2]
            d1 = geo_data.extent[0]
            d2 = geo_data.extent[2]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = geo_data.resolution[0]
            r2 = geo_data.resolution[1]

        for edge in G.edges_iter():
            a, b = edge

            plt.plot(np.array([centroids[a][c1], centroids[b][c1]]) * e1 / r1 + d1,
                          np.array([centroids[a][c2], centroids[b][c2]]) * e2 / r2 + d2, "black", linewidth=0.75)

            for node in G.nodes_iter():
                plt.plot(centroids[node][c1] * e1 / r1 + d1, centroids[node][c2] * e2 / r2 +d2,
                         marker="o", color="black", markersize=10, alpha=0.75)
                plt.text(centroids[node][c1] * e1 / r1 + d1,
                         centroids[node][c2] * e2 / r2 + d2, str(node), color="white", size=6, ha="center", va="center",
                         weight="ultralight", family="monospace")

    # TODO: Incorporate to the class
    @staticmethod
    def annotate_plot(frame, label_col, x, y, **kwargs):
        """
        Annotate the plot of a given DataFrame using one of its columns

        Should be called right after a DataFrame or series plot method,
        before telling matplotlib to show the plot.

        Parameters
        ----------
        frame : pandas.DataFrame

        plot_col : str
            The string identifying the column of frame that was plotted

        label_col : str
            The string identifying the column of frame to be used as label

        kwargs:
            Other key-word args that should be passed to plt.annotate

        Returns
        -------
        None

        Notes
        -----
        After calling this function you should call plt.show() to get the
        results. This function only adds the annotations, it doesn't show
        them.
        """
        import matplotlib.pyplot as plt  # Make sure we have pyplot as plt

        for label, x, y in zip(frame[label_col], frame[x], frame[y]):
            plt.annotate(label, xy=(x + 0.2, y + 0.15), **kwargs)


class steno3D():

    def __init__(self, geo_data):
        self._data = geo_data

    def plot3D_steno(self, block,  project, plot=True, **kwargs):
        import steno3d
        import numpy as np
        steno3d.login()

        description = kwargs.get('description', 'Nothing')
        proj = steno3d.Project(
            title=project,
            description=description,
            public=True,
        )

        mesh = steno3d.Mesh3DGrid(h1=np.ones(self._data.resolution[0]) * (self._data.extent[0] - self._data.extent[1]) /
                                                                         (self._data.resolution[0] - 1),
                                  h2=np.ones(self._data.resolution[1]) * (self._data.extent[2] - self._data.extent[3]) /
                                                                         (self._data.resolution[1] - 1),
                                  h3=np.ones(self._data.resolution[2]) * (self._data.extent[4] - self._data.extent[5]) /
                                                                         (self._data.resolution[2] - 1))

        data = steno3d.DataArray(
            title='Lithologies',
            array=block)

        vol = steno3d.Volume(project=proj, mesh=mesh, data=[dict(location='CC', data=data)])
        vol.upload()

        if plot:
            return vol.plot()


class vtkVisualization:
    """
    Class to visualize data and results in 3D. Init will create all the render properties while the method render
    model will lunch the window. Using set_interfaces, set_orientations and set_surfaces in between can be chosen what
    will be displayed.

    Args:
        geo_data(gempy.InputData): All values of a DataManagement object
        ren_name (str): Name of the renderer window
        verbose (int): Verbosity for certain functions

    Attributes:
        renWin(vtk.vtkRenderWindow())
        camera_list (list): list of cameras for the distinct renderers
        ren_list (list): list containing the vtk renderers
    """
    def __init__(self, geo_data, ren_name='GemPy 3D-Editor', verbose=0, color_lot=color_lot, real_time=False, bg_color=None, ve=1):
        self.ve = ve

        self.real_time = real_time
        self.geo_data = geo_data#copy.deepcopy(geo_data)
        self.interp_data = None
        self.C_LOT = color_lot
        # Number of renders
        self.n_ren = 4
        self.formation_number = geo_data.interfaces['formation number'].unique()
        self.formation_name = geo_data.interfaces['formation'].unique()

        # Extents
        self.extent = self.geo_data.extent
        self.extent[-1] = ve * self.extent[-1]
        self.extent[-2] = ve * self.extent[-2]
        _e = self.geo_data.extent
        self._e_dx = _e[1] - _e[0]
        self._e_dy = _e[3] - _e[2]
        self._e_dz = _e[5] - _e[4]
        self._e_d_avrg = (self._e_dx + self._e_dy + self._e_dz) / 3

        # Resolution
        self.res = self.geo_data.resolution

        # create render window, settings
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName(ren_name)

        # Set 4 renderers. ie 3D, X,Y,Z projections
        self.ren_list = self.create_ren_list()

        # create interactor and set interactor style, assign render window
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renwin)

        # 3d model camera for the 4 renders
        self.camera_list = self._create_cameras(self.extent, verbose=verbose)
        # Setting the camera and the background color to the renders
        self.set_camera_backcolor(color=bg_color)

        # Creating the axis
        for e, r in enumerate(self.ren_list):
            # add axes actor to all renderers
            axe = self._create_axes(self.camera_list[e])

            r.AddActor(axe)
            r.ResetCamera()

    def render_model(self, size=(1920, 1080), fullscreen=False):
        """
        Method to launch the window

        Args:
            size (tuple): Resolution of the window
            fullscreen (bool): Launch window in full screen or not
        Returns:

        """
        # initialize and start the app

        if fullscreen:
            self.renwin.FullScreenOn()
        self.renwin.SetSize(size)
        self.interactor.Initialize()
        self.interactor.Start()

        # close_window(interactor)
        del self.renwin, self.interactor

    def create_surface_points(self, vertices):
        """
        Method to create the points that form the surfaces
        Args:
            vertices (numpy.array): 2D array (XYZ) with the coordinates of the points

        Returns:
            vtk.vtkPoints: with the coordinates of the points
        """
        Points = vtk.vtkPoints()
        for v in vertices:
            v[-1] = self.ve * v[-1]
            Points.InsertNextPoint(v)
        return Points

    @staticmethod
    def create_surface_triangles(simplices):
        """
        Method to create the Triangles that form the surfaces
        Args:
            simplices (numpy.array): 2D array with the value of the vertices that form every single triangle

        Returns:
            vtk.vtkTriangle
        """

        Triangles = vtk.vtkCellArray()
        Triangle = vtk.vtkTriangle()

        for s in simplices:
            Triangle.GetPointIds().SetId(0, s[0])
            Triangle.GetPointIds().SetId(1, s[1])
            Triangle.GetPointIds().SetId(2, s[2])

            Triangles.InsertNextCell(Triangle)
        return Triangles

    def create_surface(self, vertices, simplices, fn, alpha=1):
        """
        Method to create the polydata that define the surfaces

        Args:
            vertices (numpy.array): 2D array (XYZ) with the coordinates of the points
            simplices (numpy.array): 2D array with the value of the vertices that form every single triangle
            fn (int): Formation number
            alpha (float): Opacity

        Returns:
            vtk.vtkActor, vtk.vtkPolyDataMapper, vtk.vtkPolyData
        """
        vertices_c = copy.deepcopy(vertices)
        simplices_c = copy.deepcopy(simplices)

        surf_polydata = vtk.vtkPolyData()

        surf_polydata.SetPoints(self.create_surface_points(vertices_c))
        surf_polydata.SetPolys(self.create_surface_triangles(simplices_c))
        surf_polydata.Modified()

        surf_mapper = vtk.vtkPolyDataMapper()
        surf_mapper.SetInputData(surf_polydata)
        surf_mapper.Update()

        surf_actor = vtk.vtkActor()
        surf_actor.SetMapper(surf_mapper)
        surf_actor.GetProperty().SetColor(self.C_LOT[fn])
        surf_actor.GetProperty().SetOpacity(alpha)

        return surf_actor, surf_mapper, surf_polydata

    def create_sphere(self, X, Y, Z, fn, n_sphere=0, n_render=0, n_index=0, r=0.03):
        """
        Method to create the sphere that represent the interfaces points
        Args:
            X: X coord
            Y: Y coord
            Z: Z corrd
            fn (int): Formation number
            n_sphere (int): Number of the sphere
            n_render (int): Number of the render where the sphere belongs
            n_index (int): index value in the PandasDataframe of InupData.interfaces
            r (float): radio of the sphere

        Returns:
            vtk.vtkSphereWidget
        """
        s = vtk.vtkSphereWidget()
        s.SetInteractor(self.interactor)
        s.SetRepresentationToSurface()
        Z = Z * self.ve
        s.r_f = self._e_d_avrg * r
        s.PlaceWidget(X - s.r_f, X + s.r_f, Y - s.r_f, Y + s.r_f, Z - s.r_f, Z + s.r_f)
        s.GetSphereProperty().SetColor(self.C_LOT[fn])

        s.SetCurrentRenderer(self.ren_list[n_render])
        s.n_sphere = n_sphere
        s.n_render = n_render
        s.index = n_index
        s.AddObserver("EndInteractionEvent", self.sphereCallback)  # EndInteractionEvent

        s.On()

        return s

    def create_foliation(self, X, Y, Z, fn,
                         Gx, Gy, Gz,
                         n_plane=0, n_render=0, n_index=0, alpha=0.5):
        """
        Method to create a plane given a foliation

        Args:
            X : X coord
            Y: Y coord
            Z: Z corrd
            fn (int): Formation number
            Gx (str): Component of the gradient x
            Gy (str): Component of the gradient y
            Gz (str): Component of the gradient z
            n_plane (int): Number of the plane
            n_render (int): Number of the render where the plane belongs
            n_index (int): index value in the PandasDataframe of InupData.interfaces
            alpha: Opacity of the plane

        Returns:
            vtk.vtkPlaneWidget
        """

        Z = Z * self.ve

        d = vtk.vtkPlaneWidget()
        d.SetInteractor(self.interactor)
        d.SetRepresentationToSurface()

        # Position
        source = vtk.vtkPlaneSource()
        source.SetCenter(X, Y, Z)
        source.SetNormal(Gx, Gy, Gz)
        source.Update()
        d.SetInputData(source.GetOutput())
        d.SetHandleSize(0.05)
        min_extent = np.min([self._e_dx, self._e_dy, self._e_dz])
        d.SetPlaceFactor(min_extent/10)
        d.PlaceWidget()
        d.SetNormal(Gx, Gy, Gz)
        d.GetPlaneProperty().SetColor(self.C_LOT[fn])
        d.GetHandleProperty().SetColor(self.C_LOT[fn])
        d.GetHandleProperty().SetOpacity(alpha)
        d.SetCurrentRenderer(self.ren_list[n_render])
        d.n_plane = n_plane
        d.n_render = n_render
        d.index = n_index
        d.AddObserver("EndInteractionEvent", self.planesCallback)

        d.On()

        return d

    def set_surfaces(self, vertices, simplices,
                      #formations, fns,
                       alpha=1):
        """
        Create all the surfaces and set them to the corresponding renders for their posterior visualization with
        render_model

        Args:
            vertices (list): list of 3D numpy arrays containing the points that form each plane
            simplices (list): list of 3D numpy arrays containing the verticies that form every triangle
            formations (list): ordered list of strings containing the name of the formations to represent
            fns (list): ordered list of formation numbers (int)
            alpha: Opacity of the plane

        Returns:
            None
        """
        self.surf_rend_1 = []

        formations = self.formation_name
        fns = self.formation_number
        assert type(
            vertices) is list, 'vertices and simpleces have to be a list of arrays even when only one formation' \
                               'is passed'
        assert 'DefaultBasement' not in formations, 'Remove DefaultBasement from the list of formations'

        for v, s, fn in zip(vertices, simplices, fns):
            act, map, pol = self.create_surface(v, s, fn, alpha)
            self.surf_rend_1.append(act)

            self.ren_list[0].AddActor(act)
            self.ren_list[1].AddActor(act)
            self.ren_list[2].AddActor(act)
            self.ren_list[3].AddActor(act)

    def set_interfaces(self):
        """
        Create all the interfaces points and set them to the corresponding renders for their posterior visualization
         with render_model

        Returns:
            None
        """
        self.s_rend_1 = []
        self.s_rend_2 = []
        self.s_rend_3 = []
        self.s_rend_4 = []
        for e, val in enumerate(self.geo_data.interfaces.iterrows()):
            index = val[0]
            row = val[1]
            self.s_rend_1.append(self.create_sphere(row['X'], row['Y'], row['Z'], row['formation number'],
                                                    n_sphere=e, n_render=0, n_index=index))
            self.s_rend_2.append(self.create_sphere(row['X'], row['Y'], row['Z'], row['formation number'],
                                                    n_sphere=e, n_render=1, n_index=index))
            self.s_rend_3.append(self.create_sphere(row['X'], row['Y'], row['Z'], row['formation number'],
                                                    n_sphere=e, n_render=2, n_index=index))
            self.s_rend_4.append(self.create_sphere(row['X'], row['Y'], row['Z'], row['formation number'],
                                                    n_sphere=e, n_render=3, n_index=index))

    def set_orientations(self):
        """
        Create all the orientations and set them to the corresponding renders for their posterior visualization with
        render_model
        Returns:
            None
        """
        self.f_rend_1 = []
        self.f_rend_2 = []
        self.f_rend_3 = []
        self.f_rend_4 = []
        for e, val in enumerate(self.geo_data.orientations.iterrows()):
            index = val[0]
            row = val[1]
            self.f_rend_1.append(self.create_foliation(row['X'], row['Y'], row['Z'], row['formation number'],
                                                       row['G_x'], row['G_y'], row['G_z'],
                                                       n_plane=e, n_render=0, n_index=index))
            self.f_rend_2.append(self.create_foliation(row['X'], row['Y'], row['Z'], row['formation number'],
                                                       row['G_x'], row['G_y'], row['G_z'],
                                                       n_plane=e, n_render=1, n_index=index))
            self.f_rend_3.append(self.create_foliation(row['X'], row['Y'], row['Z'], row['formation number'],
                                                       row['G_x'], row['G_y'], row['G_z'],
                                                       n_plane=e, n_render=2, n_index=index))
            self.f_rend_4.append(self.create_foliation(row['X'], row['Y'], row['Z'], row['formation number'],
                                                       row['G_x'], row['G_y'], row['G_z'],
                                                       n_plane=e, n_render=3, n_index=index))

    def create_slider_rep(self, min=0, max=10, val=0):

        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(min)
        slider_rep.SetMaximumValue(max)
        slider_rep.SetValue(val)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0, 40)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(300, 40)

        self.slider_w = vtk.vtkSliderWidget()
        self.slider_w.SetInteractor(self.interactor)
        self.slider_w.SetRepresentation(slider_rep)
        self.slider_w.SetCurrentRenderer(self.ren_list[0])
        self.slider_w.SetAnimationModeToAnimate()
        self.slider_w.On()
        self.slider_w.AddObserver('EndInteractionEvent', self.slideCallback)

    def slideCallback(self, obj, event):

        self.post.change_input_data(self.interp_data, obj.GetRepresentation().GetValue())
        try:
            for surf in self.surf_rend_1:
                self.ren_list[0].RemoveActor(surf)
                self.ren_list[1].RemoveActor(surf)
                self.ren_list[2].RemoveActor(surf)
                self.ren_list[3].RemoveActor(surf)

            ver, sim = self.update_surfaces_real_time(self.interp_data)
            self.set_surfaces(ver, sim)
        except AttributeError:
            print('no surf')
            pass
        try:
            for sph in zip(self.s_rend_1, self.s_rend_2, self.s_rend_3, self.s_rend_4, self.geo_data.interfaces.iterrows()):

                row_i = sph[4][1]
                sph[0].PlaceWidget(row_i['X'] - sph[0].r_f, row_i['X'] + sph[0].r_f,
                                   row_i['Y'] - sph[0].r_f, row_i['Y'] + sph[0].r_f,
                                   row_i['Z'] - sph[0].r_f, row_i['Z'] + sph[0].r_f)

                sph[1].PlaceWidget(row_i['X'] - sph[1].r_f, row_i['X'] + sph[1].r_f,
                                   row_i['Y'] - sph[1].r_f, row_i['Y'] + sph[1].r_f,
                                   row_i['Z'] - sph[1].r_f, row_i['Z'] + sph[1].r_f)

                sph[2].PlaceWidget(row_i['X'] - sph[2].r_f, row_i['X'] + sph[2].r_f,
                                   row_i['Y'] - sph[2].r_f, row_i['Y'] + sph[2].r_f,
                                   row_i['Z'] - sph[2].r_f, row_i['Z'] + sph[2].r_f)

                sph[3].PlaceWidget(row_i['X'] - sph[3].r_f, row_i['X'] + sph[3].r_f,
                                   row_i['Y'] - sph[3].r_f, row_i['Y'] + sph[3].r_f,
                                   row_i['Z'] - sph[3].r_f, row_i['Z'] + sph[3].r_f)
        except AttributeError:
            pass
        try:
            for fol in (zip(self.f_rend_1, self.f_rend_2, self.f_rend_3, self.f_rend_4, self.geo_data.orientations.iterrows())):
                row_f = fol[4][1]

                fol[0].SetCenter(row_f['X'], row_f['Y'], row_f['Z'])
                fol[0].SetNormal(row_f['G_x'], row_f['G_y'], row_f['G_z'])

        except AttributeError:
            pass

    def sphereCallback(self, obj, event):
        """
        Function that rules what happens when we move a sphere. At the moment we update the other 3 renderers and
        update the pandas data frame.
        """

        # Resetting the xy camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[1].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[1].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[1].GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.ren_list[1].GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)

        # Resetting the yz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[2].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[2].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[2].GetActiveCamera().SetPosition(fp[0] + dist, fp[1], fp[2])
        self.ren_list[2].GetActiveCamera().SetViewUp(0.0, -1.0, 0.0)
        self.ren_list[2].GetActiveCamera().Roll(90)

        # Resetting the xz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[3].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[3].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[3].GetActiveCamera().SetPosition(fp[0], fp[1] - dist, fp[2])
        self.ren_list[3].GetActiveCamera().SetViewUp(-1.0, 0.0, 0.0)
        self.ren_list[3].GetActiveCamera().Roll(90)

        # Get new position of the sphere
        new_center = obj.GetCenter()

        # Get which sphere we are moving
        index = obj.index

        # Modify Pandas DataFrame
        self.geo_data.modify_interface(index, X=new_center[0], Y=new_center[1], Z=new_center[2])

        # Check what render we are working with
        render = obj.n_render
        r_f = obj.r_f

        # Update the other renderers
        if render != 0:
            self.s_rend_1[obj.n_sphere].PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                                                    new_center[1] - r_f, new_center[1] + r_f,
                                                    new_center[2] - r_f, new_center[2] + r_f)

        if render != 1:
            self.s_rend_2[obj.n_sphere].PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                                                    new_center[1] - r_f, new_center[1] + r_f,
                                                    new_center[2] - r_f, new_center[2] + r_f)
        if render != 2:
            self.s_rend_3[obj.n_sphere].PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                                                    new_center[1] - r_f, new_center[1] + r_f,
                                                    new_center[2] - r_f, new_center[2] + r_f)
        if render != 3:
            self.s_rend_4[obj.n_sphere].PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                                                    new_center[1] - r_f, new_center[1] + r_f,
                                                    new_center[2] - r_f, new_center[2] + r_f)

        if self.real_time:
            for surf in self.surf_rend_1:
                self.ren_list[0].RemoveActor(surf)
                self.ren_list[1].RemoveActor(surf)
                self.ren_list[2].RemoveActor(surf)
                self.ren_list[3].RemoveActor(surf)

            vertices, simpleces = self.update_surfaces_real_time(self.interp_data)
            self.set_surfaces(vertices, simpleces)

    def planesCallback(self, obj, event):
        """
        Function that rules what happens when we move a plane. At the moment we update the other 3 renderers and
        update the pandas data frame
        """

        # Resetting the xy camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[1].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[1].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[1].GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.ren_list[1].GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)

        # Resetting the yz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[2].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[2].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[2].GetActiveCamera().SetPosition(fp[0] + dist, fp[1], fp[2])
        self.ren_list[2].GetActiveCamera().SetViewUp(0.0, -1.0, 0.0)
        self.ren_list[2].GetActiveCamera().Roll(90)

        # Resetting the xz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[3].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[3].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[3].GetActiveCamera().SetPosition(fp[0], fp[1] - dist, fp[2])
        self.ren_list[3].GetActiveCamera().SetViewUp(-1.0, 0.0, 0.0)
        self.ren_list[3].GetActiveCamera().Roll(90)

        # Get new position of the plane and GxGyGz
        new_center = obj.GetCenter()
        new_normal = obj.GetNormal()
        # Get which plane we are moving
        index = obj.index

        new_source = vtk.vtkPlaneSource()
        new_source.SetCenter(new_center)
        new_source.SetNormal(new_normal)
        new_source.Update()

        # Modify Pandas DataFrame
        # update the gradient vector components and its location
        self.geo_data.modify_orientation(index, X=new_center[0], Y=new_center[1], Z=new_center[2],
                                       G_x=new_normal[0], G_y=new_normal[1], G_z=new_normal[2])
        # update the dip and azimuth values according to the new gradient
        self.geo_data.calculate_orientations()

        # Update the other renderers
        render = obj.n_render

        # TODO: get the plane source, change the stuff there, and then update the widget accordingly?
        if render != 0:
            self.f_rend_1[obj.n_plane].SetInputData(new_source.GetOutput())
            self.f_rend_1[obj.n_plane].SetNormal(new_normal)
            self.f_rend_1[obj.n_plane].SetCenter(new_center[0], new_center[1], new_center[2])
        if render != 1:
            self.f_rend_2[obj.n_plane].SetInputData(new_source.GetOutput())
            self.f_rend_2[obj.n_plane].SetNormal(new_normal)
            self.f_rend_2[obj.n_plane].SetCenter(new_center[0], new_center[1], new_center[2])
        if render != 2:
            self.f_rend_3[obj.n_plane].SetInputData(new_source.GetOutput())
            self.f_rend_3[obj.n_plane].SetNormal(new_normal)
            self.f_rend_3[obj.n_plane].SetCenter(new_center[0], new_center[1], new_center[2])
        if render != 3:
            self.f_rend_4[obj.n_plane].SetInputData(new_source.GetOutput())
            self.f_rend_4[obj.n_plane].SetNormal(new_normal)
            self.f_rend_4[obj.n_plane].SetCenter(new_center[0], new_center[1], new_center[2])

        if self.real_time:
            for surf in self.surf_rend_1:
                self.ren_list[0].RemoveActor(surf)
                self.ren_list[1].RemoveActor(surf)
                self.ren_list[2].RemoveActor(surf)
                self.ren_list[3].RemoveActor(surf)

            vertices, simpleces = self.update_surfaces_real_time(self.interp_data)
            #  print(vertices[0][60])
            self.set_surfaces(vertices, simpleces)

    def create_ren_list(self):
        """
        Create a list of the 4 renderers we use. One general view and 3 cartersian projections
        Returns:
            list: list of renderers
        """

        # viewport dimensions setup
        xmins = [0, 0.6, 0.6, 0.6]
        xmaxs = [0.6, 1, 1, 1]
        ymins = [0, 0, 0.33, 0.66]
        ymaxs = [1, 0.33, 0.66, 1]

        # create list of renderers, set vieport values
        ren_list = []
        for i in range(self.n_ren):
            # append each renderer to list of renderers
            ren_list.append(vtk.vtkRenderer())
            # add each renderer to window
            self.renwin.AddRenderer(ren_list[-1])
            # set viewport for each renderer
            ren_list[-1].SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        return ren_list

    def _create_cameras(self, extent, verbose=0):
        """
        Create the 4 cameras for each renderer
        """
        _e = extent
        _e_dx = _e[1] - _e[0]
        _e_dy = _e[3] - _e[2]
        _e_dz = _e[5] - _e[4]
        _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
        _e_max = np.argmax(_e)

        # General camera
        model_cam = vtk.vtkCamera()
        model_cam.SetPosition(_e[_e_max] * 5, _e[_e_max] * 5, _e[_e_max] * 5)
        model_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                                np.min(_e[2:4]) + _e_dy / 2,
                                np.min(_e[4:]) + _e_dz / 2)

        model_cam.SetViewUp(-0.239, 0.155, 0.958)

        # XY camera RED
        xy_cam = vtk.vtkCamera()

        xy_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                           np.min(_e[2:4]) + _e_dy / 2,
                           _e[_e_max] * 4)

        xy_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)

        # YZ camera GREEN
        yz_cam = vtk.vtkCamera()
        yz_cam.SetPosition(_e[_e_max] * 4,
                           np.min(_e[2:4]) + _e_dy / 2,
                           np.min(_e[4:]) + _e_dz / 2)

        yz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)
        yz_cam.SetViewUp(0, -1, 0)
        yz_cam.Roll(90)

        # XZ camera BLUE
        xz_cam = vtk.vtkCamera()
        xz_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                           - _e[_e_max] * 4,
                           np.min(_e[4:]) + _e_dz / 2)

        xz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)
        xz_cam.SetViewUp(0, 1, 0)
        xz_cam.Roll(0)

        # camera position debugging
        if verbose == 1:
            print("RED XY:", xy_cam.GetPosition())
            print("RED FP:", xy_cam.GetFocalPoint())
            print("GREEN YZ:", yz_cam.GetPosition())
            print("GREEN FP:", yz_cam.GetFocalPoint())
            print("BLUE XZ:", xz_cam.GetPosition())
            print("BLUE FP:", xz_cam.GetFocalPoint())

        return [model_cam, xy_cam, yz_cam, xz_cam]

    def set_camera_backcolor(self, color=None):
        """
        define background colors of the renderers
        """
        if color is None:
            color = (66 / 250, 66 / 250, 66 / 250)

        ren_color = [color for i in range(self.n_ren)]

        for i in range(self.n_ren):
            # set active camera for each renderer
            self.ren_list[i].SetActiveCamera(self.camera_list[i])
            # set background color for each renderer
            self.ren_list[i].SetBackground(ren_color[i][0], ren_color[i][1], ren_color[i][2])

    def _create_axes(self, camera, verbose=0, tick_vis=True):
        """
        Create the axes boxes
        """
        cube_axes_actor = vtk.vtkCubeAxesActor()
        cube_axes_actor.SetBounds(self.geo_data.extent)
        cube_axes_actor.SetCamera(camera)
        if verbose == 1:
            print(cube_axes_actor.GetAxisOrigin())

        # set axes and label colors
        cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
        cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)

        cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
        cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

        cube_axes_actor.DrawXGridlinesOn()
        cube_axes_actor.DrawYGridlinesOn()
        cube_axes_actor.DrawZGridlinesOn()

        if not tick_vis:
            cube_axes_actor.XAxisMinorTickVisibilityOff()
            cube_axes_actor.YAxisMinorTickVisibilityOff()
            cube_axes_actor.ZAxisMinorTickVisibilityOff()

        cube_axes_actor.SetXTitle("X")
        cube_axes_actor.SetYTitle("Y")
        cube_axes_actor.SetZTitle("Z")

        cube_axes_actor.SetXAxisLabelVisibility(1)
        cube_axes_actor.SetYAxisLabelVisibility(1)
        cube_axes_actor.SetZAxisLabelVisibility(1)

        # only plot grid lines furthest from viewpoint
        # ensure platform compatibility for the grid line options
        if sys.platform == "win32":
            cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
        else:  # rather use elif == "linux" ? but what about other platforms
            try:  # apparently this can also go wrong on linux, maybe depends on vtk version?
                cube_axes_actor.SetGridLineLocation(vtk.VTK_GRID_LINES_FURTHEST)
            except AttributeError:
                pass

        return cube_axes_actor

    @staticmethod
    def update_surfaces_real_time(interp_data):

        lith_block, fault_block = gp.compute_model(interp_data)
        try:
            v_l, s_l = gp.get_surfaces(interp_data, lith_block[1], fault_block[1::2], original_scale=False)
        except IndexError:
            v_l, s_l = gp.get_surfaces(interp_data, lith_block[1], None, original_scale=False)
        return v_l, s_l

    @staticmethod
    def export_vtk_lith_block(geo_data, lith_block, path=None):
        """
        Export data to a vtk file for posterior visualizations

        Args:
            geo_data(gempy.InputData): All values of a DataManagement object
            block(numpy.array): 3D array containing the lithology block
            path (str): path to the location of the vtk

        Returns:
            None
        """

        from pyevtk.hl import gridToVTK

        import numpy as np

        # Dimensions

        nx, ny, nz = geo_data.resolution

        lx = geo_data.extent[1] - geo_data.extent[0]
        ly = geo_data.extent[3] - geo_data.extent[2]
        lz = geo_data.extent[5] - geo_data.extent[4]

        dx, dy, dz = lx / nx, ly / ny, lz / nz

        ncells = nx * ny * nz

        npoints = (nx + 1) * (ny + 1) * (nz + 1)

        # Coordinates
        x = np.arange(geo_data.extent[0], geo_data.extent[1] + 0.1, dx, dtype='float64')

        y = np.arange(geo_data.extent[2], geo_data.extent[3] + 0.1, dy, dtype='float64')

        z = np.arange(geo_data.extent[4], geo_data.extent[5] + 0.1, dz, dtype='float64')

        lith = lith_block.reshape((nx, ny, nz))

        # Variables

        if not path:
            path = "./default"

        gridToVTK(path+'_lith_block', x, y, z, cellData={"Lithology": lith})

    @staticmethod
    def export_vtk_surfaces(vertices, simplices, path=None, name='_surfaces', alpha=1):
        """
        Export data to a vtk file for posterior visualizations

        Args:
            geo_data(gempy.InputData): All values of a DataManagement object
            block(numpy.array): 3D array containing the lithology block
            path (str): path to the location of the vtk

        Returns:
            None
        """
        import vtk

        for s_n in range(len(vertices)):
            # setup points and vertices
            Points = vtk.vtkPoints()
            Triangles = vtk.vtkCellArray()
            Triangle = vtk.vtkTriangle()
            for p in vertices[s_n]:
                Points.InsertNextPoint(p)

            # Unfortunately in this simple example the following lines are ambiguous.
            # The first 0 is the index of the triangle vertex which is ALWAYS 0-2.
            # The second 0 is the index into the point (geometry) array, so this can range from 0-(NumPoints-1)
            # i.e. a more general statement is triangle->GetPointIds()->SetId(0, PointId);
            for i in simplices[s_n]:
                Triangle.GetPointIds().SetId(0, i[0])
                Triangle.GetPointIds().SetId(1, i[1])
                Triangle.GetPointIds().SetId(2, i[2])

                Triangles.InsertNextCell(Triangle)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(Points)
            polydata.SetPolys(Triangles)

            polydata.Modified()
            if vtk.VTK_MAJOR_VERSION <= 5:
                polydata.Update()

            writer = vtk.vtkXMLPolyDataWriter();

            # Add colors
            surf_mapper = vtk.vtkPolyDataMapper()
            surf_mapper.SetInputData(polydata)
            surf_mapper.Update()

            surf_actor = vtk.vtkActor()
            surf_actor.SetMapper(surf_mapper)
            surf_actor.GetProperty().SetColor(color_lot[s_n])
            surf_actor.GetProperty().SetOpacity(alpha)

            if not path:
                path = "./default_"

            writer.SetFileName(path+'_surfaces'+str(s_n)+'.vtp')
            if vtk.VTK_MAJOR_VERSION <= 5:
                writer.SetInput(polydata)
            else:
                writer.SetInputData(polydata)
            writer.Write()


def _create_color_lot(geo_data, cd_rgb):
    """Returns color [r,g,b] LOT for formation numbers."""
    if "formation number" not in geo_data.interfaces or "formation number" not in geo_data.orientations:
        geo_data.set_formation_number()  # if not, set formation numbers

    c_names = ["indigo", "red", "yellow", "brown", "orange",
                "green", "blue", "amber", "pink", "light-blue",
                "lime", "blue-grey", "deep-orange", "grey", "cyan",
                "deep-purple", "purple", "teal", "light-green"]

    lot = {}
    ci = 0  # use as an independent running variable because of fault formations
    # get unique formation numbers
    fmt_numbers = np.unique([val for val in geo_data.interfaces['formation number'].unique()])
    # get unique fault formation numbers
    fault_fmt_numbers = np.unique(geo_data.interfaces[geo_data.interfaces["isFault"] == True]["formation number"])
    # iterate over all unique formation numbers
    for i, n in enumerate(fmt_numbers):
        # if its a fault formation set it to black by default
        if n in fault_fmt_numbers:
            lot[n] = cd_rgb["black"]["400"]
        # if not, just go through
        else:
            lot[n] = cd_rgb[c_names[ci]]["400"]
            ci += 1

    return lot
