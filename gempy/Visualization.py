"""
Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016

@author: Miguel de la Varga
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from os import path
import sys
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from IPython.core.debugger import Pdb
from gempy.colors import color_dict_rgb, color_dict_hex

# TODO: inherit pygeomod classes
# import sys, os


class PlotData(object):
    """
    Class to make the different plot related with gempy

    Args:
        _data(GeMpy_core.DataManagement): All values of a DataManagement object
        block(theano shared): 3D array containing the lithology block
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        potential_field(numpy.ndarray): 3D array containing a individual potential field
        verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
    """

    def __init__(self, _data, cd_rgb=color_dict_rgb, cd_hex=color_dict_hex, **kwargs):

        self._data = _data

        if 'potential_field' in kwargs:
            self._potential_field_p = kwargs['potential_field']

            # TODO planning the whole visualization scheme. Only data, potential field
            # and block. 2D 3D? Improving the iteration
            # with pandas framework
        self._set_style()

        self._cd_rgb = cd_rgb
        self._cd_hex = cd_hex

        # TODO: Map colors to formations and integer values for plots

        c_names = ["indigo", "red", "yellow", "brown", "orange",
                   "green", "blue", "amber", "pink", "light-blue",
                   "lime", "blue-grey", "deep-orange", "grey", "cyan",
                   "deep-purple", "purple", "teal", "light-green"]

        # c_subnames = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900']
        #               'a100','a200', 'a400', 'a700']

        self._cmap = matplotlib.colors.ListedColormap([self._cd_rgb[key]["400"] for key in c_names])
        self._sns_palette = [self._cd_rgb[key]["400"] for key in c_names]

        bounds = [i for i in range(len(c_names))]
        self._norm = matplotlib.colors.BoundaryNorm(bounds, self._cmap.N)
        # TODO: Are colors correctly mapped between voxel plot and data plot?

    def _set_style(self):
        """
        Private function to set some plotting options

        """

        plt.style.use(['seaborn-white', 'seaborn-paper'])
        # sns.set_context("paper")
        # matplotlib.rc("font", family="Helvetica")

    def plot_data(self, direction="y", series="all", **kwargs):
        """
        Plot the projecton of the raw data (interfaces and foliations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            series(str): series to plot
            **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

        Returns:
            Data plot

        """

        x, y, Gx, Gy = self._slice(direction)[4:]

        if series == "all":
            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"].
                isin(self._data.series.columns.values)]
            series_to_plot_f = self._data.foliations[self._data.foliations["series"].
                isin(self._data.series.columns.values)]

        else:

            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"] == series]
            series_to_plot_f = self._data.foliations[self._data.foliations["series"] == series]

        sns.lmplot(x, y,
                   data=series_to_plot_i,
                   fit_reg=False,
                   hue="formation",
                   scatter_kws={"marker": "D",
                                "s": 100},
                   legend=True,
                   legend_out=True,
                   palette=self._sns_palette,
                   **kwargs)

        # Plotting orientations
        plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                   series_to_plot_f[Gx], series_to_plot_f[Gy],
                   pivot="tail")

        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or potential field) in the specific direction selected in the plotting functions

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
                           plot_data = False, **kwargs):
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

        plt.imshow(plot_block[_a, _b, _c].T, origin="bottom", cmap=self._cmap, norm=self._norm,
                   extent=extent_val,
                   interpolation=interpolation, **kwargs)

        plt.xlabel(x)
        plt.ylabel(y)

    def plot_potential_field(self, potential_field, cell_number, n_pf=0,
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
            Potential field plot
        """

        if plot_data:
            self.plot_data(direction, 'all')

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
        plt.contour(potential_field.reshape(
            self._data.resolution[0], self._data.resolution[1], self._data.resolution[2])[_a, _b, _c].T,
                    cell_number,
                    extent=extent_val, *args,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.title(self._data.interfaces['series'].unique()[n_pf])
        plt.xlabel(x)
        plt.ylabel(y)

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

        # mesh = steno3d.Mesh3DGrid(h1=np.diff(np.linspace(self._data.extent[0], self._data.extent[1], self._data.resolution[0])),
        #                           h2=np.diff(np.linspace(self._data.extent[2], self._data.extent[3], self._data.resolution[1])),
        #                           h3=np.diff(np.linspace(self._data.extent[4], self._data.extent[5], self._data.resolution[2])))

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

    # def plot3D_ipyvolume(block, geo_data, cm=plt.cm.viridis.colors):
    #     import ipyvolume.pylab as p3
    #     n_lith = np.unique(block).shape[0]
    #     p3.figure()
    #     for e, i in enumerate(np.unique(sol)):  # np.unique(block)):
    #
    #         bool_pos = block == i
    #         p3.scatter(
    #             geo_data.grid.grid[:, 0][bool_pos],
    #             geo_data.grid.grid[:, 1][bool_pos],
    #             geo_data.grid.grid[:, 2][bool_pos],
    #             marker='box',
    #             color=cm[np.linspace(0, 255, n_lith, dtype=int)[e]])
    #
    #     p3.xlim(np.min(geo_data.grid.grid[:, 0]), np.max(geo_data.grid.grid[:, 0]))
    #     p3.ylim(np.min(geo_data.grid.grid[:, 1]), np.max(geo_data.grid.grid[:, 1]))
    #     p3.zlim(np.min(geo_data.grid.grid[:, 2]), np.max(geo_data.grid.grid[:, 2]))
    #
    #     p3.show()

    def export_vtk_structured(self):
        """
        export vtk
        :return:
        """

        # from evtk.hl import gridToVTK
        #
        # import numpy as np
        #
        # import random as rnd
        #
        # # Dimensions
        #
        # nx, ny, nz = 50, 50, 50
        #
        # lx, ly, lz = 10., 10., -10.0
        #
        # dx, dy, dz = lx / nx, ly / ny, lz / nz
        #
        # ncells = nx * ny * nz
        #
        # npoints = (nx + 1) * (ny + 1) * (nz + 1)
        #
        # # Coordinates
        #
        # X = np.arange(0, lx + 0.1 * dx, dx, dtype='float64')
        #
        # Y = np.arange(0, ly + 0.1 * dy, dy, dtype='float64')
        #
        # Z = np.arange(0, lz + 0.1 * dz, dz, dtype='float64')
        #
        # x = np.zeros((nx + 1, ny + 1, nz + 1))
        #
        # y = np.zeros((nx + 1, ny + 1, nz + 1))
        #
        # z = np.zeros((nx + 1, ny + 1, nz + 1))
        #
        # # We add some random fluctuation to make the grid more interesting
        #
        # for k in range(nz + 1):
        #
        #     for j in range(ny + 1):
        #         for i in range(nx + 1):
        #             x[i, j, k] = X[i] + (0.5 - rnd.random()) * 0.2 * dx
        #
        #             y[i, j, k] = Y[j] + (0.5 - rnd.random()) * 0.2 * dy
        #
        #             z[i, j, k] = Z[k] + (0.5 - rnd.random()) * 0.2 * dz
        #
        #             # Variables
        #
        # pressure = sol[0, 0, :].reshape((nx, ny, nz))
        #
        # gridToVTK("./structured", x, y, z, cellData={"pressure": pressure})

    def export_vtk_rectilinear(self):
        """
        ajapg
        Returns:

        """
        # from evtk.hl import gridToVTK
        #
        # import numpy as np
        #
        # # Dimensions
        #
        #
        # nx, ny, nz = 50, 50, 50
        #
        # lx, ly, lz = 10., 10., -10.0
        #
        # dx, dy, dz = lx / nx, ly / ny, lz / nz
        #
        # ncells = nx * ny * nz
        #
        # npoints = (nx + 1) * (ny + 1) * (nz + 1)
        #
        # # Coordinates
        #
        # x = np.arange(0, lx + 0.1 * dx, dx, dtype='float64')
        #
        # y = np.arange(0, ly + 0.1 * dy, dy, dtype='float64')
        #
        # z = np.arange(0, lz + 0.1 * dz, dz, dtype='float64')
        #
        # # Variables
        #
        # pressure = sol[0, 0, :].reshape((nx, ny, nz))
        #
        # temp = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1))
        #
        # gridToVTK("./rectilinear", x, y, z, cellData={"lithology": pressure}, )

    def plot_layers(self):
        """
        plot layers
        Returns:

        """

        # import vtk
        # from vtk import *
        #
        # # setup points and vertices
        # Points = vtk.vtkPoints()
        # Triangles = vtk.vtkCellArray()
        # Triangle = vtk.vtkTriangle()
        #
        # for p in vertices * 0.4:
        #     Points.InsertNextPoint(p)
        #
        # # Unfortunately in this simple example the following lines are ambiguous.
        # # The first 0 is the index of the triangle vertex which is ALWAYS 0-2.
        # # The second 0 is the index into the point (geometry) array, so this can range from 0-(NumPoints-1)
        # # i.e. a more general statement is triangle->GetPointIds()->SetId(0, PointId);
        #
        # for i in simplices:
        #     Triangle.GetPointIds().SetId(0, i[0])
        #     Triangle.GetPointIds().SetId(1, i[1])
        #     Triangle.GetPointIds().SetId(2, i[2])
        #
        #     Triangles.InsertNextCell(Triangle)
        #
        # polydata = vtk.vtkPolyData()
        # polydata.SetPoints(Points)
        # polydata.SetPolys(Triangles)
        #
        # polydata.Modified()
        # if vtk.VTK_MAJOR_VERSION <= 5:
        #     polydata.Update()
        #
        # writer = vtk.vtkXMLPolyDataWriter();
        # writer.SetFileName("Fabian_f.vtp");
        # if vtk.VTK_MAJOR_VERSION <= 5:
        #     writer.SetInput(polydata)
        # else:
        #     writer.SetInputData(polydata)
        # writer.Write()