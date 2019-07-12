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

@author: Miguel de la Varga, Elisa Heim
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedFormatter, FixedLocator
import seaborn as sns
from os import path
import sys
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gempy.core.solution import Solution
import gempy.plot.helpers as plothelp
sns.set_context('talk')
plt.style.use(['seaborn-white', 'seaborn-talk'])


#try:
    #import mplstereonet
    #MPLST_IMPORT = True
#except ImportError:
    #MPLST_IMPORT = False


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

    def __init__(self, model, **kwargs):

        self.model = model

        self._color_lot = dict(zip(self.model.surfaces.df['surface'], self.model.surfaces.df['color'])) # model.surfaces.colors.colordict
        self._cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))
        self._norm = mcolors.Normalize(vmin=0.5, vmax=len(self._cmap.colors)+0.5)

        self._set_style()

    @staticmethod
    def _set_style():
        """
        Private function to set some plot options

        """

        plt.style.use(['seaborn-white', 'seaborn-talk'])
        sns.set_context("talk")

    def plot_data(self, direction="y", data_type='all', series="all", legend_font_size=10, ve=1, **kwargs):
        """
        Plot the projecton of the raw data (surface_points and orientations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            data_type (str): type of data to plot. 'all', 'surface_points' or 'orientations'
            series(str): series to plot
            ve(float): Vertical exageration
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

        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        # apply vertical exageration
        if direction == 'x' or direction == 'y':
            aspect /= ve

        if aspect < 1:
            min_axis = 'width'
        else:
            min_axis = 'height'
        if series == "all":
            series_to_plot_i = self.model.surface_points.df[self.model.surface_points.df["series"].
                isin(self.model.series.df.index.values)]
            series_to_plot_f = self.model.orientations.df[self.model.orientations.df["series"].
                isin(self.model.series.df.index.values)]

        else:

            series_to_plot_i = self.model.surface_points[self.model.surface_points.df["series"] == series]
            series_to_plot_f = self.model.orientations[self.model.orientations.df["series"] == series]

        if data_type == 'all':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="surface",
                           #scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette= self._color_lot,
                           **kwargs)

            p.axes[0, 0].set_ylim(extent[2], extent[3])
            p.axes[0, 0].set_xlim(extent[0], extent[1])

            # Plotting orientations
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=10)

        if data_type == 'surface_points':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="surface",
                           #scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette=self._color_lot,
                           **kwargs)
            p.axes[0, 0].set_ylim(extent[2], extent[3])
            p.axes[0, 0].set_xlim(extent[0], extent[1])

        if data_type == 'orientations':
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=15)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions

        """
        _a, _b, _c = (slice(0, self.model.grid.regular_grid.resolution[0]),
                      slice(0, self.model.grid.regular_grid.resolution[1]),
                      slice(0, self.model.grid.regular_grid.resolution[2]))
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self.model.grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def _slice2D(self, cell_number, direction):
        if direction == 'x':
            _slice = np.s_[cell_number, :, :]
            extent = self.model.grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == 'y':
            _slice = np.s_[:, cell_number, :]
            extent = self.model.grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == 'z':
            _slice = np.s_[:, :, cell_number]
            extent = self.model.grid.regular_grid.extent[[1, 2, 3, 4]]
        else:
            print('not a direction')
        return _slice, extent

    def plot_topography(self, cell_number, direction):
        # Todo replace this function
        line = self.model.grid.topography._line_in_section(cell_number=cell_number, direction=direction)
        if direction == 'x':
            ext = self.model.grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == 'y':
            ext = self.model.grid.regular_grid.extent[[0, 1, 4, 5]]
        # add corners
        line = np.append(line, ([ext[1], line[0, -1]], [ext[1], ext[3]], [ext[0], ext[3]], [ext[0], line[0, 1]])).reshape(-1,2)
        plt.fill(line[:, 0], line[:, 1], color='k')#, alpha=0.5)


    def extract_fault_lines(self, cell_number=25, direction='y'):
        faults = list(self.model.faults.df[self.model.faults.df['isFault'] == True].index)
        _slice, extent = self._slice2D(cell_number, direction)
        for fault in faults:
            f_id = int(self.model.series.df.loc[fault, 'order_series']) - 1
            block = self.model.solutions.scalar_field_matrix[f_id]
            level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
                self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]
            plt.contour(block.reshape(self.model.grid.regular_grid.resolution)[_slice].T, 0, extent=extent, levels=level,
                        colors=self._cmap.colors[f_id], linestyles='solid')

    def plot_map(self, solution: Solution, contour_lines=True):
        # maybe add contour kwargs
        assert solution.geological_map is not None, 'Geological map not computed. Activate the topography grid.'
        geomap = solution.geological_map.reshape(self.model.grid.topography.values_3D[:,:,2].shape)
        fig, ax = plt.subplots()
        plt.imshow(geomap, origin="lower", extent=self.model.grid.topography.extent, cmap=self._cmap, norm=self._norm)
        if contour_lines == True:
            CS = ax.contour(self.model.grid.topography.values_3D[:, :, 2],  cmap='Greys', linestyles='solid',
                            extent=self.model.grid.topography.extent)
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            cbar = plt.colorbar(CS)
            cbar.set_label('elevation [m]')
        plt.title("Geological map", fontsize=15)
        plt.xlabel('X')
        plt.ylabel('Y')

    def plot_block_section(self, solution:Solution, cell_number:int, block:np.ndarray=None, direction:str="y",
                           interpolation:str='none', show_data:bool=False, show_faults:bool=False, show_topo:bool=False,
                           block_type=None, ve:float=1, **kwargs):
        """Plot a section of the block model

        Args:
            solution (Solution): [description]
            cell_number (int): Section position of the array to plot.
            block (np.ndarray, optional): Lithology block. Defaults to None.
            direction (str, optional): Cartesian direction to be plotted
                ("x", "y", "z"). Defaults to "y".
            interpolation (str, optional): Type of interpolation of plt.imshow.
                Defaults to 'none'. Acceptable values are ('none' ,'nearest',
                'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
                'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
                'bessel', 'mitchell', 'sinc', 'lanczos'.
            show_data (bool, optional): Plots input data on-top of block
                section. Defaults to False.
            show_faults (bool, optional): Plot fault line on-top of block
                section. Defaults to False.
            show_topo (bool, optional): Plots block section with topography.
                Defaults to True.
            block_type ([type], optional): [description]. Defaults to None.
            ve (float, optional): Vertical exaggeration. Defaults to 1.

        Returns:
            (gempy.plot.visualization_2d.PlotData2D) Block section plot.
        """
        if block is None:
            _block = solution.lith_block
        else:
            _block = block
            if _block.dtype == bool:
                kwargs['cmap'] = 'viridis'
                kwargs['norm'] = None

        if block_type is not None:
            raise NotImplementedError

        plot_block = _block.reshape(self.model.grid.regular_grid.resolution[0],
                                    self.model.grid.regular_grid.resolution[1],
                                    self.model.grid.regular_grid.resolution[2])
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        if show_data:
            self.plot_data(direction, 'all')
        # TODO: plot_topo option - need fault_block for that

        # apply vertical exageration
        if direction in ("x", "y"):
            aspect = ve
        else:
            aspect = 1

        if 'cmap' not in kwargs:
            kwargs['cmap'] = self._cmap
        if 'norm' not in kwargs:
            kwargs['norm'] = self._norm

        im = plt.imshow(plot_block[_a, _b, _c].T,
                        origin="bottom",
                        extent=extent_val,
                        interpolation=interpolation,
                        aspect=aspect,
                        **kwargs)

        if extent_val[3] < 0:           # correct vertical orientation of plot
            plt.gca().invert_yaxis()    # if maximum vertical extent negative

        if show_faults:
            self.extract_fault_lines(cell_number, direction)

        if show_topo:
            if self.model.grid.topography is not None:
                if direction == 'z':
                    plt.contour(self.model.grid.topography.values_3D[:, :, 2], extent=extent_val, cmap='Greys')
                else:
                    self.plot_topography(cell_number=cell_number, direction=direction)

        if not show_data:
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=color, label=surface) for surface, color in self._color_lot.items()]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.xlabel(x)
        plt.ylabel(y)
        return plt.gcf()

    def plot_scalar_field(self, solution, cell_number, series=0, N=20,
                          direction="y", plot_data=True, *args, **kwargs):
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

        if isinstance(solution, Solution):
            scalar_field = solution.scalar_field_matrix[series]
        else:
            warnings.warn('Passing the block directly will get deprecated in the next version. Please use Solution'
                          'and block_type instead', FutureWarning)
            scalar_field = solution

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'magma'

        if plot_data:
            self.plot_data(direction, 'all')

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        plt.contour(scalar_field.reshape(
            self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[_a, _b, _c].T,
                    N,
                    extent=extent_val, *args,
                    **kwargs)

        plt.contourf(scalar_field.reshape(
            self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[_a, _b, _c].T,
                    N,
                    extent=extent_val, alpha=0.6, *args,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.xlabel(x)
        plt.ylabel(y)

    @staticmethod
    def plot_topo_g(geo_model, G, centroids, direction="y",
                    label_kwargs=None, node_kwargs=None, edge_kwargs=None):
        res = geo_model.grid.regular_grid.resolution
        if direction == "y":
            c1, c2 = (0, 2)
            e1 = geo_model.grid.regular_grid.extent[1] - geo_model.grid.regular_grid.extent[0]
            e2 = geo_model.grid.regular_grid.extent[5] - geo_model.grid.regular_grid.extent[4]
            d1 = geo_model.grid.regular_grid.extent[0]
            d2 = geo_model.grid.regular_grid.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = res[0]
            r2 = res[2]
        elif direction == "x":
            c1, c2 = (1, 2)
            e1 = geo_model.grid.regular_grid.extent[3] - geo_model.grid.regular_grid.extent[2]
            e2 = geo_model.grid.regular_grid.extent[5] - geo_model.grid.regular_grid.extent[4]
            d1 = geo_model.grid.regular_grid.extent[2]
            d2 = geo_model.grid.regular_grid.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = res[1]
            r2 = res[2]
        elif direction == "z":
            c1, c2 = (0, 1)
            e1 = geo_model.grid.regular_grid.extent[1] - geo_model.grid.regular_grid.extent[0]
            e2 = geo_model.grid.regular_grid.extent[3] - geo_model.grid.regular_grid.extent[2]
            d1 = geo_model.grid.regular_grid.extent[0]
            d2 = geo_model.grid.regular_grid.extent[2]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = res[0]
            r2 = res[1]

        nkw = {
            "marker": "o",
            "color": "black",
            "markersize": 20,
            "alpha": 0.75
        }
        if node_kwargs is not None:
            nkw.update(node_kwargs)

        tkw = {
            "color": "white",
            "size": 10,
            "ha": "center",
            "va": "center",
            "weight": "ultralight",
            "family": "monospace"
        }
        if label_kwargs is not None:
            tkw.update(label_kwargs)

        lkw = {
            "linewidth": 0.75,
            "color": "black"
        }
        if edge_kwargs is not None:
            lkw.update(edge_kwargs)

        for edge in G.edges():
            a, b = edge
            # plot edges
            plt.plot(np.array([centroids[a][c1], centroids[b][c1]]) * e1 / r1 + d1,
                          np.array([centroids[a][c2], centroids[b][c2]]) * e2 / r2 + d2, **lkw)

            for node in G.nodes():
                plt.plot(centroids[node][c1] * e1 / r1 + d1, centroids[node][c2] * e2 / r2 +d2,
                         marker="o", color="black", markersize=10, alpha=0.75)
                plt.text(centroids[node][c1] * e1 / r1 + d1,
                         centroids[node][c2] * e2 / r2 + d2, str(node), **tkw)

    def plot_gradient(self, scalar_field, gx, gy, gz, cell_number, quiver_stepsize=5, #maybe call r sth. like "stepsize"?
                      direction="y", plot_scalar = True, *args, **kwargs): #include plot data?
        """
            Plot the gradient of the scalar field in a given direction.

            Args:
                geo_data (gempy.DataManagement.InputData): Input data of the model
                scalar_field(numpy.array): scalar field to plot with the gradient
                gx(numpy.array): gradient in x-direction
                gy(numpy.array): gradient in y-direction
                gz(numpy.array): gradient in z-direction
                cell_number(int): position of the array to plot
                quiver_stepsize(int): step size between arrows to indicate gradient
                direction(str): xyz. Caartesian direction to be plotted
                plot_scalar(bool): boolean to plot scalar field
                **kwargs: plt.contour kwargs

            Returns:
                None
        """
        if direction == "y":
            if plot_scalar:
                self.plot_scalar_field(scalar_field, cell_number, direction=direction, plot_data=False)
            U = gx.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize,
                 cell_number, ::quiver_stepsize].T
            V = gz.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize,
                 cell_number, ::quiver_stepsize].T
            plt.quiver(self.model.grid.values[:, 0].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, cell_number, ::quiver_stepsize].T,
                   self.model.grid.values[:, 2].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, cell_number, ::quiver_stepsize].T, U, V, pivot="tail",
                   color='blue', alpha=.6)
        elif direction == "x":
            if plot_scalar:
                self.plot_scalar_field(scalar_field, cell_number, direction=direction, plot_data=False)
            U = gy.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize, ::quiver_stepsize].T
            V = gz.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize, ::quiver_stepsize].T
            plt.quiver(self.model.grid.values[:, 1].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1],
                                                            self.model.grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize,  ::quiver_stepsize].T,
                       self.model.grid.values[:, 2].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1],
                                                            self.model.grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize,  ::quiver_stepsize].T, U, V,
                       pivot="tail",
                       color='blue', alpha=.6)
        elif direction== "z":
            if plot_scalar:
                self.plot_scalar_field(scalar_field, cell_number, direction=direction, plot_data=False)
            U = gx.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize, cell_number].T
            V = gy.reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1], self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize, cell_number].T
            plt.quiver(self.model.grid.values[:, 0].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1],
                                                            self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize, cell_number].T,
                       self.model.grid.values[:, 1].reshape(self.model.grid.regular_grid.resolution[0], self.model.grid.regular_grid.resolution[1],
                                                            self.model.grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize, cell_number].T, U, V,
                       pivot="tail",
                       color='blue', alpha=.6)
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")

    def plot_stereonet(self, litho=None, planes=True, poles=True, single_plots=False,
                       show_density=False):
        '''
        Plot an equal-area projection of the orientations dataframe using mplstereonet.

        Args:
            geo_model (gempy.DataManagement.InputData): Input data of the model
            series_only: To select whether a stereonet is plotted per series or per formation
            litho: selection of formation or series names, as list. If None, all are plotted
            planes: If True, azimuth and dip are plotted as great circles
            poles: If True, pole points (plane normal vectors) of azimuth and dip are plotted
            single_plots: If True, each formation is plotted in a single stereonet
            show_density: If True, density contour plot around the pole points is shown

        Returns:
            None
        '''

        try:
            import mplstereonet
        except ImportError:
            warnings.warn('mplstereonet package is not installed. No stereographic projection available.')

        from collections import OrderedDict
        import pandas as pn

        if litho is None:
            litho = self.model.orientations.df['surface'].unique()

        if single_plots is False:
            fig, ax = mplstereonet.subplots(figsize=(5, 5))
            df_sub2 = pn.DataFrame()
            for i in litho:
                df_sub2 = df_sub2.append(self.model.orientations.df[self.model.orientations.df['surface'] == i])

        for formation in litho:
            if single_plots:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection='stereonet')
                ax.set_title(formation, y=1.1)

            #if series_only:
                #df_sub = self.model.orientations.df[self.model.orientations.df['series'] == formation]
            #else:
            df_sub = self.model.orientations.df[self.model.orientations.df['surface'] == formation]

            if poles:
                ax.pole(df_sub['azimuth'] - 90, df_sub['dip'], marker='o', markersize=7,
                        markerfacecolor=self._color_lot[formation],
                        markeredgewidth=1.1, markeredgecolor='gray', label=formation + ': ' + 'pole point')
            if planes:
                ax.plane(df_sub['azimuth'] - 90, df_sub['dip'], color=self._color_lot[formation],
                         linewidth=1.5, label=formation + ': ' + 'azimuth/dip')
            if show_density:
                if single_plots:
                    ax.density_contourf(df_sub['azimuth'] - 90, df_sub['dip'],
                                        measurement='poles', cmap='viridis', alpha=.5)
                else:
                    ax.density_contourf(df_sub2['azimuth'] - 90, df_sub2['dip'], measurement='poles', cmap='viridis',
                                        alpha=.5)

            fig.subplots_adjust(top=0.8)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.9, 1.1))
            ax.grid(True, color='black', alpha=0.25)

class PlotSolution:
    # Todo plot data filtered for sections and map
    def __init__(self, model):

        self.model = model

        self._color_lot = dict(zip(self.model.surfaces.df['surface'], self.model.surfaces.df['color']))
        self._cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))
        self._norm = mcolors.Normalize(vmin=0.5, vmax=len(self._cmap.colors) + 0.5)

    def plot_map(self, solution: Solution = None, contour_lines=True, show_faults=True, show_data=False):
        if solution is not None:
            assert solution.geological_map is not None, 'Geological map not computed. Activate the topography grid.'
        else:
            solution = self.model.solutions
        geomap = solution.geological_map.reshape(self.model.grid.topography.values_3D[:, :, 2].shape)
        if show_data:
            self.plot_data(direction='z')
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
        im = plt.imshow(geomap, origin='lower', extent=self.model.grid.topography.extent, cmap=self._cmap,
                        norm=self._norm)
        if contour_lines == True and show_data == False:
            CS = ax.contour(self.model.grid.topography.values_3D[:, :, 2], cmap='Greys', linestyles='solid',

                            extent=self.model.grid.topography.extent)
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            plothelp.add_colorbar(im=im, label='elevation [m]', cs=CS, aspect=35)
        if show_faults == True:
            self.extract_section_fault_lines('topography')
        plt.title("Geological map", fontsize=15)
        plt.xlabel('X')
        plt.ylabel('Y')

    def extract_section_fault_lines(self, section_name=None, axes=None):
        # Todo merge this with extract fault lines
        faults = list(self.model.faults.df[self.model.faults.df['isFault'] == True].index)
        if section_name == 'topography':
            shape = self.model.grid.topography.resolution
            a = self.model.solutions.geological_map_scalfield
        else:
            l0, l1 = self.model.grid.sections.get_section_args(section_name)
            j = np.where(self.model.grid.sections.names == section_name)[0][0]
            shape = [self.model.grid.sections.resolution[j][0], self.model.grid.sections.resolution[j][1]]
            a = self.model.solutions.sections_scalfield[:, l0:l1]
        for fault in faults:
            f_id = int(self.model.series.df.loc[fault, 'order_series']) - 1
            block = a[f_id]
            level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
                self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]
            if section_name == 'topography':
                plt.contour(block.reshape(shape), 0, levels=level, colors=self._cmap.colors[f_id],
                            linestyles='solid',
                            extent=self.model.grid.topography.extent)
            else:
                if axes is not None:
                    axes.contour(block.reshape(shape).T, 0, levels=level, colors=self._cmap.colors[f_id],
                                 linestyles='solid', origin='lower',
                                 extent=[0, self.model.grid.sections.dist[j],
                                         self.model.grid.regular_grid.extent[4],
                                         self.model.grid.regular_grid.extent[5]])
                else:
                    plt.contour(block.reshape(shape).T, 0, levels=level, colors=self._cmap.colors[f_id],
                                linestyles='solid', origin='lower',
                                extent=[0, self.model.grid.sections.dist[j],
                                        self.model.grid.regular_grid.extent[4],
                                        self.model.grid.regular_grid.extent[5]])

    def plot_sections(self, show_traces=True, show_data=False, section_names=None, show_faults=True, show_topo=True,
                      figsize=(12, 12)):
        assert self.model.solutions.sections is not None, 'no sections for plotting defined'
        if self.model.grid.topography is None:
            show_topo = False
        if section_names is not None:
            if type(section_names) == list:
                section_names = np.array(section_names)
        else:
            section_names = self.model.grid.sections.names
        if show_traces:
            self.plot_section_traces(show_data=show_data, section_names=section_names, contour_lines=False)
        shapes = self.model.grid.sections.resolution
        fig, axes = plt.subplots(nrows=len(section_names), ncols=1, figsize=figsize)
        for i, section in enumerate(section_names):
            j = np.where(self.model.grid.sections.names == section)[0][0]
            l0, l1 = self.model.grid.sections.get_section_args(section)
            if len(section_names) == 1:
                if show_faults:
                    self.extract_section_fault_lines(section, axes)
                if show_topo:
                    xy = self.make_topography_overlay_4_sections(j)
                    axes.fill(xy[:, 0], xy[:, 1], 'k', zorder=10)

                axes.imshow(self.model.solutions.sections[0][l0:l1].reshape(shapes[j][0], shapes[j][1]).T,
                            origin='lower',
                            cmap=self._cmap, norm=self._norm, extent=[0, self.model.grid.sections.dist[j],
                                                                      self.model.grid.regular_grid.extent[4],
                                                                      self.model.grid.regular_grid.extent[5]])

                labels, axname = self._make_section_xylabels(section, len(axes.get_xticklabels()) - 2)
                pos_list = np.linspace(0, self.model.grid.sections.dist[j], len(labels))
                axes.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
                axes.xaxis.set_major_formatter(FixedFormatter((labels)))
                axes.set(title=self.model.grid.sections.names[j], xlabel=axname, ylabel='Z')

            else:
                if show_faults:
                    self.extract_section_fault_lines(section, axes[i])
                if show_topo:
                    xy = self.make_topography_overlay_4_sections(j)
                    axes[i].fill(xy[:, 0], xy[:, 1], 'k', zorder=10)
                axes[i].imshow(self.model.solutions.sections[0][l0:l1].reshape(shapes[j][0], shapes[j][1]).T,
                               origin='lower',
                               cmap=self._cmap, norm=self._norm, extent=[0, self.model.grid.sections.dist[j],
                                                                         self.model.grid.regular_grid.extent[4],
                                                                         self.model.grid.regular_grid.extent[5]])

                labels, axname = self._make_section_xylabels(section, len(axes[i].get_xticklabels()) - 2)
                pos_list = np.linspace(0, self.model.grid.sections.dist[j], len(labels))
                axes[i].xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
                axes[i].xaxis.set_major_formatter(FixedFormatter((labels)))
                axes[i].set(title=self.model.grid.sections.names[j], xlabel=axname, ylabel='Z')
        fig.tight_layout()

    def _slice_topo_4_sections(self, p1, p2):
        xy = self.model.grid.sections.calculate_line_coordinates_2points(p1, p2,
                                                                         self.model.grid.topography.resolution[0],
                                                                         self.model.grid.topography.resolution[1])
        z = self.model.grid.topography.interpolate_zvals_at_xy(xy)
        return xy[:, 0], xy[:, 1], z

    def make_topography_overlay_4_sections(self, j):
        startend = list(self.model.grid.sections.section_dict.values())[j]
        p1, p2 = startend[0], startend[1]
        x, y, z = self._slice_topo_4_sections(p1, p2)
        pseudo_x = np.linspace(0, self.model.grid.sections.dist[j][0], z.shape[0])
        a = np.vstack((pseudo_x, z)).T
        a = np.append(a,
                      ([self.model.grid.sections.dist[j][0], a[:, 1][-1]],
                       [self.model.grid.sections.dist[j][0], self.model.grid.regular_grid.extent[5]],
                       [0, self.model.grid.regular_grid.extent[5]],
                       [0, a[:, 1][0]]))
        return a.reshape(-1, 2)

    def _make_section_xylabels(self, section_name, n=5):
        if n > 5:
            n = 3  # todo I don't know why but sometimes it wants to make a lot of xticks
        j = np.where(self.model.grid.sections.names == section_name)[0][0]
        startend = list(self.model.grid.sections.section_dict.values())[j]
        p1, p2 = startend[0], startend[1]
        xy = self.model.grid.sections.calculate_line_coordinates_2points(p1, p2, n, n)
        if len(np.unique(xy[:, 0])) == 1:
            labels = xy[:, 1].astype(int)
            axname = 'Y'
        elif len(np.unique(xy[:, 1])) == 1:
            labels = xy[:, 0].astype(int)
            axname = 'X'
        else:
            labels = [str(xy[:, 0].astype(int)[i]) + ',\n' + str(xy[:, 1].astype(int)[i]) for i in
                      range(xy[:, 0].shape[0])]
            axname = 'X,Y'
        return labels, axname

    def plot_section_traces(self, show_data=False, section_names=None, contour_lines=True):
        if section_names is None:
            section_names = self.model.grid.sections.names

        if self.model.solutions.geological_map is not None:
            self.plot_map(self.model.solutions, contour_lines=contour_lines, show_data=show_data)
        elif self.model.solutions.lith_block.shape[0] != 0:
            pass
            # self.plot_block_section(self.model.solutions, cell_number=self.model.grid.regular_grid.resolution[2] - 1,
            # direction='z', show_faults=False, show_topo=False, show_data=show_data)
            # plt.title('Section traces, z direction')
        else:
            # fig = plt.figure()
            # plt.title('Section traces, z direction')
            if show_data:
                self.plot_data('z', 'all')
        for section in section_names:
            j = np.where(self.model.grid.sections.names == section)[0][0]
            plt.plot([self.model.grid.sections.points[j][0][0], self.model.grid.sections.points[j][1][0]],
                     [self.model.grid.sections.points[j][0][1], self.model.grid.sections.points[j][1][1]],
                     label=section, linestyle='--')

            plt.xlim(self.model.grid.regular_grid.extent[:2])
            plt.ylim(self.model.grid.regular_grid.extent[2:4])
            # plt.set_aspect(np.diff(geo_model.grid.regular_grid.extent[:2])/np.diff(geo_model.grid.regular_grid.extent[2:4]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_data(self, direction="y", data_type='all', series="all", legend_font_size=10, ve=1, **kwargs):
        """
        Plot the projecton of the raw data (surface_points and orientations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            data_type (str): type of data to plot. 'all', 'surface_points' or 'orientations'
            series(str): series to plot
            ve(float): Vertical exageration
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

        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        # apply vertical exageration
        if direction == 'x' or direction == 'y':
            aspect /= ve

        if aspect < 1:
            min_axis = 'width'
        else:
            min_axis = 'height'
        if series == "all":
            series_to_plot_i = self.model.surface_points.df[self.model.surface_points.df["series"].
                isin(self.model.series.df.index.values)]
            series_to_plot_f = self.model.orientations.df[self.model.orientations.df["series"].
                isin(self.model.series.df.index.values)]

        else:

            series_to_plot_i = self.model.surface_points[self.model.surface_points.df["series"] == series]
            series_to_plot_f = self.model.orientations[self.model.orientations.df["series"] == series]

        if data_type == 'all':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="surface",
                           # scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette=self._color_lot,
                           **kwargs)

            p.axes[0, 0].set_ylim(extent[2], extent[3])
            p.axes[0, 0].set_xlim(extent[0], extent[1])

            # Plotting orientations
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=10)

        if data_type == 'surface_points':
            p = sns.lmplot(x, y,
                           data=series_to_plot_i,
                           fit_reg=False,
                           aspect=aspect,
                           hue="surface",
                           # scatter_kws=scatter_kws,
                           legend=False,
                           legend_out=False,
                           palette=self._color_lot,
                           **kwargs)
            p.axes[0, 0].set_ylim(extent[2], extent[3])
            p.axes[0, 0].set_xlim(extent[0], extent[1])

        if data_type == 'orientations':
            plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                       series_to_plot_f[Gx], series_to_plot_f[Gy],
                       pivot="tail", scale_units=min_axis, scale=15)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions

        """
        _a, _b, _c = (slice(0, self.model.grid.regular_grid.resolution[0]),
                      slice(0, self.model.grid.regular_grid.resolution[1]),
                      slice(0, self.model.grid.regular_grid.resolution[2]))
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self.model.grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy




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

