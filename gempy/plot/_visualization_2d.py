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
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedFormatter, FixedLocator
import seaborn as sns
from os import path
import sys

# This is for sphenix to find the packages
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from gempy.core.solution import Solution
from  gempy.plot import helpers as plothelp
sns.set_context('talk')
plt.style.use(['seaborn-white', 'seaborn-talk'])
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as mpatches



class PlotData2D:
    def __init__(self, model):
        self.model = model
        self._color_lot = dict(zip(self.model._surfaces.df['surface'], self.model._surfaces.df['color']))
        self._cmap = mcolors.ListedColormap(list(self.model._surfaces.df['color']))
        self._norm = mcolors.Normalize(vmin=0.5, vmax=len(self._cmap.colors) + 0.5)

    def get_plot_data(self, series="all", at="everywhere", direction=None, cell_number=None, radius='default',
                      show_all_data=False):

        if radius == 'default':
            radius = None
        else:
            if not isinstance(radius, int):
                raise AttributeError('You need to pass a number (in model extent) for the radius to take more '
                                     'or less data into account.')

        if series == "all":
            series_to_plot_i = self.model._surface_points.df[self.model._surface_points.df["series"].
                isin(self.model._stack.df.index.values)]
            series_to_plot_f = self.model._orientations.df[self.model._orientations.df["series"].
                isin(self.model._stack.df.index.values)]

        else:
            series_to_plot_i = self.model._surface_points.df[self.model._surface_points.df["series"] == series]
            series_to_plot_f = self.model._orientations.df[self.model._orientations.df["series"] == series]

        if show_all_data:
            at = 'everywhere'
        if type(at) == str:
            if at == 'topography':
                mask_surfpoints, mask_orient = self.get_mask_surface_data(radius=radius)
            elif at == 'block_section':
                mask_surfpoints, mask_orient = self.get_mask_block_section(cell_number=cell_number, direction=direction,
                                                                           radius=radius)
            elif at == 'everywhere':
                mask_surfpoints = np.ones(series_to_plot_i.shape[0], dtype=bool)
                mask_orient = np.ones(series_to_plot_f.shape[0], dtype=bool)
            else:  # see if it is a section name
                try:
                    j = np.where(self.model._grid.sections.names == at)[0][0]
                    mask_surfpoints, mask_orient = self.get_mask_sections(j, radius=radius)

                except:
                    raise AttributeError  # 'must be topography, a section name or block_section'

        elif type(at) == list:  # should be a list of section names but must be asserted
            try:
                mask_surfpoints = np.zeros(series_to_plot_i.shape[0], dtype=bool)
                mask_orient = np.zeros(series_to_plot_f.shape[0], dtype=bool)
                for i in at:
                    j = np.where(self.model._grid.sections.names == i)[0][0]
                    mask_surfpoints_i, mask_orient_i = self.get_mask_sections(j, radius=radius)
                    mask_surfpoints[mask_surfpoints_i] = True
                    mask_orient[mask_orient_i] = True
            except AttributeError:
                print('must be topography, a section name or block_section')
        else:
            print('problem')

        return series_to_plot_i[mask_surfpoints], series_to_plot_f[mask_orient]

    def plot_data(self, cell_number=2, direction="y", data_type='all', series="all", show_legend=True, ve=1,
                  at='everywhere', radius='default', show_all_data=False, **kwargs):
        """
        Plot the projecton of the raw data (surface_points and orientations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            data_type (str): type of data to plot. 'all', 'surface_points' or 'orientations'
            series(str): series to plot
            ve(float): Vertical exageration
            show_all_data:
            at:
            **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

        Returns:
            Data plot

        """

        if 'scatter_kws' not in kwargs:
            kwargs['scatter_kws'] = {"marker": "o",
                                     "s": 100,
                                     "edgecolors": "black",
                                     "linewidths": 1}

        x, y, Gx, Gy = self._slice(direction)[4:]
        extent = self._slice(direction)[3]

        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        # apply vertical exageration
        if direction == 'x' or direction == 'y':
            aspect /= ve

        min_axis = 'width' if aspect < 1 else 'height'

        plot_surfpoints, plot_orient = self.get_plot_data(series=series, at=at, direction=direction,
                                                          radius=radius, cell_number=cell_number,
                                                          show_all_data=show_all_data)

        if data_type == 'all':
            self._plot_surface_points(x, y, plot_surfpoints, aspect, extent, kwargs)
            self._plot_orientations(x, y, Gx, Gy, plot_orient, min_axis, extent, False)

        if data_type == 'surface_points':
            self._plot_surface_points(x, y, plot_surfpoints, aspect, extent, kwargs)

        if data_type == 'orientations':
            self._plot_orientations(x, y, Gx, Gy, plot_orient, min_axis, extent, True, aspect)

        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.xlabel(x)
        plt.ylabel(y)

    def plot_section_data(self, section_name, show_all_data=False, radius='default', **kwargs):
        if show_all_data:
            at = 'everywhere'
        else:
            at = section_name
        plot_surfpoints, plot_orient = self.get_plot_data(at=at, radius=radius)
        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        # convert the data to the new coord system
        # define x,y, Gx, Gy #orientations is difficult?
        # direction is to fix plot to x and z column, but we replace x column with projected values
        extent = [0, self.model._grid.sections.dist[j][0],
                  self.model._grid.regular_grid.extent[4], self.model._grid.regular_grid.extent[5]]

        a = self.model._grid.sections.points[j][0]  # startpoint = 0, we need distance to that point
        bs_i = plot_surfpoints[['X', 'Y']].values
        bs_o = plot_orient[['X', 'Y']].values
        new_x_points_i = np.linalg.norm(bs_i - a, axis=1)
        new_x_points_o = np.linalg.norm(bs_o - a, axis=1)
        plot_surfpoints['X'] = new_x_points_i
        plot_orient['X'] = new_x_points_o
        y = 'Z'
        x = 'X'
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        kwargs['scatter_kws'] = {"marker": "o",
                                 "s": 100,
                                 "edgecolors": "black",
                                 "linewidths": 1}
        # print(plot_surfpoints)
        self._plot_surface_points(x, y, plot_surfpoints, aspect, extent, kwargs)
        Gx = 'G_x'
        Gy = 'G_z'
        min_axis = 'width' if aspect < 1 else 'height'
        self._plot_orientations(x, y, Gx, Gy, plot_orient, min_axis, extent, False, aspect)
        warnings.warn('the orientations are not converted to apparent dip.')

    def _plot_surface_points(self, x, y, series_to_plot_i, aspect, extent, kwargs):
        if series_to_plot_i.shape[0] != 0:
            # size = fig.get_size_inches() * fig.dpi
            # print(size)
            # print(aspect)
            try:
                p = sns.FacetGrid(series_to_plot_i, hue="surface",
                                  palette=self._color_lot,
                                  ylim=[extent[2], extent[3]],
                                  xlim=[extent[0], extent[1]],
                                  legend_out=False,
                                  aspect=aspect,
                                  height=6)
            except KeyError:  # for kriging dataframes
                p = sns.FacetGrid(series_to_plot_i, hue=None,
                                  palette='k',
                                  ylim=[extent[2], extent[3]],
                                  xlim=[extent[0], extent[1]],
                                  legend_out=False,
                                  aspect=aspect,
                                  height=6)

            p.map(plt.scatter, x, y,
                  **kwargs['scatter_kws'],
                  zorder=10)
        else:
            self._show_legend = True

    def _plot_orientations(self, x, y, Gx, Gy, series_to_plot_f, min_axis, extent, p, aspect=None, ax=None):
        if series_to_plot_f.shape[0] != 0:
            # print('hello')
            if p is False:
                # size = fig.get_size_inches() * fig.dpi
                # print('before plot orient', size)
                surflist = list(series_to_plot_f['surface'].unique())
                for surface in surflist:
                    to_plot = series_to_plot_f[series_to_plot_f['surface'] == surface]
                    plt.quiver(to_plot[x], to_plot[y],
                               to_plot[Gx], to_plot[Gy],
                               pivot="tail", scale_units=min_axis, scale=30, color=self._color_lot[surface],
                               edgecolor='k', headwidth=8, linewidths=1)
                    # ax.Axes.set_ylim([extent[2], extent[3]])
                    # ax.Axes.set_xlim([extent[0], extent[1]])
                # fig = plt.gcf()
                # fig.set_size_inches(20,10)
                # if aspect is not None:
                # ax = plt.gca()
                # ax.set_aspect(aspect)

            else:
                p = sns.FacetGrid(series_to_plot_f, hue="surface",
                                  palette=self._color_lot,
                                  ylim=[extent[2], extent[3]],
                                  xlim=[extent[0], extent[1]],
                                  legend_out=False,
                                  aspect=aspect,
                                  height=6)
                p.map(plt.quiver, x, y, Gx, Gy, pivot="tail", scale_units=min_axis, scale=10, edgecolor='k',
                      headwidth=4, linewidths=1)
        else:
            # print('no orient')
            pass

        # size = fig.get_size_inches() * fig.dpi
        # print('after plot_orientations', size)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions

        """
        _a, _b, _c = (slice(0, self.model._grid.regular_grid.resolution[0]),
                      slice(0, self.model._grid.regular_grid.resolution[1]),
                      slice(0, self.model._grid.regular_grid.resolution[2]))

        if direction == "x":
            _a, x, y, Gx, Gy = cell_number, "Y", "Z", "G_y", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            _b, x, y, Gx, Gy = cell_number, "X", "Z", "G_x", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            _c, x, y, Gx, Gy = cell_number, "X", "Y", "G_x", "G_y"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def _slice2D(self, cell_number, direction):
        if direction == 'x':
            _slice = np.s_[cell_number, :, :]
            extent = self.model._grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == 'y':
            _slice = np.s_[:, cell_number, :]
            extent = self.model._grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == 'z':
            _slice = np.s_[:, :, cell_number]
            extent = self.model._grid.regular_grid.extent[[1, 2, 3, 4]]
        else:
            print('not a direction')
        return _slice, extent

    def get_mask_surface_data(self, radius=None):
        points_interf = np.vstack(
            (self.model._surface_points.df['X'].values, self.model._surface_points.df['Y'].values)).T
        points_orient = np.vstack((self.model._orientations.df['X'].values, self.model._orientations.df['Y'].values)).T

        mask_interf = self.get_data_within_extent(points_interf)
        mask_orient = self.get_data_within_extent(points_orient)

        xj = self.model._grid.topography.values_3D[:, :, 0][0, :]
        yj = self.model._grid.topography.values_3D[:, :, 1][:, 0]
        zj = self.model._grid.topography.values_3D[:, :, 2].T

        interpolate = RegularGridInterpolator((xj, yj), zj)

        Z_interf_interp = interpolate(points_interf[mask_interf])
        Z_orient_interp = interpolate(points_orient[mask_orient])

        if radius is None:
            radius = np.diff(zj).max()
        print(radius)

        dist_interf = np.abs(Z_interf_interp - self.model._surface_points.df['Z'].values[mask_interf])
        dist_orient = np.abs(Z_orient_interp - self.model._orientations.df['Z'].values[mask_orient])

        surfmask_interf = dist_interf < radius
        surfmask_orient = dist_orient < radius
        surf_indexes = np.flatnonzero(mask_interf)[surfmask_interf]
        orient_indexes = np.flatnonzero(mask_orient)[surfmask_orient]

        mask_surfpoints = np.zeros(points_interf.shape[0], dtype=bool)
        mask_orient = np.zeros(points_orient.shape[0], dtype=bool)

        mask_surfpoints[surf_indexes] = True
        mask_orient[orient_indexes] = True

        return mask_surfpoints, mask_orient

    def get_mask_block_section(self, cell_number=3, direction='y', radius=None):
        if direction == 'x':
            column = 'X'
            start = self.model._grid.regular_grid.extent[0]
            end = self.model._grid.regular_grid.extent[1]
            r_o_inf = self.model._grid.regular_grid.dx
        elif direction == 'y':
            column = 'Y'
            start = self.model._grid.regular_grid.extent[2]
            end = self.model._grid.regular_grid.extent[3]
            r_o_inf = self.model._grid.regular_grid.dy
        elif direction == 'z':
            column = 'Z'
            start = self.model._grid.regular_grid.extent[4]
            end = self.model._grid.regular_grid.extent[5]
            r_o_inf = self.model._grid.regular_grid.dz
        else:
            raise

        if cell_number < 0:
            cell_number = end + cell_number + 1

        if radius is None:
            radius = r_o_inf
        coord = start + radius * cell_number
        mask_surfpoints = np.abs(self.model._surface_points.df[column].values - coord) < radius
        mask_orient = np.abs(self.model._orientations.df[column].values - coord) < radius
        return mask_surfpoints, mask_orient

    def get_mask_sections(self, j, radius=None):
        points_interf = np.vstack(
            (self.model._surface_points.df['X'].values, self.model._surface_points.df['Y'].values)).T
        points_orient = np.vstack((self.model._orientations.df['X'].values,
                                   self.model._orientations.df['Y'].values)).T
        if radius is None:
            radius = self.model._grid.sections.dist[j] / self.model._grid.sections.resolution[j][0]

        p1, p2 = np.array(self.model._grid.sections.points[j][0]), np.array(self.model._grid.sections.points[j][1])

        d_interf = np.abs(np.cross(p2 - p1, points_interf - p1) / np.linalg.norm(p2 - p1))
        d_orient = np.abs(np.cross(p2 - p1, points_orient - p1) / np.linalg.norm(p2 - p1))

        mask_surfpoints = d_interf < radius
        mask_orient = d_orient < radius
        return mask_surfpoints, mask_orient

    def get_data_within_extent(self, pts, ext=None):
        # ext = geo_model.grid.regular_grid.extent[:4]
        if ext is None:
            #extent must be the one of the topography (is half cell size smaller than regualar grid) because of
            #interpolation function
            ext = np.array([self.model._grid.topography.values_3D[:, :, 0][0, :][[0, -1]],
                            self.model._grid.topography.values_3D[:, :, 1][:, 0][[0, -1]]]).ravel()
        mask_x = np.logical_and(pts[:, 0] >= ext[0], pts[:, 0] <= ext[1])
        mask_y = np.logical_and(pts[:, 1] >= ext[2], pts[:, 1] <= ext[3])
        return np.logical_and(mask_x, mask_y)

    def plot_section_traces(self, section_names=None, contour_lines=True, show_data=True, show_all_data=False):
        if section_names is None:
            section_names = list(self.model._grid.sections.names)

        if show_data:
            if not show_all_data:
                self.plot_data(direction='z', at=section_names)
            else:
                self.plot_data(direction='z', at='everywhere')

        for section in section_names:
            j = np.where(self.model._grid.sections.names == section)[0][0]
            plt.plot([self.model._grid.sections.points[j][0][0], self.model._grid.sections.points[j][1][0]],
                     [self.model._grid.sections.points[j][0][1], self.model._grid.sections.points[j][1][1]],
                     label=section, linestyle='--')

            plt.xlim(self.model._grid.regular_grid.extent[:2])
            plt.ylim(self.model._grid.regular_grid.extent[2:4])
            # plt.set_aspect(np.diff(geo_model.grid.regular_grid.extent[:2])/np.diff(geo_model.grid.regular_grid.extent[2:4]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_stereonet(self, litho=None, planes=True, poles=True, single_plots=False,
                       show_density=False):
        try:
            import mplstereonet
        except ImportError:
            warnings.warn('mplstereonet package is not installed. No stereographic projection available.')

        from collections import OrderedDict
        import pandas as pn

        if litho is None:
            litho = self.model._orientations.df['surface'].unique()

        if single_plots is False:
            fig, ax = mplstereonet.subplots(figsize=(5, 5))
            df_sub2 = pn.DataFrame()
            for i in litho:
                df_sub2 = df_sub2.append(self.model._orientations.df[self.model._orientations.df['surface'] == i])

        for formation in litho:
            if single_plots:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection='stereonet')
                ax.set_title(formation, y=1.1)

            # if series_only:
            # df_sub = self.model.orientations.df[self.model.orientations.df['series'] == formation]
            # else:
            df_sub = self.model._orientations.df[self.model._orientations.df['surface'] == formation]

            if poles:
                ax.pole(df_sub['azimuth'] - 90, df_sub['dip'], marker='o', markersize=10,
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
            ax._grid(True, color='black', alpha=0.25)


class PlotSolution(PlotData2D):

    def __init__(self, model):
        self.model = model
        # super().__init__(self)
        self._color_lot = dict(zip(self.model._surfaces.df['surface'], self.model._surfaces.df['color']))
        self._cmap = mcolors.ListedColormap(list(self.model._surfaces.df['color']))
        self._norm = mcolors.Normalize(vmin=0.5, vmax=len(self._cmap.colors) + 0.5)
        self._show_legend = False

    def plot_map(self, solution: Solution = None, contour_lines=False, show_data=True,
                 show_all_data=False, show_hillshades: bool = False, figsize=(12, 12), **kwargs):
        """

        Args:
            solution:
            contour_lines:
            show_data:
            show_all_data:
            show_hillshades: Calculate and add hillshading using elevation data
            figsize:
            **kwargs:
                - azdeg: float = azimuth of sun for hillshade
                - altdeg: float = altitude in degrees of sun for hillshade

        """
        azdeg: float = kwargs.get('azdeg', 315.0)
        altdeg: float = kwargs.get('altdeg', 45.0)
        if solution is None:
            solution = self.model.solutions

        if solution.geological_map is None:
            raise AttributeError('Geological map not computed. Activate the topography grid.')

        try:
            geomap = solution.geological_map[0].reshape(self.model._grid.topography.values_3D[:, :, 2].shape)
        except AttributeError:
            warnings.warn('Geological map not computed. Activate the topography grid.')

        if show_data == True and show_hillshades == True:
            self.plot_data(direction='z', at='topography', show_all_data=show_all_data)
            ls = LightSource(azdeg=azdeg, altdeg=altdeg)
            hillshade_topography = ls.hillshade(self.model.grid.topography.values_3D[:, :, 2])
            plt.imshow(hillshade_topography, origin='lower', extent=self.model.grid.topography.extent, alpha=0.75)
        elif show_data == True and show_hillshades == False:
            self.plot_data(direction='z', at='topography', show_all_data=show_all_data)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        im = plt.imshow(geomap, origin='lower', extent=self.model._grid.topography.extent, cmap=self._cmap,
                        norm=self._norm, zorder=-100)

        if contour_lines == True and show_data == False:
            CS = ax.contour(self.model._grid.topography.values_3D[:, :, 2], cmap='Greys', linestyles='solid',
                            extent=self.model._grid.topography.extent, zlevel=200)
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            plothelp.add_colorbar(im=im, label='elevation [m]', cs=CS, aspect=35)

        # self.extract_section_lines('topography')

        plt.title("Geological map", fontsize=15)
        plt.xlabel('X')
        plt.ylabel('Y')

    def extract_section_lines(self, section_name=None, axes=None, zorder=2, faults_only=False):
        # Todo merge this with extract fault lines
        faults = list(self.model._faults.df[self.model._faults.df['isFault'] == True].index)
        if section_name == 'topography':
            shape = self.model._grid.topography.resolution
            a = self.model.solutions.geological_map[1]
            extent = self.model._grid.topography.extent
        else:
            l0, l1 = self.model._grid.sections.get_section_args(section_name)
            j = np.where(self.model._grid.sections.names == section_name)[0][0]
            shape = [self.model._grid.sections.resolution[j][0], self.model._grid.sections.resolution[j][1]]
            a = self.model.solutions.sections[1][:, l0:l1]
            extent = [0, self.model._grid.sections.dist[j],
                      self.model._grid.regular_grid.extent[4],
                      self.model._grid.regular_grid.extent[5]]

        counter = a.shape[0]
        if faults_only:
            counters = np.arange(0, len(faults), 1)
        else:
            counters = np.arange(0, counter, 1)

        c_id = 0  # color id startpoint
        for f_id in counters:
            block = a[f_id]
            level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
                self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]

            levels = np.insert(level, 0, block.max())
            c_id2 = c_id + len(level)
            if f_id == counters.max():
                levels = np.insert(levels, level.shape[0], block.min())
                c_id2 = c_id + len(levels)  # color id endpoint
            if section_name == 'topography':
                block = block.reshape(shape)
            else:
                block = block.reshape(shape).T

            zorder = zorder - (f_id + len(level))
            if axes is None:
                axes = plt.gca()

            if f_id >= len(faults):
                axes.contourf(block, 0, levels=np.sort(levels), colors=self._cmap.colors[c_id:c_id2][::-1],
                              linestyles='solid', origin='lower',
                              extent=extent, zorder=zorder)
            else:
                axes.contour(block, 0, levels=np.sort(levels), colors=self._cmap.colors[c_id:c_id2][0],
                             linestyles='solid', origin='lower',
                             extent=extent, zorder=zorder)
            c_id += len(level)

    def extract_fault_lines(self, cell_number=25, direction='y'):
        faults = list(self.model._faults.df[self.model._faults.df['isFault'] == True].index)
        if len(faults) == 0:
            pass
        else:
            _slice, extent = self._slice2D(cell_number, direction)
            for fault in faults:
                f_id = int(self.model._stack.df.loc[fault, 'order_series']) - 1
                block = self.model.solutions.scalar_field_matrix[f_id]
                level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
                    self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]
                level.sort()
                plt.contour(block.reshape(self.model._grid.regular_grid.resolution)[_slice].T, 0, extent=extent,
                            levels=level,
                            colors=self._cmap.colors[f_id], linestyles='solid')

    def plot_section_by_name(self, section_name, show_data=True, show_faults=True, show_topo=True,
                             show_all_data=False, contourplot=True, radius='default', **kwargs):

        if self.model.solutions.sections is None:
            raise AttributeError('no sections for plotting defined')
        if section_name not in self.model._grid.sections.names:
            raise AttributeError(f'Section "{section_name}" is not defined. '
                                 f'Available sections for plotting: {self.model._grid.sections.names}')

        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        l0, l1 = self.model._grid.sections.get_section_args(section_name)
        shape = self.model._grid.sections.resolution[j]

        image = self.model.solutions.sections[0][0][l0:l1].reshape(shape[0], shape[1]).T
        extent = [0, self.model._grid.sections.dist[j][0],
                  self.model._grid.regular_grid.extent[4], self.model._grid.regular_grid.extent[5]]

        if show_data:
            self.plot_section_data(section_name=section_name, show_all_data=show_all_data, radius=radius)

        axes = plt.gca()
        axes.imshow(image, origin='lower', zorder=-100,
                    cmap=self._cmap, norm=self._norm, extent=extent)
        if show_faults and not contourplot:
            self.extract_section_lines(section_name, axes, faults_only=True)
        else:
            self.extract_section_lines(section_name, axes, faults_only=False)
        if show_topo:
            if self.model._grid.topography is not None:
                alpha = kwargs.get('alpha', 1)
                xy = self.make_topography_overlay_4_sections(j)
                axes.fill(xy[:, 0], xy[:, 1], 'k', zorder=10, alpha=alpha)

        labels, axname = self._make_section_xylabels(section_name, len(axes.get_xticklabels()) - 1)
        pos_list = np.linspace(0, self.model._grid.sections.dist[j], len(labels))
        axes.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
        axes.xaxis.set_major_formatter(FixedFormatter((labels)))
        axes.set(title=self.model._grid.sections.names[j], xlabel=axname, ylabel='Z')

    def plot_all_sections(self, show_data=False, section_names=None, show_topo=True,
                          figsize=(12, 12)):
        if self.model.solutions.sections is None:
            raise AttributeError('no sections for plotting defined')
        if self.model._grid.topography is None:
            show_topo = False
        if section_names is not None:
            if isinstance(section_names, list):
                section_names = np.array(section_names)
        else:
            section_names = self.model._grid.sections.names

        shapes = self.model._grid.sections.resolution
        fig, axes = plt.subplots(nrows=len(section_names), ncols=1, figsize=figsize)
        for i, section in enumerate(section_names):
            j = np.where(self.model._grid.sections.names == section)[0][0]
            l0, l1 = self.model._grid.sections.get_section_args(section)

            self.extract_section_lines(section, axes[i], faults_only=False)

            if show_topo:
                xy = self.make_topography_overlay_4_sections(j)
                axes[i].fill(xy[:, 0], xy[:, 1], 'k', zorder=10)

            # if show_data:
            #    section = str(section)
            #    print(section)
            #    self.plot_section_data(section_name=section)

            axes[i].imshow(self.model.solutions.sections[0][0][l0:l1].reshape(shapes[j][0], shapes[j][1]).T,
                           origin='lower', zorder=-100,
                           cmap=self._cmap, norm=self._norm, extent=[0, self.model._grid.sections.dist[j],
                                                                     self.model._grid.regular_grid.extent[4],
                                                                     self.model._grid.regular_grid.extent[5]])

            labels, axname = self._make_section_xylabels(section, len(axes[i].get_xticklabels()) - 1)
            pos_list = np.linspace(0, self.model._grid.sections.dist[j], len(labels))
            axes[i].xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
            axes[i].xaxis.set_major_formatter(FixedFormatter((labels)))
            axes[i].set(title=self.model._grid.sections.names[j], xlabel=axname, ylabel='Z')

        fig.tight_layout()

    def plot_section_scalarfield(self, section_name, sn, levels=50, show_faults=True, show_topo=True, lithback=True):
        if self.model.solutions.sections is None:
            raise AttributeError('no sections for plotting defined')
        if self.model._grid.topography is None:
            show_topo = False
        shapes = self.model._grid.sections.resolution
        fig = plt.figure(figsize=(16, 10))
        axes = fig.add_subplot(1, 1, 1)
        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        l0, l1 = self.model._grid.sections.get_section_args(section_name)
        if show_faults:
            self.extract_section_fault_lines(section_name, zorder=9)

        if show_topo:
            xy = self.make_topography_overlay_4_sections(j)
            axes.fill(xy[:, 0], xy[:, 1], 'k', zorder=10)

        axes.contour(self.model.solutions.sections[1][sn][l0:l1].reshape(shapes[j][0], shapes[j][1]).T,
                     # origin='lower',
                     levels=levels, cmap='autumn', extent=[0, self.model._grid.sections.dist[j],
                                                           self.model._grid.regular_grid.extent[4],
                                                           self.model._grid.regular_grid.extent[5]], zorder=8)
        axes.set_aspect('equal')
        if lithback:
            axes.imshow(self.model.solutions.sections[0][0][l0:l1].reshape(shapes[j][0], shapes[j][1]).T,
                        origin='lower',
                        cmap=self._cmap, norm=self._norm, extent=[0, self.model._grid.sections.dist[j],
                                                                  self.model._grid.regular_grid.extent[4],
                                                                  self.model._grid.regular_grid.extent[5]])

        labels, axname = self._make_section_xylabels(section_name, len(axes.get_xticklabels()))
        pos_list = np.linspace(0, self.model._grid.sections.dist[j], len(labels))
        axes.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
        axes.xaxis.set_major_formatter(FixedFormatter((labels)))
        axes.set(title=self.model._grid.sections.names[j], xlabel=axname, ylabel='Z')

    def _slice_topo_4_sections(self, p1, p2, resx, resy):
        xy = self.model._grid.sections.calculate_line_coordinates_2points(np.array(p1),
                                                                          np.array(p2),
                                                                          resx)
        z = self.model._grid.topography.interpolate_zvals_at_xy(xy)
        return xy[:, 0], xy[:, 1], z

    def make_topography_overlay_4_sections(self, j):
        startend = list(self.model._grid.sections.section_dict.values())[j]
        p1, p2 = startend[0], startend[1]
        x, y, z = self._slice_topo_4_sections(p1, p2, self.model._grid.topography.resolution[0],
                                              self.model._grid.topography.resolution[1])
        pseudo_x = np.linspace(0, self.model._grid.sections.dist[j][0], z.shape[0])
        a = np.vstack((pseudo_x, z)).T
        a = np.append(a,
                      ([self.model._grid.sections.dist[j][0], a[:, 1][-1]],
                       [self.model._grid.sections.dist[j][0], self.model._grid.regular_grid.extent[5]],
                       [0, self.model._grid.regular_grid.extent[5]],
                       [0, a[:, 1][0]]))
        return a.reshape(-1, 2)

    def make_topography_overlay_4_blockplot(self, cell_number, direction):
        p1, p2 = self.calculate_p1p2(direction, cell_number)
        resx = self.model._grid.topography.resolution[0]
        resy = self.model._grid.topography.resolution[1]
        print('p1', p1, 'p2', p2)
        x, y, z = self._slice_topo_4_sections(p1, p2, resx, resy)
        if direction == 'x':
            a = np.vstack((y, z)).T
            ext = self.model._grid.regular_grid.extent[[2, 3]]
        elif direction == 'y':
            a = np.vstack((x, z)).T
            ext = self.model._grid.regular_grid.extent[[0, 1]]
        a = np.append(a,
                      ([ext[1], a[:, 1][-1]],
                       [ext[1], self.model._grid.regular_grid.extent[5]],
                       [ext[0], self.model._grid.regular_grid.extent[5]],
                       [ext[0], a[:, 1][0]]))
        line = a.reshape(-1, 2)
        plt.fill(line[:, 0], line[:, 1], color='k')

    def calculate_p1p2(self, direction, cell_number):
        if direction == 'y':
            y = self.model._grid.regular_grid.extent[2] + self.model._grid.regular_grid.dy * cell_number
            p1 = [self.model._grid.regular_grid.extent[0], y]
            p2 = [self.model._grid.regular_grid.extent[1], y]

        elif direction == 'x':
            x = self.model._grid.regular_grid.extent[0] + self.model._grid.regular_grid.dx * cell_number
            p1 = [x, self.model._grid.regular_grid.extent[2]]
            p2 = [x, self.model._grid.regular_grid.extent[3]]

        else:
            raise NotImplementedError
        return p1, p2

    def _make_section_xylabels(self, section_name, n=5):
        if n > 10:
            n = n - 2  # todo I don't know why but sometimes it wants to make a lot of xticks
        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        startend = list(self.model._grid.sections.section_dict.values())[j]
        p1, p2 = startend[0], startend[1]
        xy = self.model._grid.sections.calculate_line_coordinates_2points(p1,
                                                                          p2,
                                                                          n)
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

    def plot_block_section(self, solution: Solution, cell_number: int, block: np.ndarray = None, direction: str = "y",
                           interpolation: str = 'none', show_data: bool = False, show_faults: bool = False,
                           show_topo: bool = False,
                           block_type=None, ve: float = 1, show_legend: bool = True, show_all_data: bool = False,
                           ax=None,
                           **kwargs):
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
            show_legend (bool, optional): Plot or hide legend - only available
                if no data is plotted.
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

        plot_block = _block.reshape(self.model._grid.regular_grid.resolution)
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        if show_data:
            if show_all_data:
                at = 'everywhere'
            else:
                at = 'block_section'
            self.plot_data(cell_number=cell_number, direction=direction, at=at, show_legend=show_legend)

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

        sliced_block = plot_block[_a, _b, _c].T

        imshow_kwargs = kwargs.copy()
        if 'show_grid' in imshow_kwargs:
            imshow_kwargs.pop('show_grid')
        if 'grid_linewidth' in imshow_kwargs:
            imshow_kwargs.pop('grid_linewidth')

        im = plt.imshow(sliced_block,
                        origin="lower",
                        extent=extent_val,
                        interpolation=interpolation,
                        aspect=aspect,
                        **imshow_kwargs)

        if extent_val[3] < extent_val[2]:  # correct vertical orientation of plot
            plt.gca().invert_yaxis()  # if maximum vertical extent negative

        if show_faults:
            self.extract_fault_lines(cell_number, direction)

        if show_topo:
            if self.model._grid.topography is not None:
                if direction == 'z':
                    plt.contour(self.model._grid.topography.values_3D[:, :, 2], extent=extent_val, cmap='Greys')
                else:
                    self.make_topography_overlay_4_blockplot(cell_number=cell_number, direction=direction)

        if self._show_legend and show_legend:
            show_data = False  # to plot legend even when there are no data points in the section
        if not show_data and show_legend:
            patches = [mpatches.Patch(color=color, label=surface) for surface, color in self._color_lot.items()]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if 'show_grid' in kwargs:
            # TODO This only works fine for the y projection
            ax = plt.gca();
            ax.set_xticks(np.linspace(extent_val[0], extent_val[1], sliced_block.shape[0] + 1));
            ax.set_yticks(np.linspace(extent_val[2], extent_val[3], sliced_block.shape[1] + 1));

            grid_linewidth = kwargs.get('grid_linewidth', 1)
            ax._grid(color='w', linestyle='-', linewidth=grid_linewidth)

        plt.xlabel(x)
        plt.ylabel(y)
        return plt.gcf()

    def plot_scalar_field(self, solution, cell_number, series=0, N=20, block=None,
                          direction="y", alpha=0.6, show_data=True, show_all_data=False,
                          *args, **kwargs):
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

        if show_data:
            if show_all_data:
                at = 'everywhere'
            else:
                at = 'block_section'
            self.plot_data(cell_number=cell_number, direction=direction, at=at)

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        plt.contour(scalar_field.reshape(
            self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
            self.model._grid.regular_grid.resolution[2])[_a, _b, _c].T,
                    N,
                    extent=extent_val, *args,
                    **kwargs)

        plt.contourf(scalar_field.reshape(
            self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
            self.model._grid.regular_grid.resolution[2])[_a, _b, _c].T,
                     N,
                     extent=extent_val, alpha=alpha, *args,
                     **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.xlabel(x)
        plt.ylabel(y)

    @staticmethod
    def plot_topo_g(geo_model, edges, centroids, direction="y", scale=False,
                    label_kwargs=None, edge_kwargs=None):
        res = geo_model._grid.regular_grid.resolution
        if direction == "y":
            c1, c2 = (0, 2)
            e1 = geo_model._grid.regular_grid.extent[1] - geo_model._grid.regular_grid.extent[0]
            e2 = geo_model._grid.regular_grid.extent[5] - geo_model._grid.regular_grid.extent[4]
            d1 = geo_model._grid.regular_grid.extent[0]
            d2 = geo_model._grid.regular_grid.extent[4]
            # if len(list(centroids.items())[0][1]) == 2:
            #     c1, c2 = (0, 1)
            r1 = res[0]
            r2 = res[2]
        elif direction == "x":
            c1, c2 = (1, 2)
            e1 = geo_model._grid.regular_grid.extent[3] - geo_model._grid.regular_grid.extent[2]
            e2 = geo_model._grid.regular_grid.extent[5] - geo_model._grid.regular_grid.extent[4]
            d1 = geo_model._grid.regular_grid.extent[2]
            d2 = geo_model._grid.regular_grid.extent[4]
            # if len(list(centroids.items())[0][1]) == 2:
            #     c1, c2 = (0, 1)
            r1 = res[1]
            r2 = res[2]
        elif direction == "z":
            c1, c2 = (0, 1)
            e1 = geo_model._grid.regular_grid.extent[1] - geo_model._grid.regular_grid.extent[0]
            e2 = geo_model._grid.regular_grid.extent[3] - geo_model._grid.regular_grid.extent[2]
            d1 = geo_model._grid.regular_grid.extent[0]
            d2 = geo_model._grid.regular_grid.extent[2]
            # if len(list(centroids.items())[0][1]) == 2:
            #     c1, c2 = (0, 1)
            r1 = res[0]
            r2 = res[1]

        tkw = {
            "color": "white",
            "fontsize": 13,
            "ha": "center",
            "va": "center",
            "weight": "ultralight",
            "family": "monospace",
            "verticalalignment": "center",
            "horizontalalignment": "center",
            "bbox": dict(boxstyle='round', facecolor='black', alpha=1),
        }
        if label_kwargs is not None:
            tkw.update(label_kwargs)

        lkw = {
            "linewidth": 1,
            "color": "black"
        }
        if edge_kwargs is not None:
            lkw.update(edge_kwargs)

        for a, b in edges:
            # plot edges
            x = np.array([centroids[a][c1], centroids[b][c1]])
            y = np.array([centroids[a][c2], centroids[b][c2]])
            if scale:
                x = x * e1 / r1 + d1
                y = y * e2 / r2 + d2
            plt.plot(x, y, **lkw)

        for node in np.unique(list(edges)):
            x = centroids[node][c1]
            y = centroids[node][c2]
            if scale:
                x = x * e1 / r1 + d1
                y = y * e2 / r2 + d2
            plt.text(x, y, str(node), **tkw)

    def plot_gradient(self, scalar_field, gx, gy, gz, cell_number, quiver_stepsize=5,
                      # maybe call r sth. like "stepsize"?
                      direction="y", plot_scalar=True, *args, **kwargs):  # include plot data?
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
            U = gx.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[::quiver_stepsize,
                cell_number, ::quiver_stepsize].T
            V = gz.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[::quiver_stepsize,
                cell_number, ::quiver_stepsize].T
            plt.quiver(self.model._grid.values[:, 0].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                            self.model._grid.regular_grid.resolution[2])[
                       ::quiver_stepsize, cell_number, ::quiver_stepsize].T,
                       self.model._grid.values[:, 2].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                            self.model._grid.regular_grid.resolution[2])[
                       ::quiver_stepsize, cell_number, ::quiver_stepsize].T, U, V, pivot="tail",
                       color='blue', alpha=.6)
        elif direction == "x":
            if plot_scalar:
                self.plot_scalar_field(scalar_field, cell_number, direction=direction, plot_data=False)
            U = gy.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize,
                ::quiver_stepsize].T
            V = gz.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[cell_number, ::quiver_stepsize,
                ::quiver_stepsize].T
            plt.quiver(self.model._grid.values[:, 1].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                             self.model._grid.regular_grid.resolution[2])[cell_number,
                       ::quiver_stepsize, ::quiver_stepsize].T,
                       self.model._grid.values[:, 2].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                             self.model._grid.regular_grid.resolution[2])[cell_number,
                       ::quiver_stepsize, ::quiver_stepsize].T, U, V,
                       pivot="tail",
                       color='blue', alpha=.6)
        elif direction == "z":
            if plot_scalar:
                self.plot_scalar_field(scalar_field, cell_number, direction=direction, plot_data=False)
            U = gx.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize,
                cell_number].T
            V = gy.reshape(self.model._grid.regular_grid.resolution[0], self.model._grid.regular_grid.resolution[1],
                           self.model._grid.regular_grid.resolution[2])[::quiver_stepsize, ::quiver_stepsize,
                cell_number].T
            plt.quiver(self.model._grid.values[:, 0].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                             self.model._grid.regular_grid.resolution[2])[
                       ::quiver_stepsize, ::quiver_stepsize, cell_number].T,
                       self.model._grid.values[:, 1].reshape(self.model._grid.regular_grid.resolution[0],
                                                            self.model._grid.regular_grid.resolution[1],
                                                             self.model._grid.regular_grid.resolution[2])[
                       ::quiver_stepsize, ::quiver_stepsize, cell_number].T, U, V,
                       pivot="tail",
                       color='blue', alpha=.6)
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
