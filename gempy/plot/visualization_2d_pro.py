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

Created on 23/09/2019

@author: Miguel de la Varga, Elisa Heim
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib.gridspec as gridspect


import seaborn as sns
from os import path
import sys
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gempy.core.solution import Solution
# import gempy.plot.helpers as plothelp

sns.set_context('talk')
plt.style.use(['seaborn-white', 'seaborn-talk'])
from scipy.interpolate import RegularGridInterpolator
from arviz.plots.jointplot import _var_names, _scale_fig_size


class Plot2D:
    def __init__(self, model, cmap=None, norm=None):
        self.model = model
        self._color_lot = dict(zip(self.model.surfaces.df['surface'], self.model.surfaces.df['color']))

        if cmap is None:
            self.cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))
        else:
            self.cmap=cmap

        if norm is None:
            self.norm = mcolors.Normalize(vmin=0.5, vmax=len(self.cmap.colors) + 0.5)
        else:
            self.norm = norm

    def _make_section_xylabels(self, section_name, n=5):
        """
        @elisa heim

        Args:
            section_name:
            n:

        Returns:

        """
        if n > 5:
            n = 3  # todo I don't know why but sometimes it wants to make a lot of xticks
        elif n < 0:
            n = 3

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

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions

        """
        _a, _b, _c = (slice(0, self.model.grid.regular_grid.resolution[0]),
                      slice(0, self.model.grid.regular_grid.resolution[1]),
                      slice(0, self.model.grid.regular_grid.resolution[2]))

        if direction == "x":
            _a, x, y, Gx, Gy = cell_number, "Y", "Z", "G_y", "G_z"
            extent_val = self.model.grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            _b, x, y, Gx, Gy = cell_number, "X", "Z", "G_x", "G_z"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            _c, x, y, Gx, Gy = cell_number, "X", "Y", "G_x", "G_y"
            extent_val = self.model.grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def create_figure(self, figsize=None, textsize=None):

        figsize, self.ax_labelsize, _, self.xt_labelsize, self.linewidth, _ = _scale_fig_size(figsize, textsize)
        self.fig, self.axes = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)

        # TODO make grid variable
        self.gs_0 = gridspect.GridSpec(3, 2, figure=self.fig, hspace=.1)

        return self.fig, self.axes, self.gs_0

    def set_section(self, section_name=None, cell_number=None, direction='y', ax=None, ax_pos=111, ve=1):

        if ax is None:
            # TODO
            ax = self.fig.add_subplot(ax_pos)

        if section_name is not None:
            if section_name == 'topography':
                ax.set_title('Geological map')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                extent_val = self.model.grid.topography.extent
            else:

                dist = self.model.grid.sections.df.loc[section_name, 'dist']

                extent_val = [0, dist,
                          self.model.grid.regular_grid.extent[4], self.model.grid.regular_grid.extent[5]]

                labels, axname = self._make_section_xylabels(section_name, len(ax.get_xticklabels()) - 2)
                pos_list = np.linspace(0, dist, len(labels))
                ax.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
                ax.xaxis.set_major_formatter(FixedFormatter((labels)))
                ax.set(title=section_name, xlabel=axname, ylabel='Z')

        elif cell_number is not None:
            _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        else:
            raise AttributeError

        if extent_val[3] < extent_val[2]:  # correct vertical orientation of plot
            ax.gca().invert_yaxis()
        aspect = (extent_val[3] - extent_val[2]) / (extent_val[1] - extent_val[0])/ve
        print(aspect)
        ax.set_xlim(extent_val[0], extent_val[1])
        ax.set_ylim(extent_val[2], extent_val[3])
        ax.set_aspect('equal')

        return ax

    def plot_lith(self, ax, section_name=None, cell_number=None, direction='y',
                  block=None, **kwargs):
        """

        Args:
            section_name:
            cell_number:
            direction:
            ax:
            extent_val:
            **kwargs: imshow kwargs

        Returns:

        """
        # # TODO force passing axis
        extent_val = [*ax.get_xlim(), *ax.get_ylim()]

        if section_name is not None:
            if section_name == 'topography':
                try:
                    image = self.model.solution.geological_map.reshape(
                        self.model.grid.topography.values_3D[:, :, 2].shape)
                except AttributeError:
                    raise AttributeError('Geological map not computed. Activate the topography grid.')
            else:
                assert type(section_name) == str, 'section name must be a string of the name of the section'
                assert self.model.solutions.sections is not None, 'no sections for plotting defined'

                l0, l1 = self.model.grid.sections.get_section_args(section_name)
                shape = self.model.grid.sections.df.loc[section_name, 'resolution']
                image = self.model.solutions.sections[0][l0:l1].reshape(shape[0], shape[1]).T

        elif cell_number is not None or block is not None:
            _a, _b, _c, _, x, y = self._slice(direction, cell_number)[:-2]
            if block is None:
                _block = self.model.solutions.lith_block
            else:
                _block = block

            plot_block = _block.reshape(self.model.grid.regular_grid.resolution)
            image = plot_block[_a, _b, _c].T
        else:
            raise AttributeError

        ax.imshow(image, origin='lower', zorder=-100,
                  cmap=self.cmap, norm=self.norm, extent=extent_val)

        return ax

    def plot_scalar_field(self, ax, section_name=None, cell_number=None, sn=0, direction='y',
                          block=None, **kwargs):
        extent_val = [*ax.get_xlim(), *ax.get_ylim()]

        if section_name is not None:
            l0, l1 = self.model.grid.sections.get_section_args(section_name)
            shape = self.model.grid.sections.df.loc[section_name, 'resolution']
            image = self.model.solutions.sections_scalfield[sn][l0:l1].reshape(shape).T

        elif cell_number is not None or block is not None:
            _a, _b, _c, _, x, y = self._slice(direction, cell_number)[:-2]
            if block is None:
                _block = self.model.solutions.scalar_field_matrix[sn]
            else:
                _block = block

            plot_block = _block.reshape(self.model.grid.regular_grid.resolution)
            image = plot_block[_a, _b, _c].T
        else:
            raise AttributeError

        ax.contour(image, cmap='autumn', extent=extent_val, zorder=8, **kwargs)

    def plot_data(self, ax, section_name=None, cell_number=None, direction='y', projection_distance=1e10):

        points = self.model.surface_points.df.copy()
        orientations = self.model.orientations.df.copy()

        if section_name is not None:
            # Project points:
            shift = np.asarray(self.model.grid.sections.df.loc[section_name, 'start'])
            end_point = np.atleast_2d(np.asarray(self.model.grid.sections.df.loc[section_name, 'stop']) - shift)
            A_rotate = np.dot(end_point.T, end_point)/self.model.grid.sections.df.loc[section_name, 'dist']**2

            cartesian_point_dist = np.sqrt(((np.dot(A_rotate, (points[['X', 'Y']]).T).T - points[['X', 'Y']])**2).sum(axis=1))
            cartesian_ori_dist = np.sqrt(((np.dot(A_rotate, (orientations[['X', 'Y']]).T).T - orientations[['X', 'Y']])**2).sum(axis=1))

            # This are the coordinates of the data projected on the section
            cartesian_point = np.dot(A_rotate, (points[['X', 'Y']] - shift).T).T
            cartesian_ori = np.dot(A_rotate, (orientations[['X', 'Y']] - shift).T).T

            # Since we plot only the section we want the norm of those coordinates
            points[['X']] = np.linalg.norm(cartesian_point, axis=1)
            orientations[['X']] = np.linalg.norm(cartesian_ori, axis=1)

        else:

            if cell_number is None:
                cell_number = int(self.model.grid.regular_grid.resolution[0]/2)

            if direction == 'x' or direction == 'X':
                arg_ = 0
                dx = self.model.grid.regular_grid.dx
                dir = 'X'
            elif direction == 'y' or direction == 'Y':
                arg_ = 2
                dx = self.model.grid.regular_grid.dy
                dir = 'Y'
            elif direction == 'z' or direction == 'Z':
                arg_ = 4
                dx = self.model.grid.regular_grid.dz
                dir = 'Z'
            else:
                raise AttributeError('Direction must be x, y, z')

            _loc = self.model.grid.regular_grid.extent[arg_] + dx * cell_number
            cartesian_point_dist = points[dir] - _loc
            cartesian_ori_dist = orientations[dir] - _loc

        select_projected_p = cartesian_point_dist < projection_distance
        select_projected_o = cartesian_ori_dist < projection_distance

        sns.scatterplot(data=points[select_projected_p], x='X', y='Z', hue='id', ax=ax, )

        sel_ori = orientations[select_projected_o]
        ax.quiver(sel_ori['X'], sel_ori['Y'], sel_ori['G_x'], [sel_ori['G_z']],
                  pivot="tail",
                  scale=10, edgecolor='k',
                  headwidth=4, linewidths=1)
        try:
            ax.legend_.set_frame_on(True)
        except AttributeError:
            pass

    def calculate_p1p2(self, direction, cell_number):
        if direction == 'y':
            y = self.model.grid.regular_grid.extent[2] + self.model.grid.regular_grid.dy * cell_number
            p1 = [self.model.grid.regular_grid.extent[0], y]
            p2 = [self.model.grid.regular_grid.extent[1], y]

        elif direction == 'x':
            x = self.model.grid.regular_grid.extent[0] + self.model.grid.regular_grid.dx * cell_number
            p1 = [x, self.model.grid.regular_grid.extent[2]]
            p2 = [x, self.model.grid.regular_grid.extent[3]]

        else:
            raise NotImplementedError
        return p1, p2

    def _slice_topo_4_sections(self, p1, p2, resx, resy):
        xy = self.model.grid.sections.calculate_line_coordinates_2points(p1, p2, resx, resy)
        z = self.model.grid.topography.interpolate_zvals_at_xy(xy)
        return xy[:, 0], xy[:, 1], z

    def plot_topography(self, ax, section_name=None, cell_number=None, direction='y', block=None):

        if section_name is not None:

            p1 = self.model.grid.sections.df.loc[section_name, 'start']
            p2 = self.model.grid.sections.df.loc[section_name, 'stop']
            x, y, z = self._slice_topo_4_sections(p1, p2, self.model.grid.topography.resolution[0],
                                                  self.model.grid.topography.resolution[1])

            pseudo_x = np.linspace(0, self.model.grid.sections.df.loc[section_name, 'dist'], z.shape[0])
            a = np.vstack((pseudo_x, z)).T
            xy = np.append(a,
                          ([self.model.grid.sections.df.loc[section_name, 'dist'], a[:, 1][-1]],
                           [self.model.grid.sections.df.loc[section_name, 'dist'], self.model.grid.regular_grid.extent[5]],
                           [0, self.model.grid.regular_grid.extent[5]],
                           [0, a[:, 1][0]])).reshape(-1, 2)

            ax.fill(xy[:,0], xy[:, 1], 'k', zorder=10)

        elif cell_number is not None or block is not None:
            p1, p2 = self.calculate_p1p2(direction, cell_number)
            resx = self.model.grid.topography.resolution[0]
            resy = self.model.grid.topography.resolution[1]
            print('p1', p1, 'p2', p2)
            x, y, z = self._slice_topo_4_sections(p1, p2, resx, resy)
            if direction == 'x':
                a = np.vstack((y, z)).T
                ext = self.model.grid.regular_grid.extent[[2, 3]]
            elif direction == 'y':
                a = np.vstack((x, z)).T
                ext = self.model.grid.regular_grid.extent[[0, 1]]
            else:
                raise NotImplementedError
            a = np.append(a,
                          ([ext[1], a[:, 1][-1]],
                           [ext[1], self.model.grid.regular_grid.extent[5]],
                           [ext[0], self.model.grid.regular_grid.extent[5]],
                           [ext[0], a[:, 1][0]]))
            line = a.reshape(-1, 2)
            ax.fill(line[:, 0], line[:, 1], color='k')

    def plot_contacts(self, ax, section_name=None, cell_number=None, direction='y', block=None, **kwargs):

        faults = list(self.model.faults.df[self.model.faults.df['isFault'] == True].index)
        extent_val = [*ax.get_xlim(), *ax.get_ylim()]
        zorder = kwargs.get('zorder', 100)

        if section_name is not None:
            if section_name == 'topography':
                shape = self.model.grid.topography.resolution
                a = self.model.solutions.geological_map_scalfield
                extent = self.model.grid.topography.extent

            else:
                l0, l1 = self.model.grid.sections.get_section_args(section_name)
                shape = self.model.grid.sections.df.loc[section_name, 'resolution']
                faults_scalar = self.model.solutions.sections_scalfield[:, l0:l1]#.reshape(shape).T

                c_id = 0  # color id startpoint

                for e, block in enumerate(faults_scalar):
                    level = self.model.solutions.scalar_field_at_surface_points[e][np.where(
                        self.model.solutions.scalar_field_at_surface_points[e] != 0)]

                    c_id2 = c_id + len(level)  # color id endpoint
                    ax.contour(block.reshape(shape).T, 0, levels=np.sort(level),
                               colors=self.cmap.colors[c_id:c_id2],
                                 linestyles='solid', origin='lower',
                                 extent=extent_val, zorder=zorder - (e+len(level))
                               )

        elif cell_number is not None or block is not None:

            if len(faults) == 0:
                pass
            else:
                _slice = self._slice(direction, cell_number)[:3]

                for fault in faults:
                    f_id = int(self.model.series.df.loc[fault, 'order_series']) - 1
                    block = self.model.solutions.scalar_field_matrix[f_id]
                    level = self.model.solutions.scalar_field_at_surface_points[f_id][np.where(
                        self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]
                    level.sort()
                    block = block.reshape(self.model.grid.regular_grid.resolution)[_slice].T

                    ax.contour(block, 0, extent=extent_val,
                               levels=level,
                               colors=self.cmap.colors[f_id], linestyles='solid')




