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
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib.gridspec as gridspect
import matplotlib as mpl
import scipy.spatial.distance as dd
import seaborn as sns

sns.set_context('talk')
plt.style.use(['seaborn-white', 'seaborn-talk'])

warnings.filterwarnings("ignore", message="No contour levels were found")


class Plot2D:
    """
    Class with functionality to plot 2D gempy sections

    Args:
        model: gempy.Model object
        cmap: Color map to pass to matplotlib
    """

    def __init__(self, model, cmap=None, norm=None, **kwargs):
        self.model = model
        self._color_lot = dict(zip(self.model._surfaces.df['surface'],
                                   self.model._surfaces.df['color']))
        self.axes = list()

        if cmap is None:
            self.cmap = mcolors.ListedColormap(list(self.model._surfaces.df['color']))
            self._custom_colormap = False
        else:
            self.cmap = cmap
            self._custom_colormap = True

        if norm is None:
            self.norm = mcolors.Normalize(vmin=0.5, vmax=len(self.cmap.colors) + 0.5)
        else:
            self.norm = norm

    def update_colot_lot(self, color_dir=None):
        if color_dir is None:
            color_dir = dict(zip(self.model._surfaces.df['surface'], self.model._surfaces.df['color']))

        self._color_lot = color_dir
        if self._custom_colormap is False:
            self.cmap = mcolors.ListedColormap(list(self.model._surfaces.df['color']))
            self.norm = mcolors.Normalize(vmin=0.5, vmax=len(self.cmap.colors) + 0.5)

    @staticmethod
    def remove(ax):
        while len(ax.collections) != 0:
            list(map(lambda x: x.remove(), ax.collections))

    def _make_section_xylabels(self, section_name, n=5):
        """
        @elisa heim
        Setting the axis labels to any combination of vertical crossections

        Args:
            section_name: name of a defined gempy crossection. See gempy.Model().grid.section
            n:

        Returns:

        """
        if n > 5:
            n = 3  # todo I don't know why but sometimes it wants to make a lot of xticks
        elif n < 0:
            n = 3

        j = np.where(self.model._grid.sections.names == section_name)[0][0]
        startend = list(self.model._grid.sections.section_dict.values())[j]
        p1, p2 = startend[0], startend[1]
        xy = self.model._grid.sections.calculate_line_coordinates_2points(p1, p2, n)
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
        _a, _b, _c = (slice(0, self.model._grid.regular_grid.resolution[0]),
                      slice(0, self.model._grid.regular_grid.resolution[1]),
                      slice(0, self.model._grid.regular_grid.resolution[2]))

        if direction == "x":
            cell_number = int(self.model._grid.regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number
            _a, x, y, Gx, Gy = cell_number, "Y", "Z", "G_y", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[2, 3, 4, 5]]
        elif direction == "y":
            cell_number = int(self.model._grid.regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number
            _b, x, y, Gx, Gy = cell_number, "X", "Z", "G_x", "G_z"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 4, 5]]
        elif direction == "z":
            cell_number = int(self.model._grid.regular_grid.resolution[2] / 2) if cell_number == 'mid' else cell_number
            _c, x, y, Gx, Gy = cell_number, "X", "Y", "G_x", "G_y"
            extent_val = self.model._grid.regular_grid.extent[[0, 1, 2, 3]]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def create_figure(self, figsize=None, textsize=None, **kwargs):
        """
        Create the figure.

        Args:
            figsize:
            textsize:

        Returns:
            figure, list axes, subgrid values
        """
        cols = kwargs.get('cols', 1)
        rows = kwargs.get('rows', 1)

        figsize, self.ax_labelsize, _, self.xt_labelsize, self.linewidth, _ = _scale_fig_size(
            figsize, textsize, rows, cols)
        self.fig = plt.figure( figsize=figsize, constrained_layout=False)
        self.fig.is_legend = False
        # TODO make grid variable
        # self.gs_0 = gridspect.GridSpec(2, 2, figure=self.fig, hspace=.9)

        return self.fig, self.axes  # , self.gs_0

    def add_section(self, section_name=None, cell_number=None, direction='y', ax=None, ax_pos=111,
                    ve=1., **kwargs):

        extent_val = kwargs.get('extent', None)
        self.update_colot_lot()

        if ax is None:
            ax = self.fig.add_subplot(ax_pos)

        if section_name is not None:
            if section_name == 'topography':
                ax.set_title('Geological map')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                extent_val = self.model._grid.topography.extent
            else:

                dist = self.model._grid.sections.df.loc[section_name, 'dist']

                extent_val = [0, dist,
                              self.model._grid.regular_grid.extent[4], self.model._grid.regular_grid.extent[5]]

                labels, axname = self._make_section_xylabels(section_name, len(ax.get_xticklabels()) - 2)
                pos_list = np.linspace(0, dist, len(labels))
                ax.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
                ax.xaxis.set_major_formatter(FixedFormatter((labels)))
                ax.set(title=section_name, xlabel=axname, ylabel='Z')

        elif cell_number is not None:
            _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set(title='Cell Number: ' + str(cell_number) + ' Direction: ' + str(direction))

        if extent_val is not None:
            if extent_val[3] < extent_val[2]:  # correct vertical orientation of plot
                ax.invert_yaxis()
            self._aspect = (extent_val[3] - extent_val[2]) / (extent_val[1] - extent_val[0]) / ve
            ax.set_xlim(extent_val[0], extent_val[1])
            ax.set_ylim(extent_val[2], extent_val[3])
        ax.set_aspect('equal')

        # Adding some properties to the axes to make easier to plot
        ax.section_name = section_name
        ax.cell_number = cell_number
        ax.direction = direction
        ax.tick_params(axis='x', labelrotation=30)
        self.axes = np.append(self.axes, ax)
        self.fig.tight_layout()

        return ax

    @staticmethod
    def _check_default_section(ax, section_name, cell_number, direction):

        if section_name is None:
            try:
                section_name = ax.section_name
            except AttributeError:
                pass
        if cell_number is None:
            try:
                cell_number = ax.cell_number
                direction = ax.direction
            except AttributeError:
                pass

        return section_name, cell_number, direction

    def plot_regular_grid(self, ax, section_name=None, cell_number=None, direction='y',
                          block: np.ndarray = None, resolution=None, **kwargs):
        """Generic function to plot all regular data

        Args:
            block:
            section_name:
            cell_number:
            direction:
            ax:
            **kwargs: imshow kwargs

        Returns:

        """
        self.update_colot_lot()
        extent_val = [*ax.get_xlim(), *ax.get_ylim()]
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = self.cmap
        if 'norm' in kwargs:
            norm = kwargs['norm']
        else:
            norm = self.norm

        section_name, cell_number, direction = self._check_default_section(ax, section_name, cell_number, direction)

        if section_name is not None:
            if section_name == 'topography':
                try:
                    image = self.model.solutions.geological_map[0].reshape(
                        self.model._grid.topography.values_2d[:, :, 2].shape)

                except AttributeError:
                    raise AttributeError('Geological map not computed. Activate the topography grid.')
            else:
                assert type(section_name) == str or type(
                    section_name) == np.str_, 'section name must be a string of the name of the section'
                assert self.model.solutions.sections is not None, 'no sections for plotting defined'

                l0, l1 = self.model._grid.sections.get_section_args(section_name)
                shape = self.model._grid.sections.df.loc[section_name, 'resolution']
                image = self.model.solutions.sections[0][0][l0:l1].reshape(shape[0], shape[1]).T

        elif cell_number is not None or block is not None:
            _a, _b, _c, _, x, y = self._slice(direction, cell_number)[:-2]
            if resolution is None:
                resolution = self.model._grid.regular_grid.resolution

            plot_block = block.reshape(self.model._grid.regular_grid.resolution)
            image = plot_block[_a, _b, _c].T
        else:
            raise AttributeError

        ax.imshow(image, origin='lower', zorder=-100,
                  cmap=cmap, norm=norm, extent=extent_val)
        return ax

    def plot_lith(self, ax, section_name=None, cell_number=None, direction='y', **kwargs):
        block = self.model.solutions.lith_block
        self.plot_regular_grid(ax, section_name, cell_number, direction, block=block)

    def plot_values(self, ax, series_n=0, section_name=None, cell_number=None,
                    direction='y', **kwargs):
        block = self.model.solutions.values_matrix[series_n]
        self.plot_regular_grid(ax, section_name, cell_number, direction, block=block,
                               **kwargs)

    def plot_block(self, ax, series_n=0, section_name=None, cell_number=None, direction='y',
                   **kwargs):

        block = self.model.solutions.block_matrix[series_n]
        self.plot_regular_grid(ax, section_name, cell_number, direction, block=block)

    def plot_scalar_field(self, ax, section_name=None, cell_number=None, series_n=0, direction='y',
                          block=None, **kwargs):
        """
        Plot the scalar field of a section.

        Args:
            ax:
            section_name:
            cell_number:
            series_n:
            direction:
            block:
            **kwargs:

        Returns:

        """

        extent_val = [*ax.get_xlim(), *ax.get_ylim()]
        section_name, cell_number, direction = self._check_default_section(ax, section_name, cell_number, direction)

        if section_name is not None:
            if section_name == 'topography':
                try:
                    image = self.model.solutions.geological_map[1][series_n].reshape(
                        self.model._grid.topography.values_3D[:, :, 2].shape)
                except AttributeError:
                    raise AttributeError('Geological map not computed. Activate the topography grid.')
            else:
                l0, l1 = self.model._grid.sections.get_section_args(section_name)
                shape = self.model._grid.sections.df.loc[section_name, 'resolution']
                image = self.model.solutions.sections[1][series_n][l0:l1].reshape(shape).T

        elif cell_number is not None or block is not None:
            _a, _b, _c, _, x, y = self._slice(direction, cell_number)[:-2]
            if block is None:
                _block = self.model.solutions.scalar_field_matrix[series_n]
            else:
                _block = block

            plot_block = _block.reshape(self.model._grid.regular_grid.resolution)
            image = plot_block[_a, _b, _c].T
        else:
            raise AttributeError

        ax.contour(image, cmap='autumn', extent=extent_val, zorder=8, **kwargs)
        if 'N' in kwargs:
            kwargs.pop('N')
        ax.contourf(image, cmap='autumn', extent=extent_val, zorder=7, alpha=.8,
                    **kwargs)

    def plot_data(self, ax, section_name=None, cell_number=None, direction='y',
                  legend=True,
                  projection_distance=None, **kwargs):
        """
        Plot data--i.e. surface_points and orientations--of a section.

        Args:
            ax:
            section_name:
            cell_number:
            direction:
            legend: bool or 'force'
            projection_distance:
            **kwargs:

        Returns:

        """
        if projection_distance is None:
            projection_distance = 0.2 * self.model._rescaling.df['rescaling factor'].values[0]

        self.update_colot_lot()

        points = self.model._surface_points.df.copy()
        orientations = self.model._orientations.df.copy()
        section_name, cell_number, direction = self._check_default_section(ax, section_name, cell_number, direction)

        if section_name is not None:
            if section_name == 'topography':

                topo_comp = kwargs.get('topo_comp', 5000)
                decimation_aux = int(self.model._grid.topography.values.shape[0] / topo_comp)
                tpp = self.model._grid.topography.values[::decimation_aux + 1, :]
                cartesian_point_dist = (dd.cdist(tpp, self.model._surface_points.df[['X', 'Y', 'Z']])
                                        < projection_distance).sum(axis=0).astype(bool)
                cartesian_ori_dist = (dd.cdist(tpp, self.model._orientations.df[['X', 'Y', 'Z']])
                                      < projection_distance).sum(axis=0).astype(bool)
                x, y, Gx, Gy = 'X', 'Y', 'G_x', 'G_y'

            else:
                # Project points:
                shift = np.asarray(self.model._grid.sections.df.loc[section_name, 'start'])
                end_point = np.atleast_2d(np.asarray(self.model._grid.sections.df.loc[section_name, 'stop']) - shift)
                A_rotate = np.dot(end_point.T, end_point) / self.model._grid.sections.df.loc[section_name, 'dist'] ** 2

                cartesian_point_dist = np.sqrt(((np.dot(
                    A_rotate, (points[['X', 'Y']]).T).T - points[['X', 'Y']]) ** 2).sum(axis=1))

                cartesian_ori_dist = np.sqrt(((np.dot(
                    A_rotate, (orientations[['X', 'Y']]).T).T - orientations[['X', 'Y']]) ** 2).sum(axis=1))

                # This are the coordinates of the data projected on the section
                cartesian_point = np.dot(A_rotate, (points[['X', 'Y']] - shift).T).T
                cartesian_ori = np.dot(A_rotate, (orientations[['X', 'Y']] - shift).T).T

                # Since we plot only the section we want the norm of those coordinates
                points[['X']] = np.linalg.norm(cartesian_point, axis=1)
                orientations[['X']] = np.linalg.norm(cartesian_ori, axis=1)
                x, y, Gx, Gy = 'X', 'Z', 'G_x', 'G_z'

        else:

            if cell_number is None:
                cell_number = int(self.model._grid.regular_grid.resolution[0] / 2)
            elif cell_number == 'mid':
                cell_number = int(self.model._grid.regular_grid.resolution[0] / 2)

            if direction == 'x' or direction == 'X':
                arg_ = 0
                dx = self.model._grid.regular_grid.dx
                dir = 'X'
            elif direction == 'y' or direction == 'Y':
                arg_ = 2
                dx = self.model._grid.regular_grid.dy
                dir = 'Y'
            elif direction == 'z' or direction == 'Z':
                arg_ = 4
                dx = self.model._grid.regular_grid.dz
                dir = 'Z'
            else:
                raise AttributeError('Direction must be x, y, z')

            _loc = self.model._grid.regular_grid.extent[arg_] + dx * cell_number
            cartesian_point_dist = points[dir] - _loc
            cartesian_ori_dist = orientations[dir] - _loc

            x, y, Gx, Gy = self._slice(direction)[4:]

        select_projected_p = cartesian_point_dist < projection_distance
        select_projected_o = cartesian_ori_dist < projection_distance

        # Hack to keep the right X label:
        temp_label = copy.copy(ax.xaxis.label)

        points_df = points[select_projected_p]
        points_df['colors'] = points_df['surface'].map(self._color_lot)

        points_df.plot.scatter(x=x, y=y, ax=ax, c='colors', s=70,  zorder=102,
                               edgecolors='white',
                               colorbar=False)
        # points_df.plot.scatter(x=x, y=y, ax=ax, c='white', s=80,  zorder=101,
        #                        colorbar=False)

        if self.fig.is_legend is False and legend is True or legend == 'force':

            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o',
                                  linestyle='') for color in
                       self._color_lot.values()]
            ax.legend(markers, self._color_lot.keys(), numpoints=1)
            self.fig.is_legend = True
        ax.xaxis.label = temp_label

        sel_ori = orientations[select_projected_o]

        aspect = np.subtract(*ax.get_ylim()) / np.subtract(*ax.get_xlim())
        min_axis = 'width' if aspect < 1 else 'height'

        # Eli options
        ax.quiver(sel_ori[x], sel_ori[y], sel_ori[Gx], sel_ori[Gy],
                  pivot="tail", scale_units=min_axis, scale=30, color=sel_ori['surface'].map(self._color_lot),
                  edgecolor='k', headwidth=8, linewidths=1, zorder=102)

        try:
            ax.legend_.set_frame_on(True)
            ax.legend_.set_zorder(10000)
        except AttributeError:
            pass

    def calculate_p1p2(self, direction, cell_number):

        if direction == 'y':
            cell_number = int(self.model._grid.regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number

            y = self.model._grid.regular_grid.extent[2] + self.model._grid.regular_grid.dy * cell_number
            p1 = [self.model._grid.regular_grid.extent[0], y]
            p2 = [self.model._grid.regular_grid.extent[1], y]

        elif direction == 'x':
            cell_number = int(self.model._grid.regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number

            x = self.model._grid.regular_grid.extent[0] + self.model._grid.regular_grid.dx * cell_number
            p1 = [x, self.model._grid.regular_grid.extent[2]]
            p2 = [x, self.model._grid.regular_grid.extent[3]]

        else:
            raise NotImplementedError
        return p1, p2

    def _slice_topo_4_sections(self, p1, p2, resx, method='interp2d'):
        """
        Slices topography along a set linear section

        Args:
            :param p1: starting point (x,y) of the section
            :param p2: end point (x,y) of the section
            :param resx: resolution of the defined section
            :param method: interpolation method, 'interp2d' for cubic scipy.interpolate.interp2d
                                             'spline' for scipy.interpolate.RectBivariateSpline

        Returns:
            :return: returns x,y,z values of the topography along the section
        """
        xy = self.model._grid.sections.calculate_line_coordinates_2points(p1, p2, resx)
        z = self.model._grid.sections.interpolate_zvals_at_xy(xy, self.model._grid.topography, method)
        return xy[:, 0], xy[:, 1], z

    def plot_topography(self, ax, fill_contour=False,
                        contour=True,
                        section_name=None,
                        cell_number=None, direction='y', block=None, **kwargs):

        hillshade = kwargs.get('hillshade', True)
        azdeg = kwargs.get('azdeg', 0)
        altdeg = kwargs.get('altdeg', 0)
        cmap = kwargs.get('cmap', 'terrain')

        self.update_colot_lot()
        section_name, cell_number, direction = self._check_default_section(ax, section_name, cell_number, direction)

        if section_name is not None and section_name != 'topography':

            p1 = self.model._grid.sections.df.loc[section_name, 'start']
            p2 = self.model._grid.sections.df.loc[section_name, 'stop']
            x, y, z = self._slice_topo_4_sections(p1, p2, self.model._grid.topography.resolution[0])

            pseudo_x = np.linspace(0, self.model._grid.sections.df.loc[section_name, 'dist'], z.shape[0])
            a = np.vstack((pseudo_x, z)).T
            xy = np.append(a,
                           ([self.model._grid.sections.df.loc[section_name, 'dist'], a[:, 1][-1]],
                            [self.model._grid.sections.df.loc[section_name, 'dist'],
                             self.model._grid.regular_grid.extent[5]],
                            [0, self.model._grid.regular_grid.extent[5]],
                            [0, a[:, 1][0]])).reshape(-1, 2)

            ax.fill(xy[:, 0], xy[:, 1], 'k', zorder=10)

        elif section_name == 'topography':
            import skimage
            from gempy.plot.helpers import add_colorbar
            topo = self.model._grid.topography
            topo_super_res = skimage.transform.resize(
                topo.values_2d,
                (1600, 1600),
                order=3,
                mode='edge',
                anti_aliasing=True, preserve_range=False)

            values = topo_super_res[:, :, 2].T
            if contour is True:
                CS = ax.contour(values, extent=(topo.extent[:4]),
                                colors='k', linestyles='solid', origin='lower')
                ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            if fill_contour is True:
                CS2 = ax.contourf(values, extent=(topo.extent[:4]), cmap=cmap)
                add_colorbar(axes=ax, label='elevation [m]', cs=CS2)

            if hillshade is True:
                from matplotlib.colors import LightSource

                ls = LightSource(azdeg=azdeg, altdeg=altdeg)
                hillshade_topography = ls.hillshade(values)
                ax.imshow(hillshade_topography, origin='lower', extent=topo.extent[:4], alpha=0.5, zorder=11,
                          cmap='gray')

        elif cell_number is not None or block is not None:
            p1, p2 = self.calculate_p1p2(direction, cell_number)
            resx = self.model._grid.regular_grid.resolution[0]
            resy = self.model._grid.regular_grid.resolution[1]

            try:
                x, y, z = self._slice_topo_4_sections(p1, p2, resx)
                if direction == 'x':
                    a = np.vstack((y, z)).T
                    ext = self.model._grid.regular_grid.extent[[2, 3]]
                elif direction == 'y':
                    a = np.vstack((x, z)).T
                    ext = self.model._grid.regular_grid.extent[[0, 1]]
                else:
                    raise NotImplementedError
                a = np.append(a,
                              ([ext[1], a[:, 1][-1]],
                               [ext[1], self.model._grid.regular_grid.extent[5]],
                               [ext[0], self.model._grid.regular_grid.extent[5]],
                               [ext[0], a[:, 1][0]]))
                line = a.reshape(-1, 2)
                ax.fill(line[:, 0], line[:, 1], color='k')
            except IndexError:
                warnings.warn('Topography needs to be a raster to be able to plot it'
                                     'in 2D sections')
        return ax

    def plot_contacts(self, ax, section_name=None, cell_number=None, direction='y', block=None,
                      only_faults=False, **kwargs):
        self.update_colot_lot()
        section_name, cell_number, direction = self._check_default_section(ax, section_name, cell_number, direction)

        if only_faults:
            contour_idx = list(self.model._faults.df[self.model._faults.df['isFault'] == True].index)
        else:
            contour_idx = list(self.model._surfaces.df.index)
        extent_val = [*ax.get_xlim(), *ax.get_ylim()]
        zorder = kwargs.get('zorder', 100)

        if section_name is not None:
            if section_name == 'topography':
                shape = self.model._grid.topography.resolution

                scalar_fields = self.model.solutions.geological_map[1]
                c_id = 0  # color id startpoint

                for e, block in enumerate(scalar_fields):
                    level = self.model.solutions.scalar_field_at_surface_points[e][np.where(
                        self.model.solutions.scalar_field_at_surface_points[e] != 0)]

                    c_id2 = c_id + len(level)  # color id endpoint
                    ax.contour(block.reshape(shape), 0, levels=np.sort(level),
                               colors=self.cmap.colors[c_id:c_id2][::-1],
                               linestyles='solid', origin='lower',
                               extent=extent_val, zorder=zorder - (e + len(level))
                               )
                    c_id = c_id2

            else:
                l0, l1 = self.model._grid.sections.get_section_args(section_name)
                shape = self.model._grid.sections.df.loc[section_name, 'resolution']
                scalar_fields = self.model.solutions.sections[1][:, l0:l1]

                c_id = 0  # color id startpoint

                for e, block in enumerate(scalar_fields):
                    level = self.model.solutions.scalar_field_at_surface_points[e][np.where(
                        self.model.solutions.scalar_field_at_surface_points[e] != 0)]

                    # Ignore warning about some scalars not being on the plot since it is very common
                    # that an interface does not exit for a given section
                    c_id2 = c_id + len(level)  # color id endpoint
                    color_list = self.model._surfaces.df.groupby('isActive').get_group(True)['color'][c_id:c_id2][::-1]

                    ax.contour(block.reshape(shape).T, 0, levels=np.sort(level),
                               # colors=self.cmap.colors[self.model.surfaces.df['isActive']][c_id:c_id2],
                               colors=color_list,
                               linestyles='solid', origin='lower',
                               extent=extent_val, zorder=zorder - (e + len(level))
                               )
                    c_id = c_id2

        elif cell_number is not None or block is not None:
            _slice = self._slice(direction, cell_number)[:3]
            shape = self.model._grid.regular_grid.resolution
            c_id = 0  # color id startpoint

            for e, block in enumerate(self.model.solutions.scalar_field_matrix):
                level = self.model.solutions.scalar_field_at_surface_points[e][np.where(
                    self.model.solutions.scalar_field_at_surface_points[e] != 0)]
                # c_id = e
                c_id2 = c_id + len(level)
                #    print(c_id, c_id2)

                color_list = self.model._surfaces.df.groupby('isActive').get_group(True)['color'][c_id:c_id2][::-1]
                #    print(color_list)

                ax.contour(block.reshape(shape)[_slice].T, 0, levels=np.sort(level),
                           colors=color_list,
                           linestyles='solid', origin='lower',
                           extent=extent_val, zorder=zorder - (e + len(level))
                           )
                c_id = c_id2

    def plot_section_traces(self, ax, section_names=None, show_data=True, **kwargs):

        if section_names is None:
            section_names = list(self.model._grid.sections.names)

        if show_data:
            self.plot_data(ax, section_name='topography', **kwargs)

        for section in section_names:
            j = np.where(self.model._grid.sections.names == section)[0][0]
            x1, y1 = np.asarray(self.model._grid.sections.df.loc[section, 'start'])
            x2, y2 = np.asarray(self.model._grid.sections.df.loc[section, 'stop'])
            ax.plot([x1, x2], [y1, y2], label=section, linestyle='--')
            ax.legend(frameon=True)

    def plot_topo_g(self, ax, G, centroids, direction="y",
                    label_kwargs=None, node_kwargs=None, edge_kwargs=None):
        res = self.model._grid.regular_grid.resolution
        if direction == "y":
            c1, c2 = (0, 2)
            e1 = self.model._grid.regular_grid.extent[1] - self.model._grid.regular_grid.extent[0]
            e2 = self.model._grid.regular_grid.extent[5] - self.model._grid.regular_grid.extent[4]
            d1 = self.model._grid.regular_grid.extent[0]
            d2 = self.model._grid.regular_grid.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = res[0]
            r2 = res[2]
        elif direction == "x":
            c1, c2 = (1, 2)
            e1 = self.model._grid.regular_grid.extent[3] - self.model._grid.regular_grid.extent[2]
            e2 = self.model._grid.regular_grid.extent[5] - self.model._grid.regular_grid.extent[4]
            d1 = self.model._grid.regular_grid.extent[2]
            d2 = self.model._grid.regular_grid.extent[4]
            if len(list(centroids.items())[0][1]) == 2:
                c1, c2 = (0, 1)
            r1 = res[1]
            r2 = res[2]
        elif direction == "z":
            c1, c2 = (0, 1)
            e1 = self.model._grid.regular_grid.extent[1] - self.model._grid.regular_grid.extent[0]
            e2 = self.model._grid.regular_grid.extent[3] - self.model._grid.regular_grid.extent[2]
            d1 = self.model._grid.regular_grid.extent[0]
            d2 = self.model._grid.regular_grid.extent[2]
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
            ax.plot(np.array([centroids[a][c1], centroids[b][c1]]) * e1 / r1 + d1,
                    np.array([centroids[a][c2], centroids[b][c2]]) * e2 / r2 + d2, **lkw)

            for node in G.nodes():
                ax.plot(centroids[node][c1] * e1 / r1 + d1, centroids[node][c2] * e2 / r2 + d2,
                        marker="o", color="black", markersize=10, alpha=0.75)
                ax.text(centroids[node][c1] * e1 / r1 + d1,
                        centroids[node][c2] * e2 / r2 + d2, str(node), **tkw)

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

        raise NotImplementedError


def _scale_fig_size(figsize, textsize, rows=1, cols=1):
    """Scale figure properties according to rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    ax_labelsize : int
        fontsize for axes label
    titlesize : int
        fontsize for title
    xt_labelsize : int
        fontsize for axes ticks
    linewidth : int
        linewidth
    markersize : int
        markersize
    """
    params = mpl.rcParams
    rc_width, rc_height = tuple(params["figure.figsize"])
    rc_ax_labelsize = params["axes.labelsize"]
    rc_titlesize = params["axes.titlesize"]
    rc_xt_labelsize = params["xtick.labelsize"]
    rc_linewidth = params["lines.linewidth"]
    rc_markersize = params["lines.markersize"]
    if isinstance(rc_ax_labelsize, str):
        rc_ax_labelsize = 15
    if isinstance(rc_titlesize, str):
        rc_titlesize = 16
    if isinstance(rc_xt_labelsize, str):
        rc_xt_labelsize = 14

    if figsize is None:
        width, height = rc_width, rc_height
        sff = 1 if (rows == cols == 1) else 1.2
        width = width * cols * sff
        height = height * rows * sff
    else:
        width, height = figsize

    if textsize is not None:
        scale_factor = textsize / rc_xt_labelsize
    elif rows == cols == 1:
        scale_factor = ((width * height) / (rc_width * rc_height)) ** 0.5
    else:
        scale_factor = 1

    ax_labelsize = rc_ax_labelsize * scale_factor
    titlesize = rc_titlesize * scale_factor
    xt_labelsize = rc_xt_labelsize * scale_factor
    linewidth = rc_linewidth * scale_factor
    markersize = rc_markersize * scale_factor

    return (width, height), ax_labelsize, titlesize, xt_labelsize, linewidth, markersize
