"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    gempy is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to visualized structural geology data and
    potential fields of the regional modelling based on the potential field
    method. Tested on Windows 10

    Created on 08.04.2020

    @author: Miguel de la Varga, Bane Sullivan, Alexander Schaaf, Jan von Harten
"""

import warnings
# insys.path.append("../../pyvista")
from copy import deepcopy
from typing import Union, Dict, List, Iterable, Set, Tuple

import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd

# TODO Check if this is necessary if it is implemented in the API
try:
    import pyvista as pv
    from pyvista.plotting.theme import parse_color

    PYVISTA_IMPORT = True
except ImportError:
    PYVISTA_IMPORT = False

import gempy as gp
from gempy.plot.vista_widgets import WidgetsCallbacks
from nptyping import Array
from logging import debug

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False


class GemPyToVista(WidgetsCallbacks):

    def __init__(self, model, plotter_type: str = 'basic', extent=None, lith_c=None, live_updating=False, **kwargs):
        """ GemPy 3-D visualization using pyVista.

        Args:
            model (gp.Model): Geomodel instance with solutions.
            extent (List[float], optional): Custom extent. Defaults to None.
            lith_c (pn.DataFrame, optional): Custom color scheme in the form
              of a look-up table. Defaults to None.
            live_updating (bool, optional): Toggles real-time updating of the plot.
              Defaults to False.
            plotter_type (str["basic", "background"], optional): Set the
              plotter type. Defaults to 'basic'.
        """

        # Override default notebook value
        kwargs['notebook'] = kwargs.get('notebook', True)

        # Model properties
        self.model = model
        self.extent = model.grid.regular_grid.extent if extent is None else extent

        # plotting options
        self.live_updating = live_updating

        # Choosing plotter
        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
        elif plotter_type == 'notebook':
            self.p = pv.PlotterITK()
        elif plotter_type == 'background':
            self.p = pv.BackgroundPlotter(**kwargs)
        else:
            raise AttributeError('Plotter type must be basic, background or notebook.')

        self.p.plotter_type = plotter_type
        # Default camera and bounds
        self.set_bounds()
        self.p.view_isometric(negative=False)

        # Actors containers
        self.surface_polydata = {}
        self.surface_points_polydata = None
        self.surface_points_widgets = []
        self.orientations_polydata = None
        self.orientations_widgets = []

        # Topology properties
        self.topo_edges = None
        self.topo_ctrs = None

    def _get_color_lot(self, lith_c=None, faults=True):
        """ Method to get the right color list depending on the type of plot."""
        if lith_c is None:
            surf_df = self.model.surfaces.df.set_index('surface')
            unique_surf_points = np.unique(self.model.surface_points.df['id'])

            if len(unique_surf_points) != 0:
                bool_surf_points = np.zeros(surf_df.shape[0], dtype=bool)
                bool_surf_points[unique_surf_points - 1] = True

                surf_df['isActive'] = (surf_df['isActive'] | bool_surf_points)

                if faults is True:

                    lith_c = surf_df.groupby('isActive').get_group(True)['color']
                else:
                    lith_c = surf_df.groupby(('isActive', 'isFault')).get_group((True, False))['color']

        color_lot = lith_c
        return color_lot

    def set_bounds(
            self,
            extent: list = None,
            grid: bool = False,
            location: str = 'furthest',
            **kwargs
    ):
        """Set and toggle display of bounds of geomodel.

        Args:
            extent (list, optional): [description]. Defaults to None.
            grid (bool, optional): [description]. Defaults to False.
            location (str, optional): [description]. Defaults to 'furthest'.
        """
        if self.p.plotter_type != 'notebook':
            if extent is None:
                extent = self.extent
            self.p.show_bounds(
                bounds=extent, location=location, grid=grid, **kwargs
            )

    def plot_data(self, surface_points=None, orientations=None, **kwargs):
        """ Plot all the geometric data

        Args:
            surface_points:
            orientations:
            **kwargs:

        Returns:

        """
        self.plot_surface_points(surface_points, **kwargs)
        self.plot_orientations(orientations, **kwargs)

    def _select_surface_points(self, surfaces):
        if surfaces == 'all':
            surface_points = self.model.surface_points.df
        else:
            surface_points = pd.concat(
                [self.model.surface_points.df.groupby('surface').get_group(group)
                 for group in surfaces])
        return surface_points

    def create_sphere_widgets(self, surface_points, colors, test_callback=True, **kwargs):
        radius = kwargs.get('radius', None)
        if radius is None:
            _e = self.extent
            _e_dx = _e[1] - _e[0]
            _e_dy = _e[3] - _e[2]
            _e_dz = _e[5] - _e[4]
            _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
            radius = _e_d_avrg * .01

            # This is Bane way. It gives me some error with index slicing
            centers = surface_points[['X', 'Y', 'Z']]

            # This is necessary to change the color of the widget if change id
            self._color_lot = colors
            s = self.p.add_sphere_widget(self.call_back_sphere,
                                         center=centers, color=colors, pass_widget=True,
                                         test_callback=test_callback,
                                         indices=surface_points.index.values,
                                         radius=radius, **kwargs)
            return s

    def plot_surface_points(self, surfaces: Union[str, Iterable[str]] = 'all',
                            surface_points: pd.DataFrame = None,
                            clear=True, colors=None, render_points_as_spheres=True, **kwargs):
        # Set the colors
        if colors is None:
            colors = self._get_color_lot().values

        # Selecting the surfaces to plot
        if surface_points is None:
            surface_points = self._select_surface_points(surfaces)

        if self.live_updating is True:
            if clear is True:
                self.p.clear_sphere_widgets()

            sphere_widgets = self.create_sphere_widgets(surface_points, colors, **kwargs)
            self.sphere_widgets = dict(zip(surface_points.index, sphere_widgets))
        else:
            poly = pv.PolyData(surface_points[["X", "Y", "Z"]].values)
            poly['id'] = surface_points['id']
            cmap = mcolors.ListedColormap(list(self._get_color_lot(faults=False)))
            self.p.add_mesh(poly, cmap=cmap)

        self.set_bounds()
