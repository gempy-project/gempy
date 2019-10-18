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
Tested on Ubuntu 18, MacOS 12

Created on 16.10.2019

@author: Miguel de la Varga, Bane Sullivan, Alex Schaaf
"""

import os
import matplotlib.colors as mcolors
import copy
import pandas as pn
import numpy as np
import sys
import gempy as gp
import warnings
# sys.path.append("../../pyvista")
from typing import Union
import pyvista as pv

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False


class Vista:
    """
    Class to visualize data and results in 3D. Init will create all the render properties while the method render
    model will lunch the window. Using set_surface_points, set_orientations and set_surfaces in between can be chosen what
    will be displayed.
    """

    def __init__(self, model, extent=None, lith_c:pn.DataFrame = None, real_time=False, **kwargs):
        self.model = model
        self.extent = model.grid.regular_grid.extent if extent is None else extent
        self.lith_c = model.surfaces.df.set_index('id')['color'] if lith_c is None else lith_c

        self.s_widget = pn.DataFrame(columns=['val'])
        self.vista_rgrid = None

        self.real_time =real_time

        self.p = pv.Plotter(**kwargs)

    def create_structured_grid(self, regular_grid=None):

        if regular_grid is None:
            regular_grid = self.model.grid.regular_grid

        g_values = regular_grid.values
        g_3D = g_values.reshape(*regular_grid.resoution, 3).T

        self.vista_rgrid = pv.StructuredGrid(*g_3D)

    def set_structured_grid_data(self, data: Union[dict, gp.Solution, str] = 'Default'):
        if data == 'Default':
            data = self.model.solution

        if isinstance(data, gp.Solution):
            data = {'lith': data.lith_block}

        for key in data:
            self.vista_rgrid.point_arrays[key] = data[key]

    def call_back_sphere(self, obj, event):
        new_center = obj.GetCenter()

        # Get which sphere we are moving
        index = obj.index

        # Check what render we are working with
        render = obj.n_render

        # This must be the radio
        # r_f = obj.r_f

        self.call_back_sphere_change_df(index, new_center)
        self.call_back_sphere_move_changes(index)

        if self.real_time:
            try:
                self.update_surfaces_real_time()
                # vertices, simpleces =
                # self.set_surfaces(vertices, simpleces)
            except AssertionError:
                print('Not enough data to compute the model')

    # region Surface Points
    def call_back_sphere_change_df(self, index, new_center):
        index = np.atleast_1d(index)
        # Modify Pandas DataFrame
        self.model.modify_surface_points(index, X=[new_center[0]], Y=[new_center[1]], Z=[new_center[2]])

    def call_back_sphere_move_changes(self, indices):
        df_changes = self.model.surface_points.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z', 'id']]
        for index, df_row in df_changes.iterrows():
            new_center = df_row[['X', 'Y', 'Z']].values

            # Update renderers
            s1 = self.s_widget.loc[index, 'val']

            s1.PlaceWidget(new_center[0] - s1.r_f, new_center[0] + s1.r_f,
                           new_center[1] - s1.r_f, new_center[1] + s1.r_f,
                           new_center[2] - s1.r_f, new_center[2] + s1.r_f)

            s1.GetSphereProperty().SetColor(mcolors.hex2color(self.lith_c[df_row['id']])
                #self.geo_model.surfaces.df.set_index('id')['color'][df_row['id']]))#self.C_LOT[df_row['id']])

    def call_back_sphere_delete_point(self, ind_i):
        """
        Warning this deletion system will lead to memory leaks until the vtk object is reseted (hopefully). This is
        mainly a vtk problem to delete objects
        """
        ind_i = np.atleast_1d(ind_i)
        # Deactivating widget
        for i in ind_i:
            self.s_widget.loc[i, 'val'].Off()

        self.s_widget.drop(ind_i)

    # endregion

    def set_interactive_data(self, surface_points=None, orientations=None, **kwargs):
        # TODO: When we call this method we need to set the plotter.notebook to False!
        self.set_surface_points(surface_points, **kwargs)

    def set_surface_points(self, surface_points, radio = None, **kwargs):
        # Calculate default surface_points radio
        if radio is None:
            _e = self.extent
            _e_dx = _e[1] - _e[0]
            _e_dy = _e[3] - _e[2]
            _e_dz = _e[5] - _e[4]
            _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
            radio = _e_d_avrg * .03

        if surface_points is None:
            surface_points = self.model.surface_points

        for e, val in surface_points.df.iterrows():
            c = self.lith_c[val['id']]
            self.s_widget.at[e] = self.p.add_sphere_widget(self.call_back_sphere_change_df,
                                                           center=val[['X', 'Y', 'Z']], color=c,
                                                           radius=radio, **kwargs)





























