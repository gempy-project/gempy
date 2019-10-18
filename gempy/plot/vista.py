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
# insys.path.append("../../pyvista")
from typing import Union
import pyvista as pv
from pyvista.plotting.theme import parse_color

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

    def __init__(self, model, extent=None, lith_c: pn.DataFrame = None, real_time=False,
                 plotter_type='basic', **kwargs):
        self.model = model
        self.extent = model.grid.regular_grid.extent if extent is None else extent
        self.lith_c = model.surfaces.df.set_index('id')['color'] if lith_c is None else lith_c

        self.s_widget = pn.DataFrame(columns=['val'])
        self.p_widget = pn.DataFrame(columns=['val'])
        self.surf_polydata = pn.DataFrame(columns=['val'])

        self.vista_rgrid = None

        self.real_time =real_time

        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
        self.p.bound = self.extent

    def create_structured_grid(self, regular_grid=None):

        if regular_grid is None:
            regular_grid = self.model.grid.regular_grid

        g_values = regular_grid.values
        g_3D = g_values.reshape(*regular_grid.resolution, 3).T

        self.vista_rgrid = pv.StructuredGrid(*g_3D)
        self.p.add_mesh(self.vista_rgrid)

    def set_structured_grid_data(self, data: Union[dict, gp.Solution, str] = 'Default'):
        if data == 'Default':
            data = self.model.solutions

        if isinstance(data, gp.Solution):
            data = {'lith': data.lith_block}

        for key in data:
            self.vista_rgrid.point_arrays[key] = data[key]

    def call_back_sphere(self, *args):

      #  print(args, args[0],args[-1], args[-1].WIDGET_INDEX)
        new_center = args[0]
        obj = args[-1]
        # Get which sphere we are moving
        index = obj.WIDGET_INDEX
        try:
            self.call_back_sphere_change_df(index, new_center)
            self.call_back_sphere_move_changes(index)

        except:
            pass

        if self.real_time:
            try:
                self.update_surfaces_real_time()

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
            r_f = s1.GetRadius() * 2
            s1.PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                           new_center[1] - r_f, new_center[1] + r_f,
                           new_center[2] - r_f, new_center[2] + r_f)

            s1.GetSphereProperty().SetColor(mcolors.hex2color(self.lith_c[df_row['id']]))
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
        self.set_orientations(orientations, **kwargs)

    def set_surface_points(self, surface_points=None, radio=None, **kwargs):
        # Calculate default surface_points radio
        if radio is None:
            _e = self.extent
            _e_dx = _e[1] - _e[0]
            _e_dy = _e[3] - _e[2]
            _e_dz = _e[5] - _e[4]
            _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
            radio = _e_d_avrg * .01

        if surface_points is None:
            surface_points = self.model.surface_points

        # This is Bane way. It gives me some error with index slicing
        if True:
            centers = surface_points.df[['X', 'Y', 'Z']]
            colors = self.lith_c[surface_points.df['id']].values
            s = self.p.add_sphere_widget(self.call_back_sphere,
                                         center=centers, color=colors,
                                         indices=surface_points.df.index.values,
                                         radius=radio, **kwargs)
            self.s_widget = pn.DataFrame(data=s, index=surface_points.df.index, columns=['val'])

        # for e, val in surface_points.df.iterrows():
        #     c = self.lith_c[val['id']]
        #     s = self.p.add_sphere_widget(self.call_back_sphere,
        #                                  center=val[['X', 'Y', 'Z']], color=c,
        #                                  radius=radio, **kwargs)
        #     # Add index
        #     s.WIDGET_INDEX = e
        #     # Add radio
        #     # s.radio = radio * 2
        #
        #     self.s_widget.at[e] = s
        return self.s_widget

    def call_back_plane(self, obj, event):
        """
              Function that rules what happens when we move a plane. At the moment we update the other 3 renderers and
              update the pandas data frame
              """

        # Get new position of the plane and GxGyGz
        new_center = obj.GetCenter()
        new_normal = obj.GetNormal()

        # Get which plane we are moving
        index = obj.index

        self.call_back_plane_change_df(index, new_center, new_normal)
        # TODO: rethink why I am calling this. Technically this happens outside. It is for sanity check?
        self.call_back_plane_move_changes(index)

        if self.real_time:

            try:
                self.update_surfaces_real_time()

            except AssertionError:
                print('Not enough data to compute the model')

        return True

    def call_back_plane_change_df(self, index, new_center, new_normal):
        # Modify Pandas DataFrame
        # update the gradient vector components and its location
        self.model.modify_orientations(index, X=new_center[0], Y=new_center[1], Z=new_center[2],
                                       G_x=new_normal[0], G_y=new_normal[1], G_z=new_normal[2])
        return True

    def call_back_plane_move_changes(self, indices):
        df_changes = self.model.orientations.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z',
                                                                             'G_x', 'G_y', 'G_z', 'id']]
        for index, new_values_df in df_changes.iterrows():
            new_center = new_values_df[['X', 'Y', 'Z']].values
            new_normal = new_values_df[['G_x', 'G_y', 'G_z']].values
            new_source = vtk.vtkPlaneSource()
            new_source.SetCenter(new_center)
            new_source.SetNormal(new_normal)
            new_source.Update()

            plane1 = self.p_widget.loc[index, 'val']
            #  plane1.SetInputData(new_source.GetOutput())
            plane1.SetNormal(new_normal)
            plane1.SetCenter(new_center[0], new_center[1], new_center[2])

            plane1.GetPlaneProperty().SetColor(
                parse_color(self.model.surfaces.df.set_index('id')['color'][new_values_df['id']]))  # self.C_LOT[new_values_df['id']])
            plane1.GetHandleProperty().SetColor(
                parse_color(self.model.surfaces.df.set_index('id')['color'][new_values_df['id']]))
        return True

    def set_orientations(self, orientations=None, **kwargs):

        factor = kwargs.get('factor', 0.1)

        if orientations is None:
            orientations = self.model.orientations
        for e, val in orientations.df.iterrows():
            c = self.lith_c[val['id']]
            p = self.p.add_plane_widget_simple(self.call_back_plane,
                                               normal=val[['G_x', 'G_y', 'G_z']],
                                               origin=val[['X', 'Y', 'Z']], color=c,
                                               bounds=self.model.grid.regular_grid.extent,
                                               factor=factor, **kwargs)
            self.p_widget.at[e] = p
        return self.p_widget

    def set_surfaces(self, surfaces=None, **kwargs):
        if surfaces is None:
            surfaces = self.model.surfaces
            for idx, val in surfaces.df[['vertices', 'edges', 'color']].dropna().iterrows():

                surf = pv.PolyData(val['vertices'], np.insert(val['edges'], 0, 3, axis=1).ravel())
                self.surf_polydata.at[idx] = surf
                self.p.add_mesh(surf, parse_color(val['color']), **kwargs)

        return self.surf_polydata

    def update_surfaces(self):
        surfaces = self.model.surfaces
        # TODO add the option of update specific surfaces
        for idx, val in surfaces.df[['vertices', 'edges', 'color']].dropna().iterrows():
            self.surf_polydata.loc[idx, 'val'].points = val['vertices']
            self.surf_polydata.loc[idx, 'val'].faces = np.insert(val['edges'], 0, 3, axis=1).ravel()

        return True

    def add_surface(self):
        raise NotImplementedError

    def delete_surface(self, id):
        raise NotImplementedError

    def update_surfaces_real_time(self, delete=True):

        try:
            gp.compute_model(self.model, sort_surfaces=False, compute_mesh=True)
        except IndexError:
            print('IndexError: Model not computed. Laking data in some surface')
        except AssertionError:
            print('AssertionError: Model not computed. Laking data in some surface')
        try:
            self.update_surfaces()
        except KeyError:
            self.set_surfaces()

        # if self.geo_model.solutions.geological_map is not None:
        #     try:
        #         self.set_geological_map()
        #     except AttributeError:
        #         pass
        return True




























