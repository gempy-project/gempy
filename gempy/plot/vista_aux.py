import numpy as np
import matplotlib.colors as mcolors
import vtk


class WidgetsCallbacks:
    """This class purpose is merely function encapsulation"""

    def call_back_sphere(self, *args):
        new_center = args[0]
        obj = args[-1]
        # Get which sphere we are moving
        index = obj.WIDGET_INDEX
        try:
            self.call_back_sphere_change_df(index, new_center)
            #self.call_back_sphere_move_changes(index)

        except KeyError as e:
            print('call_back_sphere error:', e)

        if self.live_updating:
            try:
                self.update_surfaces(recompute=True)

            except AssertionError:
                print('Not enough data to compute the model')

    def call_back_sphere_change_df(self, index, new_center):
        index = np.atleast_1d(index)
        # Modify Pandas DataFrame
        self.model.modify_surface_points(index, X=[new_center[0]], Y=[new_center[1]], Z=[new_center[2]])

    def call_back_sphere_delete_point(self, ind_i):
        """
        Warning this deletion system will lead to memory leaks until the vtk object is reseted (hopefully). This is
        mainly a vtk problem to delete objects
        """
        ind_i = np.atleast_1d(ind_i)
        # Deactivating widget
        for i in ind_i:
            del_widg = self.sphere_wigets.pop(i)
            del_widg.Off()
        return del_widg

    # region plane
    def call_back_plane(self, normal, origin, obj):
        """
              Function that rules what happens when we move a plane. At the moment we update the other 3 renderers and
              update the pandas data frame
              """
        # Get new position of the plane and GxGyGz
        new_center = origin
        new_normal = normal

        # Get which plane we are moving
        index = obj.WIDGET_INDEX

        self.call_back_plane_change_df(index, new_center, new_normal)
        # TODO: rethink why I am calling this. Technically this happens outside.
        #  It is for sanity check?
       # self.call_back_plane_move_changes(index)
        if self.live_updating:
            try:
                self.update_surfaces(recompute=True)

            except AssertionError:
                print('Not enough data to compute the model')

        return True

    def call_back_plane_change_df(self, index, new_center, new_normal):
        # Modify Pandas DataFrame
        # update the gradient vector components and its location
        self.model.modify_orientations(index, X=new_center[0], Y=new_center[1], Z=new_center[2],
                                       G_x=new_normal[0], G_y=new_normal[1], G_z=new_normal[2])
        return True

    # endregion


class RenderChanges:
    def call_back_plane_move_changes(self, indices):
        df_changes = self.model._orientations.df.loc[np.atleast_1d(indices).astype(int)][['X', 'Y', 'Z',
                                                                                          'G_x', 'G_y', 'G_z', 'id']]
        for index, new_values_df in df_changes.iterrows():
            new_center = new_values_df[['X', 'Y', 'Z']].values
            new_normal = new_values_df[['G_x', 'G_y', 'G_z']].values
            new_source = vtk.vtkPlaneSource()
            new_source.SetCenter(new_center)
            new_source.SetNormal(new_normal)
            new_source.Update()

            plane1 = self.orientations_widgets[index]
            #  plane1.SetInputData(new_source.GetOutput())
            plane1.SetNormal(new_normal)
            plane1.SetCenter(new_center[0], new_center[1], new_center[2])

            _color_lot = self._get_color_lot(is_faults=True, is_basement=False, index='id')
            plane1.GetPlaneProperty().SetColor(mcolors.hex2color(_color_lot[int(new_values_df['id'])]))
            plane1.GetHandleProperty().SetColor(mcolors.hex2color(_color_lot[int(new_values_df['id'])]))

    def call_back_sphere_move_changes(self, indices):
        df_changes = self.model._surface_points.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z', 'id']]
        for index, df_row in df_changes.iterrows():
            new_center = df_row[['X', 'Y', 'Z']].values

            # Update renderers
            s1 = self.surface_points_widgets[index]
            r_f = s1.GetRadius() * 2
            s1.PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                           new_center[1] - r_f, new_center[1] + r_f,
                           new_center[2] - r_f, new_center[2] + r_f)

            _color_lot = self._get_color_lot(is_faults=True, is_basement=False, index='id')
            s1.GetSphereProperty().SetColor(mcolors.hex2color(_color_lot[(df_row['id'])]))

    def render_move_surface_points(self, indices):
        self.call_back_sphere_move_changes(indices)
        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_add_surface_points(self, indices):
        indices = np.atleast_1d(indices)
        surface_points = self.model._surface_points.df.loc[indices]
        self.plot_surface_points(surface_points = surface_points, clear=False)
        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_delete_surface_points(self, index):
        sw = self.surface_points_widgets.pop(index)
        sw.Off()
        del sw

        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_move_orientations(self, indices):
        self.call_back_plane_move_changes(indices)
        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_add_orientations(self, indices):
        indices = np.atleast_1d(indices)
        orientations = self.model._orientations.df.loc[indices]
        self.plot_orientations(orientations = orientations, clear=False)
        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_delete_orientations(self, index):
        ow = self.orientations_widgets.pop(index)
        ow.Off()
        del ow

        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.p.interactor.Render()

    def render_topography(self):
        if self.live_updating is True:
            self.update_surfaces(recompute=True)
        self.plot_topography()
        self.p.interactor.Render()
