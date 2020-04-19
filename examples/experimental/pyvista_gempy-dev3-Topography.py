# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
sys.path = list(np.insert(sys.path, 0, "../../../pyvista"))


#sys.path("../../../pyvista")

import pyvista

""
path_to_data = os.pardir+"/data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50], 
                        path_o = path_to_data + "model5_orientations.csv",
                        path_i = path_to_data + "model5_surface_points.csv") 

""
gp.map_series_to_surfaces(geo_data, {"Fault_Series":'fault', 
                         "Strat_Series": ('rock2','rock1')})
geo_data.set_is_fault(['Fault_Series'])

""
# %matplotlib inline
gp.plot.plot_data(geo_data, direction='y')

""
geo_data.set_topography(source='random',fd=1.9, d_z=np.array([1000,700]), resolution=np.array([200,200]))

""
geo_data.orientations

""
geo_data.surfaces

""
interp_data = gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile')
sol = gp.compute_model(geo_data)
# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=25,
                         direction='y', show_data=False, show_all_data=True)

"""
## Pyvista
"""

from gempy.plot import vista
import pyvista as pv
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data, plotter_type='background', notebook=False, real_time=True)


""
np.unique(np.round(geo_data.solutions.geological_map).astype(int)[0])

""
geo_data.surfaces

""
gv.set_topography(scalars='topography')

""
gv.set_surfaces();

""
gv.set_structured_grid(opacity=.1, annotations = {2:'rock2', 3:'rock1', 4:'basement'})

""
gv.set_surface_points();

""
gv.set_orientations();

###############################################################################
# ## Key strokes: Normal renderer

gv.p.add_key_event('f', toogle_real_time)

""
gv.p.reset_key_events()

""
gv.p._key_press_event_callbacks['f'][0]()


""
def toogle_real_time():
    print(gv.real_time)
    gv.real_time = 5#gv.real_time ^ True
    return


""
gv.p.show()

""
gv.real_time

###############################################################################
# ### Key strokes: Background

gv.p.iren.AddObserver("KeyPressEvent", key_callbacks)


""
def key_callbacks(self, obj, event):
    print(obj, event)
    key = gv.p.iren.interactor.GetKeySym().lower()
    print(key)
    if key == 'r':
        gv.real_time = 8
        print(5)


""
gv.real_time

""


""


###############################################################################
# ## Clip

topography = geo_data.grid.topography.values
cloud = pv.PolyData(topography)
mesh = cloud.delaunay_2d()

""
rg = gv.vista_rgrids_mesh['lith']

""
rg.clip_surface(mesh, invert=False)

""
clipped = Out[108]

""
gv.p.remove_actor(gv.vista_rgrids_actors['lith'])

###############################################################################
# ## Add text

gv.p.add_text('foo')

###############################################################################
# ## Add toolbar

tb = gv.p.app_window.addToolBar('surfaces')

""
tb.addAction('rock2', something)

""
# gv.p.grabMouse??

""
a = 0


""

def something():
    print(gv.p.iren.GetEventPosition())


""
import vtk

colors = vtk.vtkNamedColors()
NUMBER_OF_SPHERES = 10


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)

        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()

        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())

        # get the new
        self.NewPickedActor = picker.GetActor()

        # If something was selected
        if self.NewPickedActor:
            # If we picked something before, reset its property
            if self.LastPickedActor:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

            # Save the property of the picked actor so that we can
            # restore it next time
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
            # Highlight the picked actor by changing its properties
            self.NewPickedActor.GetProperty().SetColor(colors.GetColor3d('Red'))
            self.NewPickedActor.GetProperty().SetDiffuse(1.0)
            self.NewPickedActor.GetProperty().SetSpecular(0.0)

            # save the last picked actor
            self.LastPickedActor = self.NewPickedActor

        self.OnLeftButtonDown()
        return


def main():
    # A renderer and render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('SteelBlue'))

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)

    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renwin)

    # add the custom style
    style = MouseInteractorHighLightActor()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    # Add spheres to play with
    for i in range(NUMBER_OF_SPHERES):
        source = vtk.vtkSphereSource()

        # random position and radius
        x = vtk.vtkMath.Random(-5, 5)
        y = vtk.vtkMath.Random(-5, 5)
        z = vtk.vtkMath.Random(-5, 5)
        radius = vtk.vtkMath.Random(.5, 1.0)

        source.SetRadius(radius)
        source.SetCenter(x, y, z)
        source.SetPhiResolution(11)
        source.SetThetaResolution(21)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        r = vtk.vtkMath.Random(.4, 1.0)
        g = vtk.vtkMath.Random(.4, 1.0)
        b = vtk.vtkMath.Random(.4, 1.0)
        actor.GetProperty().SetDiffuseColor(r, g, b)
        actor.GetProperty().SetDiffuse(.8)
        actor.GetProperty().SetSpecular(.5)
        actor.GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetSpecularPower(30.0)

        renderer.AddActor(actor)

    # Start
    interactor.Initialize()
    renwin.Render()
    interactor.Start()




""
main()

""


""


""
gv.p.iren.GetEventPosition()

""
gv.p.remove_actor(Out[102])

""
gv.p.add_mesh(clipped)

""
gv.set_topography()

""
mesh

""

