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
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data, plotter_type='basic')
gv.create_structured_grid()
gv.set_structured_grid_data()
#gv.p.add_mesh(gv.vista_rgrid)

###############################################################################
# **When a sclar valur is added after passing the mesh to the plotter, the plotter does not display it. If you set the scalar before it does. Why?**

gv.p.mesh is gv.vista_rgrid

""
gv.p.show(use_panel=True)

""
gv.vista_rgrid.active_scalar_name

""
gv.vista_rgrid.plot()

###############################################################################
# **Is there a way to recover the renderer?**

gv.p.show()

###############################################################################
# ### Surfaces:

from gempy.plot import vista
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data)

""
gv.set_surfaces()
gv.p.show_grid()


###############################################################################
# **Is there a way to plot the grid with panel?** Prob no because is a widget.

gv.p.show(use_panel=True)

###############################################################################
# ### Plot data

from gempy.plot import vista
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data, plotter_type='background')
#gv.set_surface_points()
#gv.set_surfaces()
#gv.p.show_bounds(bounds=geo_data.grid.regular_grid.extent, location='furthest', use_2d=False, grid=False)
#o = 


""
#gv.p.view_isometric(negative=True)

""
gv.set_orientations();

""
gv.set_surface_points();

""
gv.p.show_bounds(bounds=geo_data.grid.regular_grid.extent, location='furthest', use_2d=False, grid=True)

###############################################################################
# ### Data and surfaces: Background

from gempy.plot import vista
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data, plotter_type='background', notebook=True, real_time=False)
#gv.set_surface_points()
#gv.set_orientations()
#gv.set_surfaces()
#gv.p.show_grid()


""
gv.set_surface_points();

""
gv.set_orientations();

""
gv.set_surfaces();

""
geo_data.orientations

""
geo_data.modify_surface_points(0, Z=1000)

""
gv.update_surfaces_real_time()

""
# gv.p.show()

###############################################################################
# ### Interaction

from gempy.plot import vista
from importlib import reload
reload(vista)

""
gv = vista.Vista(geo_data, notebook=False, plotter_type='background', real_time=True)
gv.set_surfaces()
gv.set_surface_points()
gv.set_orientations()

###############################################################################
# ### Adding point

import sys

from PyQt5 import Qt
import numpy as np

import pyvista as pv


class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        self.vtk_widget = pv.QtInteractor(self.frame)
        vlayout.addWidget(self.vtk_widget)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = Qt.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.vtk_widget.add_mesh(sphere)
        self.vtk_widget.reset_camera()


""
app = Qt.QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec_())

###############################################################################
# ### Bane code

import pyvista as pv
from pyvista import examples

mesh = examples.download_kitchen()

plotter = pv.BackgroundPlotter()
plotter.add_mesh_clip_plane(mesh)

""
streamlines = mesh.streamlines(n_points=40, source_center=(0.08, 3, 0.71))
plotter.add_mesh(streamlines.tube(radius=0.01), scalars="velocity", lighting=False)

""

