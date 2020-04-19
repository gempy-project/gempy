# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:topology]
#     language: python
#     name: conda-env-topology-py
# ---

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda"

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
# sys.path = list(np.insert(sys.path, 0, "../../../pyvista"))
import pyvista as pv

""
###############################################################################
# %matplotlib qt

""
geo_model = gp.load_model(
    'Tutorial_ch1-9b_Fault_relations', 
    path= '../data/gempy_models', 
    recompile=False
)

gp.set_interpolation_data(
    geo_model, 
    output='geology', 
    compile_theano=True, 
    theano_optimizer='fast_run', # fast_compile, fast_run
    dtype="float32",  # float64 for model stability
    verbose=[]
)

gp.compute_model(geo_model, compute_mesh=True)

""
from gempy.plot import vista
from importlib import reload
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

""
geo_model.surfaces.df

""
p = pv.Plotter(notebook=False)


def callback(*args):
    for arg in args:
        print(arg)
    
p.add_mesh(
    pv.PolyData(
        np.random.randn(300).reshape(-1, 3)
    )
)

# p.add_line_widget(
#     callback,
#     use_vertices=True,
#     center=100
    
# )
p.add_plane_widget(
    callback, 
    factor=0.1,

)

p.show()

""
# geo_model.modify_orientations?

""
reload(vista)
gpv = vista.Vista(geo_model, plotter_type="basic", **dict(notebook=False))

gpv.set_bounds()
# gpv.plot_surface_points("rock4")
# gpv.plot_surface_points_all()
# gpv.plot_surface_points_interactive("rock1", radius=20)
# gpv.plot_surface_points_interactive_all(radius=15)
# gpv.plot_orientations("rock4")

# gpv.plot_surface("rock4")
# gpv.plot_surfaces_all()

# gpv.plot_orientations_all()

# gpv.plot_orientations_interactive("rock3")
gpv.plot_surfaces_all()
gpv.plot_orientations_interactive_all()
gpv._live_updating = True



# gpv.plot_structured_grid("lith")
# gpv.plot_structured_grqid("scalar")



gpv.p.show()

""
# pv.Sphere?

""
hasattr(actor, "points")

""
actor = pv.PolyData(np.ones((3, 3)))

""
actors = [pv.PolyData(np.ones((3, 3))).points.tostring() for i in range(5)]

""
actor.points.tostring() in actors

""
actor.points.tostring()

""


""
actor = pv.PolyData(np.ones((3, 3)))
actors = [actor]
actors_hash = [hash(actor)]


""
actor in actors

""
pv.PolyData(np.ones((3, 3))) in actors

""
hash(pv.PolyData(np.ones((3, 3)))) in actors_hash

""
hash(actor) in actors_hash

""


""
import pyvista as pv
from nptyping import Array


class PyvistaPlotter:
    def __init__(self, geo_model, plotter_kwargs={}):
        self.geo_model = geo_model
        self.p = pv.Plotter(**plotter_kwargs)
        self.colors = {
            fmt:c for fmt, c in zip(geo_model.surfaces.df.surface.values, 
                                    geo_model.surfaces.df.color.values)
        }
        
        self._surface_points = False
        self._orientations = False
        self._surfaces = False
        
        self.formations = geo_model.surfaces.df.surface[:-1].values
    
    def add_surface_points(self, fmt:str, mesh_kwargs={}):
        i = self.geo_model.surface_points.df.groupby("surface").groups[fmt]
        self.p.add_mesh(
            pv.PolyData(
                self.geo_model.surface_points.df.loc[i][["X", "Y", "Z"]].values
            ),
            color=self.colors[fmt],
            **mesh_kwargs
        )
        
    def add_surface_points_widget(self, fmt:str):
        i = self.geo_model.surface_points.df.groupby("surface").groups[fmt]
        
        for _, pt in self.geo_model.surface_points.df.loc[i].iterrows():
            self.p.add_sphere_widget(
                callback,
                center=pt[["X", "Y", "Z"]]
            )
    
    def add_surface_orientations(self, fmt:str, mesh_kwargs={}):
        i = self.geo_model.orientations.df.groupby("surface").groups[fmt]
        pts = self.geo_model.orientations.df.loc[i][["X", "Y", "Z"]].values
        nrms = self.geo_model.orientations.df.loc[i][["G_x", "G_y", "G_z"]].values
        for pt, nrm in zip(pts, nrms):
            self.p.add_mesh(
                pv.Line(pointa=pt, pointb=pt+200*nrm),
                color=self.colors[fmt],
                **mesh_kwargs
            )
            
    def add_surface(self, fmt:str, mesh_kwargs={}):
        i = np.where(self.geo_model.surfaces.df.surface == fmt)[0][0] 
        ver = self.geo_model.solutions.vertices[i]
        
        sim = self._simplices_to_pv_tri_simplices(
            self.geo_model.solutions.edges[i]
        )
        
        self.p.add_mesh(
            pv.PolyData(ver, sim),
            color=self.colors[fmt],
            **mesh_kwargs
        )
            
    def add_surfaces(self, mesh_kwargs={}):
        if self._surfaces:
            return
        vertices = self.geo_model.solutions.vertices
        simplices = self.geo_model.solutions.edges
        for fmt in geo_model.surfaces.df.surface[:-1].values:
            self.add_surface(fmt, mesh_kwargs=mesh_kwargs)
        self._surfaces = True
        
    def add_surfaces_points(self, mesh_kwargs={}):
        if self._surface_points:
            return
        for fmt in self.formations:
            self.add_surface_points(fmt, mesh_kwargs=mesh_kwargs)
        self._surface_points = True
            
    def add_surfaces_orientations(self, mesh_kwargs={}):
        if self._orientations:
            return
        for fmt in self.formations:
            self.add_surface_orientations(fmt, mesh_kwargs=mesh_kwargs)
        self._orientations = True    
        
    def _simplices_to_pv_tri_simplices(self, sim:Array[int, ..., 3]) -> Array[int, ..., 4]:
        """Convert triangle simplices (n, 3) to pyvista-compatible
        simplices (n, 4)."""
        n_edges = np.ones(sim.shape[0]) * 3
        return np.append(n_edges[:, None], sim, axis=1)


""


""


""


""


""


""


""


""


"""
#### Point picking
"""



""
hash(pv.PolyData(np.ones((3,3)))) == hash(pv.PolyData(np.ones((3,3))))

""


""
def callback(polydata, pid):
    print(pid)

p.enable_point_picking(callback, use_mesh=True) 

""


""


""


""


""
# gp.plot.plot_section(geo_model, show_data=True)

""
import pyvista as pv
from PyQt5 import Qt
import sys
import numpy as np
from nptyping import Array


class MainWindow(Qt.QMainWindow):
    def __init__(self, geo_model, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.geo_model = geo_model
        
        self.colors = self.geo_model.surfaces.df.color.values
        
        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        self.vtk_widget = pv.QtInteractor(self.frame)
        vlayout.addWidget(self.vtk_widget)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Plot')
        self.add_surfacepoints_action = Qt.QAction('Add Surface Points', self)
        self.add_surfacepoints_action.triggered.connect(self.add_surface_points)
        meshMenu.addAction(self.add_surfacepoints_action)
        
        self.add_surfaces_action = Qt.QAction('Add Surfaces', self)
        self.add_surfaces_action.triggered.connect(self.add_surfaces)
        meshMenu.addAction(self.add_surfaces_action)
        
        self.vtk_widget.enable_point_picking() 

        if show:
            self.show()
        
    def add_surface_points(self):
        points = pv.PolyData(
            self.geo_model.surface_points.df[["X", "Y", "Z"]].values
        )
        self.vtk_widget.add_mesh(points)
        
    def add_surfaces(self):
        vertices = self.geo_model.solutions.vertices
        simplices = self.geo_model.solutions.edges
        
        for i, (ver, sim) in enumerate(zip(self.geo_model.solutions.vertices, 
                            self.geo_model.solutions.edges)):
            sim = self._simplices_to_pv_tri_simplices(sim)
            
            self.vtk_widget.add_mesh(pv.PolyData(ver, sim), color=self.colors[i])
            
    def _simplices_to_pv_tri_simplices(self, sim:Array[int, ..., 3]) -> Array[int, ..., 4]:
        """Convert triangle simplices (n, 3) to pyvista-compatible
        simplices (n, 4)."""
        n_edges = np.ones(sim.shape[0]) * 3
        return np.append(n_edges[:, None], sim, axis=1)



""
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

""
# %gui qt

app = QtCore.QCoreApplication.instance()
if app is None:
    app = Qt.QApplication(sys.argv)
    
w = QWidget()
w.setWindowTitle('Simple')
w.show()
window = MainWindow(geo_model)
# sys.exit(app.exec_())
app.exec_()

""


""
from gempy.plot import vista
from importlib import reload
reload(vista)

""
import pyvista as pv

""
from create_geomodel_gullfaks import create_geomodel

""
gv = vista.Vista(geo_model, notebook=False, plotter_type="background")

""
gv.plot_surfaces()

""


""
class Vista2:
    def __init__(self, geo_model, **kwargs):
        self.geo_model = geo_model
        self.p = pv.BackgroundPlotter(**kwargs)
        self.entities = []
        
    def plot_surfaces(self):
        for idx, val in self.geo_model.surfaces.df.dropna().iterrows():
            if idx in self.entities:
                continue
                
            surf = pv.PolyData(
                val.vertices, 
                np.insert(val['edges'], 0, 3, axis=1).ravel()
            )
            self.p.add_mesh(surf, color=val['color'])
            self.entities.append(idx)
            

gv2 = Vista2(geo_model)

""
gv2.plot_surfaces()

""


""


""


""

