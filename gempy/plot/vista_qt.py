import pyvista as pv
import pyvistaqt as pvqt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from gempy.plot._vista import Vista


class MainWindow(QMainWindow):  # QtWidgets.QWidget
    def __init__(self, geo_model, parent=None):
        super(MainWindow, self).__init__(parent)

        self.create_menu_bar()
        self.bar = None

        self.main_widget = MainWidget(geo_model, parent=self)
        self.setCentralWidget(self.main_widget)

    def create_menu_bar(self):
        self.bar = self.menuBar()

        file = self.bar.addMenu("File")
        _quit = QAction("Quit", self)
        file.addAction(_quit)


class MainWidget(QWidget):
    def __init__(self, geo_model, parent=None):
        super(MainWidget, self).__init__(parent)
        self.model = geo_model

        # init base ui
        hbox = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        self.init_tree()
        splitter.addWidget(self.tree)

        plot = QFrame()
        self.Vista = Vista(geo_model,
                           plotter_type="basic")  # init Vista plotter
        self.vtk_widget = pvqt.QtInteractor(plot)
        self.Vista.p = self.vtk_widget  # set Plotter to the vtk widget Plotter

        splitter.addWidget(self.vtk_widget)
        hbox.addWidget(splitter)
        self.setLayout(hbox)

        # self.Vista.plot_surface_points_all()

    def init_tree(self):
        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)

        self.tree_items = {"surfaces": {}}
        self.tree_actors = {"surfaces": {}}

        for id_, row in self.model._surfaces.df.iterrows():
            item = QTreeWidgetItem([row.surface])
            item.setCheckState(0, Qt.Unchecked)
            self.tree.addTopLevelItem(item)
            self.tree_items["surfaces"][row.surface] = item

        self.tree.itemClicked.connect(self._check_tree_status)

    def _check_tree_status(self):
        for name, item in self.tree_items["surfaces"].items():

            if item.checkState(0) == Qt.Checked:
                actor = self.Vista.plot_surface(name)
                # self.Vista.p.renderer.add_actor(actor)
                self.tree_actors["surfaces"][name] = actor


            elif item.checkState(0) == Qt.Unchecked:
                actor = self.tree_actors["surfaces"].get(name)
                if actor:
                    self.Vista.p.remove_actor(actor)
                    # self.Vista.p.renderer.remove_actor(actor)
                    self.tree_actors["surfaces"][name] = None
