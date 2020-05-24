"""
Chapter 4: Analyzing Geomodel Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import gempy as gp
from gempy.assets import topology as tp

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# %%
# Load example Model
# ^^^^^^^^^^^^^^^^^^
# 
# First let's set up a very simple example model. For that we initialize
# the geo_data object with the correct model extent and the resolution we
# like. Then we load our data points from csv files and set the series and
# order the formations (stratigraphic pile).
# 

# %% 
geo_model = gp.create_model("Model_Tutorial6")

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
gp.init_data(
    geo_model, [0, 3000, 0, 20, 0, 2000], [50, 10, 67], 
    path_i=data_path+"data/input_data/tut_chapter6/ch6_data_interf.csv", 
    path_o=data_path+"data/input_data/tut_chapter6/ch6_data_fol.csv"
)
gp.map_stack_to_surfaces(
    geo_model,
    {
        "fault": "Fault",
        "Rest": ('Layer 2', 'Layer 3', 'Layer 4', 'Layer 5')
    }
)
geo_model.set_is_fault(["fault"]);
gp.set_interpolator(geo_model)
sol = gp.compute_model(geo_model, compute_mesh=True)

# %% 
gp.plot_2d(geo_model, cell_number=[5])


# %%
# Analyzing Topology
# ^^^^^^^^^^^^^^^^^^
# 
# GemPy sports in-built functionality to analyze the topology of its
# models. All we need for this is our geo_data object, lithology block and
# the fault block. We input those into *gp.topology_compute* and get
# several useful outputs:
# 
# -  an adjacency graph **G**, representing the topological relationships
#    of the model
# -  the **centroids** of the all the unique topological regions in the
#    model (x,y,z coordinates of their center)
# -  a list of all the unique labels (labels_unique)
# -  two look-up-tables from the lithology id's to the node labels, and
#    vice versa
# 

# %% 
edges, centroids = tp.compute_topology(geo_model)


# %%
# The first output of the topology function is the ``set`` of edges
# representing topology relationships between unique geobodies of the
# block model. An edge is represented by a ``tuple`` of two ``int``
# geobody (or node) labels:
# 

# %% 
edges


# %%
# The second output is the centroids ``dict``, mapping the unique geobody
# id's (graph node id's) to the geobody centroid position in grid
# coordinates:
# 

# %% 
centroids


# %%
# After computing the model topology, we can overlay the topology graph
# over a model section:
# 


# %%
# Visualizing topology
# ~~~~~~~~~~~~~~~~~~~~
# 
# 2-D Visualization of the Topology Graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
gp.plot.plot_topology(geo_model, edges, centroids)
plt.show()

# %% 
gp.plot_2d(geo_model, cell_number=[5], show=False)
gp.plot.plot_topology(geo_model, edges, centroids, scale=True)
plt.show()

# %%
# Adjacency Matrix
# ~~~~~~~~~~~~~~~~
# 
# Another way to encode and visualize the geomodel topology is using an
# adjacency graph:
# 

# %% 
M = tp.get_adjacency_matrix(geo_model, edges, centroids)
print(M)

# %% 
tp.plot_adjacency_matrix(geo_model, M)


# %%
# 3-D Visualization of the Topology Graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# You can also plot the topology in 3-D using GemPy's 3-D visualization
# toolkit powered by ``pyvista``:
# 

# %% 
from gempy.plot._vista import Vista
gpv = Vista(geo_model)
gpv.plot_topology(edges, centroids)
gpv.show()


# %%
# Look-up tables
# ~~~~~~~~~~~~~~
# 


# %%
# The ``topology`` asset provides several look-up tables to work with the
# unique geobody topology id's.
# 
# Mapping node id's back to lithology / surface id's:
# 

# %% 
lith_lot = tp.get_lot_node_to_lith_id(geo_model, centroids)
lith_lot


# %%
# Figuring out which nodes are in which fault block:
# 

# %% 
fault_lot = tp.get_lot_node_to_fault_block(geo_model, centroids)
fault_lot


# %%
# We can also easily map the lithology id to the corresponding topology
# id's:
# 

# %% 
tp.get_lot_lith_to_node_id(lith_lot)


# %%
# Detailed node labeling
# ~~~~~~~~~~~~~~~~~~~~~~
# 

# %%
# sphinx_gallery_thumbnail_number = 4
dedges, dcentroids = tp.get_detailed_labels(geo_model, edges, centroids)
gp.plot_2d(geo_model, cell_number=[5], show=False)
gp.plot.plot_topology(geo_model, dedges, dcentroids, scale=True)
plt.show()

# %% 
dedges

# %% 
dcentroids


# %%
# Checking adjacency
# ~~~~~~~~~~~~~~~~~~
# 


# %%
# So lets say we want to check if the purple layer (id 5) is connected
# across the fault to the yellow layer (id 3). For this we can make easy
# use of the detailed labeling and the ``check_adjacency`` function:
# 

# %% 
tp.check_adjacency(dedges, "5_1", "3_0")


# %%
# We can also check all geobodies that are adjacent to the purple layer
# (id 5) on the left side of the fault (fault id 1):
# 

# %% 
tp.get_adjacencies(dedges, "5_1")

# %% 
