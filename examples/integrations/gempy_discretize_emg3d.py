"""
Use ``GemPy`` to create a geological model as input to ``emg3d``
================================================================

"""

# %%
import numpy as np
import gempy as gempy
from matplotlib.colors import LogNorm, SymLogNorm

# %% 
import emg3d
import pyvista
import discretize

# %%
# Creating a geological model
# ---------------------------
# 
# We start by using the model given in `Chapter
# 1.1 <https://nbviewer.jupyter.org/github/cgre-aachen/gempy/blob/master/notebooks/tutorials/ch1-1_Basics.ipynb>`__
# of the GemPy documentation. It is a nice, made-up model of a folded
# structure with a fault.
# 
# **Changes made:** We load the csv-files from the above example in
# Chapter 1.1. I changed the stratigraphic unit names, and moved the model
# 2 km down.
# 
# Instead of reading a csv-file we could initiate an empty instance and
# then add points and orientations after that by, e.g., providing numpy
# arrays.
# 

# %% 
# Initiate a model
geo_model = gempy.create_model('gempy-discretize-emg3d')

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

# Importing the data from CSV-files and setting extent and resolution
gempy.init_data(
    geo_model,
    [0, 2000., 0, 2000., -2000, 40.], [50, 50, 51],  # This is a regular grid, mainly for plotting purposes
    path_o=data_path+"/data/input_data/tut_chapter1/simple_fault_model_orientations_geophy.csv",
    path_i=data_path+"/data/input_data/tut_chapter1/simple_fault_model_points_geophy.csv",
)

# %%
# Initiate the stratigraphies and faults, and add resistivities to lithology
# --------------------------------------------------------------------------
# 

# %% 
# Add an air-layer: Horizontal layer at z=0m
geo_model.add_surfaces('air')
geo_model.add_surface_points(0, 0, 0, 'air')
geo_model.add_surface_points(0, 0, 0, 'air')
geo_model.add_orientations(0, 0, 0, 'air', [0, 0, 1])

# Add a Series for the air layer; this series will not be cut by the fault
geo_model.add_series('Air_Series')
geo_model.modify_order_series(2, 'Air_Series')
gempy.map_series_to_surfaces(geo_model, {'Air_Series': 'air'})

# Map the different series
gempy.map_series_to_surfaces(
    geo_model,
    {
        "Fault_Series": 'fault',
        "Air_Series": ('air'),
        "Strat_Series": ('seawater', 'overburden', 'target', 'underburden', 'basement')
    },
    remove_unused_series=True
)

# %% 
geo_model.rename_series({'Main_Fault': 'Fault_Series'})

# %% 
# Set which series the fault series is cutting
geo_model.set_is_fault('Fault_Series')
geo_model.faults.faults_relations_df

# %%
# Model generation
# ----------------
# 

# %% 
gempy.set_interpolator(
    geo_model,
    compile_theano=True,
    theano_optimizer='fast_compile',
    verbose=[]
)

# %% 
sol = gempy.compute_model(geo_model, compute_mesh=True)

# %% 
# Plot lithologies (colour-code corresponds to lithologies)
_ = gempy.plot_2d(geo_model, cell_number=25, direction='y', show_data=True)

# %%
# Let’s start with a discretize mesh for a CSEM survey
# ----------------------------------------------------
# 
# Source location and frequency
# 

# %% 
src = [1000, 1000, -500, 0, 0]  # x-directed e-source at (1000, 1000, -500)
freq = 1.0  # Frequency

# %% 
# Get calculation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.utils.get_domain(
        x0=src[0], freq=freq, limits=[0, 2000], min_width=[5, 100])
hz_min, zdomain = emg3d.utils.get_domain(
        freq=freq, limits=[-2000, 0], min_width=[5, 20], fact_pos=40)

# Create stretched grid
nx = 2 ** 6
hx = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src[0])
hy = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src[1])
hz = emg3d.utils.get_stretched_h(hz_min, zdomain, nx * 2, x0=src[2], x1=0)
grid = discretize.TensorMesh(
        [hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))
grid

# %%
# Create a smaller grid without the massive air-layer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# (The massive air-layer is used to push the boundary far away for the
# calculation.)
# 

# %% 
c_nz = 70
cgrid = discretize.TensorMesh([grid.hx, grid.hy, grid.hz[:-c_nz]], x0=grid.x0)

# %%
# Put resistivities to stratigraphic units
# ----------------------------------------
# 
# We could define the resistivities before, but currently it is difficult
# for GemPy to interpolate for something like resistivities with a very
# wide range of values (several orders of magnitudes). So we can simply
# map it here to the ``id`` (Note: GemPy does not do interpolation for
# cells which lie in different stratigraphies).
# 

# %% 
# First, we have to get the id's for our mesh
sol = gempy.compute_model(geo_model, at=grid.gridCC)

# %% 
geo_model.surfaces

# %% 
# Now, we convert the id's to resistivities
res = sol.custom[0][0, :grid.nC]

res[res == 1] = 2e14  # air
# id=2 is the fault
res[np.round(res) == 3] = 0.3  # sea water
res[np.round(res) == 4] = 1  # overburden
res[np.round(res) == 5] = 50  # target
res[np.round(res) == 6] = 1.5  # underburden
res[np.round(res) == 7] = 200  # basement

# %%
# Plot the input model
# --------------------
# 

# %%
# sphinx_gallery_thumbnail_number = 2
dataset = grid.toVTK({'res': np.log10(res)})
dataset = dataset.clip_box(bounds=(0, 2000, 0, 2000, -2000, 0), invert=False)

# Create the rendering scene and add a grid axes
p = pyvista.Plotter(notebook=True)
p.show_grid(location='outer')

# Add spatially referenced data to the scene
dparams = {'rng': np.log10([0.3, 200]), 'cmap': 'viridis', 'show_edges': False}
xyz = (1500, 500, -1500)
p.add_mesh(dataset.slice('x', xyz), name='x-slice', **dparams)
p.add_mesh(dataset.slice('y', xyz), name='y-slice', **dparams)
# p.add_mesh(dataset.slice('z', xyz), name='z-slice', **dparams)

# Add a layer as 3D
p.add_mesh(dataset.threshold([1.69, 1.7]), name='vol', **dparams)

# Show the scene!
p.show()

# %%
# Calculate the resistivities
# ---------------------------
# 

# %% 
# Create model
model = emg3d.utils.Model(grid, res)

# Source field
sfield = emg3d.utils.get_source_field(grid, src, freq, 0)

# Calculate the efield
pfield = emg3d.solve(grid, model, sfield, sslsolver=True, verb=3)

# %% 
cgrid.plot_3d_slicer(
    pfield.fx[:, :, :-70].ravel('F'), zslice=-1000, view='abs', vType='Ex',
    clim=[1e-13, 1e-8], pcolorOpts={'cmap': 'viridis', 'norm': LogNorm()})

# %% 
emg3d.__version__
