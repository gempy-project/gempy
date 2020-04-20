"""
Chapter 1.8: Onlap relationships
--------------------------------

"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")
#sys.path.insert(0, '/home/miguel/anaconda3/lib/python3.6/site-packages/scikit_image-0.15.dev0-py3.6-linux-x86_64.egg/')
import skimage
# Importing gempy
import gempy as gp
import matplotlib.pyplot as plt
# Embedding matplotlib figures into the notebooks
#%matplotlib inline


# Aux imports
import numpy as np
import pandas as pn
import matplotlib
import theano
import qgrid

#%matplotlib widget


######################################################################
# We import a model from an existing folder, representing a subduction
# zone with onlap relationships. The theano function is automatically
# recombiled to allow changes.
# 

geo_model = gp.load_model('Tutorial_ch1-8_Onlap_relations', path= '../../data/gempy_models', recompile=False)

geo_model.additional_data

geo_model.surfaces


######################################################################
# Displaying the input data:
# 

geo_model.series

gp.plot.plot_data(geo_model, direction='y')

gp.set_interpolation_data(geo_model, verbose=[])

geo_model.set_regular_grid([-200,1000,-500,500,-1000,0], [100,100,100])

gp.compute_model(geo_model, compute_mesh=True)

geo_model.solutions.compute_all_surfaces();

geo_model.solutions.scalar_field_at_surface_points

gp.plot.plot_section(geo_model, 2, block=geo_model.solutions.lith_block, show_data=True)


######################################################################
# Marching cubes explanation.
# ---------------------------
# 
# The geological model above is done ovelying several fields. This is a
# common geometry in geological models due to tectonics and similar
# effects over the history of a region.
# 

# Example of block of rock1 and rock 2
gp.plot.plot_section(geo_model, 2, block=geo_model.solutions.block_matrix[1], show_data=True)

# Example of block for onlap surface
gp.plot.plot_section(geo_model, 20, block=geo_model.solutions.block_matrix[2], show_data=True)


######################################################################
# This discretizations are coming for an interpolated scalar field:
# 

# Example of scalar field of rock1 and rock 2
gp.plot.plot_scalar_field(geo_model, 25, series=1)
plt.colorbar()

# Example of scalar field of onlap series
gp.plot.plot_scalar_field(geo_model, 25, series=2)
plt.colorbar()


######################################################################
# The way to overlap this different fields is given by boolean matrices
# that encode their stratigraphic relations:
# 

# Example of block of rock1 and rock 2
plt.imshow(geo_model.solutions.mask_matrix[1].reshape(100,100,100)[:,20,:].T, origin='bottom')

# Example of onlap
plt.imshow(geo_model.solutions.mask_matrix[3].reshape(100,100,100)[:,20,:].T, origin='bottom')


######################################################################
# But actually this boolean arrays are within the volume! The surfaces
# where we want to perform the marching cube are at the interfaces of the
# boolean arrays. To do so we add some padding:
# 

geo_model.solutions.mask_matrix[0]

from gempy.utils.input_manipulation import find_interfaces_from_block_bottoms

# Example of block of rock1 and rock 2
mp1 = find_interfaces_from_block_bottoms(geo_model.solutions.mask_matrix[1].reshape(100,100,100), True)

plt.imshow(mp1[:,20,:].T, origin='bottom')

# Example of block of rock1 and rock 2
mp2 = find_interfaces_from_block_bottoms(geo_model.solutions.mask_matrix[3].reshape(100,100,100), True)

plt.imshow(mp2[:,20,:].T, origin='bottom')


######################################################################
# Performing marching cubes
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now lets go back to the original model. For this example lets forcus on
# the green, purple and magenta surfaces:
# 

gp.plot.plot_section(geo_model, 2, block=geo_model.solutions.lith_block, show_data=True)


######################################################################
# If we just perform the marching cube in the scalar fields we saw above
# the mesh will be continuos:
# 


######################################################################
# Classic marching cubes
# ^^^^^^^^^^^^^^^^^^^^^^
# 

from skimage import measure

scalar_field = geo_model.solutions.scalar_field_matrix
level = geo_model.solutions.scalar_field_at_surface_points

v0, s0, normals, values = measure.marching_cubes_lewiner(
    scalar_field[0].reshape(geo_model.grid.regular_grid.resolution[0],
                         geo_model.grid.regular_grid.resolution[1],
                         geo_model.grid.regular_grid.resolution[2]),
    level[0,0],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),

)


v1, s1, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                         geo_model.grid.regular_grid.resolution[1],
                         geo_model.grid.regular_grid.resolution[2]),
    level[1,1],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
)

v2, s2, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                         geo_model.grid.regular_grid.resolution[1],
                         geo_model.grid.regular_grid.resolution[2]),
    level[1,2],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
)

v3, s3, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                         geo_model.grid.regular_grid.resolution[1],
                         geo_model.grid.regular_grid.resolution[2]),
    level[2, 3],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
)

v4, s4, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                         geo_model.grid.regular_grid.resolution[1],
                         geo_model.grid.regular_grid.resolution[2]),
    level[3, 4],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
)


fig = gp.plot.ipyvolumeVisualization(geo_model)
fig.plot_surfaces()


######################################################################
# .. figure:: ../../data/figures/ipv.png
#    :alt: foo2
# 
#    foo2
# 


######################################################################
# However what we want is that the layers end in the other layers:
# 


######################################################################
# Masked marching cubes
# ^^^^^^^^^^^^^^^^^^^^^
# 

from skimage import measure

scalar_field = geo_model.solutions.scalar_field_matrix
level = geo_model.solutions.scalar_field_at_surface_points

v0, s0, normals, values = measure.marching_cubes_lewiner(
    scalar_field[0].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[0,0],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[0],
)

v1, s1, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,1],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[1],
)

v2, s2, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,2],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[1],
)

v3, s3, normals, values = measure.marching_cubes_lewiner(
    scalar_field[2].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[2, 3],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[3],
)

v4, s4, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[3, 4],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[3],
)



fig = gp.plot.ipyvolumeVisualization(geo_model)
fig.plot_surfaces()


######################################################################
# .. figure:: ../../data/figures/ipyvolume.png
#    :alt: foo
# 
#    foo
# 


######################################################################
# Speed comparison
# ~~~~~~~~~~~~~~~~
# 


######################################################################
# Original
# ^^^^^^^^
# 

scalar_field = geo_model.solutions.scalar_field_matrix
level = geo_model.solutions.scalar_field_at_surface_points

v0, s0, normals, values = measure.marching_cubes_lewiner(
    scalar_field[0].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[0,0],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=None,
)


v1, s1, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,1],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=None,
)

v2, s2, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,2],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=None,
)

v3, s3, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[2, 3],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=None,
)

v4, s4, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[3, 4],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=None,
)


######################################################################
# Masked
# ^^^^^^
# 

# %%timeit
scalar_field = geo_model.solutions.scalar_field_matrix
level = geo_model.solutions.scalar_field_at_surface_points

v0, s0, normals, values = measure.marching_cubes_lewiner(
    scalar_field[0].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[0,0],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[0],
)

v1, s1, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,1],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[1],
)

v2, s2, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,2],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[1],
)

v3, s3, normals, values = measure.marching_cubes_lewiner(
    scalar_field[2].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[2, 3],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[3],
)

v4, s4, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[3, 4],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=geo_model.solutions.mask_matrix_pad[3],
)


# Number of Trues
print(geo_model.solutions.mask_matrix_pad[0].sum(), geo_model.solutions.mask_matrix_pad[0].sum()/1e4,'%')
print(geo_model.solutions.mask_matrix_pad[1].sum(), geo_model.solutions.mask_matrix_pad[1].sum()/1e4,'%')
print(geo_model.solutions.mask_matrix_pad[3].sum(), geo_model.solutions.mask_matrix_pad[3].sum()/1e4,'%')


######################################################################
# Masked all True
# ^^^^^^^^^^^^^^^
# 


######################################################################
# But if we make the masking all Trues it takes a bit longer than just
# calling the unmodified function (which we do when we pass None). (Not in
# my new laptop apparently)
# 

# %%timeit
scalar_field = geo_model.solutions.scalar_field_matrix
level = geo_model.solutions.scalar_field_at_surface_points

v0, s0, normals, values = measure.marching_cubes_lewiner(
    scalar_field[0].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[0,0],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=np.ones_like(geo_model.solutions.mask_matrix_pad[0]),
)

v1, s1, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,1],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=np.ones_like(geo_model.solutions.mask_matrix_pad[0]),
)

v2, s2, normals, values = measure.marching_cubes_lewiner(
    scalar_field[1].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[1,2],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=np.ones_like(geo_model.solutions.mask_matrix_pad[0]),
)

v3, s3, normals, values = measure.marching_cubes_lewiner(
    scalar_field[2].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[2, 3],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=np.ones_like(geo_model.solutions.mask_matrix_pad[0]),
)

v4, s4, normals, values = measure.marching_cubes_lewiner(
    scalar_field[3].reshape(geo_model.grid.regular_grid.resolution[0],
                            geo_model.grid.regular_grid.resolution[1],
                            geo_model.grid.regular_grid.resolution[2]),
    level[3, 4],
    spacing=geo_model.grid.regular_grid.get_dx_dy_dz(),
    mask=np.ones_like(geo_model.solutions.mask_matrix_pad[0]),
)


gp.plot.plot_3D(geo_model)


######################################################################
# Update if any changes were made:
# 

#geo_model.update_to_interpolator()
#gp.compute_model(geo_model, compute_mesh=False)


######################################################################
# Save model if any changes were made:
# 

#geo_model.save_model('Tutorial_ch1-8_Onlap_relations')