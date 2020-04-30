# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:08:44 2020

@author: stuetz
"""

import sys
import gempy as gp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gempy.plot import visualization_2d_pro as vv
from gempy.plot import vista
# import pyvista as pv
from importlib import reload
reload(vista)
sys.path.append("D:/Documents/Python Scripts/")
import stuetz as st

# %% read data
path_sp = "Ori-NN_Faults_Poi.csv"
df_sp = pd.read_csv(path_sp)

geo_data = gp.create_model('Ori-NN_F')
geo_data.set_surface_points(df_sp, add_basement=False)
geo_data.surface_points.df['smooth'] = 0

# calculate orientation
# geo_data.orientations.create_orientation_from_nn(geo_data.surface_points.df, 2)

# %% Set surfaces
gp.map_series_to_surfaces(geo_data, {"Fault1": ('f'), "Fault2": ('v'),
                                     "Surface": ('s', 't')},
                          sort_geometric_data=True,
                          remove_unused_series=True)

geo_data.set_is_fault(["Fault1", "Fault2"], change_color=False)

# %% Calculate orientations and defnition of faults

# define neighbourhood
neigh_f = 2

# find faults and calculate orientations
id_f = geo_data.faults.df.index.categories[geo_data.faults.df.isFault.values]  # detect fault names
fault_poi = geo_data.surface_points.df[geo_data.surface_points.df.series.isin(id_f)]  # find fault points
geo_data.orientations.create_orientation_from_nn(fault_poi, neigh_f)  # calculate fault orientations

fault_dist = pd.DataFrame(columns=['G_x', 'G_y', 'G_z', 'const'], index=id_f)  # define empty fault dataframe (Hesse normal form)
for fault in id_f: # for every fault
    all_poi_f = geo_data.orientations.df[geo_data.orientations.df.series.eq(fault)] # find points of one fault
    fault_dist.at[fault] = all_poi_f.iloc[0,6:9]  # paste normal vector
    fault_dist.at[fault, 'const'] = \  # calculate distance to origin
        sum(all_poi_f.iloc[0,6:9].reset_index(drop=True)*
            all_poi_f.iloc[0,0:3].reset_index(drop=True))*-1


# %% Calculate orientations surfaces
neigh_p = 50.  # define neighbourhood
id_s = geo_data.faults.df.index.categories[np.logical_not(geo_data.faults.df.isFault.values)] # detect series names
id_p = geo_data.surface_points.df.index[geo_data.surface_points.df.series.isin(id_s)]  # find surface points
# create group-ID variable for every surface point (without fault points)
poi_id = pd.DataFrame(columns=['group'],
                      data=[0]*int(geo_data.surface_points.df.shape[0]
                                   - geo_data.orientations.df.shape[0]))
poi_id['ID'] = id_p
poi_id = poi_id.set_index('ID')  # choose same index as in geo_data.surface_points

for sur in geo_data.surfaces.df.surface:  # for every surface
    x = 1  # ID counter
    # detect to which series belongs the surface
    ser = geo_data.surfaces.df.series[geo_data.surfaces.df.surface == sur]
    # load fault relations
    rel_faults = geo_data.faults.faults_relations_df.loc[:,ser]
    if any(rel_faults.values):  # only for surfaces (not faults!)
        name_rel_faults = rel_faults.index[rel_faults.iloc[:,0]]  # faults which influences surface
        all_poi_sur = geo_data.surface_points.df[geo_data.surface_points.df.surface==sur]  # find all points of one surface
        for fault in name_rel_faults:  # split points in groups for every fault
            func = lambda x: sum(np.asarray(x) * np.asarray(fault_dist.loc[fault,'G_x':'G_z'])) \
                                 + fault_dist.at[fault, 'const']  # define plane hesse normal form function
            dist = all_poi_sur.iloc[:,0:3].apply(func, axis=1)  # calculate dist to fault
            for i, j in zip(dist, all_poi_sur.index):  # if dist < 0 set new group-ID for point
                if i < 0:
                    poi_id.loc[j] = poi_id.loc[j] + x
            x = x*10

for i in np.unique(poi_id):  # for every ID-group
    poi_group = geo_data.surface_points.df.loc[poi_id.index[(poi_id==i).group.values].tolist()]  # find points in same group
    geo_data.orientations.create_orientation_from_nn(poi_group, neigh_p)  # calculate orientations for every group

# %% Set surfaces again for the orientations
gp.map_series_to_surfaces(geo_data, {"Fault1": ('f'), "Fault2": ('v'),\
                                     "Surface": ('s', 't')},
                          sort_geometric_data=True,
                          remove_unused_series=True)

geo_data.set_is_fault(["Fault1", "Fault2"], change_color=False)

# %% create grid
st.create_regular_grid_auto(geo_data, 50, 5)

# %% Interpolation
gp.set_interpolator(geo_data, compile_theano=True,
                    theano_optimizer='fast_run', verbose=[])
gp.compute_model(geo_data, sort_surfaces=False, compute_mesh=True)

# %% plot 3D

gv = vista.Vista(geo_data, plotter_type='background', notebook=False,
                  real_time=False)
gv.plot_surface_points()
gv.plot_orientations()
gv.plot_surfaces()
