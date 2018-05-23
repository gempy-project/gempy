 # cd notebooks\tutorials
 # import setting
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:\ProgramData\Anaconda3\Lib\site-packages\graphviz'
sys.path.append("../..")
import gempy as gp
from copy import copy, deepcopy
# Aux imports

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

def compile_dis(a, b, c, loc):
    eu = np.sqrt((a - loc[0]) ** 2 + (b - loc[1]) ** 2 + (c - loc[2]) ** 2)
    return eu

def get_T_mat(Xn, Yn, Zn, rxLoc):
    eps = 1e-10  # add a small value to the locations to avoid /0

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    Tx = np.zeros((1, 3 * nC))
    Ty = np.zeros((1, 3 * nC))
    Tz = np.zeros((1, 3 * nC))

    dz2 = rxLoc[2] - Zn[:, 0]
    dz1 = rxLoc[2] - Zn[:, 1]

    dy2 = Yn[:, 1] - rxLoc[1]
    dy1 = Yn[:, 0] - rxLoc[1]

    dx2 = Xn[:, 1] - rxLoc[0]
    dx1 = Xn[:, 0] - rxLoc[0]

    R1 = (dy2 ** 2 + dx2 ** 2) + eps
    R2 = (dy2 ** 2 + dx1 ** 2) + eps
    R3 = (dy1 ** 2 + dx2 ** 2) + eps
    R4 = (dy1 ** 2 + dx1 ** 2) + eps

    arg1 = np.sqrt(dz2 ** 2 + R2)
    arg2 = np.sqrt(dz2 ** 2 + R1)
    arg3 = np.sqrt(dz1 ** 2 + R1)
    arg4 = np.sqrt(dz1 ** 2 + R2)
    arg5 = np.sqrt(dz2 ** 2 + R3)
    arg6 = np.sqrt(dz2 ** 2 + R4)
    arg7 = np.sqrt(dz1 ** 2 + R4)
    arg8 = np.sqrt(dz1 ** 2 + R3)

    Tx[0, 0:nC] = np.arctan2(dy1 * dz2, (dx2 * arg5)) + \
                  - np.arctan2(dy2 * dz2, (dx2 * arg2)) + \
                  np.arctan2(dy2 * dz1, (dx2 * arg3)) + \
                  - np.arctan2(dy1 * dz1, (dx2 * arg8)) + \
                  np.arctan2(dy2 * dz2, (dx1 * arg1)) + \
                  - np.arctan2(dy1 * dz2, (dx1 * arg6)) + \
                  np.arctan2(dy1 * dz1, (dx1 * arg7)) + \
                  - np.arctan2(dy2 * dz1, (dx1 * arg4))

    Ty[0, 0:nC] = np.log((dz2 + arg2) / (dz1 + arg3)) + \
                  -np.log((dz2 + arg1) / (dz1 + arg4)) + \
                  np.log((dz2 + arg6) / (dz1 + arg7)) + \
                  -np.log((dz2 + arg5) / (dz1 + arg8))

    Ty[0, nC:2 * nC] = np.arctan2(dx1 * dz2, (dy2 * arg1)) + \
                       - np.arctan2(dx2 * dz2, (dy2 * arg2)) + \
                       np.arctan2(dx2 * dz1, (dy2 * arg3)) + \
                       - np.arctan2(dx1 * dz1, (dy2 * arg4)) + \
                       np.arctan2(dx2 * dz2, (dy1 * arg5)) + \
                       - np.arctan2(dx1 * dz2, (dy1 * arg6)) + \
                       np.arctan2(dx1 * dz1, (dy1 * arg7)) + \
                       - np.arctan2(dx2 * dz1, (dy1 * arg8))

    R1 = (dy2 ** 2 + dz1 ** 2) + eps
    R2 = (dy2 ** 2 + dz2 ** 2) + eps
    R3 = (dy1 ** 2 + dz1 ** 2) + eps
    R4 = (dy1 ** 2 + dz2 ** 2) + eps

    Ty[0, 2 * nC:] = np.log((dx1 + np.sqrt(dx1 ** 2 + R1)) /
                            (dx2 + np.sqrt(dx2 ** 2 + R1))) + \
                     -np.log((dx1 + np.sqrt(dx1 ** 2 + R2)) / (dx2 + np.sqrt(dx2 ** 2 + R2))) + \
                     np.log((dx1 + np.sqrt(dx1 ** 2 + R4)) / (dx2 + np.sqrt(dx2 ** 2 + R4))) + \
                     -np.log((dx1 + np.sqrt(dx1 ** 2 + R3)) / (dx2 + np.sqrt(dx2 ** 2 + R3)))

    R1 = (dx2 ** 2 + dz1 ** 2) + eps
    R2 = (dx2 ** 2 + dz2 ** 2) + eps
    R3 = (dx1 ** 2 + dz1 ** 2) + eps
    R4 = (dx1 ** 2 + dz2 ** 2) + eps

    Tx[0, 2 * nC:] = np.log((dy1 + np.sqrt(dy1 ** 2 + R1)) /
                            (dy2 + np.sqrt(dy2 ** 2 + R1))) + \
                     -np.log((dy1 + np.sqrt(dy1 ** 2 + R2)) / (dy2 + np.sqrt(dy2 ** 2 + R2))) + \
                     np.log((dy1 + np.sqrt(dy1 ** 2 + R4)) / (dy2 + np.sqrt(dy2 ** 2 + R4))) + \
                     -np.log((dy1 + np.sqrt(dy1 ** 2 + R3)) / (dy2 + np.sqrt(dy2 ** 2 + R3)))

    Tz[0, 2 * nC:] = -(Ty[0, nC:2 * nC] + Tx[0, 0:nC])
    Tz[0, nC:2 * nC] = Ty[0, 2 * nC:]
    Tx[0, nC:2 * nC] = Ty[0, 0:nC]
    Tz[0, 0:nC] = Tx[0, 2 * nC:]

    Tx = Tx / (4 * np.pi)
    Ty = Ty / (4 * np.pi)
    Tz = Tz / (4 * np.pi)

    return Tx, Ty, Tz

## Uncertainty Quatification
geo_data = gp.create_data([706000-20000,746000+20000,6864000-20000,6924000+20000,-18000,2000],
                          [40,50,100],
                         path_f = os.pardir+"/input_data/legacy/a_Foliations.csv",
                         path_i = os.pardir+"/input_data/legacy/a_Points.csv")
gp.set_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite',
                              "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                              "SimpleMafic_Series":'SimpleMafic1'},
                      order_series = ["EarlyGranite_Series",
                                      "BIF_Series",
                                      "SimpleMafic_Series"],
                      order_formations= ['EarlyGranite', 'SimpleMafic2', 'SimpleBIF', 'SimpleMafic1'],
              verbose=1)
plt.show()

# gp.plot_data(geo_data, direction="y")
# plt.show()

interp_data = gp.InterpolatorData(geo_data, dtype='float64', output='gravity', compile_theano=True)

# Gravity
gp.set_geophysics_obj(interp_data,
                      [706000, 746000, 6864000, 6924000, -18000, 2000], # Extent
                      [20,30])                            # Resolution

# Setting density and precomputations afafa
tz, select = gp.precomputations_gravity(interp_data, 20,
                               [2.61, 2.92, 3.1, 2.92, 2.61])

vx, vy, vz, select = gp.precomputation_magnetic(interp_data, 20,
                               [0, 0, 0.1, 0, 0], [63.4, 0, 50])

lith, fault, gravi = gp.compute_model(interp_data, output='gravity')

## Topology
topo = gp.topology_compute(geo_data, lith[0], fault)
gp.plot_section(geo_data, lith[0],20, plot_data=True, direction='y')
gp.plot_topology(geo_data, topo[0], topo[1])
# plt.xlim(0, 19000)
# plt.ylim(-10000, 0)
# save topology state for likelihood use
topo_G = copy(topo[0])
plt.show()

# # Get real value
# real_mag = pn.read_csv(os.pardir+'/input_data/Sandstone_geophys/Sst_TMI_2000.csv', sep=',', header=None, names=["x", "y", "z", "mag"])
# real_mag = np.array([real_mag.x, real_mag.y, real_mag.z, real_mag.mag])
# res_mag = real_mag
# res_mag[0] = (real_mag[0]-interp_data.centers[0])/interp_data.rescaling_factor + 0.5001
# res_mag[1] = (real_mag[1]-interp_data.centers[1])/interp_data.rescaling_factor + 0.5001
# res_mag[2] = (real_mag[2]-interp_data.centers[2])/interp_data.rescaling_factor + 0.5001
# mag = [0,0,0.1,0,0]

# # Get coordinates
# grid = interp_data.geo_data_res.x_to_interp_given
# x, y, z = grid[:,0], grid[:,1], grid[:,2]
# dis = max(z)-min(z)
#
# # Define susceptibility
# k = np.ones(x.shape[0])
# k[np.where(np.round(lith[0]) == 1)] = k[np.where(np.round(lith[0]) == 1)] * mag[0] / (1 + mag[0] * (0.242))
# k[np.where(np.round(lith[0]) == 2)] = k[np.where(np.round(lith[0]) == 2)] * mag[1] / (1 + mag[1] * (0.242))
# k[np.where(np.round(lith[0]) == 3)] = k[np.where(np.round(lith[0]) == 3)] * mag[2] / (1 + mag[2] * (0.242))
# k[np.where(np.round(lith[0]) == 4)] = k[np.where(np.round(lith[0]) == 4)] * mag[3] / (1 + mag[3] * (0.242))
# k[np.where(np.round(lith[0]) == 5)] = k[np.where(np.round(lith[0]) == 5)] * mag[4] / (1 + mag[4] * (0.242))
# # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1298918
#
# # Compute T matrix
# ind_all = np.zeros((real_mag.shape[1],40*50*100))
# for i in range(real_mag.shape[1]):
#     def set_vox_size(extent, resolution):
#         x_ex = extent[1] - extent[0]
#         y_ex = extent[3] - extent[2]
#         z_ex = extent[5] - extent[4]
#         vox_size = np.array([x_ex, y_ex, z_ex]) / resolution
#         return vox_size
#
#     vox_size = set_vox_size(interp_data.geo_data_res.extent, interp_data.geo_data_res.resolution)
#
#     def get_index(x, y, z, real_mag, dis):
#         ind = np.zeros(x)
#         ind[np.where(compile_dis(x, y, z, real_mag[0:3]) <= dis)] = 1
#         return ind
#
#     ind = get_index(x, y, z, real_mag[:,i], dis)
#     ind_all[i,:] = ind
#
#     xn = x[np.where(ind == 1)]
#     yn = y[np.where(ind == 1)]
#     zn = z[np.where(ind == 1)]
#
#     Xn = np.c_[xn - vox_size[0], xn + vox_size[0]]
#     Yn = np.c_[yn - vox_size[1], yn + vox_size[1]]
#     Zn = np.c_[zn - vox_size[2], zn + vox_size[2]]
#
#     nC = Xn.shape[0]
#     Tx, Ty, Tz = get_T_mat(Xn, Yn, Zn, real_mag[:,i])
#
#     if i == 0:
#         Tx_all = Tx
#         Ty_all = Ty
#         Tz_all = Tz
#
#     else:
#         Tx_all = np.vstack((Tx_all, Tx))
#         Ty_all = np.vstack((Ty_all, Ty))
#         Tz_all = np.vstack((Tz_all, Tz))
#
# # Global field and direction
# # Bz component in Korea
# Inc = 63.4
# Dec = 0
# # Dec = (450. - float(DDec)) % 360
# Btot = 50
#
# l = np.cos(Inc / 180. * np.pi) * np.cos(Dec / 180. * np.pi)
# m = np.cos(Inc / 180. * np.pi) * np.sin(Dec / 180. * np.pi)
# n = -np.sin(Inc / 180. * np.pi)
# dir = Btot * (10/(4*np.pi)) * np.array([l, m, n])[:, np.newaxis]
#
# M = np.zeros((3 * nC,nC))
# M[0:nC, 0:nC] = np.diag(np.ones(nC) * dir[0])
# M[nC:2 * nC, 0:nC] = np.diag(np.ones(nC) * dir[1])
# M[2 * nC:3 * nC, 0:nC] = np.diag(np.ones(nC) * dir[2])
#
# # # Convert Bdecination from north to cartesian
# newD = (450. - float(Dec)) % 360.
# # Projection matrix
# Ptmi = np.r_[np.cos(Inc / 180. * np.pi) * np.cos(newD / 180. * np.pi),
#              np.cos(Inc / 180. * np.pi) * np.sin(newD / 180. * np.pi),
#              np.sin(Inc / 180. * np.pi)].T
# Ptmi = Ptmi.flatten(order='F')[:, np.newaxis].T
#
# fwr_out = np.zeros(real_mag.shape[1])
# for i in range(real_mag.shape[1]):
#     fwr_out[i] = (Ptmi.dot(np.vstack((Tx[i,:], Ty[i,:], Tz[i,:]))).dot(M)).dot(k[np.where(ind[i,:]==1)])

## PYMC

import pymc
geo_data.interfaces.head()
from copy import deepcopy
geo_data_stoch_init = deepcopy(interp_data.geo_data_res)
interp_data.geo_data_res.interfaces.tail()

# Positions (rows) of the data we want to make stochastic
ids = range(0,70)
# List with the stochastic parameters
interface_Z_modifier = [pymc.Normal("interface_Z_mod_"+str(i), 0., 1./0.01**2) for i in ids]

# Plotting the first element of the list
samples = [interface_Z_modifier[0].rand() for i in range(10000)]
plt.hist(samples, bins=24, normed=True);

# Deterministic functions
@pymc.deterministic(trace=True)
def input_data(value=0,
               interface_Z_modifier=interface_Z_modifier,
               geo_data_stoch_init=geo_data_stoch_init,
               ids=ids,
               verbose=0):
    # First we extract from our original intep_data object the numerical data that is necessary for the interpolation.
    # geo_data_stoch is a pandas Dataframe
    geo_data_stoch = gp.get_data(geo_data_stoch_init, numeric=True)

    # Now we loop each id which share the same uncertainty variable. In this case, each layer.
    for num, i in enumerate(ids):
        # We add the stochastic part to the initial value
        interp_data.geo_data_res.interfaces.set_value(i, "Z", geo_data_stoch_init.interfaces.iloc[i]["Z"] +
                                                      interface_Z_modifier[num])

    if verbose > 0:
        print(geo_data_stoch)

    # then return the input data to be input into the modeling function. Due to the way pymc2 stores the traces
    # We need to save the data as numpy arrays
    return [interp_data.geo_data_res.interfaces[["X", "Y", "Z"]].values,
            interp_data.geo_data_res.orientations[
                ["G_x", "G_y", "G_z", "X", "Y", "Z", 'dip', 'azimuth', 'polarity']].values]

@pymc.deterministic(trace=False)
def gempy_model(value=0,
                input_data=input_data, verbose=False):
    # modify input data values accordingly
    interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = input_data[0]

    # Gx, Gy, Gz are just used for visualization. The theano function gets azimuth dip and polarity!!!
    interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", 'dip', 'azimuth', 'polarity']] = \
    input_data[1]

    try:
        # try to compute model
        lb, fb, grav = gp.compute_model(interp_data, output='gravity')
        if verbose:
            gp.plot_section(interp_data.geo_data_res, lb[0], 5, plot_data=False)
        # gp.plot_data(interp_data.geo_data_res, direction='y')

        return lb, fb, grav

    except np.linalg.linalg.LinAlgError as err:
        # if it fails (e.g. some input data combinations could lead to
        # a singular matrix and thus break the chain) return an empty model
        # with same dimensions (just zeros)
        if verbose:
            print("Exception occured.")
        return np.zeros_like(lith), np.zeros_like(fault), np.zeros_like(grav_i)

@pymc.deterministic(trace=True)
def gempy_surfaces(value=0, gempy_model=gempy_model):
    vert, simp = gp.get_surfaces(interp_data, gempy_model[0][1], gempy_model[1][1], original_scale=True)

    return vert

@pymc.deterministic(trace=True)
def gempy_topo(value=0, gm=gempy_model, verbose=False):
    G, c, lu, lot1, lot2 = gp.topology_compute(geo_data, gm[0][0], gm[1], cell_number=0, direction="y")

    if verbose:
        gp.plot_section(geo_data, gm[0][0], 0)
        gp.plot_topology(geo_data, G, c)

    return G, c, lu, lot1, lot2

@pymc.deterministic
def e_sq(value = original_grav, model_grav = gempy_model[2], verbose = 0):
    square_error =  np.sqrt(np.sum((value*10**-7 - (model_grav*10**-7))**2))
  #  print(square_error)
    return square_error

## Likelihood functions
@pymc.stochastic
def like_topo_jaccard_cauchy(value=0, gempy_topo=gempy_topo, G=topo_G):
    """Compares the model output topology with a given topology graph G using an inverse Jaccard-index embedded in a half-cauchy likelihood."""
    j = gp.topology.compare_graphs(G, gempy_topo[0])  # jaccard-index comparison
    return pymc.half_cauchy_like(1 - j, 0, 0.001)  # the last parameter adjusts the "strength" of the likelihood

@pymc.observed
def inversion(value = 1, e_sq = e_sq):
    return pymc.half_cauchy_like(e_sq,0,0.1)
