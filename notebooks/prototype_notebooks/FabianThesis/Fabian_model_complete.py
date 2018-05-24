import sys
##sys.path.insert(0, '/home/bl9/gempy')
import gempy as gp

#%matplotlib inline
import pymc
import numpy as np
import math

from IPython.core.display import Image

from pymc.Matplot import plot
from pymc import graph

from IPython.core.pylabtools import figsize
figsize(15, 6)

import scipy.optimize as sop
import scipy.stats as stats
from scipy.signal import argrelextrema

import matplotlib.mlab as mlab
from matplotlib import pyplot as plt

import importlib

from operator import itemgetter

from mpl_toolkits.mplot3d import Axes3D

import vtk
import evtk

from scipy.interpolate import griddata

import decision_making as dm

# Importing the data from csv files and setting extent and resolution
geo_data = gp.create_data([0,2000,0,2000,0,2000],[50,50,50],
                         path_o = "./reservoir_model_orientations.csv",
                         path_i = "./reservoir_model_interfaces.csv")
geo_data.n_faults = 1

gp.set_series(geo_data, {"fault":'MainFault',
                      "Rest":('Base_Top', 'Res_Top', 'Seal_Top', 'SecRes_Top')},
                       order_series = ["fault","Rest",], order_formations=['MainFault',
                                         'SecRes_Top', 'Seal_Top', 'Res_Top','Base_Top',
                                         ])

# DECLARING SOME MODEL VARIABLES
resolution = geo_data.resolution[1] #resolution, standard: 50
model_size = geo_data.extent[:2][1] # 'real' model extent, here: 2000 m - cubic (what if not cubic?)
scale_factor = (model_size/resolution) # scale factor used for calculating voxel volumes in [m]
                                        # here: 2000/50 = 40
#rescale_f = interp_data.rescaling_factor # rescaling factor from geo_data to interp_data

minmax_buffer = True # buffer around local min and max values [on/off] - not used atm

SSF_c = 3

# Creating a row label 'fault side' to distinguish between footwall (FW) and hanging wall (HW)
def set_fault_sides(geo_data, fault_border, fault_name):
    geo_data.interfaces['fault side'] = 'nan'
    HW_border = fault_border # distance of middle of fault (border HW/FW) from footwall border of model
                # (X = 2000 - 1300 = 700)
    nonfault_cond = geo_data.interfaces['formation'] != fault_name
    fault_cond = geo_data.interfaces['formation'] == fault_name

    fw_x_cond = geo_data.interfaces['X'] > (geo_data.extent[:2][1] - HW_border) # condition for FW points

    hw_x_cond = geo_data.interfaces['X'] < (geo_data.extent[:2][1] - HW_border) # condition for HW points

    geo_data.interfaces.loc[fw_x_cond,'fault side'] = 'footwall' # setting FW
    geo_data.interfaces.loc[hw_x_cond,'fault side'] = 'hanging wall' #setting HW
    geo_data.interfaces.loc[fault_cond,'fault side'] = 'nan'; # reverting fault points to 'nan'
    return geo_data


# FUNCTION TO FIND ANTICLINAL SPILL POINT AND CROSS-FAULT LEAK POINT

def spill_leak_P(interp_data, res_surf, lith, fault, print_figures=True):
    # creating a grid with uniform distances for vertices of the reservoir surface
    grid_x, grid_y = np.meshgrid(np.unique(interp_data.geo_data_res.grid.values[:, 0]),
                                 np.unique(interp_data.geo_data_res.grid.values[:, 1]))

    # grid_x=(grid_x*rescale_f)-(np.min(grid_x)*rescale_f)
    # grid_y=(grid_y*rescale_f)-(np.min(grid_y)*rescale_f)
    grid_x = (grid_x) - (np.min(grid_x))
    grid_y = (grid_y) - (np.min(grid_y))

    grid_z0 = griddata(res_surf[:, :2], res_surf[:, 2],
                       (grid_x, grid_y), method='linear')

    # order of values that serve to find relative extrema (min/max)
    rel_order_maxX = 5
    rel_order_maxY = 10
    rel_order_minX = 5
    rel_order_minY = 5

    # check grid_z0 for max and min in directions x and y
    # direction X
    minX1, minX2 = argrelextrema(grid_z0, np.less, order=rel_order_minX, axis=1)
    maxX1, maxX2 = argrelextrema(grid_z0, np.greater, order=rel_order_maxX, axis=1)
    grid_minX = np.zeros_like(grid_z0)
    grid_minX[minX1, minX2] = 1  # grid of min in X
    grid_maxX = np.zeros_like(grid_z0)
    grid_maxX[maxX1, maxX2] = 1  # grid of max in X

    # direction Y
    minY1, minY2 = argrelextrema(grid_z0, np.less, order=rel_order_minY, axis=0)
    maxY1, maxY2 = argrelextrema(grid_z0, np.greater, order=rel_order_maxY, axis=0)
    grid_minY = np.zeros_like(grid_z0)
    grid_minY[minY1, minY2] = 1  # grid of min in Y
    grid_maxY = np.zeros_like(grid_z0)
    grid_maxY[maxY1, maxY2] = 1  # grid of max in Y

    # fault leak line: defining line of juxtaposition, point of cross-fault leakage to be found on it
    # check for minima line that is on hanging wall side compared to max contact of layer top with fault
    fault_max_line_bool = np.copy(grid_maxX)
    fault_max_line = fault_max_line_bool.argmax(axis=1)
    fault_max = np.max(fault_max_line)  # max of fault-layer contact as threshold
    fleak_line = np.copy(grid_minX).astype(int)
    fleak_line[:, fault_max:] = 0  # only returns minima at hanging wall side

    # minmax buffering
    # to set neighboring values of min and max to min and max respectively, too
    if minmax_buffer:
        minXroll1 = np.logical_or(grid_minX, np.roll(grid_minX, 1, axis=0))
        minXroll1[:, :fault_max] = 0
        minXroll2 = np.logical_or(grid_minX, np.roll(grid_minX, -1, axis=0))
        minXroll2[:, :fault_max] = 0
        minXbuffer = np.logical_or(minXroll1, minXroll2)
        grid_minX = np.logical_or(grid_minX, minXbuffer)
        # grid_maxX = np.logical_or(grid_maxX,np.roll(grid_maxX,1))
        # grid_maxX = np.logical_or(grid_maxX,np.roll(grid_maxX,-1))
        # grid_minY = np.logical_or(grid_minY,np.roll(grid_minY,1))
        # grid_minY = np.logical_or(grid_minY,np.roll(grid_minY,-1))
        grid_maxY = np.logical_or(grid_maxY, np.roll(grid_maxY, 1, axis=1))
        grid_maxY = np.logical_or(grid_maxY, np.roll(grid_maxY, -1, axis=1))

    # check for saddle points
    saddle_p1 = np.logical_and(grid_minX, grid_maxY)
    saddle_p2 = np.logical_and(grid_minY, grid_maxX)
    saddle_p_all = np.logical_or(saddle_p1, saddle_p2)

    # this should find saddle points relative to X and Y directions
    # problem of finding other points in a rotated direction?
    ### NOT FINISHED: DEFINE LEAK POINT OVER LEAK LINE MAX?

    # distinguish anticlinal spill points from fault leak points:
    pot_leak_points = np.logical_and(fleak_line, saddle_p_all)
    pot_spill_points = saddle_p_all - pot_leak_points  # substracting leak bool from saddle point bool
    # to get spill point bool
    # leak and spill point 3D coordinates
    # LEAK POINT
    pot_leak_Xcoord = grid_x[pot_leak_points]
    pot_leak_Ycoord = grid_y[pot_leak_points]
    pot_leak_Zcoord = grid_z0[pot_leak_points]
    pot_leak_3Dcoord = np.array(list(zip(pot_leak_Xcoord, pot_leak_Ycoord, pot_leak_Zcoord)))

    if pot_leak_3Dcoord.size == 0:
        fault_leak_3Dcoord = np.array([])  # if no leak coordinates found, set to empty array
    else:
        max_leak_pos = pot_leak_3Dcoord[:, 2].argmax(axis=0)
        fault_leak_3Dcoord = pot_leak_3Dcoord[max_leak_pos, :]  # max is LP

    # SPILL POINT
    pot_spill_Xcoord = grid_x[pot_spill_points]
    pot_spill_Ycoord = grid_y[pot_spill_points]
    pot_spill_Zcoord = grid_z0[pot_spill_points]
    pot_spill_3Dcoord = np.array(list(zip(pot_spill_Xcoord, pot_spill_Ycoord, pot_spill_Zcoord)))

    if pot_spill_3Dcoord.size == 0:
        anticline_spill_3Dcoord = np.array([])  # if no leak coordinates found, set to empty array
    else:
        max_spill_pos = pot_spill_3Dcoord[:, 2].argmax(axis=0)
        anticline_spill_3Dcoord = pot_spill_3Dcoord[max_spill_pos, :]  # max is SP

    # PLOTTING (for visualization and checking)
    # plot of min/max bools and all potential LPs(+) and SPs(x):
    if print_figures == True:
        figsize(15, 6)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(grid_x, grid_y, grid_z0, c="b", alpha = 0.1)
        ax.scatter(grid_x, grid_y, grid_minY, c="b", alpha=0.1)
        ax.scatter(grid_x, grid_y, grid_maxY, c="r", alpha=0.1)
        ax.scatter(grid_x, grid_y, grid_minX, c="b", alpha=0.1)
        ax.scatter(grid_x, grid_y, grid_maxX, c="r", alpha=0.1)
        ax.scatter(grid_x, grid_y, pot_spill_points, c="black", alpha=1, marker='x', s=250)
        ax.scatter(grid_x, grid_y, pot_leak_points, c="black", alpha=1, marker='+', s=250)
        # ax.scatter(grid_x, grid_y, fleak_line, c="b", alpha = 1, marker='+', s= 250)
        # ax.scatter(grid_x, grid_y, leak_max[2], c="g", alpha = 1, marker='+', s= 250)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

        # plot of reservoir top surface and position of all potential LPs(+) and SPs(x):
        plot_spill_leak(res_surf, pot_spill_points, anticline_spill_3Dcoord, fault_leak_3Dcoord, grid_x, grid_y,
                        grid_z0)

    return anticline_spill_3Dcoord, fault_leak_3Dcoord


# PLOTTING FUNCTIONS: Spill and leak point visualization
def plot_spill_leak(res_surface, pot_spills, spill_point, leak_point, grid_x, grid_y, grid_z0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_x, grid_y, grid_z0, c="b", alpha=0.1)
    # ax.scatter(grid_x, grid_y,pot_spills, c="r", alpha = 1, marker='p', s = 250)
    if spill_point.size != 0:
        ax.scatter(spill_point[0], spill_point[1], spill_point[2], c="black", alpha=1, marker='x', s=250)
    if leak_point.size != 0:
        ax.scatter(leak_point[0], leak_point[1], leak_point[2], c="black", alpha=1, marker='+', s=250)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

# LITHOLOGY BLOCK MASKING FUNCTIONS

# Masking function that confines to voxels which are part fo the reservoir formation,
# in the footwall side and above the z-horizontal defined by the spill or leak point
def res_mask(lith,fault, bottom_z, formation_bool, fault_bool):
    mask = np.ones_like(lith)
    mask[~formation_bool] = False
    mask[fault_bool] = False
    #mask[fault.astype(bool)] = False
    bottom_z = round((bottom_z/scale_factor)+0.5) #rounding up to avoid voxel connection to outside borders
    mask = mask.reshape(resolution,resolution,resolution)
    mask[:,:,:bottom_z] = False
    mask = mask.reshape(resolution**3,)
    return mask

def topo_analysis(lith, fault, seal_th, base_n, res_n, secres_n, seal_n, over_n):
    traps = []
    SSF = 0
    trap_control = 0
    # Padding of lith- and fault-block (creating border sections, formation number = 8)
    lith_pad = (np.pad(lith.reshape(resolution,resolution,resolution), 1, 'constant', constant_values = (8,8)))
    fault_pad = (np.pad(fault.reshape(resolution,resolution,resolution), 1, 'constant', constant_values = (8,8)))
    G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot, topo_block = \
    gp.topology.topology_analyze(lith_pad, fault_pad, 1, areas_bool=False, return_block=True)
    # Finding sections which are possible traps in the reservoir (formation number = 7)
    trap_keys = [int(k) for k in lith_to_labels_lot["7"].keys()] # potential trap sections
    pad_keys = [int(p) for p in lith_to_labels_lot["8"].keys()] # model border sections
    base_keys = [int(b) for b in lith_to_labels_lot["%s" % base_n].keys()]
    res_keys = [int(r) for r in lith_to_labels_lot["%s" % res_n].keys()]
    secres_keys = [int(j) for j in lith_to_labels_lot["%s" % secres_n].keys()] # sections in Secondary Reservoir
    over_keys = [int(o) for o in lith_to_labels_lot["%s" % over_n].keys()] # section in Overlying
    tot_under_keys = np.append(res_keys, base_keys) # all sections that belong to underlying below seal
    tot_over_keys = np.append(secres_keys, over_keys) # all sections that belong to overlying above seal
    bool_block = np.zeros_like(topo_block)
    gp.topology.classify_edges(G, centroids, lith.reshape(50,50,50), fault.reshape(50,50,50))
    # If too many traps, model probably broken:
    print("NUMBER OF TRAPS", len(trap_keys))
    if len(trap_keys) > 10:
        trap_mask = bool_block[1:-1,1:-1,1:-1]
        SSF = 0
        trap_control = 0
        return trap_mask, SSF, trap_control
    else:
        # Check for each possible trap section, if it is connected to a border
        for tk in trap_keys:
            print('Checking Section:', tk)
            # Check for adjacency to model border
            if gp.topology.check_adjacency(G, tk, pad_keys[0]) == False:
                # Check for connection (juxtaposition!) to overlying layers (above seal) and SSF
                fault_seal_bool, SSF, trap_control = juxta_SSF_check(G, tk, res_keys, tot_under_keys, tot_over_keys, topo_block, seal_th)
                if fault_seal_bool == True:
                    traps.append(tk)
                    print('TRAP:', tk)
        if traps == []:
            print('No trap found! =(')
        else:
            for i in traps:
                # Creating a mask from the trap sections in the label block
                top_bool = topo_block == i
                bool_block[top_bool] = True
        # Removing padding
        trap_mask = bool_block[1:-1,1:-1,1:-1]
        return trap_mask, SSF, trap_control # this mask returns True for all voxels which were accepted as trap voxels


def juxta_SSF_check(G, tk, res_keys, tot_under_keys, tot_over_keys, topo_block, seal_th):
    throw = 0
    jp = 0
    trap_c = 4
    # FIRST: check the adjacency area only for the trap: only the point of highest throw is relevant for SSF
    for tk in res_keys:
        for ok in tot_over_keys:
            if gp.topology.check_adjacency(G, tk, ok) == True:
                if G.adj[tk][ok]['edge_type'] == 'stratigraphic':
                    print("Stratigraphic adjacency, leakage assumed!")
                    sealing = False
                    SSF = 0
                    trap_c = 4
                    return sealing, SSF, trap_c  # stratigraphic adjancency assumed to always leak
                elif G.adj[tk][ok]['edge_type'] == 'fault':
                    # print('TEST0', gp.Topology.compute_adj_shape(tk,ok,topo_block).shape)
                    # print('TEST1', gp.Topology.compute_adj_shape(tk,ok,topo_block))
                    # print('TEST2', gp.Topology.compute_adj_shape(tk,ok,topo_block)[3])
                    # TEST = gp.Topology.compute_adj_shape(tk,ok,topo_block)
                    # print('TEST:', TEST)
                    trap_jshape = gp.topology.compute_adj_shape(tk, ok, topo_block)[3]
                    # fault throw at edge of trap section
                    if trap_jshape != 0:
                        y_ext = trap_jshape[1]
                        z_ext = trap_jshape[2]
                        # finding point of greatest throw for trap --> relevant for total fault throw and SSF
                        # bottom of trap is flat, so need to find only highest z-values for juxtaposition:
                        z_max = max(z_ext)
                        # finding the y.axis range in which this highest throw appears:
                        y_range = y_ext[z_ext == z_max]
                        y_min = min(y_range)
                        y_max = max(y_range)
                        jp += 1

    # SECOND: check for adjancencies between all seal-underlying and seal-overlying formations to
    # determine total fault throw at point/range defined over trap (y_bool)
    if jp != 0:
        for uk in tot_under_keys:
            for ok in tot_over_keys:
                if gp.topology.check_adjacency(G, uk, ok) == True:
                    # print("Adjacency with section:", ok)
                    if G.adj[uk][ok]['edge_type'] == 'stratigraphic':
                        print("Stratigraphic adjacency, leakage assumed!")
                        sealing = False
                        SSF = 0
                        trap_c = 4
                        return sealing, SSF, trap_c  # stratigraphic adjancency assumed to always leak
                    elif G.adj[uk][ok]['edge_type'] == 'fault':
                        # get the array for z-extent for this adj-area, then mask to relevant range
                        section_jshape = gp.topology.compute_adj_shape(uk, ok, topo_block)[3]
                        if section_jshape != 0:
                            section_z_ext = section_jshape[2]
                            # creating mask where values within this range = True --> to appl
                            section_y_ext = section_jshape[1]
                            y_ext_bool = np.logical_and(section_y_ext >= y_min, section_y_ext <= y_max)
                            y_bool_mask = np.copy(section_y_ext)
                            y_bool_mask[~y_ext_bool] = False
                            y_bool_mask[y_ext_bool] = True
                            if np.count_nonzero(y_bool_mask) != 0:
                                # the following can be used to find the throw height for one area
                                y_bool = y_bool_mask == True
                                section_z_range = section_z_ext[y_bool]
                                # get the throw height:
                                z_len = max(section_z_range) - min(section_z_range) + 1
                                # calculate throw in [m] for this adjacency and add to total
                                section_throw = z_len * scale_factor
                                throw += section_throw
                                print("Adding %s to throw, adjacency between %s and %s" % (section_throw, uk, ok))
                                jp += 1
    if jp == 0:
        print('No JP with any layer over seal.')
        SSF = 0
        sealing = True
        trap_c = 0
        return sealing, SSF, trap_c
    else:
        print('JP with layer over seal, checking SSF.')
        tot_fault_throw = throw + seal_th
        SSF = tot_fault_throw / seal_th
        print('SSF = %s / %s = %s' % (tot_fault_throw, seal_th, SSF))
        if SSF > SSF_c:
            print('SSF threshold exceeded, shale smear assumed to be incomplete.')
            sealing = False
            trap_c = 3
            return sealing, SSF, trap_c
        else:
            print('SSF in confidence range, complete shale smear sealing assumed.')
            sealing = True
            trap_c = 0
            return sealing, SSF, trap_c


# MAIN FUNCTION: Calculation of maximum fill volume in reservoir traps

def max_trap_vol(interp_data, lith, fault, res_surface, seal_th):
    trap_control = 0  # intitiating variable that indicates what kind of mechanism controls the trap volume
    # 0 = Unclear
    # 1 = Spill Point
    # 2 = Leak Point (LEAK UNDER)
    # 3 = Seal breach juxtaposition leakage (LEAK OVER)
    # 4 = Stratigraphical adjacency leakage
    base_n = int(interp_data.geo_data_res.formations.loc['basement', 'formation_number'].values[0])
    res_n = int(interp_data.geo_data_res.formations.loc['Base_Top', 'formation_number'].values[0])
    seal_n = int(interp_data.geo_data_res.formations.loc['Res_Top', 'formation_number'].values[0])
    secres_n = int(interp_data.geo_data_res.formations.loc['Seal_Top', 'formation_number'].values[0])
    over_n = int(interp_data.geo_data_res.formations.loc['SecRes_Top', 'formation_number'].values[0])
    mainfault_n = int(interp_data.geo_data_res.formations.loc['MainFault', 'formation_number'].values[0])

    spill_point, leak_point = spill_leak_P(interp_data, res_surface, lith, fault, print_figures=False)
    print("SPILL POINT AT:", spill_point)
    print("LEAK POINT AT:", leak_point)
    # CHECK: ONLY CONTINUE IF SPILL POINT FOUND, ELSE RETURN ZERO TRAP VOLUME
    if spill_point.size == 0:
        spill_z = np.nan
        print('No SPILL POINT found!')
        trap_vol = 0
        final_trap_mask = np.zeros_like(lith)
        return trap_vol, final_trap_mask, np.nan, 0
    else:
        spill_z = spill_point[2]
        # spill_z, spill_p, spill_min_line, spill_bottom = spill_point(res_surface)
        # calculate leak point
        if leak_point.size == 0:
            print('No LEAK POINT found! So FULL LEAKAGE assumed!')
            leak_z = 2000
        else:
            leak_z = leak_point[2]
        # leak_z, leak_p, leak_line, leak_bottom = leak_point(res_surface)

        # Check for "down-to" z-horizon, maximum depth of reservoir
        # Check for fault sealing and subsequent relevance of leak point
        max_z = np.nanmax([spill_z, leak_z])
        if max_z == spill_z:
            trap_control = 1
        else:
            trap_control = 2

        rounded_lith = np.around(lith).astype(int)
        rounded_fault = np.around(fault).astype(int)
        res_n_bool = rounded_lith == res_n
        fault_bool = rounded_lith == 2

        pre_trap_mask = res_mask(lith, fault, max_z, res_n_bool, fault_bool)

        # volume cells for counting
        vol_cells = 0
        lith_copy = np.copy(lith)

        pre_trap_mask_bool = (pre_trap_mask == True)
        # Check if there is anaything in the reservoir mask at all
        if np.count_nonzero(pre_trap_mask_bool) == 0:
            print("No res_formation above max_z!")
            SSF = 0
            trap_control = 0
            return 0, pre_trap_mask, SSF, trap_control
        else:
            lith_copy[pre_trap_mask_bool] = 7  # setting reservoir above bottom on footwall side
            final_trap_mask, SSF, trap_c = topo_analysis(lith_copy, fault, seal_th, base_n, res_n, secres_n, seal_n, over_n)
            if trap_c == 3:
                trap_control = trap_c
            elif trap_c == 4:
                trap_control = trap_c
            lith_copy[final_trap_mask] = 9  # setting final trap to formation value = 9
            vol_cells = np.count_nonzero(final_trap_mask)

            # calulate volume from cells
            trap_vol = ((scale_factor) ** 3) * vol_cells
            # revert to lith_block without any masks
            # lith[pre_trap_mask_bool] = res_n
            # return the maximum reservoir volume
            return trap_vol, final_trap_mask, SSF, trap_control