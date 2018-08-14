"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


@author: Fabian A. Stamm
"""

import numpy as np
import scipy.signal as sg
from matplotlib import pyplot as plt
from scipy.spatial import distance


def get_fault_mask(geo_data, fault_sol, fault_n, fault_side='both'):
    fault_block = fault_sol[0].astype(int).reshape(geo_data.resolution[0],
                                                   geo_data.resolution[1], geo_data.resolution[2])

    # boolean conditions for hanging and footwall
    hw_cond = fault_block == fault_n
    fw_cond = fault_block == (fault_n + 1)
    hw_cond = np.pad(hw_cond, (1), 'edge')
    fw_cond = np.pad(fw_cond, (1), 'edge')

    ### ROLLING
    # FW MASK 1
    roll_x11 = np.roll(hw_cond, 1, axis=0)
    roll_y11 = np.roll(hw_cond, 1, axis=1)
    roll_z11 = np.roll(hw_cond, -1, axis=2)

    roll_x11[hw_cond] = False
    roll_y11[hw_cond] = False
    roll_z11[hw_cond] = False

    # HW MASK 1
    roll_x21 = np.copy(hw_cond)
    roll_y21 = np.copy(hw_cond)
    roll_z21 = np.copy(hw_cond)

    roll_x21_cut = np.roll(hw_cond, -1, axis=0)
    roll_y21_cut = np.roll(hw_cond, -1, axis=1)
    roll_z21_cut = np.roll(hw_cond, 1, axis=2)

    roll_x21[roll_x21_cut] = False
    roll_y21[roll_y21_cut] = False
    roll_z21[roll_z21_cut] = False

    # FW MASK 2
    roll_x22 = np.copy(fw_cond)
    roll_y22 = np.copy(fw_cond)
    roll_z22 = np.copy(fw_cond)

    roll_x22_cut = np.roll(fw_cond, -1, axis=0)
    roll_y22_cut = np.roll(fw_cond, -1, axis=1)
    roll_z22_cut = np.roll(fw_cond, 1, axis=2)

    roll_x22[roll_x22_cut] = False
    roll_y22[roll_y22_cut] = False
    roll_z22[roll_z22_cut] = False

    # HW MASK 2
    roll_x12 = np.roll(fw_cond, 1, axis=0)
    roll_y12 = np.roll(fw_cond, 1, axis=1)
    roll_z12 = np.roll(fw_cond, -1, axis=2)

    roll_x12[fw_cond] = False
    roll_y12[fw_cond] = False
    roll_z12[fw_cond] = False

    # COMBINE BOTH DIRECTIONS - HW
    f_mask_hw1 = np.logical_or(roll_z21, np.logical_or(roll_x21, roll_y21))
    f_mask_hw2 = np.logical_or(roll_z12, np.logical_or(roll_x12, roll_y12))
    f_mask_hw = np.logical_or(f_mask_hw1, f_mask_hw2)[1:-1, 1:-1, 1:-1]

    # COMBINE BOTH DIRECTIONS - FW
    f_mask_fw1 = np.logical_or(roll_z11, np.logical_or(roll_x11, roll_y11))
    f_mask_fw2 = np.logical_or(roll_z22, np.logical_or(roll_x22, roll_y22))
    f_mask_fw = np.logical_or(f_mask_fw1, f_mask_fw2)[1:-1, 1:-1, 1:-1]

    # COMBINE BOTH SIDES
    f_mask_both = np.logical_or(f_mask_hw, f_mask_fw)

    if fault_side == 'both':
        return f_mask_both

    elif fault_side == 'hanging wall' or fault_side == 'hw':
        return f_mask_hw

    elif fault_side == 'footwall' or fault_side == 'fw':
        return f_mask_fw
    else:
        print('fault side has do be chosen as both, footwall or hanging wall.')
        return np.nan

def get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                       lith_n, fault_n, fault_side='both'):
    fault_block = fault_sol[0].astype(int).reshape(geo_data.resolution[0],
                                     geo_data.resolution[1],geo_data.resolution[2])
    fault_mask = get_fault_mask(geo_data, fault_sol, fault_n, fault_side)
    lith_block = lith_sol[0].astype(int).reshape(geo_data.resolution[0],
                                     geo_data.resolution[1],geo_data.resolution[2])
    lith_cond = lith_block == lith_n
    if fault_side == 'hanging wall' or fault_side == 'hw':
        fs_cond = fault_block == fault_n
    elif fault_side == 'footwall' or fault_side == 'fw':
        fs_cond = fault_block == (fault_n+1)
    elif fault_side == 'both':
        fs_cond = np.ones_like(fault_block)
    else:
        print('fault side has do be chosen as both, footwall or hanging wall.')
        return np.nan
    lith_cut = lith_cond * fs_cond
    vox_contact = fault_mask * lith_cut
    return vox_contact


def project_voxels(voxel_array, projection='automatic'):
    if projection == 'automatic':
        d_x = (np.max(voxel_array[:, 0]) - np.min(voxel_array[:, 0]))
        d_y = (np.max(voxel_array[:, 1]) - np.min(voxel_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    if projection == 'yz':
        p = 0
    elif projection == 'xz':
        p = 1
    else:
        print('Projection plane should be yz, xz or automatic.')
        p = 0
    proj = np.zeros_like(voxel_array)
    pos = np.argwhere(voxel_array == True)
    pos[:, p] = 0
    proj[pos[:, 0], pos[:, 1], pos[:, 2]] = True
    return proj


def get_extrema_line_projected(projected_array, ext_type='max'):
    if ext_type == 'max':
        roll = np.roll(projected_array, -1, axis=2)
        roll[:, :, -1] = 0
        ext_line = np.bitwise_xor(projected_array, roll)
        ext_line[~projected_array] = 0
    elif ext_type == 'min':
        roll = np.roll(projected_array, 1, axis=2)
        roll[:, :, 0] = 0
        ext_line = np.bitwise_xor(projected_array, roll)
        ext_line[~projected_array] = 0
    else:
        print('ext_type must be either max or min.')
        ext_line = np.nan
    return ext_line


def get_extrema_line_voxels(voxel_array, ext_type='max', projection='automatic', form='projected'):
    projected_array = project_voxels(voxel_array, projection)
    ext_line_p = get_extrema_line_projected(projected_array, ext_type)
    if form == 'projected':
        return ext_line_p
    elif form == 'original':
        return ext_line_o
    # ext_line_r = np.zeros_like(voxel_array)
    # if projection == 'automatic':
    #    d_x = (np.max(projected_array[:,0])-np.min(projected_array[:,0]))
    #    d_y = (np.max(projected_array[:,1])-np.min(projected_array[:,1]))
    #    if d_x > d_y:
    #        i = 0
    #    else:
    #        i = 1
    # elif projection == 'yz':
    #    i = 1
    # elif projection == 'xz':
    #    i = 0
    # rcond1 = voxel_array[:,2] == ext_line_p[:,2]
    # ext_line_r[rcond1] = 1
    ### need to find out how to come back to original form
    else:
        print('form needs to be projected or original.')


def get_juxtaposition(hw_array, fw_array):
    juxtapos = np.logical_and(hw_array, fw_array)
    return juxtapos


def plot_allan_diagram(hw_array, fw_array, projection='automatic'):
    if projection == 'automatic':
        d_x = (np.max(hw_array[:, 0]) - np.min(hw_array[:, 0]))
        d_y = (np.max(hw_array[:, 1]) - np.min(hw_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    fw_proj = project_voxels(fw_array, projection)
    fw_maxline = get_extrema_line_voxels(fw_array, ext_type='max', projection=projection)
    fw_minline = get_extrema_line_voxels(fw_array, ext_type='min', projection=projection)
    fw_between = np.bitwise_xor(fw_proj, np.logical_or(fw_maxline, fw_minline))

    hw_proj = project_voxels(hw_array, projection)
    hw_maxline = get_extrema_line_voxels(hw_array, ext_type='max', projection=projection)
    hw_minline = get_extrema_line_voxels(hw_array, ext_type='min', projection=projection)
    hw_between = np.bitwise_xor(hw_proj, np.logical_or(hw_maxline, hw_minline))

    juxtapos = np.logical_and(fw_proj, hw_proj)

    if projection == 'yz':
        diagram = np.zeros_like(hw_array[0, :, :].astype(int))

        diagram[fw_maxline[0, :, :]] = 1
        diagram[fw_minline[0, :, :]] = 1
        diagram[fw_between[0, :, :]] = 1

        diagram[hw_maxline[0, :, :]] = 2
        diagram[hw_minline[0, :, :]] = 2
        diagram[hw_between[0, :, :]] = 2

        diagram[juxtapos[0, :, :]] = 3
    elif projection == 'xz':
        diagram = np.zeros_like(hw_array[:, 0, :].astype(int))

        diagram[fw_maxline[:, 0, :]] = 1
        diagram[fw_minline[:, 0, :]] = 1
        diagram[fw_between[:, 0, :]] = 1

        diagram[hw_maxline[:, 0, :]] = 2
        diagram[hw_minline[:, 0, :]] = 2
        diagram[hw_between[:, 0, :]] = 2

        diagram[juxtapos[:, 0, :]] = 3

    else:
        return np.nan
        # RAISE ERROR
    plt.imshow(diagram.T, origin='bottom', cmap='viridis')
    return diagram


def plot_footwall_projection(fw_array, projection='automatic'):
    if projection == 'automatic':
        d_x = (np.max(fw_array[:, 0]) - np.min(fw_array[:, 0]))
        d_y = (np.max(fw_array[:, 1]) - np.min(fw_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    fw_proj = project_voxels(fw_array, projection)
    if projection == 'yz':
        plt.imshow(fw_proj[0, :, :].T, origin='bottom', cmap='viridis')
    elif projection == 'xz':
        plt.imshow(fw_proj[:, 0, :].T, origin='bottom', cmap='viridis')
    else:
        print('Projection plane should be yz, xz or automatic.')


def plot_hanging_wall_projection(hw_array, projection='automatic'):
    if projection == 'automatic':
        d_x = (np.max(fw_array[:, 0]) - np.min(fw_array[:, 0]))
        d_y = (np.max(fw_array[:, 1]) - np.min(fw_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    hw_proj = project_voxels(fw_array, projection)
    if projection == 'yz':
        plt.imshow(hw_proj[0, :, :].T, origin='bottom', cmap='viridis')
    elif projection == 'xz':
        plt.imshow(hw_proj[:, 0, :].T, origin='bottom', cmap='viridis')
    else:
        print('Projection plane should be yz, xz or automatic.')


def get_lith_fault_intersect(v_f, v_l):
    # cutting layer surface vertices (v_l) down to maximal extent of fault surface (v_f)
    f_x_extent = np.array([np.min(v_f[:, 0]), np.max(v_f[:, 0])])
    f_y_extent = np.array([np.min(v_f[:, 1]), np.max(v_f[:, 1])])
    f_z_extent = np.array([np.min(v_f[:, 2]), np.max(v_f[:, 2])])

    x_cond1 = v_l[:, 0] >= f_x_extent[0]
    x_cond2 = v_l[:, 0] <= f_x_extent[1]
    x_cond = np.logical_and(x_cond1, x_cond2)

    y_cond1 = v_l[:, 1] >= f_y_extent[0]
    y_cond2 = v_l[:, 1] <= f_y_extent[1]
    y_cond = np.logical_and(y_cond1, y_cond2)

    z_cond1 = v_l[:, 2] >= f_z_extent[0]
    z_cond2 = v_l[:, 2] <= f_z_extent[1]
    z_cond = np.logical_and(z_cond1, z_cond2)

    xyz_cond = np.logical_and(z_cond, np.logical_and(x_cond, y_cond))

    v_cut = v_l[xyz_cond]

    # find intersection between layer and fault surface
    fl_dist = distance.cdist(v_cut, v_f)
    min_dist = np.min(fl_dist, axis=0)
    fl_cut_bool = min_dist < (vox_size_diag / 2)
    fault_intersect = v_f[fl_cut_bool]
    holder = np.zeros_like(v_f)
    holder[fl_cut_bool] = 1

    fl_dist = distance.cdist(v_f, v_cut)
    min_dist = np.min(fl_dist, axis=0)
    fl_cut_bool = min_dist < (vox_size_diag / 2)
    fault_intersect2 = v_cut[fl_cut_bool]

    return fault_intersect, fault_intersect2, holder.astype(bool)


def get_layer_fault_contact(fault_vertices, layer_vertices, voxel_array, \
                            projection='yz', fault_side='footwall'):
    if projection == 'automatic':
        d_x = (np.max(voxel_array[:, 0]) - np.min(voxel_array[:, 0]))
        d_y = (np.max(voxel_array[:, 1]) - np.min(voxel_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    if projection == 'yz':
        p = 0
    elif projection == 'xz':
        p = 1
    else:
        print('Projection plane should be yz, xz or automatic.')
        p = 0
    intersection_surface = get_lith_fault_intersect(fault_vertices, layer_vertices)[0]
    maxline_vox = get_extrema_line_voxels(voxel_array, ext_type='max', projection=projection)
    maxpos_vox = np.argwhere(maxline_vox == True)
    # rescaling
    maxpos_vox[:, 0] = maxpos_vox[:, 0] * vox_size_x
    maxpos_vox[:, 1] = maxpos_vox[:, 1] * vox_size_y
    maxpos_vox[:, 2] = maxpos_vox[:, 2] * vox_size_z
    maxpos_vox_red = np.delete(maxpos_vox, p, 1)
    intersection_red = np.delete(intersection_surface, p, 1)
    mi_dist = distance.cdist(maxpos_vox_red, intersection_red)
    min_dist = np.min(mi_dist, axis=0)
    mi_cut_bool = min_dist < (vox_size_diag / 1000)
    top_line = intersection_surface[mi_cut_bool]
    return top_line


def get_contact_peaks(fault_vertices, layer_vertices, voxel_array, \
                      projection='yz', fault_side='footwall', \
                      order='automatic'):
    if order == 'automatic':
        np.int(np.round(((geo_data.resolution[0] + geo_data.resolution[1]) / 2) / 2))
    top_line = get_layer_fault_contact(fault_vertices, layer_vertices, voxel_array, \
                                       projection=projection, fault_side=fault_side)
    relmaxpos = sg.argrelextrema(top_line[:, 2], np.greater_equal, order=order)
    peaks = top_line[relmaxpos]
    return peaks
