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
    """
            Get a boolean mask (voxel block) for the fault surface contact to either the
            footwall, the hanging wall side or both.

            Args:
                geo_data (:class:`gempy.data_management.InputData`)
                fault_sol (ndarray): Computed model fault solution.
                fault_n (int): Number of the fault of interest.
                fault_side (string, optional, default='both'): The side of the fault for which the
                    contact is to be returned:
                        'footwall' or 'fw'
                        'hanging wall' or 'hw'
                        'both'.

            Returns:
                Fault surface contact boolean mask (ndarray).
            """

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
        raise AttributeError(str(form) + "must be 'footwall' ('fw'), 'hanging wall' ('hw') or 'both'.")

def get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                       lith_n, fault_n, fault_side='both'):
    """
            Get voxels of a lithology which are in contact with the fault surface, either on the footwall,
            the hanging wall or both sides.

            Args:
                geo_data (:class:`gempy.data_management.InputData`)
                lith_sol (ndarray): Computed model lithology solution.
                fault_sol (ndarray): Computed model fault solution.
                lith_n (int): Number of the lithology of interest (at the moment only one can be chosen at one time).
                fault_n (int): Number of the fault of interest.
                fault_side (string, optional, default='both'): The side of the fault for which the
                    contact is to be returned:
                        'footwall' or 'fw'
                        'hanging wall' or 'hw'
                        'both'.

            Returns:
                Lithology-fault contact voxels as boolean array.
            """

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
        raise AttributeError(str(form) + "must be 'footwall' ('fw'), 'hanging wall' ('hw') or 'both'.")
    lith_cut = lith_cond * fs_cond
    vox_contact = fault_mask * lith_cut
    return vox_contact

def project_voxels(voxel_array, projection='automatic'):
    """
            Project the 'True' voxels in a boolean array onto either the 'yz'- or the 'xz'-plane.

            Args:
                voxel_array (3D boolean array): Boolean array to be projected. Must be in 3D shape.
                projection (string, optional, default='automatic'): Choose the plane onto which the
                    voxels are to be projected:
                        'yz'
                        'xz'
                        'automatic': Automatically determines the plane which is parallel to the length
                            of the voxel spread.

            Returns:
                3D boolean array with True voxels projected onto one plane.
            """
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
        raise AttributeError(str(projection) + "must be declared as planes on 'yz', 'xz' or as 'automatic'.")
        p = 0
    proj = np.zeros_like(voxel_array)
    pos = np.argwhere(voxel_array == True)
    pos[:, p] = 0
    proj[pos[:, 0], pos[:, 1], pos[:, 2]] = True
    return proj
    ### This is currently still a 3D array. Unnecessary. Find a better format (2d array?) and adapt functions!

def get_extrema_line_projected(projected_array, extr_type='max'):
    """
            Get either the top or bottom edge of a projected lithology-fault contact (3D boolean array).

            Args:
                projected_array (3D boolean array): Previously projected array of the contact. Must be in 3D shape.
                extr_type (string, optional, default='max'): Either 'max' for top edge or 'min' for lower edge.

            Returns:
                3D boolean array with 'True' voxels for the according edge line.
            """
    if extr_type == 'max':
        roll = np.roll(projected_array, -1, axis=2)
        roll[:, :, -1] = 0
        ext_line = np.bitwise_xor(projected_array, roll)
        ext_line[~projected_array] = 0
    elif extr_type == 'min':
        roll = np.roll(projected_array, 1, axis=2)
        roll[:, :, 0] = 0
        ext_line = np.bitwise_xor(projected_array, roll)
        ext_line[~projected_array] = 0
    else:
        raise AttributeError(str(extr_type) + "must be either 'min' or 'max.")
    return ext_line

def get_extrema_line_voxels(voxel_array, extrema_type='max', projection='automatic', form='projected'):
    """
        Get either the top or bottom edge of a lithology-fault contact (3D boolean array).

        Args:
                    voxel_array (3D boolean array): Boolean array of the contact. Must be in 3D shape.
                    extr_type (string, optional, default='max'): Either 'max' for top edge or 'min' for lower edge.
                    projection (string, optional, default='automatic'): Choose the plane onto which the
                    voxels are to be projected:
                        'yz'
                        'xz'
                        'automatic': Automatically determines the plane which is parallel to the length
                            of the voxel spread.
                    form (string, optional, default='projected'): In which form the edge is to be returned,
                        either 'projected' or 'original' (not working at this moment).

                Returns:
                    3D boolean array with 'True' voxels for the according edge.
                """
    projected_array = project_voxels(voxel_array, projection)
    extr_line_p = get_extrema_line_projected(projected_array, extrema_type)
    if form == 'projected':
        return extr_line_p
    elif form == 'original':
        return extr_line_o # original form not working yet
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
        raise AttributeError(str(form) + "must be 'projected' or 'original'")

def get_juxtaposition(hw_array, fw_array):
    juxtapos = np.logical_and(hw_array, fw_array) # this should only work with projected arrays
    return juxtapos

def plot_allan_diagram(geo_data, lith_sol, fault_sol, \
                       lith_n, fault_n, projection='automatic'):
    """
            Simple Allan diagram illustration (voxel-based).

            Args:
                geo_data (:class:`gempy.data_management.InputData`)
                lith_sol (ndarray): Computed model lithology solution.
                fault_sol (ndarray): Computed model fault solution.
                lith_n (int): Number of the lithology of interest (at the moment only one can be chosen at one time).
                fault_n (int): Number of the fault of interest.
                projection (string, optional, default='automatic'): Choose the plane onto which the
                    voxels are to be projected:
                        'yz'
                        'xz'
                        'automatic': Automatically determines the plane which is parallel to the length
                            of the voxel spread.

            Returns:
                Allan diagram plot showing the layer-fault contact on footwall and hanging wall side, as well as
                the resulting juxtaposition in different colors.
            """
    fw_array = get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                       lith_n, fault_n, fault_side='fw')
    hw_array = get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                                  lith_n, fault_n, fault_side='hw')

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
        raise AttributeError(str(projection) + "must be declared as planes on 'yz', 'xz' or as 'automatic'.")

    plt.imshow(diagram.T, origin='bottom', cmap='viridis')
    return diagram

def plot_fault_contact_projection(geo_data, lith_sol, fault_sol, \
                       lith_n, fault_n, fault_side='footwall', projection='automatic'):
    """
            Voxel-based illustration of the contact of a lithology with a fault surface.

            Args:
                geo_data (:class:`gempy.data_management.InputData`)
                lith_sol (ndarray): Computed model lithology solution.
                fault_sol (ndarray): Computed model fault solution.
                lith_n (int): Number of the lithology of interest (at the moment only one can be chosen at one time).
                fault_n (int): Number of the fault of interest.
                fault_side (string, optional, default='both'): The side of the fault for which the
                    contact is to be returned:
                        'footwall' or 'fw'
                        'hanging wall' or 'hw'
                        'both'.
                projection (string, optional, default='automatic'): Choose the plane onto which the
                    voxels are to be projected:
                        'yz'
                        'xz'
                        'automatic': Automatically determines the plane which is parallel to the length
                            of the voxel spread.
            """
    if fault_side == 'hanging wall' or fault_side == 'hw':
        w_array = get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                                     lith_n, fault_n, fault_side='hw')
    elif fault_side == 'footwall' or fault_side == 'fw':
        w_array = get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                                     lith_n, fault_n, fault_side='fw')
    elif fault_side == 'both':
        w_array = get_vox_lf_contact(geo_data, lith_sol, fault_sol, \
                                     lith_n, fault_n, fault_side='both')
    else:
        raise AttributeError(str(form) + "must be 'footwall' ('fw'), 'hanging wall' ('hw') or 'both'.")


    if projection == 'automatic':
        d_x = (np.max(w_array[:, 0]) - np.min(w_array[:, 0]))
        d_y = (np.max(w_array[:, 1]) - np.min(w_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    w_proj = project_voxels(w_array, projection)
    if projection == 'yz':
        plt.imshow(w_proj[0, :, :].T, origin='bottom', cmap='viridis')
    elif projection == 'xz':
        plt.imshow(w_proj[:, 0, :].T, origin='bottom', cmap='viridis')
    else:
        raise AttributeError(str(projection) + "must be declared as planes on 'yz', 'xz' or as 'automatic'.")

def plot_direct_wall_projection(w_array, projection='automatic'):
    if projection == 'automatic':
        d_x = (np.max(w_array[:, 0]) - np.min(w_array[:, 0]))
        d_y = (np.max(w_array[:, 1]) - np.min(w_array[:, 1]))
        if d_x > d_y:
            projection = 'xz'
        else:
            projection = 'yz'
    w_proj = project_voxels(w_array, projection)
    if projection == 'yz':
        plt.imshow(w_proj[0, :, :].T, origin='bottom', cmap='viridis')
    elif projection == 'xz':
        plt.imshow(w_proj[:, 0, :].T, origin='bottom', cmap='viridis')
    else:
        raise AttributeError(str(projection) + "must be declared as planes on 'yz', 'xz' or as 'automatic'.")

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
        raise AttributeError(str(projection) + "must be declared as planes on 'yz', 'xz' or as 'automatic'.")
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
