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
from matplotlib import pyplot as plt
from skimage import measure
from scipy.spatial import distance

def get_gradient_minima(geo_data, GX,GY,GZ=np.nan, direction='z'):

    # for scaling from voxels up to original scale using the original extent
    vox_size_x = geo_data.extent[1] / geo_data.resolution[0]
    vox_size_y = geo_data.extent[3] / geo_data.resolution[1]
    vox_size_z = geo_data.extent[5] / geo_data.resolution[2]
    vox_size_diag = np.sqrt(vox_size_x ** 2 + vox_size_y ** 2 + vox_size_z ** 2)

    if direction == 'z':
        gx = GX
        gy = GY
    elif direction == 'y':
        gx = GY
        gy = GZ
    elif direction == 'x':
        gx = GZ
        gy = GY
    else:
        raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")

    # using marching cubes to aquire surfaces (vertices, simplices) that align
    # with the occurrence of zeros of the gradients (gradient minima)
    v_gx0 = measure.marching_cubes_lewiner(gx, 0)[0]
    v_gy0 = measure.marching_cubes_lewiner(gy, 0)[0]

    v_gx = v_gx0
    v_gx[:, 0] = v_gx0[:, 0] * vox_size_x
    v_gx[:, 1] = v_gx0[:, 1] * vox_size_y
    v_gx[:, 2] = v_gx0[:, 2] * vox_size_z

    v_gy = v_gy0
    v_gy[:, 0] = v_gy0[:, 0] * vox_size_x
    v_gy[:, 1] = v_gy0[:, 1] * vox_size_y
    v_gy[:, 2] = v_gy0[:, 2] * vox_size_z

    dist_gxy = distance.cdist(v_gx, v_gy, 'euclidean')

    # get distance minima and minima positions for both vertices groups
    # this way we can pair 2 vertices from gx and gy based on their
    # common distance which is to be minimal (smaller than to all other points)
    minx = np.min(dist_gxy, axis=1)
    miny = np.min(dist_gxy, axis=0)
    minx_pos = np.argmin(dist_gxy, axis=1)
    miny_pos = np.argmin(dist_gxy, axis=0)

    # set a cut-off value for minimal distance (here: 3D-diagonal of a voxel)
    gx_cut_bool = minx < vox_size_diag
    gy_cut_bool = miny < vox_size_diag

    # need to pair the vertices of one group to those of the other
    # for this we actually only need the minima positions of one vertices group
    ### pair the mins of the shorter array onto the longer array
    if len(v_gy) >= len(v_gx):
        vgy_paired = v_gy[minx_pos]
        # limit (cut down) the vertices groups to only those
        # below the defined distance threshold
        vgx_cut = v_gx[gx_cut_bool]
        vgy_cut = vgy_paired[gx_cut_bool]

        V1 = vgx_cut
        V2 = vgy_cut
        V_mean = (V1 + V2) / 2
    else:
        vgx_paired = v_gx[miny_pos]
        vgy_cut = v_gy[gy_cut_bool]
        vgx_cut = vgx_paired[gy_cut_bool]

        V1 = vgx_cut
        V2 = vgy_cut
        V_mean = (V1 + V2) / 2

    return V_mean

def get_gradmin_intersect(geo_data, surface_vertices, grad_minima):
    vox_size_x = geo_data.extent[1] / geo_data.resolution[0]
    vox_size_y = geo_data.extent[3] / geo_data.resolution[1]
    vox_size_z = geo_data.extent[5] / geo_data.resolution[2]
    vox_size_diag = np.sqrt(vox_size_x ** 2 + vox_size_y ** 2 + vox_size_z ** 2)

    #v_l = np.array(surface_vertices[0])
    v_l = np.array(surface_vertices)
    l_dist = distance.cdist(grad_minima, v_l)
    min_dist = np.min(l_dist, axis=0)
    l_cut_bool = min_dist < vox_size_diag
    intersect = v_l[l_cut_bool]
    return intersect

def get_voxel_extrema(GX, GY, GZ=np.nan, direction='z'):

    if direction == 'z':
        gx = GX
        gy = GY
    elif direction == 'y':
        gx = GY
        gy = GZ
    elif direction == 'x':
        gx = GZ
        gy = GY
    else:
        raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")

    # getting array with gradient signs (-1 for negative, 1 for positive and exactly 0)
    gx_signs = np.sign(gx)
    gy_signs = np.sign(gy)
    # empty holder arrays for inserting minima and maxima
    gx_maxima = np.zeros_like(gx)
    gy_maxima = np.zeros_like(gy)
    gx_minima = np.zeros_like(gx)
    gy_minima = np.zeros_like(gy)

    # create boolean arrays where signchange voxels are True
    signchange_gx = ((np.roll(gx_signs, 1, axis=0) - gx_signs) != 0).astype(int)
    # avoid border error from np.roll by setting relevant border to False
    signchange_gx[0, :, :] = 0

    signchange_gy = ((np.roll(gy_signs, 1, axis=1) - gy_signs) != 0).astype(int)
    signchange_gy[:, 0] = 0

    # conditions for a voxel to be recognized as maximum or minimum in given direction
    gx_max_cond = (signchange_gx == 1) & (gx_signs == 1)
    gy_max_cond = (signchange_gy == 1) & (gy_signs == 1)
    gx_min_cond = (signchange_gx == 1) & (gx_signs == -1)
    gy_min_cond = (signchange_gy == 1) & (gy_signs == -1)
    # voxels which meet the conditions are marked as True
    # for maxima and minima accordingly
    # since signchange only identifies the next index AFTER the signchange,
    # we include the voxel BEFORE the signchange by doing an according roll
    gx_maxima[gx_max_cond] = 1
    gx_maxima2 = np.roll(gx_maxima, -1, axis=1)
    gx_max_final = gx_maxima + gx_maxima2
    gy_maxima[gy_max_cond] = 1
    gy_maxima2 = np.roll(gy_maxima, -1, axis=0)
    gy_max_final = gy_maxima + gy_maxima2
    # overall maxima in BOTH directions:
    vox_maxima = np.logical_and(gx_max_final, gy_max_final)
    # NOTE: 0 gives a POSITIVE sign! Might have to correct for this?
    # analogous process for minima:
    gx_minima[gx_min_cond] = 1
    gx_minima2 = np.roll(gx_minima, -1, axis=1)
    gx_min_final = gx_minima + gx_minima2
    gy_minima[gy_min_cond] = 1
    gy_minima2 = np.roll(gy_minima, -1, axis=0)
    gy_min_final = gy_minima + gy_minima2
    vox_minima = np.logical_and(gx_min_final, gy_min_final)
    # saddle points as max in one and min in the other direction
    vox_saddles = np.logical_or(np.logical_and(gx_min_final, gy_max_final),
                                np.logical_and(gy_min_final, gx_max_final))

    return vox_minima, vox_maxima, vox_saddles

def get_surface_extrema(geo_data, surface_vertices, GX, GY, plot_figure=False):
    vox_size_x = geo_data.extent[1] / geo_data.resolution[0]
    vox_size_y = geo_data.extent[3] / geo_data.resolution[1]
    vox_size_z = geo_data.extent[5] / geo_data.resolution[2]
    vox_size_diag = np.sqrt(vox_size_x ** 2 + vox_size_y ** 2 + vox_size_z ** 2)

    grad_minima = get_gradient_minima(geo_data, GX,GY)
    intersect = get_gradmin_intersect(geo_data, surface_vertices, grad_minima)
    vox_minima, vox_maxima, vox_saddles = get_voxel_extrema(GX, GY)

    # get the coordinates for minima, maxima and saddles
    MAX_coord0 = np.argwhere(vox_maxima == True)
    MIN_coord0 = np.argwhere(vox_minima == True)
    SADD_coord0 = np.argwhere(vox_saddles == True)

    # rescale the coordinates to actual size of voxels
    # to use combined with the intersection coordinates from above
    MAX_coord = MAX_coord0
    MAX_coord[:, 0] = (MAX_coord0[:, 0] * vox_size_x)  # + vox_size_x/2
    MAX_coord[:, 1] = (MAX_coord0[:, 1] * vox_size_y)  # + vox_size_y/2
    MAX_coord[:, 2] = (MAX_coord0[:, 2] * vox_size_z)  # + vox_size_z/2

    MIN_coord = MIN_coord0
    MIN_coord[:, 0] = (MIN_coord0[:, 0] * vox_size_x)  # + vox_size_x/2
    MIN_coord[:, 1] = (MIN_coord0[:, 1] * vox_size_y)  # + vox_size_y/2
    MIN_coord[:, 2] = (MIN_coord0[:, 2] * vox_size_z)  # + vox_size_z/2

    SADD_coord = SADD_coord0
    SADD_coord[:, 0] = (SADD_coord0[:, 0] * vox_size_x)  # + vox_size_x/2
    SADD_coord[:, 1] = (SADD_coord0[:, 1] * vox_size_y)  # + vox_size_y/2
    SADD_coord[:, 2] = (SADD_coord0[:, 2] * vox_size_z)  # + vox_size_z/2

    # get distances between intersection and the according extrema coordinates
    dist_MIN = distance.cdist(intersect, MIN_coord, 'euclidean')
    dist_MAX = distance.cdist(intersect, MAX_coord, 'euclidean')
    dist_SADD = distance.cdist(intersect, SADD_coord, 'euclidean')

    # classify intersection extrema by limiting to distance to according voxel coordinates
    # half a voxel-diagonal to get what is "inside" a voxel (best results, yet)
    min_dist_MIN = np.min(dist_MIN, axis=1)
    cut_bool_MIN = min_dist_MIN < vox_size_diag / 2
    intersect_minima_all = intersect[cut_bool_MIN]

    min_dist_MAX = np.min(dist_MAX, axis=1)
    cut_bool_MAX = min_dist_MAX < vox_size_diag / 2
    intersect_maxima_all = intersect[cut_bool_MAX]

    min_dist_SADD = np.min(dist_SADD, axis=1)
    cut_bool_SADD = min_dist_SADD < vox_size_diag / 2
    intersect_saddles_all = intersect[cut_bool_SADD]

    if plot_figure == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(surface_vertices[:, 0], surface_vertices[:, 1], surface_vertices[:, 2], color='k', alpha=0.05)
        ax.scatter(intersect_minima_all[:, 0], intersect_minima_all[:, 1], intersect_minima_all[:, 2],
                   color='turquoise', s=200, marker='+')
        ax.scatter(intersect_maxima_all[:, 0], intersect_maxima_all[:, 1], intersect_maxima_all[:, 2], color='r', s=200,
                   marker='x')
        ax.scatter(intersect_saddles_all[:, 0], intersect_saddles_all[:, 1], intersect_saddles_all[:, 2],
                   color='violet', s=200, marker='*')

        ax.set_xlim(geo_data.extent[0], geo_data.extent[1])
        ax.set_ylim(geo_data.extent[2], geo_data.extent[3])
        ax.set_zlim(geo_data.extent[4], geo_data.extent[5])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    return intersect_minima_all, intersect_maxima_all, intersect_saddles_all

def get_spill_point(geo_data, surface_vertices, GX, GY, plot_figure=False):
    intersect_saddles_all = get_surface_extrema(geo_data, surface_vertices, GX, GY, plot_figure)[2]
    # get highest saddle point as the relevant point
    final_SADD = intersect_saddles_all[np.argmax(intersect_saddles_all[:, 2])]
    if plot_figure == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(surface_vertices[:, 0], surface_vertices[:, 1], surface_vertices[:, 2], color='k', alpha=0.05)
        ax.scatter(final_SADD[0], final_SADD[1], final_SADD[2], color='violet', s=300, marker='*')

        ax.set_xlim(geo_data.extent[0], geo_data.extent[1])
        ax.set_ylim(geo_data.extent[2], geo_data.extent[3])
        ax.set_zlim(geo_data.extent[4], geo_data.extent[5])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
    return final_SADD

def get_surface_max(geo_data, surface_vertices, GX, GY, plot_figure=True):
    intersect_maxima_all = get_surface_extrema(geo_data, surface_vertices, GX, GY, plot_figure)[1]
    final_MAX = intersect_maxima_all[np.argmax(intersect_maxima_all[:, 2])]
    if plot_figure == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(surface_vertices[:, 0], surface_vertices[:, 1], surface_vertices[:, 2], color='k', alpha=0.05)
        ax.scatter(final_MAX[0], final_MAX[1], final_MAX[2], color='r', s=200, marker='x')

        ax.set_xlim(geo_data.extent[0], geo_data.extent[1])
        ax.set_ylim(geo_data.extent[2], geo_data.extent[3])
        ax.set_zlim(geo_data.extent[4], geo_data.extent[5])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
    return final_MAX
