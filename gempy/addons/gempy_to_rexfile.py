"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


    Created on 21/02/2020

    @author: Miguel de la Varga
"""

import numpy as np

rexFileHeaderSize = 64
rexCoordSize = 22

file_header_size = 86
rexDataBlockHeaderSize = 16

file_header_and_data_header = 102
mesh_header_size = 128
all_header_size = 230

# Supported block types
# typeLineSet = 0
# typeText = 1
# typePointList = 2
typeMesh = 3
# typeImage = 4
# typeMaterial = 5
# typePeopleSimulation = 6
# typeUnityPackage = 7
# typeSceneNode = 8


n_bytes = 0


def encode(input_: list):
    global n_bytes
    block = bytearray()

    for tup in input_:
        arr = np.array(tup[0], dtype=tup[1]).tobytes()
        n_bytes += len(arr)
        block += arr

    return block


def write_header_block(n_data_blocks, size_data_blocks, version=1, start_data=86, offsets=None):
    """
    Function that writes the header block of a rexfile:

    Args:
        n_data_blocks:
        size_data_blocks:
        version:
        start_data:
        offsets:

    Returns:

    """
    reserved = '0'*42
    if offsets is None:
        offsets = [0, 0, 0]

    input_ = [('REX1', 'bytes'),
              (version, 'uint16'),
              (0, 'uint32'),
              (n_data_blocks, 'uint16'),
              (start_data, 'uint16'),
              (size_data_blocks, 'uint64'),
              (reserved, 'bytes'),
              (3876, 'uint32'),
              (4, 'uint16'),
              ('EPSG', 'bytes'),
              (offsets, 'float32')]

    block_bytes = encode(input_)
    return block_bytes


def write_data_block_header(size_data, data_id = 1, data_type=3, version_data=1):
    """
    Function to write a data block header.

    Args:
        size_data:
        data_id:
        data_type:
        version_data:

    Returns:

    """

    input_ = [(data_type, 'uint16'),
              (version_data, 'uint16'),
              (size_data, 'uint32'),
              (data_id, 'uint64')]

    block_bytes = encode(input_)
    return block_bytes


def write_mesh_header(n_vtx_coord,  n_triangles,
                      start_vtx_coord, start_nor_coord, start_tex_coord, start_vtx_colors, start_triangles,
                      name, material_id=1,  # material_id=9223372036854775807
                      n_nor_coord=0, n_tex_coord=0, n_vtx_colors=0,
                      lod=1, max_lod=1):
    """

    Args:
        n_vtx_coord:
        n_triangles:
        start_vtx_coord:
        start_nor_coord:
        start_tex_coord:
        start_vtx_colors:
        start_triangles:
        name:
        material_id:
        n_nor_coord:
        n_tex_coord:
        n_vtx_colors:
        lod:
        max_lod:

    Returns:

    """


    str_size = len(name)
    rest_name_ = ' ' * (74 - str_size)
    full_name = name + rest_name_

    input_ = [([lod, max_lod], 'uint16'),
              ([n_vtx_coord,     n_nor_coord,     n_tex_coord,     n_vtx_colors,     n_triangles,
                start_vtx_coord, start_nor_coord, start_tex_coord, start_vtx_colors, start_triangles],
               'uint32'),
              (material_id, 'uint64'),
              (str_size, 'uint16'),
              (full_name, 'bytes')]

    block_bytes = encode(input_)
    return block_bytes


def write_mesh_coordinates(vertex, triangles, normal=None, texture=None, colors=None):
    """


    Args:
        vertex:
        triangles:
        normal:
        texture:
        colors:

    Returns:

    """

    ver = vertex.ravel()
    tri = triangles.ravel()
    if normal is None:
        normal = []
    if texture is None:
        texture = []
    if colors is None:
        colors = []

    input_ = [(ver, 'float32'),
              (normal, 'float32'),
              (texture, 'float32'),
              (colors, 'float32'),
              (tri, 'uint32')]

    block_bytes = encode(input_)
    return block_bytes


def mesh_preprocess(ver, tri):
    ver_ravel = ver.ravel()
    tri_ravel = tri.ravel()
    n_vtx_coord = ver_ravel.shape[0]
    n_triangles = tri_ravel.shape[0]
    return ver_ravel, tri_ravel, n_vtx_coord, n_triangles


def write_file(bytes, path: str):
    newFile = open(path + ".rex", "wb")
    newFile.write(bytes)
    return True


def write_material_data(ka_red=255.0/255, ka_green=255.0/255, ka_blue=255.0/255, ka_texture_ID=9223372036854775807,  # ambient
                        ks_red=255.0/255, ks_green=255.0/255, ks_blue=255.0/255, ks_texture_ID=9223372036854775807,  # specular
                        kd_red=255.0/255, kd_green=255.0/255, kd_blue=255.0/255, kd_texture_ID=9223372036854775807,  # diffuse
                        ns=0.1,  #specular exponent
                        alpha=1  #opacity
                        ):
    """
    writes a standard material definition block

    Returns: bytes (size:68) representation of the material

    """

    input_ = [(ka_red, 'float32'), (ka_green, 'float32'), (ka_blue, 'float32'), (ka_texture_ID, 'uint64'),
              (ks_red, 'float32'), (ks_green, 'float32'), (ks_blue, 'float32'), (ks_texture_ID, 'uint64'),
              (kd_red, 'float32'), (kd_green, 'float32'), (kd_blue, 'float32'), (kd_texture_ID, 'uint64'),
              (ns,'float32'), (alpha, 'float32')]

    block_bytes = encode(input_)
    return block_bytes


# TODO Move to utils
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def geo_model_to_rex(geo_model, path='./gempy_rex'):
    file_names = []
    mesh_header_size = 128
    file_header_size = 86
    e = 0

    for idx, surface_vals in geo_model.surfaces.df.iterrows():
        ver = surface_vals['vertices']
        tri = surface_vals['edges']
        if tri is np.nan:
            break

        col = surface_vals['color']

        colors = (np.zeros_like(ver) + hex_to_rgb(col))/255

        ver_ = np.copy(ver)
        ver_[:, 2] = ver[:, 1]
        ver_[:, 1] = ver[:, 2]

        tri_ = np.copy(tri)

        ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver_, tri_)
        mesh_block_size_no_data_block_header = (2 * n_vtx_coord + n_triangles) * 4 + mesh_header_size
        material_block_size_no_data_block_header = 68
        # Write header
        n_data_blocks = 3
        header_bytes = write_header_block(n_data_blocks=n_data_blocks,
                                          size_data_blocks=mesh_block_size_no_data_block_header +
                                                           rexDataBlockHeaderSize +
                                                           material_block_size_no_data_block_header,
                                          start_data=file_header_size)

        # Write data block
        data_bytes = write_data_block_header(size_data=mesh_block_size_no_data_block_header,
                                             data_id=1, data_type=3, version_data=1)

        # Write mesh block
        mesh_header_bytes = write_mesh_header(n_vtx_coord / 3, n_triangles / 3, n_vtx_colors=n_vtx_coord/3,
                                              start_vtx_coord=mesh_header_size,
                                              start_nor_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_tex_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_vtx_colors=mesh_header_size + n_vtx_coord * 4,
                                              start_triangles=mesh_header_size + 2 *
                                                              (n_vtx_coord * 4),
                                              name='test_a', material_id=0)

        mesh_block_bytes = write_mesh_coordinates(ver_ravel, tri_ravel, colors=colors.ravel())

        # Write data block
        data_bytes_r = write_data_block_header(size_data=mesh_block_size_no_data_block_header,
                                               data_id=2, data_type=3, version_data=1)

        # One side of the normals
        tri_[:, 2] = tri[:, 1]
        tri_[:, 1] = tri[:, 2]

        ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver_, tri_)

        # Write mesh block
        mesh_header_bytes_r = write_mesh_header(n_vtx_coord / 3, n_triangles / 3, n_vtx_colors=n_vtx_coord / 3,
                                              start_vtx_coord=mesh_header_size,
                                              start_nor_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_tex_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_vtx_colors=mesh_header_size + n_vtx_coord * 4,
                                              start_triangles=mesh_header_size + 2 *
                                                              (n_vtx_coord * 4),
                                              name='test_a', material_id=0)

        mesh_block_bytes_r = write_mesh_coordinates(ver_ravel, tri_ravel, colors=colors.ravel())

        # Write material block
        material_header_bytes = write_data_block_header(data_type=5, version_data=1, size_data=68, data_id=0)
        material_bytes = write_material_data()

        all_bytes = header_bytes + data_bytes + mesh_header_bytes + mesh_block_bytes +\
                                 data_bytes_r + mesh_header_bytes_r + mesh_block_bytes_r +\
                                 material_header_bytes + material_bytes

        file_name = path+surface_vals['surface']
        write_file(all_bytes, file_name)
        file_names.append(file_name+'.rex')
        e += 1

    return file_names














































































