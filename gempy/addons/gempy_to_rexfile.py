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
rexDataBlockHeaderSize = 16

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


def encode(input_: list):
    block = bytearray()

    for tup in input_:
        block += np.array(tup[0], dtype=tup[1]).tobytes()

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
    reserved = [0]*42
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
              ('EPSG', 'bytes'),
              (offsets, 'float32')]

    block_bytes = encode(input_)
    return block_bytes


def write_data_block(size_data, data_id = 1, data_type=3, version_data=1):
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
                      name, material_id = 0,
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
    full_name = ' ' * 74
    full_name[:str_size] = name

    input_ = [([lod, max_lod], 'uint16'),
              ([n_vtx_coord, n_nor_coord, n_tex_coord, n_vtx_colors, n_triangles,
                start_vtx_coord, start_tex_coord, start_nor_coord, start_tex_coord, start_vtx_colors, start_triangles],
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


def write_file(bytes, name: str):
    newFile = open(name+".rex", "wb")
    newFile.write(bytes)
    return True
















































































