"""
This file is part of gempy.

Created on 21/02/2020

@author: Miguel de la Varga
"""

import numpy as np
import matplotlib.colors as mcolors
import pandas as pd

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


class GemPyToRex:
    def __init__(self, geo_model=None):
        """Writes GemPy data structures into binary Rex

         https://github.com/roboticeyes/openrex/blob/master/doc/rex-spec-v1.md
        """
        self.rex_bytes = bytearray()
        self.n_bytes = 0

        self.data_id = 0
        self.geo_model = geo_model

    def __call__(self, geo_model=None, meshes=True, material=True,
                 surfaces=None, topography=True, app='GemPlay'):
        """

        Args:
            meshes:
            surfaces (list): Subset of surfaces to send to the client
            app (str): Either RexViewer or GemPlay. Set of default values

        Returns:

        """

        byte_array = bytearray()
        byte_size = 0
        self.data_id = 0

        if geo_model is None:
            geo_model = self.geo_model
        else:
            self.geo_model = geo_model

        if surfaces is not None:
            raise NotImplementedError

        flip_yz, backside, vertex_color = self.default_values(app)
        # flip_yz, backside, vertex_color = False, True, False

        surface_df = self.grab_meshes(geo_model)
        if topography:
            topography_dict = self.grab_topography(geo_model)
        else:
            topography_dict = None

        # Data Blocks
        # -----------
        if material is True:
            # Material
            byte_array += self.gempy_color_to_rex_material(surface_df, topography)

        if meshes is True:
            # Mesh
            byte_array += self.gempy_meshes_to_rex(
                surface_df,
                topography_dict=topography_dict,
                flip_yz=flip_yz,
                backside=backside,
                vertex_color=vertex_color)

        # Size of all data blocks together
        byte_size += len(byte_array)

        # Write file header
        # -----------------
        n_data_blocks = self.data_id

        header_bytes = write_file_header_block(
            n_data_blocks=n_data_blocks,
            size_data_blocks=byte_size,
            start_data=file_header_size)

        return header_bytes + byte_array

    @staticmethod
    def grab_meshes(geo_model):
        """Check if surfaces are computed. And return a pandas.DataFrame with
         the meshes to be converted

        Args:
            geo_model:

        Returns:

        """

        try:
            # Drop basement
            surface_df = geo_model._surfaces.df.groupby(
                ['isActive', 'isBasement']).get_group((True, False))
        except (IndexError, KeyError):
            raise RuntimeError('No computed surfaces yet.')

        return surface_df[['surface', 'vertices', 'edges', 'color']]

    @staticmethod
    def grab_topography(geo_model):
        from scipy.spatial import Delaunay

        if geo_model._grid.topography is None or geo_model._grid.topography.values.shape[0] == 0:
            return None
        else:
            topography_dict = dict()
            topography_dict['surface'] = "Topography"
            topography_dict['vertices'] = geo_model._grid.topography.values
            # tri = Delaunay(geo_model._grid.topography.values)
            topography_dict['edges'] = Delaunay(geo_model._grid.topography.values[:, :2]).simplices
            topography_dict['color'] = geo_model.solutions.geological_map

            return topography_dict

    @staticmethod
    def hex_to_rgb(hex: str, normalize: bool = True) -> np.ndarray:
        """Transform colors from hex to rgb"""
        hex = hex.lstrip('#')
        hlen = len(hex)
        rgb = np.array([int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3)])
        if normalize is True:
            rgb = rgb / 255
        return rgb

    @staticmethod
    def default_values(app):
        """

        Args:
            app:

        Returns:
            list: flip_yz, backside, vertex_color
        """
        if app == 'GemPlay':
            return (False, True, False)
        elif app == 'RexView':
            return (True, True, True)
        else:
            raise AttributeError('app must be either GemPlay or RexView')

    def gempy_meshes_to_rex(self,
                            surface_df,
                            topography_dict=None,
                            flip_yz=False,
                            backside=True,
                            vertex_color=False):
        """Write mesh to Rexfile.

        Args:
            surface_df:
            topography_dict:
            flip_yz: Fliping YZ coordinates. Rexview need this
            backside: If True, create a second set of triangles on the backside of the mesh
            vertex_color

        Returns:

        Notes:
            At the moment 14.07.2020 it is not possible to write normals or texture

        """
        rex_bytes = bytearray()
        mesh_number = 0

        # Loop geological surfaces surfaces
        for idx, surface_vals in surface_df.iterrows():

            tri = surface_vals['edges']
            if tri is np.nan:
                continue

            ver = surface_vals['vertices']
            surface_name = surface_vals['surface']
            if vertex_color:
                # Hex Colors
                col_ = surface_vals['color']
            else:
                col_ = None

            rex_bytes = self.mesh_prepare_and_encode(rex_bytes, mesh_number, ver, tri,
                                                     surface_name, col_=col_,
                                                     flip_yz=flip_yz, backside=backside,
                                                     vertex_color=False)
            mesh_number += 1

        # Add topography
        if topography_dict is not None:

            rex_bytes = self.mesh_prepare_and_encode(rex_bytes, n_surface=-1,
                                                     ver=topography_dict['vertices'],
                                                     tri=topography_dict['edges'],
                                                     surface_name=topography_dict['surface'],
                                                     col_=topography_dict['color'],
                                                     flip_yz=flip_yz, backside=backside,
                                                     vertex_color=True)

        return rex_bytes

    def mesh_prepare_and_encode(self, rex_bytes, n_surface, ver, tri, surface_name, col_=None, \
                                flip_yz=False,
                                backside=True,
                                vertex_color=False):

        if flip_yz:
            # This depends. For RexViewer we need to flip XYZ. For GemPlay not really
            ver_ = np.copy(ver)
            ver[:, 2] = ver_[:, 1]
            ver[:, 1] = ver_[:, 2]

        # Pre-processing GemPy output
        ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver, tri)

        # Number of vertex colors
        if vertex_color:
            n_vtx_colors = n_vtx_coord

            # Give color to each vertex
            # TODO: Is this necessary if I pass a material
            if type(col_) is str:
                colors = np.zeros_like(ver) + self.hex_to_rgb(col_, normalize=True)
                c_r = colors.ravel()

            elif type(col_) is np.ndarray:
                surf_df = self.geo_model._surfaces.df.set_index('id')
                colors_hex = surf_df.groupby(
                    ['isActive', 'isFault']).get_group((True, False))['color']

                colors_rgb_ = colors_hex.apply(lambda val: list(mcolors.hex2color(val)))
                colors_rgb = pd.DataFrame(colors_rgb_.to_list(), index=colors_hex.index)

                sel = np.round(col_[0]).astype(int)[0]
                c_r = colors_rgb.loc[sel].values.ravel()
            else:
                raise AttributeError("col_ must be either hex string or rgb array")
        else:
            n_vtx_colors = 0
            c_r = None

        rex_bytes = self._mesh_encode(
            rex_bytes, n_surface,
            n_vtx_coord, n_triangles, n_vtx_colors,
            surface_name, ver_ravel, tri_ravel, c_r)

        if backside:
            # Coping triangles to create the backside normal of the layers
            tri_ = np.copy(tri)
            # TURN normals - One side of the normals
            tri_[:, 2] = tri[:, 1]
            tri_[:, 1] = tri[:, 2]
            # Pre-processing GemPy output
            ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver, tri_)
            # tri = np.append(tri, tri_)

            rex_bytes = self._mesh_encode(
                rex_bytes, n_surface,
                n_vtx_coord, n_triangles, n_vtx_colors,
                surface_name, ver_ravel, tri_ravel, c_r)

        return rex_bytes

    def _mesh_encode(self, rex_bytes, material_id,
                     n_vtx_coord, n_triangles, n_vtx_colors,
                     surface_name, ver_ravel, tri_ravel, c_r):
        # Write Mesh block - header
        mesh_header_bytes = write_mesh_header(
            n_vtx_coord / 3, n_triangles / 3,
            n_vtx_colors=n_vtx_colors / 3,
            start_vtx_coord=mesh_header_size,
            start_nor_coord=mesh_header_size + n_vtx_coord * 4,
            start_tex_coord=mesh_header_size + n_vtx_coord * 4,
            start_vtx_colors=mesh_header_size + n_vtx_coord * 4,
            start_triangles=mesh_header_size +
                            ((n_vtx_coord + n_vtx_colors) * 4),
            name=surface_name,
            material_id=material_id  # self.data_id + surface_df.shape[0]
        )

        # Write Mesh block - Vertex, triangles
        mesh_block_bytes = write_mesh_coordinates(ver_ravel, tri_ravel,
                                                  colors=c_r  # When using
                                                  # material we can avoid this
                                                  )

        # Calculate the size of the mesh block
        mesh_block_size_no_data_block_header = len(mesh_header_bytes) + \
                                               len(mesh_block_bytes)  # This is cte 128

        # Write data block header for Mesh 1
        data_header_bytes = write_data_block_header(
            size_data=mesh_block_size_no_data_block_header,
            data_id=self.data_id,
            data_type=3,  # 3 for mesh
            version_data=1  # Probably useful for counting
            # the operation number
        )
        self.data_id += 1
        rex_bytes += data_header_bytes + mesh_header_bytes + mesh_block_bytes

        return rex_bytes

    def gempy_color_to_rex_material(self, surface_df, topography=False):
        rex_bytes = bytearray()

        for idx, surface_vals in surface_df.iterrows():
            # Write data block header for Material 1
            data_header_bytes = write_data_block_header(
                data_type=5,  # Material data type
                version_data=1,  # Version. Probably useful for operation counter
                size_data=68,  # Size of the block is FIXED
                data_id=self.data_id  # self.data_id
            )
            self.data_id += 1

            rgb_color = self.hex_to_rgb(surface_vals['color'], normalize=True)
            # rgb_color = [1, 1, 1]
            # Write Material
            material_bytes = write_material_data(
                ka_red=rgb_color[0], ka_green=rgb_color[1], ka_blue=rgb_color[2],
                ka_texture_ID=9223372036854775807,  # ambient
                ks_red=rgb_color[0], ks_green=rgb_color[1], ks_blue=rgb_color[2],
                ks_texture_ID=9223372036854775807,  # specular
                kd_red=rgb_color[0], kd_green=rgb_color[1], kd_blue=rgb_color[2],
                kd_texture_ID=9223372036854775807,  # diffuse
                ns=0.1,  # specular exponent
                alpha=1  # opacity
            )

            rex_bytes += data_header_bytes + material_bytes

        if topography is True:
            # Write data block header for Material 1
            data_header_bytes = write_data_block_header(
                data_type=5,  # Material data type
                version_data=1,  # Version. Probably useful for operation counter
                size_data=68,  # Size of the block is FIXED
                data_id=-1  # self.data_id
            )

            rgb_color = [1, 1, 1]
            # Write Material
            material_bytes = write_material_data(
                ka_red=rgb_color[0], ka_green=rgb_color[1], ka_blue=rgb_color[2],
                ka_texture_ID=9223372036854775807,  # ambient
                ks_red=rgb_color[0], ks_green=rgb_color[1], ks_blue=rgb_color[2],
                ks_texture_ID=9223372036854775807,  # specular
                kd_red=rgb_color[0], kd_green=rgb_color[1], kd_blue=rgb_color[2],
                kd_texture_ID=9223372036854775807,  # diffuse
                ns=0.1,  # specular exponent
                alpha=1  # opacity
            )

            self.data_id += 1
            rex_bytes += data_header_bytes + material_bytes

        return rex_bytes


def encode(input_: list):
    """Encode python objects - normally Python primitives or numpy arrays - into
    its correspondent byte representation

    Args:
        input_ (List[tuples]): List of tuples: (object, type)

    Returns:
        byte: Array of bytes
    """
    global n_bytes
    block = bytearray()

    for tup in input_:
        arr = np.array(tup[0], dtype=tup[1]).tobytes()
        n_bytes += len(arr)
        block += arr

    return block


def write_file_header_block(n_data_blocks, size_data_blocks, version=1,
                            start_data=86, srid=3876, offsets=None):
    """
    Function that writes the header block of a rexfile:

    Args:
        n_data_blocks:
        size_data_blocks:
        version (int): Version of the file
        start_data (int): Position where data start. This is after the header
         and coordinate system. If everything works fine it should be 86
        srid (int): Spatial reference system identifier (srid)
        offsets:

    Returns:

    """
    reserved = '0' * 42
    if offsets is None:
        offsets = [0, 0, 0]

    input_ = [('REX1', 'bytes'),  # REX1
              (version, 'uint16'),  # file version
              (0, 'uint32'),  # CRC32
              (n_data_blocks, 'uint16'),  # Number of DATA BLOCKS
              (start_data, 'uint16'),  # StartData
              (size_data_blocks, 'uint64'),  # Size of all data blocks
              (reserved, 'bytes'),  # Reserved
              # Coordinate system block
              (srid, 'uint32'),  # Spatial reference system identifier (srid)
              (4, 'uint16'),  # Size of the name of the used system.
              ('EPSG', 'bytes'),  # name of the used system.
              (offsets, 'float32')]  # Global x, y, z offset

    block_bytes = encode(input_)
    return block_bytes


def write_data_block_header(size_data, data_id=1, data_type=3, version_data=1):
    """Function to write a DATA BLOCK header.

    Args:
        size_data: data block size (without header)
        data_id: id which is used in the database
        data_type (int): Type of data the data block contains:
            * 0	LineSet	A list of vertices which get connected by line segments
            * 1	Text	A position information and the actual text
            * 2	PointList	A list of 3D points with color information (e.g. point cloud)
            * 3	Mesh	A triangle mesh datastructure️
            * 4	Image	A single of arbitrary format can be stored in this block
            * 5	MaterialStandard	A standard (mesh) material definition
            * 6	SceneNode	A wrapper around a data block which can be used in the scenegraph
            * 7	Track	A track is a tracked position and orientation of an AR device
        version_data: version for this data block

    Returns:

    """

    input_ = [(data_type, 'uint16'),  # data type
              (version_data, 'uint16'),  # version for this data block
              (size_data, 'uint32'),  # data block size (without header)
              (data_id, 'uint64')]  # id which is used in the database

    block_bytes = encode(input_)
    return block_bytes


def write_mesh_header(n_vtx_coord, n_triangles,
                      start_vtx_coord, start_nor_coord, start_tex_coord, start_vtx_colors,
                      start_triangles,
                      name, material_id=1,  # material_id=9223372036854775807
                      n_nor_coord=0, n_tex_coord=0, n_vtx_colors=0,
                      lod=1, max_lod=1):
    """Function to write MESH DATA BLOCK header. The header size is fixed at 128 bytes.

    Args:
        n_vtx_coord: number of vertex coordinates
        n_triangles: number of triangles
        start_vtx_coord: start vertex coordinate block (relative to mesh block start)
        start_nor_coord: start vertex normals block (relative to mesh block start)
        start_tex_coord: start of texture coordinate block (relative to mesh block start)
        start_vtx_colors: start of colors block (relative to mesh block start)
        start_triangles: start triangle block for vertices (relative to mesh block start)
        name (str): Name of the mesh
        material_id (int):  id which refers to the corresponding material block in this file
        n_nor_coord:  number of normal coordinates (can be zero)
        n_tex_coord:  number of texture coordinates (can be zero)
        n_vtx_colors: number of vertex colors (can be zero)
        lod (int): level of detail for the given geometry
        max_lod (int): maximal level of detail for given geometry

    Returns:
        bytes: array of bytes
    """

    # Strings are immutable so there is no way to modify them in place
    str_size = len(name)  # Size of the actual name of the mesh
    rest_name = ' ' * (74 - str_size)  #
    full_name = name + rest_name

    input_ = [([lod, max_lod], 'uint16'),  # Level of detail
              ([n_vtx_coord,  # number of vertex coordinates
                n_nor_coord,  # number of normal coordinates (can be zero)
                n_tex_coord,  # number of texture coordinates (can be zero)
                n_vtx_colors,  # number of vertex colors (can be zero)
                n_triangles,  # number of triangles
                start_vtx_coord,  # start vertex coordinate block (relative to mesh block start)
                start_nor_coord,  # start vertex normals block (relative to mesh block start)
                start_tex_coord,  # start of texture coordinate block (relative to mesh block start)
                start_vtx_colors,  # start of colors block (relative to mesh block start)
                start_triangles  # start triangle block for vertices (relative to mesh block start)
                ],
               'uint32'),
              (material_id, 'uint64'),
              # id which refers to the corresponding material block in this file
              (str_size, 'uint16'),  # size of the following string name
              (full_name, 'bytes')]  # name of the mesh (this is user-readable)

    block_bytes = encode(input_)
    return block_bytes


def write_mesh_coordinates(vertex, triangles, normal=None, texture=None, colors=None):
    """Block with the coordinates of a mesh. This has to go with a header!

    Args:
        vertex (numpy.ndarray[float32]): Array of vertex XYZXYZ...
        triangles (numpy.ndarray[int32]): This is a list of integers which form
         one triangle. Please make sure that normal and texture coordinates are inline with the
         vertex coordinates. One index refers to the same normal and texture position. The
         triangle orientation is required to be counter-clockwise (CCW)
        normal (numpy.ndarray):
        texture (numpy.ndarray):
        colors (numpy.ndarray):

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


def write_material_data(ka_red=255.0 / 255, ka_green=255.0 / 255, ka_blue=255.0 / 255,
                        ka_texture_ID=9223372036854775807,  # ambient
                        ks_red=255.0 / 255, ks_green=255.0 / 255, ks_blue=255.0 / 255,
                        ks_texture_ID=9223372036854775807,  # specular
                        kd_red=255.0 / 255, kd_green=255.0 / 255, kd_blue=255.0 / 255,
                        kd_texture_ID=9223372036854775807,  # diffuse
                        ns=0.1,  # specular exponent
                        alpha=1  # opacity
                        ):
    """Writes a standard material definition block

    Returns: bytes (size:68) representation of the material

    """

    input_ = [(ka_red, 'float32'), (ka_green, 'float32'), (ka_blue, 'float32'),
              (ka_texture_ID, 'uint64'),
              (ks_red, 'float32'), (ks_green, 'float32'), (ks_blue, 'float32'),
              (ks_texture_ID, 'uint64'),
              (kd_red, 'float32'), (kd_green, 'float32'), (kd_blue, 'float32'),
              (kd_texture_ID, 'uint64'),
              (ns, 'float32'), (alpha, 'float32')]

    block_bytes = encode(input_)
    return block_bytes


# TODO Move to utils
def hex_to_rgb(hex):
    """Transform colors from hex to rgb"""
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def geomodel_to_rex(geo_model, backside=True):
    """

    Args:
        geo_model (gempy.Model):
    """

    # Fixed sizes
    mesh_header_size = 128
    file_header_size = 86

    # Init dict
    rex_bytes = {}

    # Check if surfaces are computed
    try:
        # Drop basement
        surface_df = geo_model._surfaces.df.groupby(
            ['isActive', 'isBasement']).get_group((True, False))
    except (IndexError, KeyError):
        raise RuntimeError('No computed surfaces yet.')

    # Loop surfaces
    for idx, surface_vals in surface_df.iterrows():
        ver = surface_vals['vertices']
        tri = surface_vals['edges']
        if tri is np.nan:
            break

        # Grab surface color
        col = surface_vals['color']

        # Give color to each vertex
        colors = (np.zeros_like(ver) + hex_to_rgb(col)) / 255

        # This depends. For RexViewer we need to flip XYZ. For GemPlay not really
        ver_ = np.copy(ver)
        ver_[:, 2] = ver[:, 1]
        ver_[:, 1] = ver[:, 2]
        # ----------------

        # Coping triangles to create the backside normal of the layers
        tri_ = np.copy(tri)

        # Preprocessing GemPy output
        ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver_, tri_)

        # Calculate the size of the mesh block
        if backside is True:
            n_sides = 2
        else:
            n_sides = 1

        mesh_block_size_no_data_block_header = (2 *  # Coordinates and colors
                                                n_vtx_coord + n_triangles) * 4 + \
                                               mesh_header_size  # This is cte 128

        # Size of a MATERIAL DATA BLOCK is cte
        material_block_size_no_data_block_header = 68

        # Write file header
        if backside is True:
            n_data_blocks = 3
        else:
            n_data_blocks = 2
        header_bytes = write_file_header_block(n_data_blocks=n_data_blocks,
                                               size_data_blocks=
                                               n_sides * mesh_block_size_no_data_block_header +
                                               rexDataBlockHeaderSize +
                                               material_block_size_no_data_block_header,
                                               start_data=file_header_size)

        # Write data block header for Mesh 1
        data_bytes = write_data_block_header(size_data=mesh_block_size_no_data_block_header,
                                             data_id=1, data_type=3, version_data=1)

        # Write Mesh 1 block - header
        mesh_header_bytes = write_mesh_header(n_vtx_coord / 3, n_triangles / 3,
                                              n_vtx_colors=n_vtx_coord / 3,
                                              start_vtx_coord=mesh_header_size,
                                              start_nor_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_tex_coord=mesh_header_size + n_vtx_coord * 4,
                                              start_vtx_colors=mesh_header_size + n_vtx_coord * 4,
                                              start_triangles=mesh_header_size + 2 *
                                                              (n_vtx_coord * 4),
                                              name='rock1', material_id=0)

        # Write Mesh 1 block - header
        mesh_block_bytes = write_mesh_coordinates(ver_ravel, tri_ravel, colors=colors.ravel())

        if backside:
            # Write data block header for Mesh 2
            data_bytes_r = write_data_block_header(size_data=mesh_block_size_no_data_block_header,
                                                   data_id=2, data_type=3, version_data=1)

            # TURN normals - One side of the normals
            tri_[:, 2] = tri[:, 1]
            tri_[:, 1] = tri[:, 2]

            ver_ravel, tri_ravel, n_vtx_coord, n_triangles = mesh_preprocess(ver_, tri_)

            # Write Mesh 2 block - header
            mesh_header_bytes_r = write_mesh_header(n_vtx_coord / 3, n_triangles / 3,
                                                    n_vtx_colors=n_vtx_coord / 3,
                                                    start_vtx_coord=mesh_header_size,
                                                    start_nor_coord=mesh_header_size + n_vtx_coord * 4,
                                                    start_tex_coord=mesh_header_size + n_vtx_coord * 4,
                                                    start_vtx_colors=mesh_header_size + n_vtx_coord * 4,
                                                    start_triangles=mesh_header_size + 2 *
                                                                    (n_vtx_coord * 4),
                                                    name='test_a', material_id=0)

            # Write Mesh 2 block - header
            mesh_block_bytes_r = write_mesh_coordinates(ver_ravel, tri_ravel,
                                                        colors=colors.ravel())

        # Write data block header for Material 1
        material_header_bytes = write_data_block_header(data_type=5, version_data=1, size_data=68,
                                                        data_id=0)

        # Write Material 1
        material_bytes = write_material_data()

        # Putting all data together
        if backside is True:
            all_bytes = header_bytes + data_bytes + mesh_header_bytes + mesh_block_bytes + \
                        data_bytes_r + mesh_header_bytes_r + mesh_block_bytes_r + \
                        material_header_bytes + material_bytes

        else:
            all_bytes = header_bytes + data_bytes + mesh_header_bytes + mesh_block_bytes + \
                        material_header_bytes + material_bytes

        # FOR REXView Saving each surface is a rexfile
        rex_bytes[surface_vals['surface']] = all_bytes
    return rex_bytes


def mesh_preprocess(ver, tri):
    """Prepare GemPy Output to be converted to rex

    Args:
        ver (numpy.ndarray):
        tri (numpy.ndarray):

    Returns:
        list: vertices raveled, triangels ravel, n vertex, n triangles
    """

    # TODO: Remove the type transform. Technically it does nothing
    ver_ravel = ver.ravel().astype('float32')
    tri_ravel = tri.ravel().astype('int32')
    n_vtx_coord = ver_ravel.shape[0]
    n_triangles = tri_ravel.shape[0]
    return ver_ravel, tri_ravel, n_vtx_coord, n_triangles


def write_file(bytes, path: str):
    """Write to disk a rexfile from its binary format"""

    newFile = open(path + ".rex", "wb")
    newFile.write(bytes)
    return True


def write_rex(rex_bytes: dict, path='./gempy_rex'):
    file_names = []
    e = 0
    for key, value in rex_bytes.items():
        file_name = path + key
        write_file(value, file_name)
        file_names.append(file_name + '.rex')
        e += 1

    return file_names
