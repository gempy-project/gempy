from typing import Union

from gempy.core.data_modules.stack import Stack
from gempy import Surfaces, Grid

try:
    import xarray as xr
    import subsurface
    from subsurface.structs.base_structures.common_data_utils import to_netcdf
except:
    print("Not subsurface compatibility available")



import numpy as np
import pandas as pd


class XSolution(object):
    """This class stores the output of the interpolation and the necessary objects
    to visualize and manipulate this data using xarray as backend.

    Depending on the activated grid (see :class:`Grid`) a different number of
     properties are returned returned:

    Args:
        grid (Grid): [s0]
        surfaces (Surfaces): [s1]
        series (Series): [s2]

    Attributes:
        grid (Grid)
        surfaces (Surfaces)
        series (Series)
        weights_vector (numpy.array): [s3]
        scalar_field_at_surface_points (numpy.array): [s4]
        block_at_surface_points (numpy.array): [s5]
        mask_at_surface_points (numpy.array): [s6]
        values_at_surface_points (numpy.array): [s7]
        lith_block (numpy.array): [s8]
        scalar_field_matrix (numpy.array): [s9]
        block_matrix (numpy.array): [s10]
        mask_matrix (numpy.array): [s11]
        mask_matrix_pad (numpy.array): mask matrix padded 2 block in order to guarantee that the layers intersect each
         other after marching cubes
        values_matrix (numpy.array): [s12]
        vertices (list[numpy.array]): [s13]
        edges (list[numpy.array]): [s14]
        geological_map (numpy.array): [s15]

    """

    def __init__(self, grid: Grid,
                 surfaces: Surfaces = None,
                 stack: Stack = None,
                 ):

        self.grid = grid
        self.stack = stack
        self.surfaces = surfaces  # Used to store ver/sim there

        # Define xarrays
        self.weights_vector = None
        self.at_surface_points = None
        self.s_regular_grid = None  # xr.Dataset()
        self.s_custom_grid = None  # xr.Dataset()
        self.s_topography = None  # xr.Dataset()
        self.s_at_surface_points = None  # xr.Dataset()
        self.s_sections = dict()
        self.meshes = None  # xr.Dataset()

    def set_meshes(self, surfaces: Surfaces = None):
        """Create xarray from the Surfaces object. In GemPy Engine we will set
         them directly.
        """

        surf_ver_sim = surfaces.df[['id', 'vertices', 'edges']]
        vertex = []
        simplex = []
        ids = []
        last_idx = 0
        self.extract_each_surface_representations(ids, last_idx, simplex,
                                                  surf_ver_sim, vertex)

        vertex_array = np.concatenate(vertex)
        simplex_array = np.concatenate(simplex)
        ids_array = np.concatenate(ids)

        self.meshes = subsurface.UnstructuredData.from_array(
            vertex=vertex_array,
            cells=simplex_array,
            attributes=pd.DataFrame(ids_array, columns=['id'])
        )

        return self.meshes

    @staticmethod
    def extract_each_surface_representations(ids, last_idx, simplex,
                                             surf_ver_sim, vertex):
        for index, row in surf_ver_sim.iterrows():
            v_ = row['vertices']
            e_ = row['edges'] + last_idx
            if v_ is not np.nan and e_ is not np.nan:
                i_ = np.ones(e_.shape[0]) * row['id']
                vertex.append(v_)
                simplex.append(e_)
                ids.append(i_)
                last_idx = e_[-20:].max() + 1

    def set_values(self,
                   values: list,
                   active_features=None,
                   surf_properties=None,
                   ):
        """ At this stage we should split values into the different grids

        Args:
            values:

        Returns:

        """

        # Get an array with all the indices for each grid
        l = self.grid.length

        coords_base, xyz = self.prepare_common_args(active_features,
                                                    surf_properties)
        self.weights_vector = values[3]

        if self.grid.active_grids_bool[0]:
            self.set_values_to_regular_grid_(values, l[0], l[1], coords_base.copy())
        if self.grid.active_grids_bool[1]:
            self.set_values_to_custom_grid(values, l[1], l[2], coords_base.copy(),
                                           xyz=xyz)
        if self.grid.active_grids_bool[2]:
            self.set_values_to_topography(values, l[2], l[3], coords_base.copy())
        if self.grid.active_grids_bool[3]:
            self.set_values_to_sections(values, l[3], l[4], coords_base.copy())
        if self.grid.active_grids_bool[4]:
            self.set_values_to_centered()

        # TODO: Add xyz from surface points
        self.set_values_to_surface_points_(values, l[-1], coords_base, xyz=None)

    def prepare_common_args(self, active_features, surf_properties):
        if active_features is None and self.stack is not None:
            active_features = self.stack.df.groupby('isActive').get_group(True).index
        if surf_properties is None and self.surfaces is not None:
            surf_properties = self.surfaces.properties_val
        coords_base = dict()
        if active_features is not None:
            coords_base['Features'] = active_features.to_list()
        if surf_properties is not None:
            coords_base['Properties'] = surf_properties.to_list()
        if self.grid.custom_grid is not None:
            xyz = self.grid.custom_grid.values
        else:
            xyz = None
        return coords_base, xyz

    def set_values_to_centered(self):
        return

    def set_values_to_surface_points_(self, values, l0, coords_base, xyz=None):
        coords = coords_base
        l1 = values[0].shape[-1]
        arrays = self.create_unstruct_xarray(values, l0, l1, xyz)

        self.s_at_surface_points = xr.Dataset(
            data_vars=arrays,
            coords=coords
        )
        return self.s_at_surface_points

    @staticmethod
    def create_struc_xarrays(values, l0, l1, res: Union[list, np.ndarray]):
        arrays = dict()

        n_dim = len(res)
        xyz = ['X', 'Y', 'Z'][:n_dim]
        if values[0] is not None:
            # This encompass lith_block and values matrix
            property_matrix = xr.DataArray(
                data=values[0][:, l0:l1].reshape(-1, *res),
                dims=['Properties', *xyz],
            )
            arrays['property_matrix'] = property_matrix

        if values[1] is not None:
            # This is the block matrix
            i, j, _ = values[1].shape
            block_matrix = xr.DataArray(
                data=values[1][:, :, l0:l1].reshape(i, j, *res),
                dims=['Features', 'Properties', *xyz],
            )
            arrays['block_matrix'] = block_matrix

            # Fault block?

        if values[4] is not None:
            # Scalar field matrix
            scalar_matrix = xr.DataArray(
                data=values[4][:, l0:l1].reshape(-1, *res),
                dims=['Features', *xyz],
            )
            arrays['scalar_field_matrix'] = scalar_matrix

        if values[6] is not None:
            # Mask matrix
            mask_matrix = xr.DataArray(
                data=values[6][:, l0:l1].reshape(-1, *res),
                dims=['Features', *xyz],
            )
            arrays['mask_matrix'] = mask_matrix

        if values[7] is not None:
            # Fault mask matrix
            fault_mask = xr.DataArray(
                data=values[7][:, l0:l1].reshape(-1, *res),
                dims=['Features', *xyz],
            )
            arrays['fault_mask'] = fault_mask

        return arrays

    @staticmethod
    def create_unstruct_xarray(values, l0, l1, xyz):
        arrays = dict()
        if xyz is not None:
            cartesian_matrix = xr.DataArray(
                data=xyz,
                dims=['Point', 'XYZ'],
                coords={'XYZ': ['X', 'Y', 'Z']}
            )
            arrays['cartesian_matrix'] = cartesian_matrix

        if values[0] is not None:
            # Values and lith block
            property_v3 = xr.DataArray(
                data=values[0][:, l0:l1],
                dims=['cell_attr', 'cell'],
            )

            arrays['property'] = property_v3

        if values[1] is not None:
            # block
            block_v3 = xr.DataArray(
                data=values[1][:, :, l0:l1],
                dims=['Features', 'cell_attr', 'cell'],
            )

            arrays['block'] = block_v3

        if values[4] is not None:
            # Scalar field
            scalar_field_v3 = xr.DataArray(
                data=values[4][:, l0:l1],
                dims=['Features', 'cell'],
            )
            arrays['scalar_field'] = scalar_field_v3

        if values[6] is not None:
            # Scalar field
            mask_v3 = xr.DataArray(
                data=values[6][:, l0:l1],
                dims=['Features', 'cell'],
            )
            arrays['mask'] = mask_v3

        return arrays

    def set_values_to_custom_grid(self, values: list, l0, l1,
                                  coords_base: dict, xyz=None):

        coords = coords_base
        arrays = self.create_unstruct_xarray(values, l0, l1, xyz=None)

        self.s_custom_grid = subsurface.UnstructuredData.from_array(
            vertex=xyz,
            cells="points",
            cells_attr=arrays,
            coords=coords,
            default_cells_attr_name="block"
        )

        return self.s_custom_grid

    def set_values_to_regular_grid_(self, values: list, l0, l1,
                                    coords_base: dict):

        coords = self.add_cartesian_coords(coords_base)

        arrays = self.create_struc_xarrays(values, l0, l1,
                                           self.grid.regular_grid.resolution)

        self.s_regular_grid = subsurface.StructuredData.from_dict(data_dict=arrays, coords=coords)

    def add_cartesian_coords(self, coords_base):
        coords = coords_base
        coords['X'] = self.grid.regular_grid.x
        coords['Y'] = self.grid.regular_grid.y
        coords['Z'] = self.grid.regular_grid.z
        return coords

    def set_values_to_topography(self,
                                 values: list,
                                 l0, l1,
                                 coords_base):
        coords = coords_base
        coords['X'] = self.grid.topography.x
        coords['Y'] = self.grid.topography.y
        resolution = self.grid.topography.resolution
        arrays = self.create_struc_xarrays(values, l0, l1, resolution)

        self.s_topography = subsurface.StructuredData.from_dict(data_dict=arrays, coords=coords)
        return self.s_topography

    def set_values_to_sections(self,
                               values: list,
                               l0, l1,
                               coords_base):
        coords = coords_base
        sections = self.grid.sections

        for e, axis_coord in enumerate(sections.generate_axis_coord()):
            resolution = sections.resolution[e]
            l0_s = sections.length[e]
            l1_s = sections.length[e + 1]
            name, xy = axis_coord
            coords['X'] = xy[:, 0]
            coords['Y'] = xy[:, 1]

            arrays = self.create_struc_xarrays(values, l0 + l0_s, l0 + l1_s, resolution)

            self.s_sections[name] = subsurface.StructuredData.from_dict(data_dict=arrays, coords=coords)
        return self.s_sections

    @property
    def data_structures(self):
        # TODO: Add sections
        args = [self.s_regular_grid, self.s_custom_grid, self.s_topography, self.meshes]
        names = ['regular_grid', 'custom_grid', 'topography', 'meshes']
        return zip(args, names)

    def to_netcdf(self, path, name, **kwargs):
        for a, n in self.data_structures:
            if a is not None:
                to_netcdf(a, f'{path}/{name}_{n}.nc', **kwargs)
