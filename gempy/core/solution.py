import numpy as np
from typing import Union, Iterable
import warnings
from skimage import measure
from gempy.utils.input_manipulation import find_interfaces_from_block_bottoms
from gempy.core.data import Grid, Surfaces
from gempy.core.data_modules.stack import Series, Stack
from gempy.utils.meta import _setdoc, _setdoc_pro
import gempy.utils.docstring as ds
import xarray as xr


@_setdoc_pro(
    [Grid.__doc__, Surfaces.__doc__, Series.__doc__, ds.weights_vector, ds.sfai,
     ds.bai, ds.mai, ds.vai,
     ds.lith_block, ds.sfm, ds.bm, ds.mm, ds.vm, ds.vertices, ds.edges,
     ds.geological_map])
class XSolution(object):
    """This class stores the output of the interpolation and the necessary objects
    to visualize and manipulate this data.

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
        # self.additional_data = additional_data

        self.grid = grid
        #  self.surface_points = surface_points
        self.stack = stack
        self.surfaces = surfaces  # Used to store ver/sim there

        # Define xarrays
        self.weights_vector = None
        self.at_surface_points = None
        self.s_regular_grid = xr.Dataset()
        self.s_custom_grid = xr.Dataset()
        self.s_topography = xr.Dataset()
        self.s_at_surface_points = xr.Dataset()
        self.s_sections = dict()
        self.meshes = None

    # Input data results
    @property
    def scalar_field_at_surface_points(self):
        return self.s_at_surface_points['scalar_field_v3'].values

    @property
    def block_at_surface_points(self):
        return self.s_at_surface_points['block_v3'].values

    @property
    def mask_at_surface_points(self):
        return self.s_at_surface_points['mask_v3'].values

    @property
    def values_at_surface_points(self):
        return self.s_at_surface_points['values_v3'].values

    @property
    def lith_block(self):
        return self.s_regular_grid['property_matrix'].loc['id'].values.reshape(1, -1)

    @property
    def scalar_field_matrix(self):
        shape = self.s_regular_grid['scalar_field_matrix'].shape
        return self.s_regular_grid['scalar_field_matrix'].values.reshape(shape[0],
                                                                         -1)

    @property
    def block_matrix(self):
        shape = self.s_regular_grid['block_matrix'].shape
        return self.s_regular_grid['block_matrix'].values.reshape(shape[0],
                                                                  shape[1],
                                                                  -1)

    @property
    def mask_matrix(self):
        shape = self.s_regular_grid['mask_matrix'].shape
        return self.s_regular_grid['mask_matrix'].values.reshape(shape[0], -1)

    # This is should be private
    # @property
    # def mask_matrix_pad(self):
    #     return

    @property
    def values_matrix(self):
        prop = self.s_regular_grid['property_matrix'].Properties.values
        sel = prop != 'id'
        values_other_than_id = prop[sel]
        array = self.s_regular_grid['property_matrix'].loc[
            values_other_than_id].values
        return array.reshape(len(values_other_than_id), -1)

    @property
    def gradient(self):
        raise NotImplementedError

    @property
    def vertices(self):
        return

    @property
    def edges(self):
        return

    @property
    def geological_map(self):
        shape = self.s_topography['scalar_field_matrix'].shape

        p = self.s_topography['property_matrix'].values.reshape(shape[0], -1)
        s = self.s_topography['scalar_field_matrix'].values.reshape(shape[0], -1)
        return np.array([p, s])

    @property
    def sections(self):
        return NotImplementedError
        # shape = self.s_sections['scalar_field_matrix'].shape
        #
        # p = self.s_topography['property_matrix'].values.reshape(shape[0], -1)
        # s = self.s_topography['scalar_field_matrix'].values.reshape(shape[0], -1)
        # return np.array([p, s])

    # @property
    # def custom(self):
    #     return
    #
    # @property
    # def fw_gravity(self):
    #     return
    #
    # @property
    # def fw_magnetics(self):
    #     return

    def set_values(self,
                   values: list,
                   active_features=None,
                   surf_properties=None,
                   attach_xyz=True):
        """ At this stage we should split values into the different grids

        Args:
            values:

        Returns:

        """

        # Get an array with all the indices for each grid
        l = self.grid.length

        coords_base, xyz = self.prepare_common_args(active_features, attach_xyz,
                                                    surf_properties)
        self.weights_vector = values[3]

        if self.grid.active_grids[0]:
            self.set_values_to_regular_grid(values, l[0], l[1], coords_base.copy())
        if self.grid.active_grids[1]:
            self.set_values_to_custom_grid(values, l[1], l[2], coords_base.copy(),
                                           xyz=xyz)
        if self.grid.active_grids[2]:
            self.set_values_to_topography(values, l[2], l[3], coords_base.copy())
        if self.grid.active_grids[3]:
            self.set_values_to_sections(values, l[3], l[4], coords_base.copy())
        if self.grid.active_grids[4]:
            self.set_values_to_centered()

        # TODO: Add xyz from surface points
        self.set_values_to_surface_points(values, l[-1], coords_base, xyz=None)

    def prepare_common_args(self, active_features, attach_xyz, surf_properties):
        if active_features is None and self.stack is not None:
            active_features = self.stack.df.groupby('isActive').get_group(True).index
        if surf_properties is None and self.surfaces is not None:
            surf_properties = self.surfaces.properties_val
        coords_base = dict()
        if active_features is not None:
            coords_base['Features'] = active_features
        if surf_properties is not None:
            coords_base['Properties'] = surf_properties
        if attach_xyz and self.grid.custom_grid is not None:
            xyz = self.grid.custom_grid.values
        else:
            xyz = None
        return coords_base, xyz

    def set_values_to_centered(self):
        return

    def set_values_to_surface_points(self, values, l0, coords_base, xyz=None):
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
                dims=['Properties', 'Point'],
            )

            arrays['property_v3'] = property_v3

        if values[1] is not None:
            # block
            block_v3 = xr.DataArray(
                data=values[1][:, :, l0:l1],
                dims=['Features', 'Properties', 'Point'],
            )

            arrays['block_v3'] = block_v3

        if values[4] is not None:
            # Scalar field
            scalar_field_v3 = xr.DataArray(
                data=values[4][:, l0:l1],
                dims=['Features', 'Point'],
            )
            arrays['scalar_field_v3'] = scalar_field_v3

        if values[6] is not None:
            # Scalar field
            mask_v3 = xr.DataArray(
                data=values[6][:, l0:l1],
                dims=['Features', 'Point'],
            )
            arrays['mask_v3'] = mask_v3

        return arrays

    def set_values_to_custom_grid(self, values: list, l0, l1,
                                  coords_base: dict, xyz=None):

        coords = coords_base
        arrays = self.create_unstruct_xarray(values, l0, l1, xyz)

        self.s_custom_grid = xr.Dataset(
            data_vars=arrays,
            coords=coords
        )
        return self.s_custom_grid

    def set_values_to_regular_grid(self, values: list, l0, l1,
                                   coords_base: dict):

        coords = self.add_cartesian_coords(coords_base)

        arrays = self.create_struc_xarrays(values, l0, l1,
                                           self.grid.regular_grid.resolution)

        self.s_regular_grid = xr.Dataset(
            data_vars=arrays,
            coords=coords
        )

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

        self.s_topography = xr.Dataset(
            data_vars=arrays,
            coords=coords
        )
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

            arrays = self.create_struc_xarrays(values, l0 + l0_s, l0 + l1_s,
                                               resolution)

            self.s_sections[name] = xr.Dataset(
                data_vars=arrays,
                coords=coords
            )
        return self.s_sections


@_setdoc_pro(
    [Grid.__doc__, Surfaces.__doc__, Series.__doc__, ds.weights_vector, ds.sfai,
     ds.bai, ds.mai, ds.vai,
     ds.lith_block, ds.sfm, ds.bm, ds.mm, ds.vm, ds.vertices, ds.edges,
     ds.geological_map])
class Solution(object):
    """This class stores the output of the interpolation and the necessary objects
    to visualize and manipulate this data.

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

    def __init__(self, grid: Grid = None,
                 surfaces: Surfaces = None,
                 series: Series = None,
                 ):

        # self.additional_data = additional_data
        self.grid = grid
        #  self.surface_points = surface_points
        self.series = series
        self.surfaces = surfaces

        # Input data results
        self.weights_vector = np.empty(0)
        self.scalar_field_at_surface_points = np.array([])
        self.block_at_surface_points = np.array([])
        self.mask_at_surface_points = np.array([])
        self.values_at_surface_points = np.array([])

        # Regular Grid
        self.lith_block = np.empty(0)
        self.scalar_field_matrix = np.array([])
        self.block_matrix = np.array([])
        self.mask_matrix = np.array([])
        self.mask_matrix_pad = []
        self.values_matrix = np.array([])
        self.gradient = np.empty(0)

        self.vertices = []
        self.edges = []

        self.geological_map = None
        self.sections = None
        self.custom = None

        # Center Grid
        self.fw_gravity = None
        self.fw_magnetics = None

    def __repr__(self):
        return '\nLithology ids \n  %s \n' \
               % (np.array2string(self.lith_block))

    def set_solution_to_regular_grid(self, values: Union[list, np.ndarray],
                                     compute_mesh: bool = True,
                                     compute_mesh_options: dict = None):
        """If regular grid is active set all the solution objects dependent on them and compute mesh.

        Args:
            values (list[np.array]): list with result of the theano evaluation (values returned by
             :func:`gempy.compute_model` function):

                 - block_matrix
                 - weights_vector
                 - scalar_field_matrix
                 - scalar field at interfaces
                 - mask_matrix

            compute_mesh (bool): if True perform marching cubes algorithm to recover the surface mesh from the
             implicit model.
            compute_mesh_options (dict): options for the marching cube function.
                - rescale: True

        Returns:
            :class:`gempy.core.solutions.Solutions`

        """
        if compute_mesh_options is None:
            compute_mesh_options = {}
        self.set_values_to_regular_grid(values)
        if compute_mesh is True:
            self.compute_all_surfaces(**compute_mesh_options)

        return self

    def set_solution_to_custom(self, values: Union[list, np.ndarray]):
        l0, l1 = self.grid.get_grid_args('custom')
        self.custom = np.array(
            [values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_solution_to_topography(self, values: Union[list, np.ndarray]):
        l0, l1 = self.grid.get_grid_args('topography')
        self.geological_map = np.array(
            [values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_solution_to_sections(self, values: Union[list, np.ndarray]):
        l0, l1 = self.grid.get_grid_args('sections')
        self.sections = np.array(
            [values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_values_to_regular_grid(self, values: Union[list, np.ndarray]):
        """Set all solution values to the correspondent attribute.

        Args:
            values (np.ndarray): values returned by `function: gempy.compute_model` function
            compute_mesh (bool): if true compute automatically the grid

        Returns:
            :class:`gempy.core.solutions.Solutions`

        """
        regular_grid_length_l0, regular_grid_length_l1 = self.grid.get_grid_args(
            'regular')

        # Lithology final block
        self.lith_block = values[0][0,
                          regular_grid_length_l0: regular_grid_length_l1]

        # Properties
        self.values_matrix = values[0][1:,
                             regular_grid_length_l0: regular_grid_length_l1]

        # Axis 0 is the series. Axis 1 is the value
        self.block_matrix = values[1][:, :,
                            regular_grid_length_l0: regular_grid_length_l1]

        self.fault_block = values[2]
        # This here does not make any sense
        self.weights_vector = values[3]

        self.scalar_field_matrix = values[4][:,
                                   regular_grid_length_l0: regular_grid_length_l1]

        self.mask_matrix = values[6][:,
                           regular_grid_length_l0: regular_grid_length_l1]

        self.fault_mask = values[7][:,
                          regular_grid_length_l0: regular_grid_length_l1]

        # TODO add topology solutions

        return self

    def set_values_to_surface_points(self, values):
        x_to_intep_length = self.grid.length[-1]
        self.scalar_field_at_surface_points = values[5]
        self.values_at_surface_points = values[0][1:, x_to_intep_length:]
        self.block_at_surface_points = values[1][:, :, x_to_intep_length:]
        # todo disambiguate below from self.scalar_field_at_surface_points
        self._scalar_field_at_surface = values[4][:, x_to_intep_length:]
        self.mask_at_surface_points = values[6][:, x_to_intep_length:]
        return self.scalar_field_at_surface_points

    def compute_marching_cubes_regular_grid(self, level: float, scalar_field,
                                            mask_array=None,
                                            rescale=False, **kwargs):
        """Compute the surface (vertices and edges) of a given surface by computing
         marching cubes (by skimage)

        Args:
            level (float): value of the scalar field at the surface
            scalar_field (np.array): scalar_field vector objects
            mask_array (np.array): mask vector with trues where marching cubes has to be performed
            rescale (bool): if True surfaces will be located between 0 and 1
            **kwargs: skimage.measure.marching_cubes_lewiner args (see below)

        Returns:
            list: vertices, simplices, normals, values

        See Also:

            :func:`skimage.measure.marching_cubes`

        """
        rg = self.grid.regular_grid
        spacing = self.grid.regular_grid.get_dx_dy_dz(rescale=rescale)
        vertices, simplices, normals, values = measure.marching_cubes(
            scalar_field.reshape(rg.resolution),
            level, spacing=spacing, mask=mask_array, **kwargs)
        idx = [0, 2, 4]
        loc_0 = rg.extent_r[idx] if rescale else rg.extent[idx]
        loc_0 = loc_0 + np.array(spacing) / 2
        vertices += np.array(loc_0).reshape(1, 3)

        return [vertices, simplices, normals, values]

    def padding_mask_matrix(self, mask_topography=True, shift=2):
        """Pad as many elements as in shift to the masking arrays. This is done
         to guarantee intersection of layers if masked marching cubes are done

        Args:
            mask_topography (bool): if True mask also the topography. Default True
            shift: Number of voxels shifted for the topology. Default 1.

        Returns:
              numpy.ndarray: masked regular grid
        """

        self.mask_matrix_pad = []
        series_type = self.series.df['BottomRelation']
        for e, mask_series in enumerate(self.mask_matrix):
            mask_series_reshape = mask_series.reshape(
                self.grid.regular_grid.resolution)

            mask_pad = (mask_series_reshape + find_interfaces_from_block_bottoms(
                mask_series_reshape, True, shift=shift))

            if series_type[e] == 'Fault':
                mask_pad = np.invert(mask_pad)

            if mask_topography and self.grid.regular_grid.mask_topo.size != 0:
                mask_pad *= np.invert(self.grid.regular_grid.mask_topo)
            self.mask_matrix_pad.append(mask_pad)
        return self.mask_matrix_pad

    def compute_all_surfaces(self, **kwargs):
        """Compute all surfaces of the model given the geological features rules.

        Args:
            **kwargs: :any:`skimage.measure.marching_cubes` args (see below)

        Returns:
            list: vertices and edges

        See Also:
            :meth:`gempy.core.solution.Solution.compute_marching_cubes_regular_grid`

        """
        self.vertices = []
        self.edges = []
        if 'mask_topography' in kwargs:
            mask_topography = kwargs.pop('mask_topography')
        else:
            mask_topography = True

        if 'masked_marching_cubes' in kwargs:
            masked_marching_cubes = kwargs.pop('masked_marching_cubes')
        else:
            masked_marching_cubes = True

        self.padding_mask_matrix(mask_topography=mask_topography)
        series_type = self.series.df['BottomRelation']
        s_n = 0
        active_indices = self.surfaces.df.groupby('isActive').groups[True]
        rescale = kwargs.pop('rescale', False)

        # We loop the scalar fields
        for e, scalar_field in enumerate(self.scalar_field_matrix):
            sfas = self.scalar_field_at_surface_points[e]
            # Drop
            sfas = sfas[np.nonzero(sfas)]
            mask_array = self.mask_matrix_pad[
                e - 1 if series_type[e - 1] == 'Onlap' else e]
            for level in sfas:
                try:
                    v, s, norm, val = self.compute_marching_cubes_regular_grid(
                        level, scalar_field, mask_array, rescale=rescale, **kwargs)
                except Exception as e:
                    warnings.warn('Surfaces not computed due to: ' + str(
                        e) + '. The surface is: Series: ' + str(e) +
                                  '; Surface Number:' + str(s_n))
                    v = np.nan
                    s = np.nan

                self.vertices.append(v)
                self.edges.append(s)
                idx = active_indices[s_n]
                self.surfaces.df.loc[idx, 'vertices'] = [v]
                self.surfaces.df.loc[idx, 'edges'] = [s]
                s_n += 1

        return self.vertices, self.edges
