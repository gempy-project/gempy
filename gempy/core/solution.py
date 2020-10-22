import numpy as np
from typing import Union
import warnings
from skimage import measure
from gempy.utils.input_manipulation import find_interfaces_from_block_bottoms
from gempy.core.data import Grid, Surfaces
from gempy.core.data_modules.stack import Series
from gempy.utils.meta import _setdoc, _setdoc_pro
import gempy.utils.docstring as ds


@_setdoc_pro([Grid.__doc__, Surfaces.__doc__, Series.__doc__, ds.weights_vector, ds.sfai, ds.bai, ds.mai, ds.vai,
              ds.lith_block, ds.sfm, ds.bm, ds.mm, ds.vm, ds.vertices, ds.edges, ds.geological_map])
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
        self.custom = np.array([values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_solution_to_topography(self, values: Union[list, np.ndarray]):
        l0, l1 = self.grid.get_grid_args('topography')
        self.geological_map = np.array([values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_solution_to_sections(self, values: Union[list, np.ndarray]):
        l0, l1 = self.grid.get_grid_args('sections')
        self.sections = np.array([values[0][:, l0: l1], values[4][:, l0: l1].astype(float)])

    def set_values_to_regular_grid(self, values: Union[list, np.ndarray]):
        """Set all solution values to the correspondent attribute.

        Args:
            values (np.ndarray): values returned by `function: gempy.compute_model` function
            compute_mesh (bool): if true compute automatically the grid

        Returns:
            :class:`gempy.core.solutions.Solutions`

        """
        regular_grid_length_l0, regular_grid_length_l1 = self.grid.get_grid_args('regular')

        # Lithology final block
        self.lith_block = values[0][0, regular_grid_length_l0: regular_grid_length_l1]

        # Properties
        self.values_matrix = values[0][1:, regular_grid_length_l0: regular_grid_length_l1]

        # Axis 0 is the series. Axis 1 is the value
        self.block_matrix = values[1][:, :, regular_grid_length_l0: regular_grid_length_l1]

        self.fault_block = values[2]
        self.weights_vector = values[3]

        self.scalar_field_matrix = values[4][:, regular_grid_length_l0: regular_grid_length_l1]

        self.mask_matrix = values[6][:, regular_grid_length_l0: regular_grid_length_l1]

        self.fault_mask = values[7][:, regular_grid_length_l0: regular_grid_length_l1]

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
            mask_series_reshape = mask_series.reshape(self.grid.regular_grid.resolution)

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
            mask_array = self.mask_matrix_pad[e-1 if series_type[e-1] == 'Onlap' else e]
            for level in sfas:
                try:
                    v, s, norm, val = self.compute_marching_cubes_regular_grid(
                        level, scalar_field, mask_array, rescale=rescale, **kwargs)
                except Exception as e:
                    warnings.warn('Surfaces not computed due to: ' + str(e) + '. The surface is: Series: ' + str(e) +
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
