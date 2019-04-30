import numpy as np
import pandas as pn
from typing import Union
import warnings
from gempy.utils.meta import _setdoc


class Solution(object):
    """
    TODO: update this
    This class store the output of the interpolation and the necessary objects to visualize and manipulate this data.
    Depending on the chosen output in Additional Data -> Options a different number of solutions is returned:
        if output is geology:
            1) Lithologies: block and scalar field
            2) Faults: block and scalar field for each faulting plane

        if output is gradients:
            1) Lithologies: block and scalar field
            2) Faults: block and scalar field for each faulting plane
            3) Gradients of scalar field x
            4) Gradients of scalar field y
            5) Gradients of scalar field z

    Attributes:
        additional_data (AdditionalData):
        surfaces (Surfaces)
        grid (Grid)
        scalar_field_at_surface_points (np.ndarray): Array containing the values of the scalar field at each interface. Axis
        0 is each series and axis 1 contain each surface in order
         lith_block (np.ndndarray): Array with the id of each layer evaluated in each point of
         `attribute:GridClass.values`
        fault_block (np.ndarray): Array with the id of each fault block evaluated in each point of
         `attribute:GridClass.values`
        scalar_field_lith(np.ndarray): Array with the scalar field of each layer evaluated in each point of
         `attribute:GridClass.values`
        scalar_field_lith(np.ndarray): Array with the scalar field of each fault segmentation evaluated in each point of
        `attribute:GridClass.values`
        values_block (np.ndarray):   Array with the properties of each layer evaluated in each point of
         `attribute:GridClass.values`. Axis 0 represent different properties while axis 1 contain each evaluated point
        gradient (np.ndarray):  Array with the gradient of the layars evaluated in each point of
        `attribute:GridClass.values`. Axis 0 contain Gx, Gy, Gz while axis 1 contain each evaluated point

    Args:
        additional_data (AdditionalData):
        surfaces (Surfaces):
        grid (Grid):
        values (np.ndarray): values returned by `function: gempy.compute_model` function
    """

    def __init__(self, additional_data = None, grid = None,
                 surface_points = None, series=None, surfaces=None):

        self.additional_data = additional_data
        self.grid = grid
        self.surface_points = surface_points
        self.series = series
        self.surfaces = surfaces

        # Lithology final block
        self.lith_block = np.empty(0)
        self.weights_vector = np.empty(0)

        self.scalar_field_matrix = np.array([])
        self.scalar_field_at_surface_points = np.array([])

        self.block_matrix = np.array([])
        self.block_at_surface_points = np.array([])

        self.mask_matrix = np.array([])
        self.mask_matrix_pad = []
        self.mask_at_surface_points = np.array([])

        self.values_matrix = np.array([])
        self.values_at_surface_points = np.array([])

        self.gradient = np.empty(0)

        self.vertices = []
        self.edges = []

    def __repr__(self):
        return '\nLithology ids \n  %s \n' \
               % (np.array2string(self.lith_block))

    def set_solution_to_regular_grid(self, values: Union[list, np.ndarray], compute_mesh: bool = True  #, sort_surfaces=True
                                     ):
        self.set_values_to_regular_grid(values)
        if compute_mesh is True:
            try:
                self.compute_all_surfaces()
            except RuntimeError:
                warnings.warn('It is not possible to compute the mesh.')

        # if sort_surfaces is True:
        #     self.set_model_order()

        return self

    # def set_model_order(self):
    #     # TODO time this function
    #     spu = self.surface_points.df['surface'].unique()
    #     sps = self.surface_points.df['series'].unique()
    #     sel = self.surfaces.df['surface'].isin(spu)
    #    # print(sel)
    #     for e, name_series in enumerate(sps):
    #         try:
    #             sfai_series = self.scalar_field_at_surface_points[e]
    #             sfai_order_aux = np.argsort(sfai_series[np.nonzero(sfai_series)])
    #             sfai_order =  (sfai_order_aux - sfai_order_aux.shape[0]) * -1
    #             # select surfaces which exist in surface_points
    #             group = self.surfaces.df[sel].groupby('series').get_group(name_series)
    #             idx = group.index
    #             surface_names = group['surface']
    #             print('idx', idx)
    #             print(sfai_order)
    #             self.surfaces.df.loc[idx, 'order_surfaces'] = self.surfaces.df.loc[idx, 'surface'].map(
    #                 pn.DataFrame(sfai_order, index=surface_names)[0])
    #             print( pn.DataFrame(sfai_order, index=surface_names)[0])
    #             print(self.surfaces.df)
    #         except IndexError:
    #             pass
    #
    #     self.surfaces.sort_surfaces()
    #     self.surfaces.set_basement()
    #     self.surface_points.df['id'] = self.surface_points.df['surface'].map(self.surfaces.df.set_index('surface')['id'])
    #     self.surface_points.sort_table()
    #
    #     return self.surfaces

    def set_values_to_regular_grid(self, values: Union[list, np.ndarray], compute_mesh: bool=True):
        # TODO ============ Set asserts of give flexibility 20.09.18 =============
        """
        Set all solution values to the correspondant attribute
        Args:
            values (np.ndarray): values returned by `function: gempy.compute_model` function
            compute_mesh (bool): if true compute automatically the grid

        Returns:

        """
        regular_grid_length_l0, regular_grid_length_l1 = self.grid.get_grid_args('regular')
        x_to_intep_length = self.grid.length[-1]

        self.scalar_field_matrix = values[3][:, regular_grid_length_l0: regular_grid_length_l1]
        self.scalar_field_at_surface_points = values[4]
        self._scalar_field_at_surface = values[3][:, x_to_intep_length:]

        self.weights_vector = values[2]

        # Axis 0 is the series. Axis 1 is the value
        self.block_matrix = values[1][:, :, regular_grid_length_l0: regular_grid_length_l1]
        self.block_at_surface_points = values[1][:, :,x_to_intep_length:]

        self.mask_matrix = values[5][:, regular_grid_length_l0: regular_grid_length_l1]
        self.mask_at_surface_points = values[5][:, x_to_intep_length:]

        # Lithology final block
        self.lith_block = values[0][0, regular_grid_length_l0: regular_grid_length_l1]

        # Properties
        self.values_matrix = values[0][1:, regular_grid_length_l0: regular_grid_length_l1]
        self.values_at_surface_points = values[0][1:, x_to_intep_length:]

        # TODO Adapt it to the gradients
        # try:
        #     if self.additional_data.options.df.loc['values', 'output'] is 'gradients':
        #         self.values_block = lith[2:-3]
        #         self.gradient = lith[-3:]
        #     else:
        #         self.values_block = lith[2:]
        # except AttributeError:
        #     self.values_block = lith[2:]
        #
        # self.scalar_field_faults = faults[1::2]
        # self.fault_blocks = faults[::2]

    def compute_surface_regular_grid(self, level: float, scalar_field, mask_array=None, **kwargs):
        """
        Compute the surface (vertices and edges) of a given surface by computing marching cubes (by skimage)
        Args:
            surface_id (int): id of the surface to be computed
            scalar_field: scalar field grid
            **kwargs: skimage.measure.marching_cubes_lewiner args

        Returns:
            list: vertices, simplices, normals, values
        """

        from skimage import measure
        # # Check that the scalar field of the surface is whithin the boundaries
        # if not scalar_field.max() > level:
        #     level = scalar_field.max()
        #     print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
        #           'in the implementation.' % surface_id)
        #
        # if not scalar_field.min() < pot_int[surface_id]:
        #     pot_int[surface_id] = scalar_field.min()
        #     print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
        #           'in the implementation.' % surface_id)

        vertices, simplices, normals, values = measure.marching_cubes_lewiner(
            scalar_field.reshape(self.grid.regular_grid.resolution[0],
                                 self.grid.regular_grid.resolution[1],
                                 self.grid.regular_grid.resolution[2]),
            level,
            spacing=self.grid.regular_grid.get_dx_dy_dz(),
            mask=mask_array,
            **kwargs
        )

        vertices += np.array([self.grid.extent[0],
                              self.grid.extent[2],
                              self.grid.extent[4]]).reshape(1, 3)

        return [vertices, simplices, normals, values]

    def mask_topo(self, mask_matrix):
        return ~self.grid.regular_grid.mask_topo * mask_matrix

    def padding_mask_matrix(self, mask_topography=True):
        self.mask_matrix_pad = []
        for mask_series in self.mask_matrix:
            mask_series_reshape = mask_series.reshape((self.grid.regular_grid.resolution[0],
                                                       self.grid.regular_grid.resolution[1],
                                                       self.grid.regular_grid.resolution[2]))
            if mask_topography and self.grid.regular_grid.mask_topo.size != 0:
                mask_series_reshape = self.mask_topo(mask_series_reshape)

            self.mask_matrix_pad.append((mask_series_reshape + self.find_interfaces_from_block_bottoms(
                mask_series_reshape, True)).T)

    @staticmethod
    def find_interfaces_from_block_bottoms(block, value, shift=3):
        """
        Find the voxel at an interface. We shift left since gempy is based on bottoms

        Args:
            block (ndarray):
            value:

        Returns:

        """
        A = block == value
        final_bool = np.zeros_like(block, dtype=bool)
        x_shift = A[:-shift, :, :] ^ A[shift:, :, :]

        # Matrix shifting along axis
        y_shift = A[:, :-shift, :] ^ A[:, shift:, :]

        # Matrix shifting along axis
        z_shift = A[:, :, :-shift] ^ A[:, :, shift:]
        final_bool[shift:, shift:, shift:] = (x_shift[:, shift:, shift:] +
                                              y_shift[shift:, :, shift:] +
                                              z_shift[shift:, shift:, :])
        return final_bool

    @_setdoc(compute_surface_regular_grid.__doc__)
    def compute_all_surfaces(self, **kwargs):
        self.vertices = []
        self.edges = []
        self.padding_mask_matrix()
       # series_type = np.append('init', self.series.df['BottomRelation'])
        series_type = self.series.df['BottomRelation']

        s_n = 0
        # We loop the scalar fields
        for e, scalar_field in enumerate(self.scalar_field_matrix):
            sfas = self.scalar_field_at_surface_points[e]
            # Drop
            sfas = sfas[np.nonzero(sfas)]
            if series_type[e-1] == 'Onlap':
                mask_array = self.mask_matrix_pad[e-1]
            elif series_type[e] == 'Fault':
                mask_array = None
            else:
                mask_array = self.mask_matrix_pad[e]

            for level in sfas:
                # print(mask_array, e)
                v, s, norm, val = self.compute_surface_regular_grid(level, scalar_field, mask_array, **kwargs)
                self.vertices.append(v)
                self.edges.append(s)
                idx = self.surfaces.df.index[s_n]
                self.surfaces.df.loc[idx, 'vertices'] = [v]
                self.surfaces.df.loc[idx, 'edges'] = [s]
                s_n += 1
        return self.vertices, self.edges
    #
    # def set_vertices(self, surface_name, vertices):
    #     self.vertices[surface_name] = vertices
    #
    # def set_edges(self, surface_name, edges):
    #     self.edges[surface_name] = edges
