import numpy as np
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
                 surface_points = None, values=None):

        self.additional_data = additional_data
        self.grid = grid
        self.surface_points = surface_points

        if values is None:

            self.scalar_field_at_surface_points = np.array([])

            self.weights_vector = np.empty(0)
            self.scalar_field_matrix = np.array([])
            self.block_matrix = np.array([])
            self.mask_matrix = np.array([])

            # Lithology final block
            self.lith_block = np.empty(0)
            self.values_matrix = np.array([])

            #self.mask_matrix = np.empty(0)
            self.gradient = np.empty(0)
        else:
            self.set_values(values)

        self.vertices = {}
        self.edges = {}

    def __repr__(self):
        return '\nLithology ids \n  %s \n' \
               'Lithology scalar field \n  %s \n' \
               % (np.array2string(self.lith_block), np.array2string(self.scalar_field_matrix))

    def set_values(self, values: Union[list, np.ndarray], compute_mesh: bool=True):
        # TODO ============ Set asserts of give flexibility 20.09.18 =============
        """
        Set all solution values to the correspondant attribute
        Args:
            values (np.ndarray): values returned by `function: gempy.compute_model` function
            compute_mesh (bool): if true compute automatically the grid

        Returns:

        """
        self.scalar_field_at_surface_points = values[4]

        self.weights_vector = values[2]
        self.scalar_field_matrix = values[3]

        # Axis 0 is the series. Axis 1 is the value
        self.block_matrix = values[1]
        self.mask_matrix = values[5]

        # Lithology final block
        self.lith_block = values[0][0]

        # Properties
        self.values_matrix = values[0][1:]

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

        # TODO I do not like this here
        if compute_mesh is True:
            try:
                self.compute_all_surfaces()
            except RuntimeError:
                warnings.warn('It is not possible to compute the mesh.')

    def compute_surface_regular_grid(self, surface_id: int, scalar_field, **kwargs):
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
        assert surface_id >= 0, 'Number of the surface has to be positive'
        # In case the values are separated by series I put all in a vector
        pot_int = self.scalar_field_at_surface_points.sum(axis=0)

        # Check that the scalar field of the surface is whithin the boundaries
        if not scalar_field.max() > pot_int[surface_id]:
            pot_int[surface_id] = scalar_field.max()
            print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
                  'in the implementation.' % surface_id)

        if not scalar_field.min() < pot_int[surface_id]:
            pot_int[surface_id] = scalar_field.min()
            print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
                  'in the implementation.' % surface_id)

        vertices, simplices, normals, values = measure.marching_cubes_lewiner(
            scalar_field.reshape(self.grid.resolution[0],
                                 self.grid.resolution[1],
                                 self.grid.resolution[2]),
            pot_int[surface_id],
            spacing=((self.grid.extent[1] - self.grid.extent[0]) / self.grid.resolution[0],
                     (self.grid.extent[3] - self.grid.extent[2]) / self.grid.resolution[1],
                     (self.grid.extent[5] - self.grid.extent[4]) / self.grid.resolution[2]),
            **kwargs
        )

        return [vertices, simplices, normals, values]

    @_setdoc(compute_surface_regular_grid.__doc__)
    def compute_all_surfaces(self, **kwargs):
        """
        Compute all surfaces.

        Args:
            **kwargs: Marching_cube args

        Returns:

        See Also:
        """
        n_surfaces = self.additional_data.structure_data.df.loc['values', 'number surfaces']
        n_faults = self.additional_data.structure_data.df.loc['values', 'number faults']

        surfaces_names = self.surface_points.df['surface'].unique()

        surfaces_cumsum = np.arange(0, n_surfaces)

        if n_faults > 0:
            for n in surfaces_cumsum[:n_faults]:
                v, s, norm, val = self.compute_surface_regular_grid(n, np.atleast_2d(self.scalar_field_faults)[n],
                                                                    **kwargs)
                self.vertices[surfaces_names[n]] = v
                self.edges[surfaces_names[n]] = s

        if n_faults < n_surfaces:

            for n in surfaces_cumsum[n_faults:]:
                # TODO ======== split each_scalar_field ===========
                v, s, norms, val = self.compute_surface_regular_grid(n, self.scalar_field_lith, **kwargs)

                # TODO Use setters for this
                self.vertices[surfaces_names[n]] = v
                self.edges[surfaces_names[n]] = s

        return self.vertices, self.edges

    def set_vertices(self, surface_name, vertices):
        self.vertices[surface_name] = vertices

    def set_edges(self, surface_name, edges):
        self.edges[surface_name] = edges
