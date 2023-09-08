import pprint
import warnings
from dataclasses import dataclass, field
from typing import Sequence, Optional

import numpy as np

import gempy_engine.core.data.grid
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from .orientations import OrientationsTable
from .structural_frame import StructuralFrame
from gempy_engine.core.data.transforms import Transform, GlobalAnisotropy
from .grid import Grid

"""
TODO:
    - [ ] StructuralFrame will all input points chunked on Elements. Here I will need a property to put all
    together to feed to InterpolationInput

"""


@dataclass
class GeoModelMeta:
    """
    Container for metadata associated with a GeoModel.

    Attributes:
        name (str): Name of the geological model.
        creation_date (str): Date of creation of the model.
        last_modification_date (str): Last modification date of the model.
        owner (str): Owner of the geological model.
    """

    name: str
    creation_date: str
    last_modification_date: str
    owner: str


@dataclass(init=False)
class GeoModel:
    """
    Class representing a geological model.

    """

    meta: GeoModelMeta  #: Meta-information about the geological model, like its name, creation and modification dates, and owner.
    structural_frame: StructuralFrame  #: The structural information of the geological model.
    grid: Grid  #: The general grid used in the geological model.

    # region GemPy engine data types
    interpolation_options: InterpolationOptions  #: The interpolation options provided by the user.

    transform: Transform = None  #: The transformation used in the geological model for input points.
    
    interpolation_grid: gempy_engine.core.data.grid.Grid = None  #: Optional grid used for interpolation. Can be seen as a cache field.
    _interpolationInput: InterpolationInput = None  #: Input data for interpolation. Fed by the structural frame and can be seen as a cache field.
    _input_data_descriptor: InputDataDescriptor = None  #: Descriptor of the input data. Fed by the structural frame and can be seen as a cache field.

    # endregion
    _solutions: gempy_engine.core.data.solutions.Solutions = field(init=False, default=None)  #: The computed solutions of the geological model. 

    legacy_model: "gpl.Project" = None  #: Legacy model (if available). Allows for backward compatibility.

    def __init__(self, name: str, structural_frame: StructuralFrame, grid: Grid,
                 interpolation_options: InterpolationOptions):
        # TODO: Fill the arguments properly
        self.meta = GeoModelMeta(
            name=name,
            creation_date=None,
            last_modification_date=None,
            owner=None
        )

        self.structural_frame = structural_frame  # ? This could be Optional

        self.grid = grid
        self.interpolation_options = interpolation_options
        self.transform = Transform.from_input_points(
            surface_points=self.surface_points,
            orientations=self.orientations
        )

    def __repr__(self):
        # TODO: Improve this
        return pprint.pformat(self.__dict__)

    def update_transform(self, auto_anisotropy: GlobalAnisotropy = GlobalAnisotropy.CUBE, anisotropy_limit: Optional[np.ndarray] = None):
        self.transform = Transform.from_input_points(
            surface_points=self.surface_points,
            orientations=self.orientations
        )
        
        self.transform.apply_anisotropy(anisotropy_type=auto_anisotropy, anisotropy_limit=anisotropy_limit)
            

    @property
    def solutions(self):
        return self._solutions

    @solutions.setter
    def solutions(self, value):
        self._solutions = value
        for e, group in enumerate(self.structural_frame.structural_groups):
            group.solution = RawArraysSolution(  # ? Maybe I need to add more fields, but I am not sure yet
                scalar_field_matrix=self._solutions.raw_arrays.scalar_field_matrix[e],
                block_matrix=self._solutions.raw_arrays.block_matrix[e],
            )

        for e, element in enumerate(self.structural_frame.structural_elements[:-1]):  # * Ignore basement

            dc_mesh = self._solutions.dc_meshes[e] if self._solutions.dc_meshes is not None else None
            # TODO: These meshes are in the order of the scalar field
            element.vertices = (self.transform.apply_inverse(dc_mesh.vertices) if dc_mesh is not None else None)
            element.edges = (dc_mesh.edges if dc_mesh is not None else None)

    @property
    def surface_points(self):
        """This is a copy! Returns a SurfacePointsTable for all surface points across the structural elements"""
        surface_points_table = self.structural_frame.surface_points
        if self.transform is not None:
            surface_points_table.model_transform = self.transform
        return surface_points_table
    
    @surface_points.setter
    def surface_points(self, value):
        self.structural_frame.surface_points = value

    @property
    def orientations(self) -> OrientationsTable:
        """This is a copy! Returns a OrientationsTable for all orientations across the structural elements"""
        orientations_table = self.structural_frame.orientations
        if self.transform is not None:
            orientations_table.model_transform = self.transform
        return orientations_table
    
    @orientations.setter
    def orientations(self, value):
        self.structural_frame.orientations = value

    @property
    def interpolation_input(self):
        if self.structural_frame.is_dirty is False:
            return self._interpolationInput
        n_octree_lvl = self.interpolation_options.number_octree_levels

        compute_octrees: bool = n_octree_lvl > 1

        # * Set regular grid to the octree resolution. ? Probably a better way to do this would be to make regular_grid resolution a property
        if compute_octrees:
            octree_leaf_resolution = np.array([2 ** n_octree_lvl] * 3)

            resolution_not_set = self.grid.regular_grid.resolution is not None
            resolution_is_octree_resolution = np.allclose(self.grid.regular_grid.resolution, octree_leaf_resolution)

            if resolution_not_set and not resolution_is_octree_resolution:
                warnings.warn(
                    message="You are using octrees and passing a regular grid. The resolution of the regular grid will be overwritten",
                    category=UserWarning
                )

            self.grid.regular_grid.set_regular_grid(
                extent=self.grid.regular_grid.extent,
                resolution=octree_leaf_resolution
            )

        self._interpolationInput = InterpolationInput.from_structural_frame(
            structural_frame=self.structural_frame,
            grid=self.grid,
            transform=self.transform,
            octrees=compute_octrees
        )

        return self._interpolationInput

    @property
    def input_data_descriptor(self) -> InputDataDescriptor:
        # TODO: This should have the exact same dirty logic as interpolation_input
        return self.structural_frame.input_data_descriptor

    def add_surface_points(self, X: Sequence[float], Y: Sequence[float], Z: Sequence[float],
                           surface: Sequence[str], nugget: Optional[Sequence[float]] = None) -> None:
        raise NotImplementedError("This method is deprecated. Use `gp.add_surface_points` instead")
