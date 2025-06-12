import datetime
import numpy as np
import pprint
import warnings
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator, ValidationError
from pydantic.functional_validators import ModelWrapValidatorHandler
from typing import Sequence, Optional, Union

from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data import Solutions
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution
from gempy_engine.core.data.transforms import Transform, GlobalAnisotropy
from .encoders.converters import instantiate_if_necessary
from .encoders.json_geomodel_encoder import encode_numpy_array
from .grid import Grid
from .orientations import OrientationsTable
from .structural_frame import StructuralFrame
from .surface_points import SurfacePointsTable
from ...modules.data_manipulation import interpolation_input_from_structural_frame


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
    last_modification_date: str | None
    owner: str | None


class GeoModel(BaseModel):
    """
    Class representing a geological model.

    """

    meta: GeoModelMeta = Field(exclude=False)  #: Meta-information about the geological model, like its name, creation and modification dates, and owner.
    structural_frame: StructuralFrame = Field(exclude=False)  #: The structural information of the geological model.
    grid: Grid = Field(exclude=False, default=None)  #: The general grid used in the geological model.

    # region GemPy engine data types
    _interpolation_options: InterpolationOptions  #: The interpolation options provided by the user.

    geophysics_input: GeophysicsInput | None = Field(default=None, exclude=False)  #: The geophysics input of the geological model.
    input_transform: Transform = Field(default=None, exclude=False)  #: The transformation used in the geological model for input points.

    interpolation_grid: EngineGrid = Field(default=None, exclude=True)  #: ptional grid used for interpolation. Can be seen as a cache field.
    _interpolationInput: InterpolationInput = PrivateAttr(default=None)  #: Input data for interpolation. Fed by the structural frame and can be seen as a cache field.
    _input_data_descriptor: InputDataDescriptor = PrivateAttr(default=None)  #: Descriptor of the input data. Fed by the structural frame and can be seen as a cache field.

    # endregion
    _solutions: Solutions = PrivateAttr(init=False, default=None)  #: The computed solutions of the geological model. 
    _taped_interpolation_input: InterpolationInput | None = PrivateAttr(default=None)

    def __repr__(self):
        # TODO: Improve this
        return pprint.pformat(self.__dict__)

    # region Properties

    @computed_field(alias="_interpolation_options")
    @property
    def interpolation_options(self) -> InterpolationOptions:
        self._infer_dense_grid_solution()
        self._interpolation_options.cache_model_name = self.meta.name
        return self._interpolation_options

    @interpolation_options.setter
    def interpolation_options(self, value):
        self._interpolation_options = value

    @property
    def solutions(self) -> Solutions:
        return self._solutions

    @solutions.setter
    def solutions(self, value: Solutions):
        # * This is set  from the gempy engine

        self._solutions = value

        # * Set solutions per element
        for e, element in enumerate(self.structural_frame.structural_elements[:-1]):  # * Ignore basement
            element.scalar_field_at_interface = value.scalar_field_at_surface_points[e]

            if self._solutions.dc_meshes is None:
                continue
            dc_mesh = self._solutions.dc_meshes[e]
            if dc_mesh is None:
                continue

            # TODO: These meshes are in the order of the scalar field 
            world_coord_vertices = self.input_transform.apply_inverse(dc_mesh.vertices)
            world_coord_vertices = self.grid.transform.apply_inverse_with_cached_pivot(world_coord_vertices)

            element.vertices = world_coord_vertices
            element.edges = (dc_mesh.edges if dc_mesh is not None else None)

        # * Reordering the elements according to the scalar field
        for e, order_per_structural_group in enumerate(self._solutions._ordered_elements):
            elements = self.structural_frame.structural_groups[e].elements
            reordered_elements = [elements[i] for i in order_per_structural_group]
            self.structural_frame.structural_groups[e].elements = reordered_elements

    @property
    def surface_points_copy(self):
        """This is a copy! Returns a SurfacePointsTable for all surface points across the structural elements"""
        surface_points_table = self.structural_frame.surface_points_copy
        return surface_points_table

    @property
    def surface_points_copy_transformed(self) -> SurfacePointsTable:
        og_sp = self.surface_points_copy

        og_sp.xyz_view = self.grid.transform.apply_with_cached_pivot(og_sp.xyz)
        og_sp.xyz_view = self.input_transform.apply(og_sp.xyz)
        return og_sp

    @property
    def surface_points(self):
        raise AttributeError("This property can only be set, not read. You can access the copy with `surface_points_copy` or"
                             "the original on the individual structural elements.")

    @surface_points.setter
    def surface_points(self, value):
        self.structural_frame.surface_points = value

    @property
    def orientations_copy(self) -> OrientationsTable:
        """This is a copy! Returns a OrientationsTable for all orientations across the structural elements"""
        orientations_table = self.structural_frame.orientations_copy
        return orientations_table

    @property
    def orientations_copy_transformed(self) -> OrientationsTable:
        # ! This is not done
        og_or = self.orientations_copy
        total_transform: Transform = self.input_transform + self.grid.transform

        og_or.xyz_view = self.grid.transform.apply_with_cached_pivot(og_or.xyz)
        og_or.xyz_view = self.input_transform.apply(og_or.xyz)

        og_or.grads_view = total_transform.transform_gradient(og_or.grads)
        return og_or

    @property
    def regular_grid_coordinates(self) -> np.ndarray:
        return self.grid.regular_grid.get_values_vtk_format(orthogonal=False)

    @property
    def regular_grid_coordinates_transformed(self) -> np.ndarray:
        values_transformed = self.grid.regular_grid.get_values_vtk_format(orthogonal=True)
        values_transformed = self.input_transform.apply(values_transformed)
        return values_transformed

    @property
    def orientations(self) -> OrientationsTable:
        raise AttributeError("This property can only be set, not read. You can access the copy with `orientations_copy` or"
                             "the original on the individual structural elements.")

    @orientations.setter
    def orientations(self, value):
        self.structural_frame.orientations = value

    @property
    def project_bounds(self) -> np.ndarray:
        return self.grid.bounding_box

    @property
    def extent_transformed_transformed_by_input(self) -> np.ndarray:
        transformed = self.input_transform.apply(self.project_bounds)  # ! grid already has the grid transform applied
        new_extents = np.array([transformed[:, 0].min(), transformed[:, 0].max(),
                                transformed[:, 1].min(), transformed[:, 1].max(),
                                transformed[:, 2].min(), transformed[:, 2].max()])
        return new_extents

    @property
    def interpolation_input_copy(self):
        warnings.warn("This property is deprecated. Use directly "
                      "`interpolation_input_from_structural_frame` instead.", DeprecationWarning)

        if self.structural_frame.is_dirty is False:
            return self._interpolationInput

        self._interpolationInput = interpolation_input_from_structural_frame(
            geo_model=self
        )

        return self._interpolationInput

    @property
    def input_data_descriptor(self) -> InputDataDescriptor:
        # TODO: This should have the exact same dirty logic as interpolation_input
        return self.structural_frame.input_data_descriptor
    
    @property
    def taped_interpolation_input(self) -> InterpolationInput:
        return self._taped_interpolation_input
    
    @taped_interpolation_input.setter
    def taped_interpolation_input(self, value: InterpolationInput):
        self._taped_interpolation_input = value

    # endregion
    # region Constructors
    @classmethod
    def from_args(cls, name: str, structural_frame: StructuralFrame, grid: Grid, interpolation_options: InterpolationOptions):
        # TODO: Fill the arguments properly
        meta = GeoModelMeta(
            name=name,
            creation_date=datetime.datetime.now().isoformat(),
            last_modification_date=None,
            owner=None
        )

        structural_frame = structural_frame  # ? This could be Optional

        grid = grid
        _interpolation_options = interpolation_options
        input_transform = Transform.from_input_points(
            surface_points=structural_frame.surface_points_copy,
            orientations=structural_frame.orientations_copy
        )

        model = GeoModel(
            meta=meta,
            structural_frame=structural_frame,
            grid=grid,
            input_transform=input_transform,
            _interpolation_options=_interpolation_options
        )

        return model

    # endregion

    # region Methods
    def update_transform(self, auto_anisotropy: GlobalAnisotropy = GlobalAnisotropy.NONE, anisotropy_limit: Optional[np.ndarray] = None):
        """Update the transformation of the geological model.

            This function updates the transformation of the geological model using the provided surface points and orientations.
            It also applies anisotropy based on the specified type and limit.

            Args:
                auto_anisotropy (GlobalAnisotropy): The type of anisotropy to apply. Defaults to GlobalAnisotropy.NONE.
                anisotropy_limit (Optional[np.ndarray]): Anisotropy limit values. If None, no limit is applied.

        """

        self.input_transform = Transform.from_input_points(
            surface_points=self.surface_points_copy,
            orientations=self.orientations_copy
        )

        self.input_transform.apply_anisotropy(anisotropy_type=auto_anisotropy, anisotropy_limit=anisotropy_limit)

    def add_surface_points(self, X: Sequence[float], Y: Sequence[float], Z: Sequence[float],
                           surface: Sequence[str], nugget: Optional[Sequence[float]] = None) -> None:
        raise NotImplementedError("This method is deprecated. Use `gp.add_surface_points` instead")

    # endregion

    # region Pydantic serialization

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=False,
        json_encoders={
                np.ndarray: encode_numpy_array,
        }
    )

    @model_validator(mode='wrap')
    @classmethod
    def deserialize_properties(cls, data: Union["GeoModel", dict], constructor: ModelWrapValidatorHandler["GeoModel"]) -> "GeoModel":
        match data:
            case GeoModel(): 
                return data
            case dict(): # 
                instance: GeoModel = constructor(data)
                instantiate_if_necessary(
                    data=data,
                    key="_interpolation_options",
                    type=InterpolationOptions
                )
                instance._interpolation_options = data.get("_interpolation_options")
                
                # * Reset geophysics if necessary
                centered_grid = instance.grid.centered_grid
                if centered_grid is not None and instance.geophysics_input is not None:
                    from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient
                    instance.geophysics_input.tz = calculate_gravity_gradient(centered_grid)
                
                return instance
            case _:
                raise ValidationError

    # endregion

    def _infer_dense_grid_solution(self):
        octrees_set: bool = Grid.GridTypes.OCTREE in self.grid.active_grids
        dense_set: bool = Grid.GridTypes.DENSE in self.grid.active_grids
        # Create a tuple representing the conditions
        match (octrees_set, dense_set):
            case (True, False):
                self._interpolation_options.block_solutions_type = RawArraysSolution.BlockSolutionType.OCTREE
            case (True, True):
                warnings.warn("Both octree levels and resolution are set. The default grid for the `raw_array_solution`"
                              "and plots will be the dense regular grid. To use octrees instead, set resolution to None in the "
                              "regular grid.")
                self._interpolation_options.block_solutions_type = RawArraysSolution.BlockSolutionType.DENSE_GRID
            case (False, True):
                self._interpolation_options.block_solutions_type = RawArraysSolution.BlockSolutionType.DENSE_GRID
            case (False, False):
                self._interpolation_options.block_solutions_type = RawArraysSolution.BlockSolutionType.NONE
