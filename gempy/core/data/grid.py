import dataclasses
import enum
import numpy as np
from pydantic import Field, model_validator, computed_field, ValidationError
from pydantic.functional_validators import ModelWrapValidatorHandler
from typing import Optional, Annotated, Union

from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.core.data.options import EvaluationOptions
from gempy_engine.core.data.transforms import Transform
from .encoders.binary_encoder import deserialize_grid
from .encoders.converters import loading_model_context
from .grid_modules import RegularGrid, CustomGrid, Sections
from .grid_modules.topography import Topography


@dataclasses.dataclass(init=True)
class Grid:
    class GridTypes(enum.Flag):
        OCTREE = 2 ** 0
        DENSE = 2 ** 1
        CUSTOM = 2 ** 2
        TOPOGRAPHY = 2 ** 3
        SECTIONS = 2 ** 4
        CENTERED = 2 ** 5
        NONE = 2 ** 10

    # ? What should we do with the extent?

    values: Annotated[np.ndarray, Field(exclude=True)] = dataclasses.field(default_factory=lambda: np.empty((0, 3)))
    length: Annotated[np.ndarray, Field(exclude=True)] = dataclasses.field(default_factory=lambda: np.empty(0))

    _octree_grid: Optional[RegularGrid] = None
    _dense_grid: Optional[RegularGrid] = None
    _custom_grid: Optional[CustomGrid] = None
    _topography: Optional[Topography] = None
    _sections: Optional[Sections] = None
    _centered_grid: Optional[CenteredGrid] = None

    _active_grids = GridTypes.NONE
    _transform: Optional[Transform] = None

    _octree_levels: int = -1

    def __init__(self, extent=None, resolution=None):

        self.values = np.empty((0, 3))
        self.length = np.empty(0)

        # Init basic grid empty
        if extent is not None and resolution is not None:
            self.dense_grid = RegularGrid(extent, resolution)

    @model_validator(mode='wrap')
    @classmethod
    def deserialize_properties(cls, data: Union["Grid", dict], constructor: ModelWrapValidatorHandler["Grid"]) -> "Grid":
        try:
            match data:
                case Grid():
                    return data
                case dict():
                    grid: Grid = constructor(data)
                    grid._active_grids = Grid.GridTypes(data["active_grids"])
                    # TODO: Digest binary data

                    metadata = data.get('binary_meta_data', {})
                    context = loading_model_context.get()

                    if 'grid_binary' not in context:
                        return grid
                    
                    custom_grid_vals, topography_vals = deserialize_grid(
                        binary_array=context['grid_binary'],
                        custom_grid_length=metadata["custom_grid_binary_length"],
                        topography_length=metadata["topography_binary_length"]
                    )
                    
                    if grid.custom_grid is not None:
                        grid.custom_grid.values = custom_grid_vals.reshape(-1, 3)
                    
                    if grid.topography is not None:
                        grid.topography.set_values2d(values=topography_vals)
                    
                    grid._update_values()
                    return grid
                case _:
                    raise ValidationError
        except ValidationError:
            raise

    @property
    def grid_binary(self):
        custom_grid_bytes = self._custom_grid.values.astype("float64").tobytes() if self._custom_grid else b''
        topography_bytes = self._topography.values.astype("float64").tobytes() if self._topography else b''
        return custom_grid_bytes + topography_bytes


    @computed_field
    def binary_meta_data(self) -> dict:
        return {
                'custom_grid_binary_length': len(self._custom_grid.values.astype("float64").tobytes()) if self._custom_grid else 0,
                'topography_binary_length': len(self._topography.values.astype("float64").tobytes()) if self._topography else 0,
        }

    @computed_field(alias="active_grids")
    @property
    def active_grids(self) -> GridTypes:
        return self._active_grids

    @active_grids.setter
    def active_grids(self, value: GridTypes):
        self._active_grids = value
        self._update_values()

    @classmethod
    def init_octree_grid(cls, extent, octree_levels):
        grid = cls()
        grid._octree_grid = RegularGrid(
            extent=extent,
            resolution=np.array([2 ** octree_levels] * 3),
        )
        grid.active_grids |= grid.GridTypes.OCTREE
        grid._update_values()
        return grid

    @classmethod
    def init_dense_grid(cls, extent, resolution):
        return cls(extent, resolution)

    def __str__(self):
        active_grid_types_str = [g_type for g_type in self.GridTypes if self.active_grids & g_type]

        grid_summary = [f"{g_type} (active: {getattr(self, g_type + '_grid_active')}): {len(getattr(self, g_type + '_grid').values)} points"
                        for g_type in active_grid_types_str]
        grid_summary_str = "\n".join(grid_summary)
        return f"Grid Object:\n{grid_summary_str}"

    @property
    def transform(self) -> Transform:
        if self.dense_grid is not None:
            return self.dense_grid.transform
        elif self.octree_grid is not None:
            return self.octree_grid.transform
        else:
            return Transform.init_neutral()

    @transform.setter
    def transform(self, value: Transform):
        self._transform = value

    @property
    def extent(self):
        if self.dense_grid is not None:
            return self.dense_grid.extent
        elif self.octree_grid is not None:
            return self.octree_grid.extent
        else:
            raise AttributeError('Extent is not defined')

    @property
    def corner_min(self):
        return self.extent[::2]

    @property
    def corner_max(self):
        return self.extent[1::2]

    @property
    def bounding_box(self):
        extents = self.extent
        # Define 3D points of the bounding box corners based on extents
        bounding_box_points = np.array([[extents[0], extents[2], extents[4]],  # min x, min y, min z
                                        [extents[0], extents[2], extents[5]],  # min x, min y, max z
                                        [extents[0], extents[3], extents[4]],  # min x, max y, min z
                                        [extents[0], extents[3], extents[5]],  # min x, max y, max z
                                        [extents[1], extents[2], extents[4]],  # max x, min y, min z
                                        [extents[1], extents[2], extents[5]],  # max x, min y, max z
                                        [extents[1], extents[3], extents[4]],  # max x, max y, min z
                                        [extents[1], extents[3], extents[5]]])  # max x, max y, max z
        return bounding_box_points

    @property
    def dense_grid(self) -> RegularGrid:
        return self._dense_grid

    @dense_grid.setter
    def dense_grid(self, value):
        self._dense_grid = value
        self.active_grids |= self.GridTypes.DENSE
        self._update_values()

    @property
    def octree_grid(self):
        return self._octree_grid

    @octree_grid.setter
    def octree_grid(self, value):
        raise AttributeError('Octree grid is not allowed to be set directly. Use init_octree_grid instead')

    def set_octree_grid(self, regular_grid: RegularGrid, evaluation_options: EvaluationOptions):
        regular_grid_resolution = regular_grid.resolution
        # Check all directions has the same res
        if not np.all(regular_grid_resolution == regular_grid_resolution[0]):
            raise AttributeError('Octree resolution must be isotropic')
        octree_levels = int(np.log2(regular_grid_resolution[0]))

        self._octree_grid = regular_grid
        self.active_grids |= self.GridTypes.OCTREE
        self._update_values()

    def set_octree_grid_by_levels(self, octree_levels: int, evaluation_options: EvaluationOptions, extent: Optional[np.ndarray] = None):
        if extent is None:
            extent = self.extent

        self._octree_grid = RegularGrid(
            extent=extent,
            resolution=np.array([2 ** octree_levels] * 3),
        )
        evaluation_options.number_octree_levels = octree_levels
        self.active_grids |= self.GridTypes.OCTREE
        self._update_values()

    @property
    def octree_levels(self):
        return self._octree_levels

    @octree_levels.setter
    def octree_levels(self, value):
        raise AttributeError('Octree levels are not allowed to be set directly. Use set_octree_grid instead')

    @property
    def custom_grid(self):
        return self._custom_grid

    @custom_grid.setter
    def custom_grid(self, value):
        self._custom_grid = value
        self.active_grids |= self.GridTypes.CUSTOM
        self._update_values()

    @property
    def topography(self):
        return self._topography

    @topography.setter
    def topography(self, value):
        self._topography = value
        self.active_grids |= self.GridTypes.TOPOGRAPHY
        self._update_values()

    @property
    def sections(self):
        return self._sections

    @sections.setter
    def sections(self, value):
        self._sections = value
        self.active_grids |= self.GridTypes.SECTIONS
        self._update_values()

    @property
    def centered_grid(self):
        return self._centered_grid

    @centered_grid.setter
    def centered_grid(self, value):
        self._centered_grid = value
        self.active_grids |= self.GridTypes.CENTERED
        self._update_values()

    @property
    def regular_grid(self):
        dense_grid_exists_and_active = self.dense_grid is not None and self.GridTypes.DENSE in self.active_grids
        octree_grid_exists_and_active = self.octree_grid is not None and self.GridTypes.OCTREE in self.active_grids

        if dense_grid_exists_and_active and octree_grid_exists_and_active:
            raise AttributeError('Both dense_grid and octree_grid are active. This is not possible.')
        elif dense_grid_exists_and_active:
            return self.dense_grid
        elif octree_grid_exists_and_active:
            return self.octree_grid
        else:
            return None

    # noinspection t
    def _update_values(self):
        values = []

        if self.GridTypes.OCTREE in self.active_grids:
            if self.octree_grid is None: raise AttributeError('Octree grid is active but not defined')
            values.append(self.octree_grid.values)
        if self.GridTypes.DENSE in self.active_grids:
            if self.dense_grid is None: raise AttributeError('Dense grid is active but not defined')
            values.append(self.dense_grid.values)
        if self.GridTypes.CUSTOM in self.active_grids:
            if self.custom_grid is None: raise AttributeError('Custom grid is active but not defined')
            values.append(self.custom_grid.values)
        if self.GridTypes.TOPOGRAPHY in self.active_grids:
            if self.topography is None: raise AttributeError('Topography grid is active but not defined')
            values.append(self.topography.values)
        if self.GridTypes.SECTIONS in self.active_grids:
            if self.sections is None: raise AttributeError('Sections grid is active but not defined')
            values.append(self.sections.values)
        if self.GridTypes.CENTERED in self.active_grids:
            if self.centered_grid is None: raise AttributeError('Centered grid is active but not defined')
            values.append(self.centered_grid.values)

        # make sure values is not empty
        if len(values) == 0:
            return self.values

        self.values = np.concatenate(values)

        return self.values

    def get_section_args(self, section_name: str):
        # TODO: This method should be part of the sections
        # assert type(section_name) is str, 'Only one section type can be retrieved'
        l0, l1 = self.get_grid_args('sections')
        where = np.where(self.sections.names == section_name)[0][0]
        return l0 + self.sections.length[where], l0 + self.sections.length[where + 1]
