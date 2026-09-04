from dataclasses import dataclass, field
from typing import Annotated, Optional, Sequence, Union

import numpy as np
from pydantic import Field

from ._data_points_helpers import generate_ids_from_names


DEFAULT_MICRO_POINT_NUGGET = 0.0
MICRO_POINT_AFFINE_TOLERANCE = 1e-6
MICRO_POINT_MAX_CONDITION_NUMBER = 1e12


@dataclass
class MicroPointsTable:
    """Micro contacts represented by local-support-to-world transforms."""

    dt = np.dtype([
            ("support_transform", "<f8", (4, 4)),
            ("element_id", "<i8"),
            ("nugget", "<f8"),
    ], align=False)

    data: Annotated[np.ndarray, Field(exclude=True)] = field(
        default_factory=lambda: np.zeros(0, dtype=MicroPointsTable.dt)
    )
    name_id_map: Optional[dict[str, int]] = None

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray) or self.data.dtype != self.dt:
            raise ValueError(f"Data array must have the following data type: {self.dt}")
        if self.data.ndim != 1:
            raise ValueError("Micro-point data must be a one-dimensional structured array")

        transforms = self.support_transforms
        nuggets = self.nugget
        if not np.all(np.isfinite(transforms)) or not np.all(np.isfinite(nuggets)):
            raise ValueError("Micro-point transforms and nuggets must be finite")
        if np.any(nuggets < 0):
            raise ValueError("Micro-point nuggets must be nonnegative")
        if len(self) == 0:
            return

        expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(
                transforms[:, 3, :],
                expected_last_row,
                atol=MICRO_POINT_AFFINE_TOLERANCE,
                rtol=0.0,
        ):
            raise ValueError("Micro-point transforms must have affine last row [0, 0, 0, 1]")

        linear_blocks = transforms[:, :3, :3]
        scales = np.linalg.norm(linear_blocks, axis=1)
        if np.any(scales <= 0):
            raise ValueError("Micro-point support scales must be positive")

        normalized_axes = linear_blocks / scales[:, None, :]
        gram_matrices = np.einsum("nji,njk->nik", normalized_axes, normalized_axes)
        if not np.allclose(
                gram_matrices,
                np.eye(3),
                atol=MICRO_POINT_AFFINE_TOLERANCE,
                rtol=0.0,
        ):
            raise ValueError("Micro-point support transforms must contain orthogonal axes without shear")

        determinants = np.linalg.det(linear_blocks)
        if np.any(determinants <= 0):
            raise ValueError("Micro-point support transforms must be right-handed and invertible")

        condition_numbers = np.linalg.cond(linear_blocks)
        if np.any(condition_numbers > MICRO_POINT_MAX_CONDITION_NUMBER):
            raise ValueError(
                f"Micro-point support transform condition number exceeds {MICRO_POINT_MAX_CONDITION_NUMBER:g}"
            )

    @classmethod
    def from_transforms(
            cls,
            support_transforms: np.ndarray,
            names: Union[Sequence[str], str],
            nugget: Optional[np.ndarray] = None,
            name_id_map: Optional[dict[str, int]] = None,
    ) -> "MicroPointsTable":
        transforms = np.asarray(support_transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError("support_transforms must have shape (N, 4, 4)")

        n_points = len(transforms)
        if isinstance(names, str):
            normalized_names: Union[Sequence[str], str] = names
        else:
            normalized_names = list(names)
            if len(normalized_names) != n_points:
                raise ValueError("names and support_transforms must have the same length")

        if nugget is None:
            nuggets = np.full(n_points, DEFAULT_MICRO_POINT_NUGGET, dtype=np.float64)
        else:
            nuggets = np.asarray(nugget, dtype=np.float64)
            if nuggets.ndim != 1 or len(nuggets) != n_points:
                raise ValueError("nugget must have shape (N,)")

        ids, resolved_name_id_map = generate_ids_from_names(
            name_id_map,
            normalized_names,
            np.empty(n_points),
        )
        if len(ids) != n_points:
            raise ValueError("names and support_transforms must have the same length")

        data = np.zeros(n_points, dtype=cls.dt)
        data["support_transform"] = transforms
        data["element_id"] = ids
        data["nugget"] = nuggets
        return cls(data=data, name_id_map=resolved_name_id_map)

    @classmethod
    def initialize_empty(cls) -> "MicroPointsTable":
        return cls(data=np.zeros(0, dtype=cls.dt), name_id_map={})

    @property
    def support_transforms(self) -> np.ndarray:
        return self.data["support_transform"]

    @property
    def xyz(self) -> np.ndarray:
        return self.support_transforms[:, :3, 3]

    @property
    def ids(self) -> np.ndarray:
        return self.data["element_id"]

    @property
    def nugget(self) -> np.ndarray:
        return self.data["nugget"]

    def get_micro_points_by_name(self, name: str) -> "MicroPointsTable":
        if self.name_id_map is None:
            raise ValueError("name_id_map is not set")
        return self.get_micro_points_by_id(self.name_id_map[name])

    def get_micro_points_by_id(self, element_id: int) -> "MicroPointsTable":
        return MicroPointsTable(
            data=self.data[self.ids == element_id],
            name_id_map=self.name_id_map,
        )

    def get_micro_points_by_id_groups(self) -> list["MicroPointsTable"]:
        return [self.get_micro_points_by_id(element_id) for element_id in np.unique(self.ids)]

    def __len__(self) -> int:
        return len(self.data)
