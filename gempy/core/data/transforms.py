import pprint
import warnings
from enum import Enum, auto

import numpy as np
from dataclasses import dataclass

from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable


class TransformOpsOrder(Enum):
    TRS = auto()  # * The order of the transformations is: scale, rotation, translation
    SRT = auto()  # * The order of the transformations is: translation, rotation, scale


@dataclass
class Transform:
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray

    _is_default_transform: bool = False

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def __post_init__(self):
        assert self.position.shape == (3,)
        assert self.rotation.shape == (3,)
        assert self.scale.shape == (3,)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        position = matrix[:3, 3]
        rotation = np.array([
            np.arctan2(matrix[2, 1], matrix[2, 2]),
            np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)),
            np.arctan2(matrix[1, 0], matrix[0, 0])
        ])
        scale = np.array([
            np.linalg.norm(matrix[0, :3]),
            np.linalg.norm(matrix[1, :3]),
            np.linalg.norm(matrix[2, :3])
        ])
        return cls(position, rotation, scale)

    @classmethod
    def from_input_points_(cls, surface_points: SurfacePointsTable, orientations: OrientationsTable) -> 'Transform':

        # ? Should we have instead a pointer to the structural frame and treat it more as a getter than a setter?

        input_points_xyz = np.concatenate([surface_points.xyz, orientations.xyz], axis=0)
        max_coord = np.max(input_points_xyz, axis=0)
        min_coord = np.min(input_points_xyz, axis=0)
        scaling_factor = 2 * np.max(max_coord - min_coord)
        center = (max_coord + min_coord) / 2
        # [1.650345e+05  3.950050e+05 - 9.470000e+00]
        # center = np.zeros(3)
        factor_ = 1 / np.array([scaling_factor, scaling_factor, scaling_factor / 100])
        # [5.56006539e-06 5.56006539e-06 5.56006539e-04]
        return cls(
            position=-center,
            rotation=np.zeros(3),
            scale=factor_
        )
    @classmethod
    def from_input_points(cls, surface_points: SurfacePointsTable, orientations: OrientationsTable) -> 'Transform':
        input_points_xyz = np.concatenate([surface_points.xyz, orientations.xyz], axis=0)
        if input_points_xyz.shape[0] == 0:
            transform = cls(position=np.zeros(3), rotation=np.zeros(3), scale=np.ones(3))
            transform._is_default_transform = True
            return transform

        max_coord = np.max(input_points_xyz, axis=0)
        min_coord = np.min(input_points_xyz, axis=0)

        # Compute the range for each dimension
        range_coord = 2 * (max_coord - min_coord)

        # Avoid division by zero; replace zero with a small number
        range_coord = np.where(range_coord == 0, 1e-10, range_coord)

        # The scaling factor for each dimension is the inverse of its range
        scaling_factors = 1 / range_coord
        # ! Be careful with toy models
        scaling_factors = np.array([scaling_factors[0], scaling_factors[0], scaling_factors[0]])
        center = (max_coord + min_coord) / 2
        return cls(
            position=-center,
            rotation=np.zeros(3),
            scale=scaling_factors
        )

    @property
    def isometric_scale(self):
        # TODO: double check how was done in old gempy
        return 1 / np.mean(self.scale)

    def get_transform_matrix(self, transform_type: TransformOpsOrder = TransformOpsOrder.SRT) -> np.ndarray:
        T = np.eye(4)
        R = np.eye(4)
        S = np.eye(4)

        T[:3, 3] = self.position

        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(rx), -np.sin(rx)],
             [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)],
             [0, 1, 0],
             [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0],
             [np.sin(rz), np.cos(rz), 0],
             [0, 0, 1]]
        )

        R[:3, :3] = Rx @ Ry @ Rz
        S[:3, :3] = np.diag(self.scale)

        match transform_type:
            case TransformOpsOrder.TRS:
                return T @ R @ S
            case TransformOpsOrder.SRT:
                return S @ R @ T
            case _:
                raise NotImplementedError(f"Transform type {transform_type} not implemented")

    def apply(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        # * NOTE: to compare with legacy we would have to add 0.5 to the coords
        assert points.shape[1] == 3
        if self._is_default_transform:
            warnings.warn(
                message="Interpolation is being done with the default transform. "
                        "If you do not know what you are doing you should probably call GeoModel.update_transform() first.",
                category=RuntimeWarning
            )

        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (self.get_transform_matrix(transform_op_order) @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def apply_inverse(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        # * NOTE: to compare with legacy we would have to add 0.5 to the coords
        assert points.shape[1] == 3
        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (np.linalg.inv(self.get_transform_matrix(transform_op_order)) @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def transform_gradient(self, gradients: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT,
                           preserve_magnitude: bool = True) -> np.ndarray:
        assert gradients.shape[1] == 3

        # Extract the 3x3 upper-left section of the transformation matrix
        transformation_3x3 = self.get_transform_matrix(transform_op_order)[:3, :3]

        # Compute the inverse transpose of this 3x3 matrix
        inv_trans_3x3 = np.linalg.inv(transformation_3x3).T

        # Multiply the gradients by this inverse transpose matrix
        transformed_gradients = (inv_trans_3x3 @ gradients.T).T

        if preserve_magnitude:
            # Compute the magnitude of the original gradients
            gradient_magnitudes = np.linalg.norm(gradients, axis=1)

            # Compute the magnitude of the transformed gradients
            transformed_gradient_magnitudes = np.linalg.norm(transformed_gradients, axis=1)

            # Compute the ratio between the two magnitudes
            magnitude_ratio = transformed_gradient_magnitudes / gradient_magnitudes

            # Multiply the transformed gradients by this ratio
            transformed_gradients /= magnitude_ratio[:, None]

        return transformed_gradients
