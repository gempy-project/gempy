import pprint
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
    def from_input_points(cls, surface_points: SurfacePointsTable, orientations: OrientationsTable) -> 'Transform':

        # ? Should we have instead a pointer to the structural frame and treat it more as a getter than a setter?
        
        input_points_xyz = np.concatenate([surface_points.xyz, orientations.xyz], axis=0)
        max_coord = np.max(input_points_xyz, axis=0)
        min_coord = np.min(input_points_xyz, axis=0)
        scaling_factor = 2 * np.max(max_coord - min_coord)
        center = (max_coord + min_coord) / 2
        # center = np.zeros(3)
        return cls(
            position=-center,
            rotation=np.zeros(3),
            scale=1 / np.array([scaling_factor, scaling_factor, scaling_factor])
        )

    @property
    def isometric_scale(self):
        # TODO: double check how was done in old gempy
        return 1/np.mean(self.scale)

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
        homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (self.get_transform_matrix(transform_op_order) @ homogeneous_points.T).T
        return transformed_points[:, :3]
    
    def apply_inverse(self, points: np.ndarray, transform_op_order: TransformOpsOrder = TransformOpsOrder.SRT):
        # * NOTE: to compare with legacy we would have to add 0.5 to the coords
        assert points.shape[1] == 3
        homogeneous_points = np.concatenate([points, -np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (np.linalg.inv(self.get_transform_matrix(transform_op_order)) @ homogeneous_points.T).T
        return transformed_points[:, :3]
    