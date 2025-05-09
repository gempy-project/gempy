from typing import Optional

import numpy as np
from numpy.linalg import svd

from gempy.core.data import SurfacePointsTable, OrientationsTable


def create_orientations_from_surface_points_coords(xyz_coords: np.ndarray,
                                                   subset: Optional[np.ndarray] = None,
                                                   element_name: Optional[str] = "Generated") -> OrientationsTable:
    # Initialize the arrays
    center = np.empty((len(subset) if subset is not None else 1, 3))
    normal = np.empty((len(subset) if subset is not None else 1, 3))

    if subset is None:
        center[0], normal[0] = _plane_fit(xyz_coords)
    else:
        for idx, i in enumerate(subset):
            center[idx], normal[idx] = _plane_fit(xyz_coords[i])

    orientations = OrientationsTable.from_arrays(
        x=center[:, 0],
        y=center[:, 1],
        z=center[:, 2],
        G_x=normal[:, 0],
        G_y=normal[:, 1],
        G_z=normal[:, 2],
        names=[element_name]
    )
    return orientations


def _plane_fit(point_list):
    """
    Fit plane to points in PointSet
    Fit an d-dimensional plane to the points in a point set.
    adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    Args:
        point_list (array_like): array of points XYZ

    Returns:
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
    """

    points = point_list.T

    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.

    # ctr = Point(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
    normal = svd(M)[0][:, -1]
    # return ctr, svd(M)[0][:, -1]
    if normal[2] < 0:
        normal = - normal

    return ctr, normal
