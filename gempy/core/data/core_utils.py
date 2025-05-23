import numpy as np

from gempy.optional_dependencies import require_scipy


def calculate_line_coordinates_2points(p1, p2, res):
    if isinstance(p1, list) or isinstance(p1, tuple):
        p1 = np.array(p1)
    if isinstance(p2, list) or isinstance(p2, tuple):
        p2 = np.array(p2)
    v = p2 - p1  # vector pointing from p1 to p2
    u = v / np.linalg.norm(v)  # normalize it
    distance = distance_2_points(p1, p2)
    steps = np.linspace(0, distance, res)
    values = p1.reshape(2, 1) + u.reshape(2, 1) * steps.ravel()
    return values.T


def distance_2_points(p1, p2):
    return np.sqrt(np.diff((p1[0], p2[0])) ** 2 + np.diff((p1[1], p2[1])) ** 2)


def interpolate_zvals_at_xy(xy, topography, method='DEP'):
    """
    Interpolates DEM values on a defined section.

    Args:
        xy (np.ndarray): Array of shape (n, 2) containing x (EW) and y (NS) coordinates of the profile.
        topography (Topography): An instance of Topography containing the DEM data.

    Returns:
        np.ndarray: z values, i.e., topography along the profile.
    """
    scipy = require_scipy()
    xj = topography.values_2d[:, 0, 0]
    yj = topography.values_2d[0, :, 1]
    zj = topography.values_2d[:, :, 2]

    spline = scipy.interpolate.RectBivariateSpline(xj, yj, zj)
    zi = spline.ev(xy[:, 0], xy[:, 1])
    
    return zi
