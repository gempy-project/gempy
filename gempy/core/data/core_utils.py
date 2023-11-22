import numpy as np

from gempy.optional_dependencies import require_scipy


def calculate_line_coordinates_2points(p1, p2, res):
    if isinstance(p1, list):
        p1 = np.array(p1)
    if isinstance(p2, list):
        p2 = np.array(p2)
    v = p2 - p1  # vector pointing from p1 to p2
    u = v / np.linalg.norm(v)  # normalize it
    distance = distance_2_points(p1, p2)
    steps = np.linspace(0, distance, res)
    values = p1.reshape(2, 1) + u.reshape(2, 1) * steps.ravel()
    return values.T


def distance_2_points(p1, p2):
    return np.sqrt(np.diff((p1[0], p2[0])) ** 2 + np.diff((p1[1], p2[1])) ** 2)


def interpolate_zvals_at_xy(xy, topography, method='interp2d'):
    """
    Interpolates DEM values on a defined section

    Args:
        xy: x (EW) and y (NS) coordinates of the profile
        topography (:class:`gempy.core.grid_modules.topography.Topography`)
        method: interpolation method, 'interp2d' for cubic scipy.interpolate.interp2d
                                         'spline' for scipy.interpolate.RectBivariateSpline

    Returns:
        numpy.ndarray: z values, i.e. topography along the profile

    """

    xj = topography.values_2d[:, 0, 0]
    yj = topography.values_2d[0, :, 1]
    zj = topography.values_2d[:, :, 2]
    scipy = require_scipy()
    if method == 'interp2d':
        f = scipy.interpolate.interp2d(xj, yj, zj.T, kind='cubic')
        zi = f(xy[:, 0], xy[:, 1])
        if xy[:, 0][0] <= xy[:, 0][-1] and xy[:, 1][0] <= xy[:, 1][-1]:
            return np.diag(zi)
        else:
            return np.flipud(zi).diagonal()
    else:
        assert xy[:, 0][0] <= xy[:, 0][
            -1], 'The xy values of the first point must be smaller than second.' \
                 'Please use interp2d as method argument. Will be fixed.'
        assert xy[:, 1][0] <= xy[:, 1][
            -1], 'The xy values of the first point must be smaller than second.' \
                 'Please use interp2d as method argument. Will be fixed.'
        f = scipy.interpolate.RectBivariateSpline(xj, yj, zj)
        zi = f(xy[:, 0], xy[:, 1])
        return np.flipud(zi).diagonal()
