import functools
import numpy as np


def ellipsoid_3d_factory(center: np.ndarray, radius: np.ndarray, max_slope: float) -> callable:
    """
    Implicit 3D ellipsoid.
    """

    implicit_ellipsoid = functools.partial(
        _implicit_3d_ellipsoid_to_slope,
        center=center,
        radius=radius,
        max_slope=max_slope,
    )

    return implicit_ellipsoid


def _implicit_3d_ellipsoid_to_slope(xyz: np.ndarray, center: np.ndarray, radius: np.ndarray,
                                    max_slope: float = 1000):
    """
    Implicit 3D ellipsoid.
    """
    # ! Finite faults needs quite a bit of fine tunning once we can compute models in real time
    # ! This function only works for elipses perpendicular to the cartesian axis
    scalar = - np.sum((xyz - center) ** 2.00 / (radius ** 2), axis=1)
    scalar_shifted = scalar - scalar.min()

    sigmoid_slope = 10  # ? This probably should be also public
    Z_x = scalar_shifted

    drift_0 = 4  # ? Making it a %. It depends on the radius
    scale_0 = max_slope
    scalar_final = scale_0 / (1 + np.exp(-sigmoid_slope * (Z_x - drift_0)))
    return scalar_final
