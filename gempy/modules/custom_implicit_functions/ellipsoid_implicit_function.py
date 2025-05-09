import functools
import numpy as np


def ellipsoid_3d_factory(center: np.ndarray, radius: np.ndarray, max_slope: float) -> callable:
    """
    Implicit 3D ellipsoid.
    """

    implicit_ellipsoid = functools.partial(
        ellipsoid_scalar_field,
        center=center,
        radii=radius,
        k_factors=max_slope,
    )

    return implicit_ellipsoid


def ellipsoid_scalar_field(xyz, center, radii, k_factors):
    """Calculate scalar field value for given coordinates.

    Parameters:
    - xyz: numpy array of shape (N, 3), where N is the number of points
    - center: numpy array of shape (3,) representing the center of the ellipsoid
    - radii: numpy array of shape (3,) representing the semiaxes a, b, and c of the ellipsoid
    - k_factors: numpy array of shape (3,) representing the slope factors for x, y, and z directions.

    Returns:
    - A numpy array of shape (N,) containing the scalar field values.
    """
    displacements = xyz - center
    values = ((displacements[:, 0] / (radii[0] * k_factors[0])) ** 2 +
              (displacements[:, 1] / (radii[1] * k_factors[1])) ** 2 +
              (displacements[:, 2] / (radii[2] * k_factors[2])) ** 2 - 1)

    return - sigmoid(values * np.prod(k_factors)) + 1 # multiplying by the product of k_factors to keep the transition sharper


def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.exp(-x))


def _implicit_3d_ellipsoid_to_slope_(xyz: np.ndarray, center: np.ndarray, radius: np.ndarray,
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
