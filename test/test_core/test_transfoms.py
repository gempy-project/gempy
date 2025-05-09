from gempy.core.data import GeoModel
from gempy_engine.core.data.transforms import Transform
from test.test_api.test_initialization_and_compute_api import _create_data
import numpy as np
import matplotlib.pyplot as plt


def test_transform_1():
    geo_data: GeoModel = _create_data()
    print(geo_data.input_transform)
    transformed_xyz = geo_data.input_transform.apply(geo_data.surface_points_copy.xyz)
    print(transformed_xyz)
    return


# Assuming you have your Transform class defined above this.

def plot_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()


def test_transform_operations_scale_move():
    transform = Transform(
        position=np.array([1, 2, 3]),
        rotation=np.array([0, 0, 0]),
        scale=np.array([2, 2, 2]))

    original_points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1]
    ])

    transformed_points = transform.apply(original_points)
    inv_transformed_points = transform.apply_inverse(transformed_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original points
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c='r', marker='o', label='Original')
    # Transformed points
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='g', marker='^', label='Transformed')
    # Inverse transformed points
    ax.scatter(inv_transformed_points[:, 0], inv_transformed_points[:, 1], inv_transformed_points[:, 2], c='b', marker='x', label='Inv Transformed')

    ax.legend()
    plt.show()

    assert np.allclose(original_points, inv_transformed_points)



def test_transform_operations_rotate():

    transform = Transform(
        position=np.array([0, 0, 0]),
        rotation=np.array([45, 45, 45]),
        scale=np.array([1, 1, 1]))

    original_points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1]
    ])

    transformed_points = transform.apply(original_points)
    inv_transformed_points = transform.apply_inverse(transformed_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Original points
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c='r', marker='o', label='Original')
    # Transformed points
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='g', marker='^', label='Transformed')
    # Inverse transformed points
    ax.scatter(inv_transformed_points[:, 0], inv_transformed_points[:, 1], inv_transformed_points[:, 2], c='b', marker='x', label='Inv Transformed')

    ax.legend()
    plt.show()

    assert np.allclose(original_points, inv_transformed_points)


