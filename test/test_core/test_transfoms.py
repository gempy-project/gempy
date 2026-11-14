from gempy.core.data import GeoModel
from gempy_engine.core.data.transforms import Transform
from test.test_api.test_initialization_and_compute_api import _create_data
import numpy as np
import pytest
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




def test_default_transform_is_derived_on_compute():
    """A model whose structural frame was empty at creation keeps the identity transform until the first compute,
    where `gp.compute_model` derives it from the data (same result as calling `update_transform` explicitly)."""
    import warnings
    import gempy as gp
    from gempy_engine.core.data.options import InterpolationOptions

    def build():
        color_gen = gp.data.structural_frame.ColorsGenerator()
        element = gp.data.StructuralElement("layer", gp.data.SurfacePointsTable.initialize_empty(),
                                            gp.data.OrientationsTable.initialize_empty(), color=next(color_gen))
        frame = gp.data.StructuralFrame(
            structural_groups=[gp.data.StructuralGroup(name="S", elements=[element],
                                                       structural_relation=gp.data.StackRelationType.ERODE)],
            color_gen=color_gen)
        model = gp.create_geomodel(project_name="default_transform", extent=[0, 2000, 0, 2000, 0, 1000],
                                   refinement=2, structural_frame=frame)
        gp.add_surface_points(model, x=[300., 1000., 1700., 400.], y=[300., 250., 400., 1100.],
                              z=[560., 540., 520., 555.], elements_names=["layer"] * 4)
        gp.add_orientations(model, x=[1000.], y=[1000.], z=[500.], elements_names=["layer"], pole_vector=[[0., 0., 1.]])
        model.interpolation_options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE
        return model

    model = build()
    assert model.input_transform._is_default_transform
    assert np.allclose(model.input_transform.scale, 1.0)

    with pytest.warns(RuntimeWarning, match="default .* input transform"):
        solutions_auto = gp.compute_model(model)

    assert not model.input_transform._is_default_transform
    assert not np.allclose(model.input_transform.scale, 1.0)

    def compute_without_transform_warning(m):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sol = gp.compute_model(m)
        assert not [w for w in caught if "input transform" in str(w.message) or "default transform" in str(w.message)]
        return sol

    # Second compute: transform already derived, no warning.
    compute_without_transform_warning(model)

    # Same result as the documented explicit call.
    model_explicit = build()
    model_explicit.update_transform(gp.data.GlobalAnisotropy.NONE)
    assert np.allclose(model_explicit.input_transform.scale, model.input_transform.scale)
    assert np.allclose(model_explicit.input_transform.position, model.input_transform.position)
    solutions_explicit = compute_without_transform_warning(model_explicit)
    np.testing.assert_allclose(solutions_explicit.raw_arrays.scalar_field_matrix,
                               solutions_auto.raw_arrays.scalar_field_matrix)
