import numpy as np
import pytest

from test.verify_helper import verify_model_serialization
from test.conftest import TEST_SPEED, TestSpeed
import gempy as gp
from gempy.core.data.enumerators import ExampleModel

pytestmark = pytest.mark.skipif(TEST_SPEED.value < TestSpeed.SECONDS.value, reason="Global test speed below this test value.")

xyz_coord = np.array(
    [[0, 0, 0],
     [1000, 0, 0],
     [0, 1000, 0],
     [1000, 1000, 0],
     [0, 0, 1000],
     [1000, 0, 1000],
     [0, 1000, 1000],
     [1000, 1000, 1000]],
    dtype=float
)


def test_custom_grid():
    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ANTICLINE,
        compute_model=False
    )

    geo_model.interpolation_options.number_octree_levels = 2

    gp.set_custom_grid(
        grid=geo_model.grid,
        xyz_coord=xyz_coord
    )

    verify_model_serialization(
        model=geo_model,
        verify_moment="after",
        file_name=f"verify/{geo_model.meta.name}"
    )
    
    sol: gp.data.Solutions = gp.compute_model(geo_model, validate_serialization=True)
    np.testing.assert_array_equal(
        sol.raw_arrays.custom,
        np.array([3., 3., 3., 3., 1., 1., 1., 1.])
    )


def test_compute_at():
    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ANTICLINE,
        compute_model=False
    )

    geo_model.interpolation_options.number_octree_levels = 2
    sol: np.ndarray = gp.compute_model_at(
        gempy_model=geo_model,
        at=xyz_coord
    )
    
    np.testing.assert_array_equal(
        sol,
        np.array([3., 3., 3., 3., 1., 1., 1., 1.])
    )
