import gempy.utils.diamond_square
# import pytest  # to add fixtures
import numpy as np  # as another testing environment


def test_class_nocrash():
    """Simply check if class can be instantiated"""
    gempy.utils.diamond_square.DiaomondSquare(size=(5, 5))


def test_grid_generation():
    """Test grid generation and extension for non-suitable grid sizes"""
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(5, 5))
    assert ds.grid.shape == (5, 5)
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(8, 10))
    assert ds.grid.shape == (9, 17)


def test_diamond_selection():
    """Test selection of diamond positions"""
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(5, 5))
    z = ds.get_selection_diamond(1)
    assert np.all(z == np.array([[2, 0, 0, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 0, 0, 2]]))
    z = ds.get_selection_diamond(0)
    assert np.all(z == np.array([[2, 0, 2, 0, 2],
                                 [0, 1, 0, 1, 0],
                                 [2, 0, 2, 0, 2],
                                 [0, 1, 0, 1, 0],
                                 [2, 0, 2, 0, 2]]))


def test_square_selection():
    """Test selection of diamond positions"""
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(5, 5))
    z = ds.get_selection_square(0)
    assert np.all(z == np.array([[0, 0, 2, 0, 2, 0, 0],
                                 [0, 2, 1, 2, 1, 2, 0],
                                 [2, 1, 2, 1, 2, 1, 2],
                                 [0, 2, 1, 2, 1, 2, 0],
                                 [2, 1, 2, 1, 2, 1, 2],
                                 [0, 2, 1, 2, 1, 2, 0],
                                 [0, 0, 2, 0, 2, 0, 0]]))
    z = ds.get_selection_square(1)
    assert np.all(z == np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 2, 0, 1, 0, 2, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [2, 0, 1, 0, 2, 0, 1, 0, 2],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 2, 0, 1, 0, 2, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 2, 0, 0, 0, 0]]))


def test_random_initialization():
    """Test random initialization of corner points"""
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(5, 6), seed=52062)
    ds.random_initialization()
    m_pow_max = min(ds.n, ds.m)
    step_size = int(2 ** m_pow_max)
    np.testing.assert_array_almost_equal(ds.grid[::step_size, ::step_size],
                                         np.array([[0.35127005, 0.55476571, 0.93745213],
                                                   [0.66668382, 0.85215985, 0.53222795]]))


def test_random_initialization_level():
    """Test random initialization on lower level"""
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(33, 33), seed=52062)
    level = 3
    ds.random_initialization(level = level)
    step_size = int(2 ** level)
    np.testing.assert_array_almost_equal(ds.grid[::step_size, ::step_size],
                                         np.array([[0.35127005, 0.55476571, 0.93745213, 0.66668382, 0.85215985],
       [0.53222795, 0.55800027, 0.20974513, 0.74837501, 0.64394326],
       [0.0359961 , 0.22723278, 0.56347804, 0.13438884, 0.32613594],
       [0.20868763, 0.03116471, 0.1498014 , 0.20755495, 0.86021482],
       [0.64707457, 0.44744272, 0.36504945, 0.52473407, 0.27948164]]))


def test_reset_grid():
    ds = gempy.utils.diamond_square.DiaomondSquare(size=(5, 6), seed=52062)
    ds.random_initialization()
    ds.reset_grid()
    np.testing.assert_array_almost_equal(ds.grid,
                                         np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
