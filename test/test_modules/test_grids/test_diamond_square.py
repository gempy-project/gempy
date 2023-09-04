import gempy.core.data.grid_modules.diamond_square
import pytest  # to add fixtures and to test error raises
import numpy as np  # as another testing environment


def test_class_nocrash():
    """Simply check if class can be instantiated"""
    gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 5))


def test_grid_generation():
    """Test grid generation and extension for non-suitable grid sizes"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 5))
    assert ds.grid.shape == (5, 5)
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(8, 10))
    assert ds.grid.shape == (9, 17)


def test_diamond_selection():
    """Test selection of diamond positions"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 5))
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
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 5))
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
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 6), seed=52062)
    ds.random_initialization()
    m_pow_max = min(ds.n, ds.m)
    step_size = int(2 ** m_pow_max)
    np.testing.assert_array_almost_equal(ds.grid[::step_size, ::step_size],
                                         np.array([[0.35127005, 0.55476571, 0.93745213],
                                                   [0.66668382, 0.85215985, 0.53222795]]))


def test_random_initialization_level():
    """Test random initialization on lower level"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(33, 33), seed=52062)
    level = 3
    ds.random_initialization(level=level)
    step_size = int(2 ** level)
    np.testing.assert_array_almost_equal(ds.grid[::step_size, ::step_size],
                                         np.array([[0.35127005, 0.55476571, 0.93745213, 0.66668382, 0.85215985],
                                                   [0.53222795, 0.55800027, 0.20974513, 0.74837501, 0.64394326],
                                                   [0.0359961, 0.22723278, 0.56347804, 0.13438884, 0.32613594],
                                                   [0.20868763, 0.03116471, 0.1498014, 0.20755495, 0.86021482],
                                                   [0.64707457, 0.44744272, 0.36504945, 0.52473407, 0.27948164]]))

def test_reset_grid():
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(5, 6), seed=52062)
    ds.random_initialization()
    ds.reset_grid()
    np.testing.assert_array_almost_equal(ds.grid,
                                         np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 0.]]))


def test_random_func():
    """Test random function implementation"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(33, 33), seed=52062)
    np.testing.assert_array_almost_equal(ds.random_func(2, 2),
                                         np.array([-0.14872995, 0.05476571]))
    # testing for correct default implementation
    ds.r_type = 'default'
    np.testing.assert_array_almost_equal(ds.random_func(2, 2),
                                         np.array([0.43745213, 0.16668382]))
    # testing long-range correlation
    ds.r_type = 'long_range'
    np.testing.assert_array_almost_equal(ds.random_func(2, 2),
                                         np.array([0.04401998, 0.00402849]))
    # testing level-scale correlation
    ds.r_type = 'level_scale'
    np.testing.assert_array_almost_equal(ds.random_func(2, 2),
                                         np.array([0.18600009, 0.06991504]))
    # testing deterministic implementation (no random value)
    ds.r_type = 'deterministic'
    assert ds.random_func(2, 2) == 0.0


def test_random_func_raises_error():
    """Test if random function raises NonImplementedError correctly"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(33, 33), seed=52062)
    ds.r_type = 'fail'

    with pytest.raises(NotImplementedError):
        ds.random_func(2, 2)


def test_interpolate():
    """Test interpolation step itself"""
    ds = gempy.core.data.grid_modules.diamond_square.DiaomondSquare(size=(9, 9), seed=52062)
    ds.interpolate()
    np.testing.assert_array_almost_equal(ds.grid,
                                         np.array([[0., 0.2951411, 0.21781267, 0.29361906, 0.01037812,
                                                    -0.0376406, -0.59889259, 0.01136296, 0.],
                                                   [-0.13102895, -0.07079394, 0.40240191, -0.24139454, -0.64535709,
                                                    -0.25358984, -0.20811689, -0.38977623, -0.02280871],
                                                   [-0.32311967, -0.08246826, 0.03236034, -0.72313104, -0.6863271,
                                                    -0.09742037, 0.16154592, -0.41643384, -0.23968483],
                                                   [-0.53344647, -0.09313507, -0.66247738, -0.42849468, -0.06519284,
                                                    -0.50628043, -0.31159035, 0.53516982, 0.07387422],
                                                   [0.23421434, 0.32817758, -0.45156142, -0.24627659, -0.2974599,
                                                    0.16071127, 0.36261452, 0.62070397, 0.60516641],
                                                   [-0.20172896, -0.05668301, -0.26331217, -0.22196496, 0.42029741,
                                                    0.35078669, 0.77129922, 0.38999358, 0.95701668],
                                                   [-0.41296032, -0.04377428, -0.23235603, 0.60954786, 0.72643437,
                                                    0.37788456, 0.62211967, 0.16198846, 0.61709021],
                                                   [0.00192109, -0.28399285, 0.28596529, 0.54081866, 1.00235637,
                                                    0.25454729, 0.1248549, 0.85789169, 0.23511424],
                                                   [0., -0.54507578, -0.33592062, 0.62216544, 0.77575097,
                                                    0.5338132, 0.22007596, 0.02926128, 0.]]))
