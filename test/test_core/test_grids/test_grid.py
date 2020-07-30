# Importing GemPy
import gempy as gp


# Importing auxiliary libraries
import numpy as np


class TestGrid:
    def test_set_regular_grid(self):
        # Test creating an empty list
        grid = gp.Grid()

        # Test set regular grid by hand
        grid.create_regular_grid([0, 2000, 0, 2000, -2000, 0], [50, 50, 50])

    def test_grid_init(self):
        # Or we can init one of the default grids since the beginning by passing
        # the correspondant attributes
        grid = gp.Grid(extent=[0, 2000, 0, 2000, -2000, 0],
                       resolution=[50, 50, 50])

    def test_section_grid(self):
        geo_data = gp.create_data('section_grid', [0, 1000, 0, 1000, 0, 1000], resolution=[10, 10, 10])
        geo_data.set_topography()
        section_dict = {'section1': ([0, 0], [1000, 1000], [100, 80]),
                        'section2': ([800, 0], [800, 1000], [150, 100]),
                        'section3': ([50, 200], [100, 500], [200, 150])}

        geo_data.set_section_grid(section_dict)

        print(geo_data._grid.sections)
        np.testing.assert_almost_equal(geo_data._grid.sections.df.loc['section3', 'dist'], 304.138127,
                                       decimal=4)

    def test_set_section_twice(self):
        geo_data = gp.create_data(extent=[0, 1000, 0, 1000, 0, 1000], resolution=[10, 10, 10])
        section_dict = {'section1': ([0, 0], [1000, 1000], [100, 80]),
                        'section2': ([800, 0], [800, 1000], [150, 100]),
                        'section3': ([50, 200], [100, 500], [200, 150])}

        geo_data.set_section_grid(section_dict)
        geo_data.set_section_grid(section_dict)
        print(geo_data._grid.sections)

    def test_custom_grid(self):
        # create custom grid
        grid = gp.Grid()
        cg = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        grid.create_custom_grid(cg)
        # make sure the custom grid is active
        assert grid.active_grids[1]
        # make sure the custom grid is equal to the provided values
        np.testing.assert_array_almost_equal(cg, grid.custom_grid.values)
        # make sure we have the correct number of values in our grid
        l0, l1 = grid.get_grid_args('custom')
        assert l0 == 0
        assert l1 == 3


