import numpy as np

from gempy import optional_dependencies
from gempy.core.data.grid_modules import RegularGrid


def test_regular_grid_extent_rotation():

    # Define the coordinates of the two points
    x1, y1 = 2, 2
    x2, y2 = 8, 8 
    x3, y3 = -2, 6

    # Define z-range (you can modify as per your requirements)
    zmin, zmax = 0, 10
    
    regular_grid = RegularGrid.from_corners_box(
        pivot=(x1, y1),
        point_x_axis=(x2, y2),
        point_y_axis= (x3, y3),
        zmin=zmin,
        zmax=zmax,
        resolution=np.array([10, 10, 1]),
    )
    
    RegularGrid.plot_rotation(regular_grid, pivot=(x1, y1), point_x_axis=(x2, y2), point_y_axis=(x3, y3))
