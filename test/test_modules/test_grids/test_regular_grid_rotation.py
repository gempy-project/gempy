import numpy as np

from gempy import optional_dependencies
from gempy.core.data.grid_modules import RegularGrid


def test_regular_grid_extent_rotation():
    plt = optional_dependencies.require_matplotlib()

    # Define the coordinates of the two points
    x1, y1 = 2, 2
    x2, y2 = 8, 8 
    x3, y3 = -2, 6

    # Define z-range (you can modify as per your requirements)
    zmin, zmax = 0, 10
    
    regular_grid = RegularGrid.from_corners_box(
        pivot=(x1, y1),
        point2=(x2, y2),
        point3= (x3, y3),
        zmin=zmin,
        zmax=zmax,
        resolution=np.array([10, 10, 1]),
    )
    
    # Plot the original corners
    plt.scatter([x1, x2, x3], [y1, y2, y3], c='r')
    
    # Plot lines with distance labels
    plt.plot([x1, x2], [y1, y2], 'r')
    plt.text((x1 + x2) / 2, (y1 + y2) / 2, f'{np.linalg.norm([x1 - x2, y1 - y2]):.2f}', color='r')
    
    plt.plot([x1, x3], [y1, y3], 'r')
    plt.text((x1 + x3) / 2, (y1 + y3) / 2, f'{np.linalg.norm([x1 - x3, y1 - y3]):.2f}', color='r')
    
    
    # Plot the transformed corners
    transformed_extent  = regular_grid.extent
    bounding_box = regular_grid.bounding_box
    plt.scatter(bounding_box[:, 0], bounding_box[:, 1], c='b')
    
    # Plot lines with distance labels
    plt.plot([bounding_box[2, 0], bounding_box[0, 0]], [bounding_box[2, 1], bounding_box[0, 1]], 'b')
    plt.text((bounding_box[2, 0] + bounding_box[0, 0]) / 2, (bounding_box[2, 1] + bounding_box[0, 1]) / 2, f'{np.linalg.norm(bounding_box[2] - bounding_box[0]):.2f}', color='b')
    
    plt.plot([bounding_box[0, 0], bounding_box[4, 0]], [bounding_box[0, 1], bounding_box[4, 1]], 'b')
    plt.text((bounding_box[0, 0] + bounding_box[4, 0]) / 2, (bounding_box[0, 1] + bounding_box[4, 1]) / 2, f'{np.linalg.norm(bounding_box[0] - bounding_box[4]):.2f}', color='b')
    
    # Plot the values of the grid
    values = regular_grid.values
    plt.scatter(values[:, 0], values[:, 1], c='g')
    
    plt.show()
    
    