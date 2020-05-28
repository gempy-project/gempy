import gempy as gp
import os
input_path = os.path.dirname(__file__) + '/../../examples/data'


def test_set_orientations():
    # Importing the data from CSV-files and setting extent and resolution
    geo_data = gp.create_data(extent=[0, 2000, 0, 2000, 0, 2000], resolution=[50, 50, 50],
                              path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                              path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')

    gp.get_data(geo_data)

    # Assigning series to formations as well as their order (timewise)
    gp.map_stack_to_surfaces(geo_data, {"Fault_Series": 'Main_Fault',
                                         "Strat_Series": ('Sandstone_2', 'Siltstone')})

    geo_data._orientations.create_orientation_from_surface_points(geo_data.surface_points, [0, 1, 2])

    gp.set_orientation_from_surface_points(geo_data, [0, 1, 2])


def test_select_nearest_surface_points(one_fault_model_no_interp):
    pass