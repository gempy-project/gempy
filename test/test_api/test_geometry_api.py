import gempy as gp
import numpy as np
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

def test_select_nearest_surface_points():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data = gp.create_data('fault', extent=[0, 1000, 0, 1000, 0, 1000],
                              resolution=[50, 50, 50],
                              path_o=path_to_data + "model5_orientations.csv",
                              path_i=path_to_data + "model5_surface_points.csv")

    # Assigning series to formations as well as their order (timewise)
    gp.map_stack_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                        "Strat_Series": ('rock2', 'rock1')})
    geo_data.set_is_fault(['Fault_Series'])

    # detect fault names
    f_id = geo_data._faults.df.index.categories[
        geo_data._faults.df.isFault.values]
    # find fault points
    fault_poi = geo_data._surface_points.df[
        geo_data._surface_points.df.series.isin(f_id)]

    # find neighbours
    knn = gp.select_nearest_surfaces_points(geo_data, fault_poi, 1)
    radius = gp.select_nearest_surfaces_points(geo_data, fault_poi, 200.)

    # sort neighbours, necessary for equal testing
    knn = [k.sort_values() for k in knn]
    radius = [r.sort_values() for r in radius]

    # define reference
    reference = [[16,17],[16,17],[18,19],[18,19],[20,21],[20,21]]

    assert np.array_equal(reference, knn) and np.array_equal(reference, radius)


def test_set_orientation_from_neighbours():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data = gp.create_data('fault', extent=[0, 1000, 0, 1000, 0, 1000],
                              resolution=[50, 50, 50],
                          path_o=path_to_data + "model5_orientations.csv",
                          path_i=path_to_data + "model5_surface_points.csv")

    # Assigning series to formations as well as their order (timewise)
    gp.map_stack_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                    "Strat_Series": ('rock2', 'rock1')})
    geo_data.set_is_fault(['Fault_Series'])

    # detect fault names
    f_id = geo_data._faults.df.index.categories[
        geo_data._faults.df.isFault.values]
    # find fault points
    fault_poi = geo_data._surface_points.df[
        geo_data._surface_points.df.series.isin(f_id)]
    # find neighbours
    neighbours = gp.select_nearest_surfaces_points(geo_data, fault_poi, 5)
    # calculate one fault orientation
    gp.set_orientation_from_neighbours(geo_data, neighbours[1])
    # find the calculated orientation
    test = geo_data._orientations.df.sort_index().iloc[-1][['dip', 'azimuth']].values

    # calculate reference
    reference = [90-np.arctan(0.5)/np.pi*180, 90]

    assert np.array_equal(reference, test)

def test_set_orientation_from_neighbours_all():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data = gp.create_data('fault', extent=[0, 1000, 0, 1000, 0, 1000],
                              resolution=[50, 50, 50],
                          path_o=path_to_data + "model5_orientations.csv",
                          path_i=path_to_data + "model5_surface_points.csv")

    # count orientations before orientation calculation
    length_pre = geo_data._orientations.df.shape[0]

    # find neighbours
    neighbours = gp.select_nearest_surfaces_points(geo_data, geo_data._surface_points.df, 2)
    # calculate all fault orientations
    gp.set_orientation_from_neighbours_all(geo_data, neighbours)

    # count orientations after orientation calculation
    length_after = geo_data._orientations.df.shape[0]

    assert np.array_equal(geo_data._surface_points.df.shape[0],
                          length_after-length_pre)
