import gempy
import os
input_path = os.path.dirname(__file__)+'/../input_data'


def test_rescaled_marching_cube(interpolator_islith_nofault):
    """
         2 Horizontal layers with drift 0
         """
    # Importing the data from csv files and settign extent and resolution
    geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    geo_data.set_theano_function(interpolator_islith_nofault)

    # Compute model
    sol = gempy.compute_model(geo_data, compute_mesh_options={'rescale': True})
    print(sol.vertices)

    return geo_data