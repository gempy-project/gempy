import gempy as gp
import matplotlib.pyplot as plt
import pandas as pn
import numpy as np
import os
import pytest
input_path = os.path.dirname(__file__)+'/../input_data'

# ## Preparing the Python environment
#
# For modeling with GemPy, we first need to import it. We should also import any other packages we want to
# utilize in our Python environment.Typically, we will also require `NumPy` and `Matplotlib` when working
# with GemPy. At this point, we can further customize some settings as desired, e.g. the size of figures or,
# as we do here, the way that `Matplotlib` figures are displayed in our notebook (`%matplotlib inline`).


# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

import sys, os

sys.path.append("../../../gempy")
import gempy as gp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test_pile_geomodel():
    ve = 3
    extent = [451e3, 456e3, 6.7820e6, 6.7840e6, -2309 * ve, -1651 * ve]

    geo_model = gp.create_model('Topology-Gullfaks')

    gp.init_data(geo_model, extent, [30, 30, 30],
                 path_o=input_path + "/filtered_orientations.csv",
                 path_i=input_path + "/filtered_surface_points.csv", default_values=True)

    series_distribution = {
        "fault3": "fault3",
        "fault4": "fault4",
        "unconformity": "BCU",
        "sediments": ("tarbert", "ness", "etive"),
    }

    gp.map_series_to_surfaces(geo_model,
                              series_distribution,
                              remove_unused_series=True)

    geo_model.reorder_series(["unconformity", "fault3", "fault4",
                              "sediments", "Basement"])

    geo_model.set_is_fault(["fault3"])
    geo_model.set_is_fault(["fault4"])

    rel_matrix = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

    geo_model.set_fault_relation(rel_matrix)

    surf_groups = pd.read_csv(input_path + "/filtered_surface_points.csv").group
    geo_model.surface_points.df["group"] = surf_groups
    orient_groups = pd.read_csv(input_path + "/filtered_orientations.csv").group
    geo_model.orientations.df["group"] = orient_groups

    geo_model.surface_points.df.reset_index(inplace=True, drop=True)
    geo_model.orientations.df.reset_index(inplace=True, drop=True)

    gp.set_interpolator(geo_model)
    gp.compute_model(geo_model)

    gp.plot.plot_section(geo_model, cell_number=25,
                         direction='y', show_data=True)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_lith_block')

    return geo_model


def test_pile_geomodel_2():
    ve = 3
    extent = [451e3, 456e3, 6.7820e6, 6.7840e6, -2309 * ve, -1651 * ve]

    geo_model = gp.create_model('Topology-Gullfaks')

    gp.init_data(geo_model, extent, [30, 30, 30],
                 path_o=input_path + "/filtered_orientations.csv",
                 path_i=input_path + "/filtered_surface_points.csv", default_values=True)

    series_distribution = {
        "fault3": "fault3",
        "fault4": "fault4",
        "unconformity": "BCU",
        "sediments": ("tarbert", "ness", "etive"),
    }

    gp.map_series_to_surfaces(geo_model,
                              series_distribution,
                              remove_unused_series=True)

    geo_model.reorder_series(["unconformity", "fault3", "fault4",
                              "sediments", "Basement"])

    geo_model.set_is_fault(["fault3"])
    geo_model.set_is_fault(["fault4"])

    rel_matrix = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

    geo_model.set_fault_relation(rel_matrix)

    surf_groups = pd.read_csv(input_path + "/filtered_surface_points.csv").group
    geo_model.surface_points.df["group"] = surf_groups
    orient_groups = pd.read_csv(input_path + "/filtered_orientations.csv").group
    geo_model.orientations.df["group"] = orient_groups

    geo_model.surface_points.df.reset_index(inplace=True, drop=True)
    geo_model.orientations.df.reset_index(inplace=True, drop=True)

    gp.set_interpolator(geo_model, verbose=['mask_matrix_loop', 'mask_e', 'nsle'])
    gp.compute_model(geo_model)

    gp.plot.plot_section(geo_model, cell_number=25,
                         direction='y', show_data=True)

    from gempy.plot.plot_api import plot_2d

    p = plot_2d(geo_model, cell_number=[25])

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_lith_block')

    return geo_model

