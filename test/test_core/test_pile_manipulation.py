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


def test_reorder_series():

    geo_model = gp.create_model('Geological_Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 4000, 0, 2775, 200, 1200], resolution=[100, 10, 100])
    # Adding a fault
    geo_model.rename_series(['Cycle1'])

    geo_model.add_series(['Fault1'])
    geo_model.set_is_fault(['Fault1'])
    geo_model.reorder_series(['Fault1', 'Cycle1'])
    assert geo_model.series.df['BottomRelation'] == ['Fault', 'Erosion']
    assert geo_model.series.df.index == ['Fault1', 'Cycle1']
    print(geo_model.series.df)


def test_complete_model():
    # ### Initializing the model:
    compute = True

    geo_model = gp.create_model('Geological_Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 4000, 0, 2775, 200, 1200], resolution=[100, 10, 100])

    if compute is True:
        gp.set_interpolator(geo_model, theano_optimizer='fast_compile', update_kriging=True,
                        verbose=[])

    from gempy.plot import visualization_2d_pro as vv

    # In this case perpendicular to the z axes
    p2d = vv.Plot2D(geo_model)
    p2d.create_figure((15, 8))
    ax = p2d.add_section(direction='z', ax_pos=121)

    ax2 = p2d.add_section(direction='y', ax_pos=122)
    ax2.set_xlim(geo_model.grid.regular_grid.extent[0], geo_model.grid.regular_grid.extent[1])
    ax2.set_ylim(geo_model.grid.regular_grid.extent[4], geo_model.grid.regular_grid.extent[5])

    geo_model.add_surfaces(['D', 'C', 'B'])

    # surface B
    geo_model.add_surface_points(X=584, Y=285, Z=500, surface='B')
    geo_model.add_surface_points(X=494, Y=696, Z=500, surface='B')
    geo_model.add_surface_points(X=197, Y=1898, Z=500, surface='B')
    geo_model.add_surface_points(X=473, Y=2180, Z=400, surface='B')
    geo_model.add_surface_points(X=435, Y=2453, Z=400, surface='B')
    # surface C
    geo_model.add_surface_points(X=946, Y=188, Z=600, surface='C')
    geo_model.add_surface_points(X=853, Y=661, Z=600, surface='C')
    geo_model.add_surface_points(X=570, Y=1845, Z=600, surface='C')
    geo_model.add_surface_points(X=832, Y=2132, Z=500, surface='C')
    geo_model.add_surface_points(X=767, Y=2495, Z=500, surface='C')
    # Surface D
    geo_model.add_surface_points(X=967, Y=1638, Z=800, surface='D')
    geo_model.add_surface_points(X=1095, Y=996, Z=800, surface='D')

    geo_model.add_orientations(X=832, Y=2132, Z=500, surface='C',
                               orientation=[98, 17.88, 1])

    p2d.plot_data(ax, direction='z')
    p2d.plot_data(ax2, direction='y')

    gp.compute_model(geo_model)
    p2d.plot_contacts(ax, direction='z', cell_number=-10)
    p2d.plot_lith(ax, direction='z', cell_number=-10)

    p2d.plot_contacts(ax2, direction='y', cell_number=5)
    p2d.plot_lith(ax2, direction='y', cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')

    # -----------
    # Adding a fault
    geo_model.rename_series(['Cycle1'])

    geo_model.add_series(['Fault1'])
    geo_model.set_is_fault(['Fault1'])
    #geo_model.reorder_series(['Fault1', 'Cycle1'])

    geo_model.modify_order_series(1, 'Fault1')
    geo_model.add_surfaces(['F1'])
    gp.map_series_to_surfaces(geo_model, {'Fault1': 'F1'})

    # Add input data of the fault
    geo_model.add_surface_points(X=1203, Y=138, Z=600, surface='F1')
    geo_model.add_surface_points(X=1250, Y=1653, Z=800, surface='F1')
    # geo_model.add_surface_points(X=1280, Y=2525, Z=500, surface='F1')
    # Add orientation
    geo_model.add_orientations(X=1280, Y=2525, Z=500, surface='F1', orientation=[272, 90, -1])



    # ----------------
    # Second cycle

    # Surface G
    geo_model.add_surface_points(X=1012, Y=1493, Z=900, surface='G')
    geo_model.add_surface_points(X=1002, Y=1224, Z=900, surface='G')
    geo_model.add_surface_points(X=1996, Y=47, Z=800, surface='G')
    geo_model.add_surface_points(X=300, Y=907, Z=700, surface='G')
    # Surface H
    geo_model.add_surface_points(X=3053, Y=727, Z=800, surface='G')
    # Orientation
    geo_model.add_orientations(X=1996, Y=47, Z=800, surface='G', orientation=[272, 5.54, 1])

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')
