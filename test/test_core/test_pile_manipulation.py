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


def test_pile_geomodel(interpolator):
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

    gp.map_stack_to_surfaces(geo_model,
                             series_distribution,
                             remove_unused_series=True)

    geo_model.reorder_features(["unconformity", "fault3", "fault4",
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
    geo_model._surface_points.df["group"] = surf_groups
    orient_groups = pd.read_csv(input_path + "/filtered_orientations.csv").group
    geo_model._orientations.df["group"] = orient_groups

    geo_model._surface_points.df.reset_index(inplace=True, drop=True)
    geo_model._orientations.df.reset_index(inplace=True, drop=True)

    geo_model.set_theano_function(interpolator)
    gp.compute_model(geo_model)

    gp.plot.plot_2d(geo_model, cell_number=25,
                    direction='y', show_data=True)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_lith_block')

    return geo_model


def test_pile_geomodel_2(interpolator):
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

    gp.map_stack_to_surfaces(geo_model,
                             series_distribution,
                             remove_unused_series=True)

    geo_model.reorder_features(["unconformity", "fault3", "fault4",
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
    geo_model._surface_points.df["group"] = surf_groups
    orient_groups = pd.read_csv(input_path + "/filtered_orientations.csv").group
    geo_model._orientations.df["group"] = orient_groups

    geo_model._surface_points.df.reset_index(inplace=True, drop=True)
    geo_model._orientations.df.reset_index(inplace=True, drop=True)

    geo_model.set_theano_function(interpolator)
    gp.compute_model(geo_model)

    gp.plot.plot_2d(geo_model, cell_number=25,
                         direction='y', show_data=True)

    from gempy.plot.plot_api import plot_2d

    p = plot_2d(geo_model, cell_number=[25])

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_lith_block')

    return geo_model


def test_reorder_series():

    geo_model = gp.create_model('Geological_Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 4000, 0, 2775, 200, 1200], resolution=[100, 10, 100])
    # Adding a fault
    geo_model.rename_features(['Cycle1'])

    geo_model.add_features(['Fault1'])
    geo_model.set_is_fault(['Fault1'])
    geo_model.reorder_features(['Fault1', 'Cycle1'])
    assert (geo_model._stack.df['BottomRelation'] == ['Fault', 'Erosion']).all()
    assert (geo_model._stack.df.index == ['Fault1', 'Cycle1']).all()
    print(geo_model._stack.df)


def test_complete_model(tmpdir, interpolator):
    # ### Initializing the model:
    compute = True

    geo_model = gp.create_model('Geological_Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 4000, 0, 2775, 200, 1200], resolution=[100, 10, 100])

    if compute is True:
        geo_model.set_theano_function(interpolator)

    from gempy.plot import visualization_2d as vv

    # In this case perpendicular to the z axes
    p2d = vv.Plot2D(geo_model)
    p2d.create_figure((15, 8))
    ax = p2d.add_section(direction='z', ax_pos=121)

    ax2 = p2d.add_section(direction='y', ax_pos=122)
    ax2.set_xlim(geo_model._grid.regular_grid.extent[0], geo_model._grid.regular_grid.extent[1])
    ax2.set_ylim(geo_model._grid.regular_grid.extent[4], geo_model._grid.regular_grid.extent[5])

    geo_model.add_surfaces(['D', 'C', 'B', 'A'])

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

    gp.compute_model(geo_model) if compute else None
    p2d.plot_contacts(ax, direction='z', cell_number=-10)
    p2d.plot_lith(ax, direction='z', cell_number=-10)

    p2d.plot_contacts(ax2, direction='y', cell_number=5)
    p2d.plot_lith(ax2, direction='y', cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')

    # -----------
    # Adding a fault
    geo_model.rename_features(['Cycle1'])

    geo_model.add_features(['Fault1'])
    geo_model.set_is_fault(['Fault1'])

    geo_model.modify_order_features(1, 'Fault1')
    geo_model.add_surfaces(['F1'])
    gp.map_stack_to_surfaces(geo_model, {'Fault1': 'F1'})

    # Add input data of the fault
    geo_model.add_surface_points(X=1203, Y=138, Z=600, surface='F1')
    geo_model.add_surface_points(X=1250, Y=1653, Z=800, surface='F1')
    # Add orientation
    geo_model.add_orientations(X=1280, Y=2525, Z=500, surface='F1', orientation=[272, 90, -1])

    gp.compute_model(geo_model)

    p2d.remove(ax)
    p2d.plot_data(ax, direction='z', cell_number=-10)
    p2d.plot_contacts(ax, direction='z', cell_number=-10)
    p2d.plot_lith(ax, direction='z', cell_number=-10)

    p2d.remove(ax2)
    # Plot
    p2d.plot_data(ax2, cell_number=5)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_contacts(ax2, cell_number=5)

    # surface B
    geo_model.add_surface_points(X=1447, Y=2554, Z=500, surface='B')
    geo_model.add_surface_points(X=1511, Y=2200, Z=500, surface='B')
    geo_model.add_surface_points(X=1549, Y=629, Z=600, surface='B')
    geo_model.add_surface_points(X=1630, Y=287, Z=600, surface='B')
    # surface C
    geo_model.add_surface_points(X=1891, Y=2063, Z=600, surface='C')
    geo_model.add_surface_points(X=1605, Y=1846, Z=700, surface='C')
    geo_model.add_surface_points(X=1306, Y=1641, Z=800, surface='C')
    geo_model.add_surface_points(X=1476, Y=979, Z=800, surface='C')
    geo_model.add_surface_points(X=1839, Y=962, Z=700, surface='C')
    geo_model.add_surface_points(X=2185, Y=893, Z=600, surface='C')
    geo_model.add_surface_points(X=2245, Y=547, Z=600, surface='C')
    # Surface D
    geo_model.add_surface_points(X=2809, Y=2567, Z=600, surface='D')
    geo_model.add_surface_points(X=2843, Y=2448, Z=600, surface='D')
    geo_model.add_surface_points(X=2873, Y=876, Z=700, surface='D')

    # Compute
    gp.compute_model(geo_model)

    # Plot
    p2d.remove(ax)
    p2d.plot_data(ax, direction='z', cell_number=-10)
    p2d.plot_contacts(ax, direction='z', cell_number=-10)
    p2d.plot_lith(ax, direction='z', cell_number=-10)

    p2d.remove(ax2)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_data(ax2, cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')

    # ----------------
    # Second cycle
    geo_model.add_features(['Cycle2'])
    geo_model.add_surfaces(['G', 'H'])
    gp.map_stack_to_surfaces(geo_model, {'Cycle2': ['G', 'H']})
    geo_model.reorder_features(['Cycle2', 'Fault1', 'Cycle1'])

    # Surface G
    geo_model.add_surface_points(X=1012, Y=1493, Z=900, surface='G')
    geo_model.add_surface_points(X=1002, Y=1224, Z=900, surface='G')
    geo_model.add_surface_points(X=1996, Y=47, Z=800, surface='G')
    geo_model.add_surface_points(X=300, Y=907, Z=700, surface='G')
    # Surface H
    geo_model.add_surface_points(X=3053, Y=727, Z=800, surface='G')
    # Orientation
    geo_model.add_orientations(X=1996, Y=47, Z=800, surface='G', orientation=[272, 5.54, 1])

    # Compute
    gp.compute_model(geo_model)

    # Plot
    p2d.remove(ax)
    p2d.plot_data(ax, direction='z', cell_number=-10)
    p2d.plot_contacts(ax, direction='z', cell_number=-10)
    p2d.plot_lith(ax, direction='z', cell_number=-10)

    p2d.remove(ax2)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_data(ax2, cell_number=5)
    p2d.plot_contacts(ax2, cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')


    # ----------------
    # Second Fault
    geo_model.add_features('Fault2')
    geo_model.set_is_fault('Fault2')
    geo_model.add_surfaces('F2')

    geo_model.reorder_features(['Cycle2', 'Fault1', 'Fault2', 'Cycle1'])
    gp.map_stack_to_surfaces(geo_model, {'Fault2': 'F2'})

    geo_model.add_surface_points(X=3232, Y=178, Z=1000, surface='F2')
    geo_model.add_surface_points(X=3132, Y=951, Z=700, surface='F2')
    # geo_model.add_surface_points(X=2962, Y=2184, Z=700, surface='F2')

    geo_model.add_orientations(X=3132, Y=951, Z=700, surface='F2', orientation=[95, 90, 1])

    # Compute
    gp.compute_model(geo_model)

    # Plot
    p2d.remove(ax)
    p2d.plot_data(ax, direction='z', cell_number=5, legend='force')
    p2d.plot_lith(ax, direction='z', cell_number=5)
    p2d.plot_contacts(ax, direction='z', cell_number=5)

    p2d.remove(ax2)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_data(ax2, cell_number=5)
    p2d.plot_contacts(ax2, cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')

    geo_model.add_surface_points(X=3135, Y=1300, Z=700, surface='D')
    geo_model.add_surface_points(X=3190, Y=969, Z=700, surface='D')

    geo_model.add_surface_points(X=3031, Y=2725, Z=800, surface='G')
    geo_model.add_surface_points(X=3018, Y=1990, Z=800, surface='G')
    geo_model.add_surface_points(X=3194, Y=965, Z=700, surface='G')

    geo_model.add_surface_points(X=3218, Y=1818, Z=890, surface='H')
    geo_model.add_surface_points(X=3934, Y=1207, Z=810, surface='H')

    # Compute
    gp.compute_model(geo_model)

    # Plot
    p2d.remove(ax)
    p2d.plot_data(ax, direction='z', cell_number=5, legend='force')
    p2d.plot_lith(ax, direction='z', cell_number=5)
    p2d.plot_contacts(ax, direction='z',cell_number=5)

    p2d.remove(ax2)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_data(ax2, cell_number=5)
    p2d.plot_contacts(ax2, cell_number=5)

    plt.savefig(os.path.dirname(__file__) + '/../figs/test_pile_complete')



