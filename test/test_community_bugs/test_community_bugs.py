import pytest

import gempy as gp
import pandas as pd
import matplotlib.pyplot as plt


def test_issue_566():
    from pyvista import set_plot_theme
    set_plot_theme('document')

    geo_model = gp.create_model('Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 791, 0, 200, -582, 0],
                             resolution=[100, 10, 100])

    geo_model.set_default_surfaces()
    geo_model.add_surface_points(X=223, Y=0.01, Z=-94, surface='surface1')


@pytest.mark.skip(reason="Expensive test. It should be ran manually.")
def test_issue_569(data_path):
    surface_points_df = df = pd.read_csv(data_path + "/coordinates_mwe.csv")
    orientations_df = pd.read_csv(data_path + "/orientations_mwe.csv")

    geo_model = gp.create_model("Deltatest")
    gp.init_data(geo_model,
                 [df.X.min() - 50, df.X.max() + 50, df.Y.min() - 50, df.Y.max() + 50,
                  df.Z.min() - 50, df.Z.max() + 50, ], [50, 50, 50],
                 surface_points_df=surface_points_df,
                 orientations_df=orientations_df,
                 default_values=True)

    fault_list = []
    series = {"Strat_Series": surface_points_df.loc[
        ~surface_points_df["formation"].str.contains(
            "fault"), "formation"].unique().tolist()}

    for fault in surface_points_df.loc[
        surface_points_df["formation"].str.contains("fault"), "formation"].unique():
        series[fault] = fault
        fault_list.append(fault)

    gp.map_stack_to_surfaces(geo_model,
                             series,
                             remove_unused_series=True)

    geo_model.set_is_fault(fault_list)

    geo_model.reorder_features(['fault_a', 'fault_b', 'Strat_Series'])
    geo_model.add_surfaces("basement")

    plot = gp.plot_2d(geo_model, show_lith=False, show_boundaries=True,
                      direction=['z'])
    plt.show(block=False)

    gp.set_interpolator(geo_model,
                        compile_theano=True,
                        theano_optimizer='fast_compile',
                        )
    gp.get_data(geo_model, 'kriging')

    sol = gp.compute_model(geo_model, sort_surfaces=True)

    gp.plot_2d(geo_model, show_scalar=True, series_n=0)
    gp.plot_2d(geo_model, series_n=0)

    gp.plot_3d(geo_model, image=True)


@pytest.mark.skip(reason="Private data missing from the repo")
def test_issue_564(data_path):
    geo_model = gp.create_model('SBPM')

    gp.init_data(geo_model, [550, 730, -200, 1000, 20, 55], [50, 50, 50],
                 path_i=data_path + "/564_Points.csv",
                 path_o=data_path + "/564_Orientations_.csv",
                 default_values=True)

    gp.map_stack_to_surfaces(geo_model,
                             {"Q": 'Quartaer',
                              "vK": 'verwKeuper',
                              "Sa": 'Sandstein',
                              "Sc": 'Schluffstein',
                              "b": 'basement'},
                             remove_unused_series=True)

    gp.set_interpolator(geo_model,
                        compile_theano=True,
                        theano_optimizer='fast_compile')

    sol = gp.compute_model(geo_model)
    gp.plot_2d(geo_model)

    gpv = gp.plot_3d(geo_model, image=True, plotter_type='basic', ve=5,
                     show_lith=False)

    geo_model.set_bottom_relation(["Q", "vK", "Sa", "Sc", "b"],
                                  ["Onlap", "Onlap", "Onlap", "Onlap",
                                   "Onlap"])

    sol = gp.compute_model(geo_model)
    gp.plot_2d(geo_model)

    gpv = gp.plot_3d(geo_model, image=True, plotter_type='basic', ve=5,
                     show_lith=True)
