import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")


def Plot_2D_scaler_field(grid, scaler_field):
    G = grid[np.where(grid[:, 1] == [grid[-1][1]])[0]]
    S = scaler_field.numpy()[np.where(grid[:, 1] == [grid[0][1]])[0]]
    XX = G[:, 0].reshape([50, 50])
    ZZ = G[:, 2].reshape([50, 50])
    S = S.reshape([50, 50])
    plt.contour(XX, ZZ, S)
    return


if __name__ == '__main__':

    from gempy.core.tensor.tensorflow_graph import TFGraph
    import tensorflow as tf
    import pandas as pd
    import gempy as gp
    from gempy.assets.geophysics import GravityPreprocessing

    geo_data = gp.create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                              path_o=os.pardir + "/../notebooks/data/input_data/jan_models/model2_orientations.csv",
                              path_i=os.pardir + "/../notebooks/data/input_data/jan_models/model2_surface_points.csv")
    gp.map_series_to_surfaces(geo_data, {"Strat_Series": (
        'rock2', 'rock1'), "Basement_Series": ('basement')})

    geo_data.add_surface_values([2.61, 3.1, 2.92])
    interpolator = geo_data.interpolator

    dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = interpolator.get_python_input_block()[
        0:-3]
    dtype = interpolator.additional_data.options.df.loc['values', 'dtype']

    len_rest_form = interpolator.additional_data.structure_data.df.loc[
        'values', 'len surfaces surface_points'] - 1
    Range = interpolator.additional_data.kriging_data.df.loc['values', 'range']
    C_o = interpolator.additional_data.kriging_data.df.loc['values', '$C_o$']
    rescale_factor = interpolator.additional_data.rescaling_data.df.loc[
        'values', 'rescaling factor']
    nugget_effect_grad = np.cast[dtype](
        np.tile(interpolator.orientations.df['smooth'], 3))
    nugget_effect_scalar = np.cast[interpolator.dtype](
        interpolator.surface_points.df['smooth'])

    TFG = TFGraph(dips_position, dip_angles, azimuth,
                  polarity, surface_points_coord, fault_drift,
                  grid, values_properties, len_rest_form, Range,
                  C_o, nugget_effect_scalar, nugget_effect_grad,
                  rescale_factor)

    Z_x = TFG.scalar_field()
    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(
        Z_x)
    formations_block = TFG.export_formation_block(
        Z_x, scalar_field_at_surface_points, values_properties)

    # regular grid
    Plot_2D_scaler_field(grid, Z_x)

    # ---------
    # Gravity test
    # ---------
    grav_res = 20
    X = np.linspace(0, 1000, grav_res)
    Y = np.linspace(0, 1000, grav_res)
    Z = 300
    xyz = np.meshgrid(X, Y, Z)
    xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
    xy_ravel

    geo_data.set_centered_grid(xy_ravel, resolution=[10, 10, 15], radius=5000)
    interpolator = geo_data.interpolator
    dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = interpolator.get_python_input_block()[
        0:-3]

    g = GravityPreprocessing(geo_data.grid.centered_grid)
    tz = g.set_tz_kernel()

    len_rest_form = interpolator.additional_data.structure_data.df.loc[
        'values', 'len surfaces surface_points'] - 1
    Range = interpolator.additional_data.kriging_data.df.loc['values', 'range']
    C_o = interpolator.additional_data.kriging_data.df.loc['values', '$C_o$']
    rescale_factor = interpolator.additional_data.rescaling_data.df.loc[
        'values', 'rescaling factor']
    nugget_effect_grad = np.cast[dtype](
        np.tile(interpolator.orientations.df['smooth'], 3))
    nugget_effect_scalar = np.cast[interpolator.dtype](
        interpolator.surface_points.df['smooth'])

    TFG = TFGraph(dips_position, dip_angles, azimuth,
                  polarity, surface_points_coord, fault_drift,
                  grid, values_properties, len_rest_form, Range,
                  C_o, nugget_effect_scalar, nugget_effect_grad,
                  rescale_factor)

    Z_x = TFG.scalar_field()
    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(
        Z_x)
    formations_block = TFG.export_formation_block(
        Z_x, scalar_field_at_surface_points, values_properties)

    lg_0 = interpolator.grid.get_grid_args('centered')[0]
    lg_1 = interpolator.grid.get_grid_args('centered')[1]
    densities = formations_block[1][lg_0:lg_1]

    grav = TFG.compute_forward_gravity(tz, lg_0, lg_1, densities)

    grav = tf.reshape(grav, [20, 20])
    # Plot gravity response
    xx, yy = np.meshgrid(X, Y)
    gp.plot.plot_data(geo_data, direction='z')
    ax = plt.gca()
    ax.scatter(xy_ravel[:, 0], xy_ravel[:, 1], s=10, zorder=1)
    ax.contourf(xx, yy, grav, zorder=-1)
