import sys
sys.path.append("/Users/zhouji/Documents/github/gempy")


if __name__ == '__main__':
    from gempy.core.tensor.tensorflow_graph import TFGraph
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import gempy as gp

    geo_data = gp.create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                              path_o="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_orientations.csv",
                              path_i="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_surface_points.csv")
    gp.map_series_to_surfaces(geo_data, {"Strat_Series": (
        'rock2', 'rock1'), "Basement_Series": ('basement')})
    # gp.plot.plot_data(geo_data, direction='y')

    interpolator = geo_data.interpolator

    interp_data = gp.set_interpolator(geo_data, compile_theano=True,
                                      theano_optimizer='fast_run',
                                      verbose=['covariance_matrix'])
    geo_data.modify_kriging_parameters('drift equations', [9, 9])

    sol = gp.compute_model(geo_data)

    dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = interpolator.get_python_input_block()[
        0:-3]
    dtype = interpolator.additional_data.options.df.loc['values', 'dtype']

    len_rest_form = interpolator.additional_data.structure_data.df.loc[
        'values', 'len surfaces surface_points']-1
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

    grid_val = TFG.x_to_interpolate(grid)
    weights = TFG.solve_kriging()

    tiled_weights = TFG.extend_dual_kriging(weights, grid_val.shape[0])

    TFG.contribution_gradient_interface(grid_val, tiled_weights)
