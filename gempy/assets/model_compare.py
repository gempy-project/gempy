import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/Users/zhouji/Documents/github/gempy")

def Plot_2D_scaler_field(grid,scaler_field):
    G = grid[np.where(grid[:,1] == [grid[-1][1]])[0]]
    S = scaler_field.numpy()[np.where(grid[:,1] == [grid[0][1]])[0]]
    XX = G[:,0].reshape([50,50])
    ZZ = G[:,2].reshape([50,50])
    S = S.reshape([50,50])
    plt.contour(XX,ZZ,S)
    return



if __name__ == '__main__':
    from gempy.core.tensor.tensorflow_graph import TFGraph
    import tensorflow as tf
    import pandas as pd
    import gempy as gp
    from gempy.assets.geophysics import GravityPreprocessing

    geo_data = gp.create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                              path_o="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_orientations.csv",
                              path_i="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_surface_points.csv")
    gp.map_series_to_surfaces(geo_data, {"Strat_Series": (
        'rock2', 'rock1'), "Basement_Series": ('basement')})
    # gp.plot.plot_data(geo_data, direction='y')
    geo_data.add_surface_values([2.61,3.1,2.92])
    interpolator = geo_data.interpolator

    interp_data = gp.set_interpolator(geo_data, compile_theano=True,
                                      theano_optimizer='fast_run',
                                      verbose=['densities'])
    # geo_data.modify_kriging_parameters('drift equations', [3, 3])

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

    sigma_0_grad = TFG.contribution_gradient_interface(grid_val, tiled_weights)
    sigma_0_interf = TFG.contribution_interface(grid_val, tiled_weights)
    f_0 = TFG.contribution_universal_drift(grid_val,weights)
    Z_x = TFG.scalar_field()
    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(Z_x)
    formations_block = TFG.export_formation_block(Z_x,scalar_field_at_surface_points,values_properties)
    print(formations_block)
    
    Plot_2D_scaler_field(grid,Z_x)
    
    
    ## Gravity test
    ## ---------
    grav_res = 20
    X = np.linspace(0, 1000, grav_res)
    Y = np.linspace(0, 1000, grav_res)
    Z= 300
    xyz= np.meshgrid(X, Y, Z)
    xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
    xy_ravel
    gp.plot.plot_data(geo_data, direction='z')
    plt.scatter(xy_ravel[:,0], xy_ravel[:, 1], s=1)
    geo_data.set_centered_grid(xy_ravel,  resolution = [10, 10, 15], radius=5000)
    interpolator = geo_data.interpolator
    dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = interpolator.get_python_input_block()[
        0:-3]
    
    g = GravityPreprocessing(geo_data.grid.centered_grid)
    tz = g.set_tz_kernel()
    gp.set_interpolator(geo_data, output=['gravity'], pos_density=1,  gradient=False,
                    theano_optimizer='fast_run',verbose=['densities'])  
    # sol = gp.compute_model(geo_data, output=['geology'])
    # grav = sol.fw_gravity
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

    sigma_0_grad = TFG.contribution_gradient_interface(grid_val, tiled_weights)
    sigma_0_interf = TFG.contribution_interface(grid_val, tiled_weights)
    f_0 = TFG.contribution_universal_drift(grid_val,weights)
    Z_x = TFG.scalar_field()
    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(Z_x)
    formations_block = TFG.export_formation_block(Z_x,scalar_field_at_surface_points,values_properties)
    
    lg_0 = interpolator.grid.get_grid_args('centered')[0]
    lg_1 = interpolator.grid.get_grid_args('centered')[1]
    densities = formations_block[1][lg_0:lg_1]
    
    grav = TFG.compute_forward_gravity(tz,lg_0,lg_1,densities)
    
    # n_devices = tf.math.floordiv((densities.shape[0]),tz.shape[0])
    # tz_rep = tf.tile(tz, [n_devices])
    # grav = densities * tz_rep
    