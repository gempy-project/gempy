import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/Users/zhouji/Documents/github/gempy")

def Plot_2D_scaler_field(grid,scaler_field):
    G = grid[np.where(grid[:,1] == [grid[0][1]])[0]]
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

    geo_data = gp.create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                              path_o="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_orientations.csv",
                              path_i="/Users/zhouji/Documents/github/gempy/notebooks/data/input_data/jan_models/model1_surface_points.csv")
    gp.map_series_to_surfaces(geo_data, {"Strat_Series": (
        'rock2', 'rock1'), "Basement_Series": ('basement')})
    # gp.plot.plot_data(geo_data, direction='y')

    interpolator = geo_data.interpolator

    interp_data = gp.set_interpolator(geo_data, compile_theano=True,
                                      theano_optimizer='fast_run',
                                      verbose=['export_formation_block'])
    geo_data.modify_kriging_parameters('drift equations', [3, 3])

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
    Z_x = sigma_0_grad+sigma_0_interf+f_0
    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(Z_x)
    formations_block = TFG.export_formation_block(Z_x,scalar_field_at_surface_points,values_properties)
    print(formations_block)

    
    Plot_2D_scaler_field(grid,Z_x)
    
    
    
# scalar_field_iter = tf.pad(tf.expand_dims(scalar_field_at_surface_points,0),[[0,0],[1,1]])[0]
# l = 50.
# n_surface_op_float_sigmoid_mask = tf.repeat(values_properties,2)
# n_surface_op_float_sigmoid = tf.expand_dims(tf.concat([tf.concat([[0],n_surface_op_float_sigmoid_mask[1:-1]],-1),[0]],-1),0)
# drift = tf.expand_dims(tf.concat([n_surface_op_float_sigmoid_mask[0:-1],[0]],-1),0)


# def compare(a, b, slice_init, Z_x, l, n_surface, drift):


#     slice_init = slice_init
#     n_surface_0 = n_surface[:, slice_init:slice_init + 1]
#     n_surface_1 = n_surface[:, slice_init + 1:slice_init + 2]
#     drift = drift[:, slice_init:slice_init + 1]



#     # The 5 rules the slope of the function
#     sigm = (-tf.reshape(n_surface_0,(-1, 1)) / (1 + tf.exp(-l * (Z_x - a)))) - \
#             (tf.reshape(n_surface_1,(-1, 1)) / (1 + tf.exp(l * (Z_x - b)))) + tf.reshape(drift,(-1, 1))

#     return sigm