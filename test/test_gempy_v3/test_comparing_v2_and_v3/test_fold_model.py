# Importing GemPy
import numpy as np

import gempy as gp
import gempy_viewer as gpv
from core.data import InterpolationOptions
from gempy.optional_dependencies import require_gempy_legacy
from gempy_3.gp3_to_gp2_input import gempy3_to_gempy2


def test_fold_model():

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"
    geo_data = gp.create_data(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 5, 50],
        path_o=path_to_data + "model2_orientations.csv",
        path_i=path_to_data + "model2_surface_points.csv"
    )

    # %% 
    geo_data.structural_frame.surface_points.df.head()  # This view needs to have pandas installed

    # %%
    # Setting and ordering the units and series:
    # 

    # %% 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # %%
    gpv.plot_2d(geo_data, direction=['y'])

    # %%
    # Calculating the model:
    # 

    # %% 
    geo_data.orientations

    # %% 

    if COMPUTE_LEGACY := False:
        gpl = require_gempy_legacy()
        legacy_model: gpl.Project = gempy3_to_gempy2(geo_data)
        gpl.set_interpolator(legacy_model, verbose=['cov_gradients', 'cov_surface_points', 'cov_interface_gradients',
                                                    'U_I', 'U_G'])
        gpl.compute_model(legacy_model)
        gpl.plot_2d(legacy_model, direction=['y'])

        gpl.plot_2d(legacy_model, direction=['y'], show_data=True, show_scalar=True)

    geo_data.interpolation_options.tensor_dtype = 'float64'
    sol = gp.compute_model(geo_data)

    # %%
    # Displaying the result in y and x direction:
    # 
    # %%
    gpv.plot_2d(geo_data, direction='y', show_data=True)
    gpv.plot_2d(geo_data, direction='y', show_scalar=True)

    # %%
    # sphinx_gallery_thumbnail_number = 2
    gpv.plot_2d(geo_data, direction='x', show_data=True)


def test_compare_input_values():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"
    geo_data = gp.create_data(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 5, 50],
        path_o=path_to_data + "model2_orientations.csv",
        path_i=path_to_data + "model2_surface_points.csv"
    )
    # * Ideally we can swap the model above and the rest of the test should still work
    
    gpl = require_gempy_legacy()
    legacy_model: gpl.Project = gempy3_to_gempy2(geo_data)
    gpl.set_interpolator(legacy_model, verbose=['cov_gradients', 'cov_surface_points', 'cov_interface_gradients',
                                                'U_I', 'U_G'])
    
    # compare surface points
    np.testing.assert_allclose(
        geo_data.interpolation_input.surface_points.sp_coords,
        legacy_model.surface_points.df[[ "X_c", "Y_c", "Z_c" ]],
        rtol=1e-3
    )
    
    # compare orientations
    np.testing.assert_allclose(
        geo_data.interpolation_input.orientations.dip_positions,
        legacy_model.orientations.df[[ "X_c", "Y_c", "Z_c" ]],
        rtol=1e-3
    )
    
    # compare grid
    new_grid= geo_data.interpolation_input.grid.values
    legacy_grid = legacy_model.grid._w.values_c
    
    np.testing.assert_allclose(
        new_grid,
        legacy_grid,
        rtol=0.02
    )
    
    # compare kriging parameters
    legacy_range = [0.8660254]
    legacy_c_o = [35.71428571]
    
    extent = geo_data.transform.apply(geo_data.grid.regular_grid.extent.reshape(-1, 3)).reshape(-1)
    default_range = np.sqrt(
        (extent[0] - extent[1]) ** 2 +
        (extent[2] - extent[3]) ** 2 +
        (extent[4] - extent[5]) ** 2)
    
    # * NOTE: Range is the same but c_o is different due to when we are rescaling
    
    interpolation_options: InterpolationOptions = InterpolationOptions(
        range=default_range,
        c_o=(default_range ** 2) / 14 / 3,
    )

    return 
    
def test_compare_results():
    # ? Maybe add here weights. What is more interesting would be cov matrix but that has to be done manually
    pass

