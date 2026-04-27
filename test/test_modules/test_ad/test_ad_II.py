import pyvista

import gempy as gp
import gempy_viewer as gpv


def test_generate_fold_model():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # Compute the geological model
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64',
            compute_grads=True
        )
    )

    foo = geo_data.solutions.dc_meshes[0].vertices_tensor
    triangle_idx = 0
    foo[triangle_idx, 2].backward(retain_graph=True, create_graph=True)
    pass

    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords
    
    p3d = gpv.plot_3d(geo_data)
    # p3d.p: pyvista.Plotter()
    p3d.p.add
