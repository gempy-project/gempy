import os

import gempy as gp
import gempy_viewer as gpv


def test_subduction():
    # Get the directory containing the test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to examples directory relative to test location
    data_path = os.path.join(current_dir, '../..', 'examples')

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Onlap_relations',
        extent=[-200, 1000, -500, 500, -1000, 0],
        resolution=[50, 50, 50],
        refinement=5,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/tut-ch1-4/tut_ch1-4_orientations.csv",
            path_to_surface_points=data_path + "/data/input_data/tut-ch1-4/tut_ch1-4_points.csv",
        )
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=0,
        structural_group_name="seafloor_series",
        elements=[geo_model.structural_frame.get_element_by_name("seafloor")],
        structural_relation=gp.data.StackRelationType.ERODE,
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=1,
        structural_group_name="right_series",
        elements=[
            geo_model.structural_frame.get_element_by_name("rock1"),
            geo_model.structural_frame.get_element_by_name("rock2"),
        ],
        structural_relation=gp.data.StackRelationType.ONLAP
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=2,
        structural_group_name="onlap_series",
        elements=[geo_model.structural_frame.get_element_by_name("onlap_surface")],
        structural_relation=gp.data.StackRelationType.ERODE
    )

    gp.add_structural_group(
        model=geo_model,
        group_index=3,
        structural_group_name="left_series",
        elements=[geo_model.structural_frame.get_element_by_name("rock3")],
        structural_relation=gp.data.StackRelationType.BASEMENT
    )

    gp.remove_structural_group_by_name(model=geo_model, group_name="default_formation")
   
   
    # %%
    from gempy_engine.core.data.options import MeshExtractionMaskingOptions
    from gempy_engine.core.data.options import EvaluationOptions
    evaluation_options: EvaluationOptions = geo_model.interpolation_options.evaluation_options
    evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    s = gp.compute_model(geo_model)

    gpv.plot_2d(geo_model)
    
    # %% 
    gpv.plot_3d(
        model=geo_model,
        show_surfaces=True,
        show_data=True,
        image=True,
        show_topography=True,
        kwargs_plot_structured_grid={'opacity': 0.1}
    )

    # %%
    # ! White are True, black are False
    if False:
        p = gpv.plot_2d(
            model=geo_model,
            cell_number=2,
            override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[0],
            show_data=True, kwargs_lithology={'cmap': 'gray', 'norm': None}
        )
        gpv.plot_2d(
            model=geo_model,
            cell_number=2,
            override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[1],
            show_data=True, kwargs_lithology={'cmap': 'gray', 'norm': None}
        )

        gpv.plot_2d(
            model=geo_model,
            cell_number=2,
            override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[2],
            show_data=True, kwargs_lithology={'cmap': 'gray', 'norm': None}
        )

        gpv.plot_2d(
            model=geo_model,
            cell_number=2,
            override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[3],
            show_data=True, kwargs_lithology={'cmap': 'gray', 'norm': None}
        )
