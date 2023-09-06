import gempy as gp
import gempy_viewer as gpv

import numpy as np
import os


def test_dual_contouring():
    data_path = os.path.abspath('../../examples')

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Onlap_relations',
        extent=[-200, 1000, -500, 500, -1000, 0],
        resolution=[50, 50, 50],
        number_octree_levels=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/tut-ch1-4/tut_ch1-4_orientations.csv",
            path_to_surface_points=data_path + "/data/input_data/tut-ch1-4/tut_ch1-4_points.csv",
        )
    )

    # gp.set_topography_from_random(grid=geo_model.grid, d_z=np.array([-600, -100]))

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
    # gp.remove_structural_group_by_name(model=geo_model, group_name="onlap_series")
    # gp.remove_structural_group_by_name(model=geo_model, group_name="left_series")
    
   
    # %%
    from gempy_engine.core.data.options import DualContouringMaskingOptions
    geo_model.interpolation_options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT
    s = gp.compute_model(geo_model)

    # %% 
    gpv.plot_3d(
        model=geo_model,
        show_surfaces=True,
        show_data=True,
        image=False,
        show_topography=True,
        kwargs_plot_structured_grid={'opacity': .2}
    )
