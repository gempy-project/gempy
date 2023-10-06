# These two lines are necessary only if GemPy is not installed
# sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np


def test_gravity():
    color_generator = gp.data.ColorsGenerator()
    element1 = gp.data.StructuralElement(
        name='surface1',
        color=next(color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([3, 9]),
            y=np.array([0, 0]),
            z=np.array([3.05, 3.05]),
            names='surface1'
        ),
        orientations=gp.data.OrientationsTable.from_arrays(
            x=np.array([6]),
            y=np.array([0]),
            z=np.array([4]),
            G_x=np.array([0]),
            G_y=np.array([0]),
            G_z=np.array([1]),
            names='surface1'
        )
    )

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([3, 9]),
            y=np.array([0, 0]),
            z=np.array([1.02, 1.02]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    frame = gp.data.StructuralFrame(
        structural_groups=[gp.data.StructuralGroup(
            name='default',
            elements=[element1, element2],
            structural_relation=gp.data.StackRelationType.ERODE
        )
        ],
        color_gen=color_generator
    )

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name="2-layers",
        extent=[0, 12, -2, 2, 0, 4],
        resolution=[500, 1, 500],
        number_octree_levels=4,
        structural_frame=frame,
    )


    gp.compute_model(geo_model)
    
    import gempy_viewer as gpv
    gpv.plot_2d(geo_model, cell_number=5)

    # Add geophysics
    # geo_model._surfaces.add_surfaces_values([2.6, 2.4, 3.2], ['density'])
    # device_loc = np.array([[6, 0, 4]])
    # 
    # geo_model.set_centered_grid(device_loc, resolution=[10, 10, 100], radius=16000)


    # print(geo_model.solutions.fw_gravity)
    # np.testing.assert_almost_equal(geo_model.solutions.fw_gravity, np.array([-1624.1714]), decimal=4)