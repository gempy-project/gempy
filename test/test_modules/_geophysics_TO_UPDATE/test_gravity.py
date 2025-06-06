# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np

from test.verify_helper import verify_model_serialization


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
        refinement=1,
        structural_frame=frame,
    )

    gp.compute_model(geo_model, validate_serialization=True)

    import gempy_viewer as gpv
    gpv.plot_2d(geo_model, cell_number=0)

    gp.set_centered_grid(
        grid=geo_model.grid,
        centers=np.array([[6, 0, 4]], dtype="float"),
        resolution=np.array([10, 10, 100], dtype="float"),
        radius=np.array([16000, 16000, 16000], dtype="float")  # ? This radius makes 0 sense but it is the original one in gempy v2
    )

    gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)
    geo_model.geophysics_input = gp.data.GeophysicsInput(
        tz=gravity_gradient,
        densities=np.array([2.6, 2.4, 3.2]),
    )

    verify_model_serialization(
        model=geo_model,
        verify_moment="after",
        file_name=f"verify/{geo_model.meta.name}"
    )

    gp.compute_model(geo_model)

    print(geo_model.solutions.gravity)
    np.testing.assert_almost_equal(geo_model.solutions.gravity, np.array([-1624.1714]), decimal=4)
