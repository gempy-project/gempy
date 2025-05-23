import os

import dotenv

import gempy as gp
from gempy.API.io_API import read_surface_points
import gempy_viewer as gpv

dotenv.load_dotenv()


def test_2025_1():
    range_ = 0.6
    orientation_loc = -286

    path_to_data = os.getenv("TEST_DATA")

    data = {
            "a": read_surface_points(f"{path_to_data}/a.dat"),
            "b": read_surface_points(f"{path_to_data}/b.dat"),
            "c": read_surface_points(f"{path_to_data}/c.dat"),
            "d": read_surface_points(f"{path_to_data}/d.dat"),
            "e": read_surface_points(f"{path_to_data}/e.dat"),
            "f": read_surface_points(f"{path_to_data}/f.dat"),
    }

    color_generator = gp.data.ColorsGenerator()
    elements = []
    for event, pts in data.items():
        orientations = gp.data.OrientationsTable.initialize_empty()
        element = gp.data.StructuralElement(
            name=event,
            color=next(color_generator),
            surface_points=pts,
            orientations=orientations,
        )
        elements.append(element)

    group = gp.data.StructuralGroup(
        name="Series1",
        elements=elements,
        structural_relation=gp.data.StackRelationType.ERODE,
        fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_FORMATIONS,
    )
    structural_frame = gp.data.StructuralFrame(
        structural_groups=[group], color_gen=color_generator
    )

    xmin = 525816
    xmax = 543233
    ymin = 5652470
    ymax = 5657860
    zmin = -780
    zmax = -636

    # * Add 20% to extent
    xmin -= 0.2 * (xmax - xmin)
    xmax += 0.2 * (xmax - xmin)
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)
    zmin -= 0.2 * (zmax - zmin)
    zmax += 0.2 * (zmax - zmin)

    geo_model = gp.create_geomodel(
        project_name="test",
        extent=[xmin, xmax, ymin, ymax, zmin, zmax],
        refinement=5,
        structural_frame=structural_frame,
    )

    if False:
        gpv.plot_3d(
            model=geo_model,
            ve=10,
            image=True,
            kwargs_pyvista_bounds={
                    'show_xlabels': False,
                    'show_ylabels': False,
            }
        )

    geo_model.interpolation_options.evaluation_options.number_octree_levels_surface = 4
    geo_model.interpolation_options.kernel_options.range = range_

    gp.add_orientations(
        geo_model=geo_model,
        x=[525825],
        y=[5651315],
        z=[orientation_loc],  # * Moving the orientation further
        pole_vector=[[0, 0, 1]],
        elements_names=["a"]
    )
    solution = gp.compute_model(
        geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy
        ),
    )

    gpv.plot_3d(
        model=geo_model,
        ve=10,
        show_lith=False,
        image=True,
        kwargs_pyvista_bounds={
                'show_xlabels': False,
                'show_ylabels': False,
                'show_zlabels': False,
        }
    )
