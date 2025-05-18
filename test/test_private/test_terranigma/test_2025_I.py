import os

import dotenv

import gempy as gp
from gempy.API.io_API import read_surface_points


dotenv.load_dotenv()

def test_2025_1():
    
    path_to_data = os.getenv("TEST_DATA")
    
    data = {
            "a" : read_surface_points(f"{path_to_data}/a.dat"),
            "b" : read_surface_points(f"{path_to_data}/b.dat"),
            "c": read_surface_points(f"{path_to_data}/c.dat"),
            "d"   : read_surface_points(f"{path_to_data}/d.dat"),
            "e": read_surface_points(f"{path_to_data}/e.dat"),
            "f" : read_surface_points(f"{path_to_data}/f.dat"),
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
        structural_relation=gp.core.data.StackRelationType.ERODE,
        fault_relations=gp.core.data.FaultsRelationSpecialCase.OFFSET_FORMATIONS,
    )
    structural_frame = gp.core.data.StructuralFrame(
        structural_groups=[group], color_gen=color_generator
    )

    xmin = 525816
    xmax = 543233
    ymin = 5652470
    ymax = 5657860
    zmin = -780
    zmax = -636
    geo_model = gp.create_geomodel(
        project_name="test",
        extent=[xmin, xmax, ymin, ymax, zmin, zmax],
        refinement=4,
        structural_frame=structural_frame,
    )
    gp.add_orientations(
        geo_model=geo_model,
        x=[525825],
        y=[5651315],
        z=[-686],
        pole_vector=[[0, 0, 1]],
        elements_names=["a"]
    )
    solution = gp.compute_model(
        geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=True,
        ),
    )
    
