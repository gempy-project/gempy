import os
import dotenv

import time
import numpy as np
import gempy as gp
from gempy.API.io_API import read_surface_points

dotenv.load_dotenv()

# Load data
def test_2025_3():

    path_to_data = os.getenv("TEST_DATA")
    # path_to_data = r"C:/Users/Benjamink/OneDrive - Mira Geoscience Limited/Documents/projects/implicit modelling/Nutrien/demo_terranigma/from_miguel"

    data = {
        "a": read_surface_points(f"{path_to_data}/a.dat"),
        "b": read_surface_points(f"{path_to_data}/b.dat"),
        "c": read_surface_points(f"{path_to_data}/c.dat"),
        "d": read_surface_points(f"{path_to_data}/d.dat"),
        "e": read_surface_points(f"{path_to_data}/e.dat"),
        "f": read_surface_points(f"{path_to_data}/f.dat"),
    }

    # Build structural frame

    color_generator = gp.core.data.ColorsGenerator()
    elements = []
    for event, pts in data.items():
        orientations = gp.data.OrientationsTable.initialize_empty()
        element = gp.core.data.StructuralElement(
            name=event,
            color=next(color_generator),
            surface_points=pts,
            orientations=orientations,
        )
        elements.append(element)

    group = gp.core.data.StructuralGroup(
        name="Series1",
        elements=elements,
        structural_relation=gp.core.data.StackRelationType.ERODE,
        fault_relations=gp.core.data.FaultsRelationSpecialCase.OFFSET_FORMATIONS,
    )
    structural_frame = gp.core.data.StructuralFrame(
        structural_groups=[group], color_gen=color_generator
    )

    # create cell centers with similar size to the BlockMesh used for Nutrien modelling

    xmin = 525816
    xmax = 543233
    ymin = 5652470
    ymax = 5657860
    zmin = -780 - 40
    zmax = -636 + 40

    x = np.arange(xmin, xmax, 50)
    y = np.arange(ymin, ymax, 50)
    z = np.arange(zmin, zmax, 1)
    X, Y, Z = np.meshgrid(x, y, z)
    centers = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

    # Create geomodel

    geo_model = gp.create_geomodel(
        project_name="test",
        extent=[xmin, xmax, ymin, ymax, zmin, zmax],
        refinement=6,
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

    # Ignore surface creation in timing lithology block creation
    geo_model.interpolation_options.evaluation_options.mesh_extraction=False

    # Time interpolation into octree cells

    tic = time.perf_counter()
    solution = gp.compute_model(
        geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=True,
        ),
    )
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"Octree interpolation runtime: {int(elapsed / 60)} minutes {int(elapsed % 60)} seconds.")

    octrees_outputs = solution.octrees_output
    n_cells = 0
    for octree_output in octrees_outputs:
        n_cells += octree_output.outputs_centers[0].exported_fields.scalar_field.size().numel()
        if len(octree_output.outputs_corners)>0: 
            n_cells += octree_output.outputs_corners[0].exported_fields.scalar_field.size().numel()
    print(f"Number of cells evaluated: {n_cells}")

    # Time extra interpolation on regular grid centers.  I was expecting/hoping that this second step
    # would just be an evaluation of the continuous scalar field solution from first step.

    tic = time.perf_counter()
    model = gp.compute_model_at(
        geo_model,
        at=centers,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=True,
        ),
    )
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"Evaluate model on regular grid centers: {int(elapsed / 60)} minutes {int(elapsed % 60)} seconds")
    print(f"Number of cells evaluated: {centers.shape[0]}")
