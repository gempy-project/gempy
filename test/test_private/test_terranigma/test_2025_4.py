import dotenv
import matplotlib.pyplot as plt

import gempy as gp
import numpy as np
from gempy_engine.core.data.stack_relation_type import StackRelationType

dotenv.load_dotenv()


def stack_data(data, n_levels, min_level):
    """
    Dump cell centers/sizes and GemPy model for all octree levels.

    :param data: gp.GeoModel containing GemPy Solution with interpolation data
        on cell centers of an octree mesh
    :param n_levels: Requested number of octree levels.
    :param min_level: Requested min_level.
    """

    centers = []
    model = []
    h = []
    for i in range(min_level, n_levels - min_level + 1):
        output = data.solutions.octrees_output[i].outputs_centers[-1]
        xyz = data.input_transform.apply_inverse(output.grid.octree_grid.values)
        h = output.grid.octree_grid.dx / data.input_transform.scale[0]
        centers.append(np.column_stack([xyz, np.repeat(h, len(xyz))]))
        model.append(output.ids_block)

    centers = np.vstack(centers)
    model = np.hstack(model)

    return centers, model


def cells_inside_mask(centers):
    """
    Find cells that contain other cells.

    :param centers: Cell centers (centers[:, :3]) and sizes (centers[:, -1]) of an
        octree mesh
    """

    # Compute faces array where each row is a bounding box for a cell in the mesh.
    h = centers[:, -1]
    faces = np.column_stack([
        centers[:, 0] - (h / 2),
        centers[:, 0] + (h / 2),
        centers[:, 1] - (h / 2),
        centers[:, 1] + (h / 2),
        centers[:, 2] - (h / 2),
        centers[:, 2] + (h / 2),
    ])

    # Loop over bounding boxes and mask if any other cell centers
    mask = np.ones(len(centers), dtype=bool)
    for i, bbox in enumerate(faces):
        contains_cells = (
                (centers[:, 0] > bbox[0])
                & (centers[:, 0] < bbox[1])
                & (centers[:, 1] > bbox[2])
                & (centers[:, 1] < bbox[3])
                & (centers[:, 2] > bbox[4])
                & (centers[:, 2] < bbox[5])
        )

        mask[i] = np.sum(contains_cells) == 1

    return mask


def test_2025_4():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"
    # Create a GeoModel instance

    n_levels = 6
    min_level = 2
    extent = 1000

    data = gp.create_geomodel(
        project_name='fault',
        extent=[0, extent, 0, extent, 0, extent],
        refinement=n_levels,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model5_orientations.csv",
            path_to_surface_points=path_to_data + "model5_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=data,
        mapping_object={
            "Fault_Series": 'fault',
            "Strat_Series": ('rock2', 'rock1')
        }
    )

    # Define fault groups
    data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
    data.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])

    # Compute the geological model
    data.interpolation_options.evaluation_options.octree_min_level = min_level
    data.interpolation_options.evaluation_options.octree_error_threshold = 0.2
    gp.compute_model(data)

    cell_centers, model = stack_data(data, n_levels, min_level)
    mask = cells_inside_mask(cell_centers)

    filtered_centers = cell_centers[mask]
    filtered_model = model[mask]

    print(
        "original n cells: ", len(cell_centers),
        "filtered n cells: ", len(filtered_centers)
    )


if __name__ == "__main__":
    test_2025_4()