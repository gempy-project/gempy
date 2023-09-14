import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel
import numpy as np
import subsurface as ss
import pandas as pd


def test_gempy_to_subsurface():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    if False:
        gpv.plot_3d(model)

    vertex: list[np.ndarray] = model.solutions.raw_arrays.vertices
    simplex_list: list[np.ndarray] = model.solutions.raw_arrays.edges

    idx_max = 0
    for simplex_array in simplex_list:
        simplex_array += idx_max
        idx_max = simplex_array.max() + 1

    id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(vertex)]

    concatenated_id_array = np.concatenate(id_array)
    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(vertex),
        cells=np.concatenate(simplex_list),
        vertex_attr=pd.DataFrame({'id': concatenated_id_array})
    )

    trisurf = ss.TriSurf(meshes)
    pyvista_mesh = ss.visualization.to_pyvista_mesh(trisurf)
    ss.visualization.pv_plot([pyvista_mesh], image_2d=False)


def test_gempy_to_subsurface_II():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    meshes: ss.UnstructuredData = model.solutions.raw_arrays.meshes_to_subsurface()

    trisurf = ss.TriSurf(meshes)
    pyvista_mesh = ss.visualization.to_pyvista_mesh(trisurf)
    ss.visualization.pv_plot([pyvista_mesh], image_2d=False)
