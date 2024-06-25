import pytest

import gempy as gp
import gempy_viewer as gpv
import subsurface.core.structs.unstructured_elements.triangular_surface
from gempy.core.data.enumerators import ExampleModel

import numpy as np

from ..conftest import REQUIREMENT_LEVEL, Requirements

pytestmark = pytest.mark.skipif(
    condition=REQUIREMENT_LEVEL.value < Requirements.DEV.value and False,
    reason="This test needs higher requirements."
)

ss = pytest.importorskip("subsurface", reason="Subsurface is not installed")
pd = pytest.importorskip("pandas", reason="Pandas is not installed")


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

    vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(vertex)]
    cell_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(simplex_list)]

    concatenated_id_array = np.concatenate(vertex_id_array)
    concatenated_cell_id_array = np.concatenate(cell_id_array)
    
    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(vertex),
        cells=np.concatenate(simplex_list),
        vertex_attr=pd.DataFrame({'id': concatenated_id_array}),
        cells_attr=pd.DataFrame({'id': concatenated_cell_id_array})
    )

    trisurf = subsurface.core.structs.unstructured_elements.triangular_surface.TriSurf(meshes)
    pyvista_mesh = ss.visualization.to_pyvista_mesh(trisurf)
    ss.visualization.pv_plot([pyvista_mesh], image_2d=True)


def test_gempy_to_subsurface_II():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution
    meshes: ss.UnstructuredData = model.solutions.raw_arrays.meshes_to_subsurface()

    trisurf = subsurface.core.structs.unstructured_elements.triangular_surface.TriSurf(meshes)
    pyvista_mesh = ss.visualization.to_pyvista_mesh(trisurf)
    ss.visualization.pv_plot([pyvista_mesh], image_2d=True)


def test_gempy_to_subsurface_III():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    meshes: ss.UnstructuredData = model.solutions.meshes_to_unstruct()

    trisurf = subsurface.core.structs.unstructured_elements.triangular_surface.TriSurf(meshes)
    pyvista_mesh = ss.visualization.to_pyvista_mesh(trisurf)
    ss.visualization.pv_plot([pyvista_mesh], image_2d=True)
