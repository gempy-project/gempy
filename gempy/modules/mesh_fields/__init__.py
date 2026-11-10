"""Mesh-backed structural groups: their signed distance field, built at compute time by a registered backend.

A group with a ``mesh_source`` takes its scalar field from that mesh. GemPy itself only stores the
mesh's address and runs whatever backend is registered here (the GemPy server registers one). Without
a backend, or before the mesh is loaded, an eroding group computes as if it were not there, with a
warning; an onlapping one cannot, because its region does not depend on its own field, so it raises.
"""
from typing import Callable, Optional
import warnings

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.stack_relation_type import StackRelationType

#: ``(vertices, faces, invert) -> CustomInterpolationFunctions``: vertices are in engine coordinates.
MeshFieldBackend = Callable[[np.ndarray, np.ndarray, bool], CustomInterpolationFunctions]

_backend: Optional[MeshFieldBackend] = None


class MeshFieldUnavailableWarning(UserWarning):
    """A mesh group was computed as absent: no backend is registered or its mesh is not loaded."""


class MeshFieldUnavailableError(NotImplementedError):
    """A mesh group cannot be computed and cannot be left out either."""


def register_mesh_field_backend(backend: Optional[MeshFieldBackend]) -> None:
    """Set (or with None, clear) the function that turns a mesh into a group's scalar field."""
    global _backend
    _backend = backend


def mesh_field_backend() -> Optional[MeshFieldBackend]:
    return _backend


def resolve_mesh_fields(geo_model) -> None:
    """Give every mesh group its ``custom_interpolation`` for the model's current transform and backend.

    Called before each compute (``interpolation_input_from_structural_frame``), after the tensor
    backend is chosen. A field is rebuilt only when the mesh, the transform or the backend changed.
    """
    groups = geo_model.structural_frame.structural_groups
    for group in groups:
        source = group.mesh_source
        if source is None:
            continue
        if _backend is None or not source.is_loaded:
            # The last group takes everything left whatever its relation, so even an onlapping one can be absent.
            if group.structural_relation is StackRelationType.ONLAP and group is not groups[-1]:
                raise MeshFieldUnavailableError(
                    f"Group '{group.name}' onlaps with a mesh field that cannot be built "
                    f"({_unavailable_reason(source)}); an onlapping group cannot be left out.")
            warnings.warn(f"Group '{group.name}' is computed as absent: {_unavailable_reason(source)}.",
                          MeshFieldUnavailableWarning, stacklevel=3)
            group.custom_interpolation = _inert_field()
            source._resolved_key = None
            continue
        key = _resolution_key(geo_model, source)
        if getattr(source, "_resolved_key", None) == key and group.custom_interpolation is not None:
            continue
        group.custom_interpolation = _backend(_to_engine_coordinates(geo_model, source.vertices),
                                              source.faces, source.invert)
        source._resolved_key = key


def _unavailable_reason(source) -> str:
    if _backend is None:
        return f"no mesh field backend is registered for '{source.uri}'"
    return f"the mesh '{source.uri}' is not loaded"


def _to_engine_coordinates(geo_model, vertices: np.ndarray) -> np.ndarray:
    """As surface points: grid transform, then input transform, so the field meets the kriged surfaces.

    An identity grid transform is skipped: going through its pivot and back would only add rounding.
    """
    grid_transform = geo_model.grid.transform
    if not np.allclose(grid_transform.get_transform_matrix(), np.eye(4), rtol=0, atol=0):
        vertices = grid_transform.apply_with_cached_pivot(vertices)
    return geo_model.input_transform.apply(vertices)


def _resolution_key(geo_model, source) -> tuple:
    return (id(source.vertices), id(source.faces), source.invert,
            geo_model.input_transform.get_transform_matrix().tobytes(),
            geo_model.grid.transform.get_transform_matrix().tobytes(),
            BackendTensor.engine_backend, str(BackendTensor.dtype))


def _inert_field() -> CustomInterpolationFunctions:
    """Below its isovalue everywhere, with no gradient: an eroding group claims nothing and meshes nothing."""

    def below(xyz):
        return BackendTensor.t.full((len(xyz),), -1.0, dtype=BackendTensor.dtype_obj)

    def flat(xyz):
        return BackendTensor.t.zeros(len(xyz), dtype=BackendTensor.dtype_obj)

    return CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=below,
        gx_function=flat,
        gy_function=flat,
        gz_function=flat,
    )
