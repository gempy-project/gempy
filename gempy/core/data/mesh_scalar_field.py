from dataclasses import dataclass, field
from typing import Annotated, Optional

import numpy as np
from pydantic import Field


@dataclass
class MeshScalarField:
    """Where a structural group's scalar field comes from instead of its surface points: a triangle mesh.

    The field is the mesh's signed distance, zero on the mesh and positive on the side its face normals
    point to; ``invert`` flips that side. Only ``uri``, ``invert`` and ``sha256`` are saved: whoever can
    read ``uri`` (e.g. ``le://<space>/static_mesh/...`` on the GemPy server) loads the mesh at runtime with
    ``load``, and a backend registered with ``gempy.modules.mesh_fields`` turns it into the field.
    """
    uri: str  #: Address of the mesh. Its scheme says who can resolve it.
    invert: bool = False  #: Positive on the side opposite to the face normals.
    sha256: Optional[str] = None  #: Optional checksum of the mesh the model was built with.

    vertices: Annotated[Optional[np.ndarray], Field(exclude=True)] = field(default=None, repr=False)  #: World coordinates (n, 3)
    faces: Annotated[Optional[np.ndarray], Field(exclude=True)] = field(default=None, repr=False)  #: Vertex indices (m, 3)

    def __post_init__(self):
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError("A mesh scalar field needs the address of its mesh.")

    def load(self, vertices: np.ndarray, faces: np.ndarray) -> "MeshScalarField":
        """Attach the mesh, in world coordinates, for this session; it is not saved with the model."""
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("A mesh needs (n, 3) vertices and (m, 3) triangle faces.")
        if len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Mesh faces must index its vertices.")
        self.vertices, self.faces = vertices, faces
        return self

    @property
    def is_loaded(self) -> bool:
        return self.vertices is not None and self.faces is not None
