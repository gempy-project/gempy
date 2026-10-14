Meshing options and fixed-weight field queries
=============================================

GemPy passes ``model.interpolation_options`` directly to GemPy Engine. With
engine 2026.2.0a0, the meshing types are available through ``gp.data``.

Configure mesh extraction
-------------------------

.. code-block:: python

    import gempy as gp
    from gempy.core.data.enumerators import ExampleModel

    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    evaluation = model.interpolation_options.evaluation_options
    evaluation.mesh_extraction = True
    evaluation.number_octree_levels = 3
    evaluation.number_octree_levels_surface = 3
    evaluation.octree_refinement_mode = gp.data.OctreeRefinementMode.BALANCED
    evaluation.deduplicate_octree_corners = True
    evaluation.triangulation_method = gp.data.TriangulationMethod.QUADS
    evaluation.mesh_extraction_overlap = "pretty"

    config = gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy, dtype="float64", compute_grads=False
    )
    gp.compute_model(model, config)

The evaluation options include:

* ``octree_refinement_mode``: ``FAST``, ``BALANCED``, or ``PRECISE`` support stencils.
* ``deduplicate_octree_corners``: evaluate shared corners once when eligible.
* ``triangulation_method``: ``LEGACY`` or ``QUADS``. Quads are split into triangles;
  output connectivity still has shape ``(N, 3)``.
* ``mesh_extraction_overlap``: ``None`` uses the process default. Explicit names
  are ``none``, ``pretty``, ``watertight``, ``joint``, and ``joint_contacts``;
  corresponding ``gp.data.DualContouringOverlap`` flags are also accepted.
* ``mesh_extraction_extent_capping``: ``gp.data.MeshExtentCapping.NONE`` or
  ``SCALAR_LESS_EQUAL`` (best-effort extent caps of the scalar sublevel region).
* ``mesh_extraction_masking_options``: ``gp.data.MeshExtractionMaskingOptions``
  with ``INTERSECT``, ``DISJOINT``, ``RAW``, and the testing-only ``NOTHING``.
* Refinement controls: ``number_octree_levels``, ``number_octree_levels_surface``,
  ``octree_min_level``, ``octree_curvature_threshold``, and ``octree_error_threshold``.

Refinement stencils and overlap mode names are not guarantees of closed or
watertight meshes. Inspect the extraction reports for the actual result.

Joint extraction
----------------

.. code-block:: python

    evaluation.mesh_extraction_overlap = "joint_contacts"
    evaluation.mesh_extraction_masking_options = gp.data.MeshExtractionMaskingOptions.INTERSECT
    evaluation.mesh_extraction_extent_capping = gp.data.MeshExtentCapping.NONE
    gp.compute_model(model, config)

``joint`` and ``joint_contacts`` establish shared erosion/onlap contact vertices.
They are exclusive overlap modes and bypass legacy triangulation. They require
float64 without autograd, ``INTERSECT`` masking, no extent capping, no null-space
stacks, and a supported face/edge-balanced dyadic leaf complex. The ``BALANCED``
refinement stencil does not itself guarantee this leaf balance.

``joint`` supports no faults or one verified independent infinite planar fault,
with separate fault banks. ``joint_contacts`` supports independent infinite
faults with pretty-style fault vertex merging. With faults, micro corrections,
external interpolation/segmentation functions, and external fault values are
unsupported. Unsupported geometry raises an engine error rather than silently
switching extraction modes.

Mesh results and persistence
----------------------------

Each computed ``StructuralElement.mesh`` exposes a mesh with world-coordinate
vertices and the engine's diagnostics and topology metadata:

.. code-block:: python

    element = model.structural_frame.structural_groups[0].elements[0]
    mesh = element.mesh
    print(mesh.support_report, mesh.capping_report, mesh.contact_report)
    # Joint modes also expose joint_vertex_ids, joint_vertex_keys, joint_seam_edges.
    # Banked joint extraction adds joint_bank_ids and joint_face_bank_ids.

``element.vertices`` and ``element.edges`` remain available. The world mesh is a
shallow view: connectivity, reports, and joint IDs are shared with the engine
mesh; engine-specific ``dc_data`` and any differentiable vertex tensors retain
engine coordinates. ``model.solutions.dc_meshes`` keeps the original engine
meshes. Recomputing with extraction disabled clears the element mesh outputs.

The ``.gempy`` format and the simplified ``JsonIO`` format preserve evaluation
options. Simplified JSON stores them in
``interpolation_options.evaluation_options``; nested values take precedence over
the legacy flat ``mesh_extraction`` and ``number_octree_levels`` keys. The nested
octree counts use the engine's serialized names ``_number_octree_levels`` and
``_number_octree_levels_surface``. Computed meshes and query snapshots are not
saved in ``.gempy`` files; recompute after loading.

Query fields without a new solve
-------------------------------

.. code-block:: python

    import numpy as np

    points = np.array([[500., 500., 300.], [500., 500., 700.]])
    evaluator = gp.prepare_field_evaluator(model)
    raw_stack_fields = evaluator(points)  # shape (number of stacks, number of points)
    stack_indices, intervals = evaluator.classify_units(points)
    # One-shot equivalent:
    raw_stack_fields = gp.evaluate_fields(model, points)

Queries accept finite world coordinates of shape ``(M, 3)``. They use the
production weights and compute-time inputs, options, and transforms retained by
``gp.compute_model``, without changing the model grid or solving again. Later
model edits do not affect the snapshot; recompute to query the edited model.
The original backend must remain active. Use float64 and disable autograd;
micro corrections, finite/dependent faults, and external fields are unsupported
by the engine's current fixed-weight query API.

Select rows with ``evaluator(points, stacks=[0, 2])``. For fault-side-fixed
queries, pass ``fault_sides`` of shape ``(F, M)`` containing -1 or +1, with rows
ordered by ``evaluator.fault_stacks``. ``evaluator.fault_values(points)`` returns
fault scalars minus their surface levels. Metadata includes ``affected_by``,
``stack_relations``, ``stack_isovalues``, and ``diagnostics``.

Classification returns compute-time stack indices and scalar intervals, not
GemPy element IDs. ``gp.classify_units(values, relations, isovalues)`` also
exposes the engine classifier for precomputed raw field arrays.
