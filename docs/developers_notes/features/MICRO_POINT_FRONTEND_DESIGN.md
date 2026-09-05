# Micro Point Front-End and Serialization Design

Status: GemPy data model and serialization implemented; engine integration proposed

This document defines how high-density micro contacts should enter GemPy, cross
the GemPy-to-engine boundary, and be serialized. It also defines the local RBF
correction that consumes those contacts.

Where this document conflicts with `../../../../gempy_engine/MICRO_ANISOTROPIC_FIELD_DEFORMATION.md` or
`../../../../gempy_engine/PLAN.md`, this document is authoritative. Those documents describe the
prototype and contain assumptions that are not valid for spatially varying
anisotropy.

Implemented in the user-facing `gempy` package:

- `MicroPointsTable` and its transform validation.
- Per-`StructuralElement` ownership and `StructuralFrame` aggregation.
- `.gempy` container format version 2 with `micro_points.bin`.
- Backward-compatible loading of legacy version 1 archives.
- Focused table, ownership, corruption, determinism, and round-trip tests.

Implementation locations:

| Concern | GemPy file |
|---|---|
| Authored table and validation | `gempy/core/data/micro_points.py` |
| Element ownership | `gempy/core/data/structural_element.py` |
| Aggregation and redistribution | `gempy/core/data/structural_frame.py` |
| Binary section decoder | `gempy/core/data/encoders/binary_encoder.py` |
| Binary loading context | `gempy/core/data/encoders/converters.py` |
| Container writer and reader | `gempy/modules/serialization/save_load.py` |
| Table tests | `test/test_core/test_micro_points.py` |
| Archive tests | `test/test_modules/test_micro_points_serialization.py` |

Not yet implemented:

- Conversion into engine coordinates and `InterpolationInput`.
- Per-stack partitioning and target/residual construction.
- The consistent local RBF solve, gradients, and octree refinement.

## 1. Purpose

Micro points are dense observations used to make an existing macro geological
model locally comply with contacts without adding every contact to the global
cokriging system.

The intended pipeline is:

```text
GemPy macro observations
    -> macro cokriging solve
    -> macro scalar field and gradients
    -> per-stack micro residual solve
    -> macro field + local micro correction
    -> activation, octree refinement, and mesh extraction
```

The macro model remains the structural hypothesis. Micro points are a local
compliance layer and do not replace surface points or orientations.

## 2. Goals

- Represent each micro point as a position, local geological frame, and local
  anisotropic support.
- Use the same representation in a Python front end and a 3D editor.
- Associate every micro point with exactly one `StructuralElement`.
- Keep authored observations separate from options and solved runtime state.
- Preserve micro points through `gempy.save_model()` and `gempy.load_model()`.
- Apply a correction only to the stack and interface to which a point belongs.
- Use one mathematically consistent operator for fitting and evaluation.
- Preserve coordinate, dtype, device, and gradient consistency.

## 3. Non-Goals

- Micro points do not participate in the macro cokriging matrix.
- Solved residuals and RBF weights are not durable model input.
- The initial engine integration will not support one micro point constraining
  multiple interfaces.
- The initial engine integration will not define corrections across fault
  blocks.
- Micro table dtype version 1 does not accept perspective transforms or
  arbitrary projective matrices.

## 4. Terminology

- **Macro point:** A standard `SurfacePointsTable` or `OrientationsTable`
  observation used by the main interpolation system.
- **Micro point:** A dense contact observation used by the additive local
  correction.
- **Support transform:** A `4x4` affine transform mapping normalized local
  support coordinates to the model's world/input coordinate system.
- **Support scale:** The correlation lengths encoded in the linear part of a
  support transform. It is not merely a display-gizmo scale.
- **Local RBF:** A radial basis function centered at one micro point and
  evaluated using that point's support transform.

## 5. Front-End Data Model

### 5.1 Ownership

`MicroPointsTable` is owned by `StructuralElement`, in the same vein as
`SurfacePointsTable` and `OrientationsTable`:

```python
class StructuralElement:
    surface_points: SurfacePointsTable
    orientations: OrientationsTable
    micro_points: MicroPointsTable
```

This establishes the interface association without a separate mutable
model-level relation. Moving an element between groups moves its micro points;
removing an element removes them. Basement elements must have an empty micro
table.

`StructuralFrame` provides derived aggregate views:

```text
micro_points_copy
number_of_micro_points_per_element
number_of_micro_points_per_group
```

The aggregate table is needed for binary serialization and engine conversion,
but it is not the authoritative mutable owner.

The implementation intentionally uses the existing integer
`StructuralElement.id` behavior, matching surface points and orientations. On
save, every micro row must match its owning element's current ID. When micro
points are present, element IDs must also be unique. Loading rejects rows whose
IDs do not resolve to a non-basement structural element.

The fallback ID remains name-derived when `_id == -1`, so renaming an element
without updating its observation tables is an existing model-wide limitation,
not something solved by this serialization change. A future stable UUID or
materialized-ID migration would need to cover surface points and orientations
at the same time. Dense surface and stack indices remain runtime values derived
from current structural order.

### 5.2 Canonical Transform

For point `i`, define:

```text
x_world_h = H_world_from_support[i] @ x_support_h
```

with:

```text
H_world_from_support = [ B_i  p_i ]
                       [ 0     1  ]
```

- `p_i` is the micro-point position.
- The columns of `B_i` are local support axes expressed in world coordinates.
- The lengths of those columns are the kernel correlation lengths.
- The third local axis is the interface-normal direction.

For the initial axisymmetric model:

```text
B_i = R_i @ diag(lateral_range_i, lateral_range_i, normal_range_i)
```

where `normal_range_i < lateral_range_i` in the common case. Rotation around
the normal has no effect when both lateral ranges are equal, but retaining a
complete frame is convenient for 3D front ends and allows future triaxial
support.

The position must not also be serialized as independent `X`, `Y`, and `Z`
fields. The translation column is authoritative. A convenience `xyz` property
may return `support_transforms[:, :3, 3]`.

### 5.3 Scale Semantics

The three support scales are physical correlation lengths in model coordinate
units. Applying the inverse transform produces dimensionless local coordinates.

This removes the ambiguous double scaling in the prototype, where
`anisotropy_matrices` contain inverse ranges and the result is divided by a
second `kernel_range`. The durable representation has one source of geometric
range: the support transform.

An optional global support multiplier may exist as an algorithm option, but it
must multiply all support lengths explicitly and must be applied identically
during fitting and evaluation.

### 5.4 Implemented Table

The public object is exported as `gp.data.MicroPointsTable` and provides:

```python
@dataclass
class MicroPointsTable:
    data: np.ndarray
    name_id_map: dict[str, int] | None = None

    @classmethod
    def from_transforms(
        cls,
        support_transforms: np.ndarray,
        names: Sequence[str] | str,
        nugget: np.ndarray | None = None,
        name_id_map: dict[str, int] | None = None,
    ) -> "MicroPointsTable": ...

    @classmethod
    def initialize_empty(cls) -> "MicroPointsTable": ...

    @property
    def support_transforms(self) -> np.ndarray: ...

    @property
    def xyz(self) -> np.ndarray: ...
```

The implemented table dtype version 1 is:

```python
np.dtype([
    ("support_transform", "<f8", (4, 4)),
    ("element_id", "<i8"),
    ("nugget", "<f8"),
], align=False)
```

This has a packed row size of 144 bytes:

| Field | Bytes | Meaning |
|---|---:|---|
| `support_transform` | 128 | Local-support-to-world affine transform |
| `element_id` | 8 | Durable association used during flatten/load |
| `nugget` | 8 | Diagonal regularization for this observation |

Little-endian encoding is explicit. The wider element ID avoids repeating the
`int32` limitation of the existing point tables.

The per-row ID remains useful even though elements own their tables: the
flattened binary table needs to be redistributed after loading, just as the
existing surface-point and orientation tables are redistributed by ID.

### 5.5 Authored Versus Derived State

Persisted input:

- Support transforms.
- Element association.
- Per-observation nugget.

Algorithm configuration:

- Enabled state.
- Kernel family.
- Correction strength.
- Macro-preservation policy.
- Solver and conditioning policy.
- Contact-driven refinement policy.

Derived runtime state:

- Engine-coordinate support transforms.
- Macro values and gradients at micro points.
- Target interface scalar values.
- Residuals.
- Local RBF weights.
- Solver diagnostics.

Derived state must not be stored in `MicroAnisotropicOptions` or serialized as
model input. It becomes stale when macro observations, transforms, structural
grouping, faults, or interpolation settings change.

## 6. Validation

`MicroPointsTable` construction must validate:

- Shape is exactly `(N, 4, 4)`.
- All matrix and nugget values are finite.
- Every last row is approximately `[0, 0, 0, 1]`.
- The `3x3` linear block is invertible.
- Authored support axes are orthogonal within tolerance for version 1.
- All three authored support scales are strictly positive.
- The authored linear block has positive determinant.
- The condition number remains below a documented limit.
- The number of names, nuggets, and transforms is identical.
- Every element association resolves to a non-basement element.
- Nugget values are nonnegative.

The implementation uses:

```text
affine and orthogonality tolerance = 1e-6
maximum condition number           = 1e12
default nugget                     = 0.0
```

Direct construction from structured data additionally requires a
one-dimensional array with the exact fixed dtype. `from_transforms()` converts
input matrices to `float64`, requires shape `(N, 4, 4)`, and resolves element
IDs from names using the same mapping mechanism as the existing observation
tables.

Micro table dtype version 1 deliberately accepts TRS support transforms, not
shear. The combined world-to-engine transform may make the engine-space linear
block non-orthogonal; orthogonality is therefore checked on authored transforms
before composition, not after it.

Normal validation errors must use regular exceptions or Pydantic validators,
not `assert`, because assertions disappear under `python -O`.

## 7. Coordinate Frames

### 7.1 Persisted Frame

Support transforms are persisted in the same world/input coordinate frame as
surface points and orientation positions. Micro points must not influence the
calculation of the model's global input transform; adding local compliance data
must not rescale the macro model.

### 7.2 World-to-Engine Conversion

Let `E` be the complete homogeneous world-to-engine transform, including the
grid transform around its cached pivot followed by `GeoModel.input_transform`:

```text
x_engine_h = E @ x_world_h
```

Then each support transforms exactly by composition:

```text
H_engine_from_support = E @ H_world_from_support
```

Runtime evaluation extracts:

```text
p_i = H_engine_from_support[:3, 3]
B_i = H_engine_from_support[:3, :3]
A_i = inverse(B_i)
```

For an engine-space query point `x`, normalized local coordinates and distance
are:

```text
u_i(x) = A_i @ (x - p_i)
r_i(x) = ||u_i(x)||
```

This guarantees coordinate invariance:

```text
||A_world @ (x_world - p_world)||
    ==
||A_engine @ (x_engine - p_engine)||
```

The conversion must use homogeneous matrix multiplication. It must not use
component-wise `Transform.__add__` or decompose the support through
`Transform.from_matrix()`, because those paths do not preserve general affine
composition under rotation and nonuniform scale.

## 8. Engine Boundary

### 8.1 Data Placement

Numerical micro observations belong in `InterpolationInput`, alongside surface
points and orientations. They do not belong in interpolation options.

Conceptually:

```python
@dataclass
class MicroPoints:
    support_transforms: ArrayLike  # (N, 4, 4), engine coordinates
    surface_indices: ArrayLike     # (N,), dense global interface indices
    nuggets: ArrayLike             # (N,)
```

`InputDataDescriptor` should carry partition metadata, preferably the number of
micro points per surface. That supports validation and deterministic slicing by
stack without pretending micro points are macro cokriging constraints.

`MicroAnisotropicOptions` should contain policy only. The prototype fields
`points`, `residuals`, `anisotropy_matrices`, and `weights` should not form the
public architecture.

### 8.2 Stack Isolation

During stack subset construction, include only micro points associated with
surfaces in the active stack. The resulting correction must not be added to:

- Unrelated stratigraphic stacks.
- Fault scalar fields unless fault micro correction is explicitly supported.
- External or null-space interpolation groups unless explicitly supported.

This is required because the current prototype stores one global correction in
evaluation options and can apply it to every evaluated scalar field.

## 9. Local RBF Correction

### 9.1 Macro Targets and Residuals

For micro point `p_i` associated with interface `s(i)`, evaluate the macro field
and use the same canonical interface scalar value consumed by activation:

```text
y_i = c_macro[s(i)] - V_macro(p_i)
```

The canonical macro interface values must remain immutable during the micro
pass. They must not be recomputed from surface-point samples after applying the
micro correction.

### 9.2 Consistent Basis

For a kernel `k`, basis `i` evaluated at query `x` is:

```text
phi_i(x) = k(r_i(x))
```

where `r_i` is calculated from support transform `i`. Construct the fitting
matrix by evaluating that exact basis at every constraint point:

```text
Phi[j, i] = phi_i(p_j)
```

Solve:

```text
(Phi + diag(nugget)) @ w = y
```

and evaluate:

```text
V_micro(x) = sum_i w_i * phi_i(x)
V_final(x) = V_macro(x) + strength * V_micro(x)
```

`Phi` is generally nonsymmetric because each column uses its source point's
support transform. The system is therefore a local RBF system, not a covariance
matrix. Dense direct solve, GMRES, BiCGSTAB, or least squares are valid solver
families; Conjugate Gradient is not valid for the general nonsymmetric system.

A nonzero nugget deliberately relaxes exact snapping. With zero nugget and a
nonsingular system, evaluating the fitted basis at the constraint points must
reproduce the residual vector.

### 9.3 Prototype Difference

The snapping prototype solves with a symmetric pair distance based on:

```text
(A_i.T @ A_i + A_j.T @ A_j) / 2
```

but evaluates with only the source matrix `A_i`. Those operators differ when
anisotropy varies between points. The identity-matrix round-trip test does not
expose the mismatch.

The production design uses the one-sided source basis in both fitting and
evaluation. It makes no positive-semidefinite covariance claim.

### 9.4 Gradients

When scalar gradients are requested, the returned gradient must include the
micro correction. For `u_i = A_i(x - p_i)` and `r_i = ||u_i||`:

```text
grad(r_i) = A_i.T @ u_i / r_i
grad(phi_i) = k'(r_i) * grad(r_i)
```

The implementation must handle `r_i = 0` using the analytic kernel limit.

```text
grad(V_final) = grad(V_macro)
              + strength * sum_i w_i * grad(phi_i)
```

Returning a corrected scalar with macro-only gradients is invalid for dual
contouring and any gradient-based refinement or diagnostics.

## 10. Model Serialization

### 10.1 Implemented Container Version

`gempy.save_model()` now writes `.gempy` container format version 2 in
`gempy/modules/serialization/save_load.py`. It is a ZIP archive with this exact
member order:

```text
model.gempy
|-- header.json
|-- input.bin
|-- micro_points.bin
|-- grid.bin
`-- liquid_earth_meta.json
```

`header.json` contains a container manifest injected by `model_to_bytes()`:

```json
{
  "serialization": {
    "format": "gempy",
    "version": 2,
    "writer_version": "<gempy-version>",
    "byte_order": "little"
  }
}
```

A missing manifest means legacy version 1. The manifest is container metadata,
not a `GeoModel` field, and is removed before `GeoModel.model_validate()`.

The reader currently accepts exactly format `"gempy"`, version `2`, and byte
order `"little"`. It rejects unsupported values instead of attempting a
best-effort decode.

`model_to_bytes()` injects the following micro section into
`structural_frame.binary_meta_data`:

```json
{
  "micro_points": {
    "dtype_version": 1,
    "row_count": 42,
    "byte_length": 6048
  }
}
```

The micro metadata is container-specific and is not emitted by a raw
`GeoModel.model_dump_json()` call. The reader selects the fixed local dtype from
`dtype_version`; it does not execute or trust an arbitrary dtype supplied by
the archive.

### 10.2 Relationship to Existing Binary Data

The existing macro input layout remains unchanged:

```text
input.bin = SurfacePointsTable bytes + OrientationsTable bytes
```

`StructuralFrame.binary_meta_data` continues to record `sp_binary_length` and
`ori_binary_length`. Micro rows are not appended to `input.bin`; they are stored
in the dedicated member because:

- Existing readers ignore trailing `input.bin` bytes.
- The micro member can be length- and schema-validated independently.
- Surface-point and orientation offsets remain backward compatible.
- Future micro dtype versions do not need to change the macro stream.

### 10.3 Save Flow

1. `StructuralFrame.validate_micro_point_ownership()` checks IDs before any
   bytes are written.
2. `StructuralElement.micro_points` is excluded from model JSON as a
   binary-only observation table; loading without a binary section uses an
   independent empty-table default.
3. `StructuralFrame.micro_points_copy` concatenates rows in structural order,
   including a zero-row basement contribution.
4. `model_to_bytes()` writes the packed rows to `micro_points.bin`.
5. The writer uses `ZIP_STORED`, timestamp `1980-01-01 00:00:00`,
   `ZipInfo.create_system = 0`, and `external_attr = 0x20` for every member.
6. Serialization validation compares the original and loaded micro tables
   directly, not through process-local `hash(bytes)` values.

These settings, fixed JSON formatting, and fixed member order make repeated
saves byte-for-byte deterministic without depending on a zlib implementation.

### 10.4 Load Flow

1. Parse `header.json`, remove the optional container manifest, and validate it
   when present.
2. Interpret a missing manifest as version 1 and do not consume an unversioned
   micro member.
3. Require `micro_points.bin` for version 2, including when it is empty.
4. Inject micro bytes through `loading_model_from_binary()` alongside
   `input.bin` and `grid.bin`.
5. Construct each `StructuralElement` with its own empty micro table default.
6. Require metadata to be an object with `dtype_version == 1`, nonnegative
   integer `row_count`, and nonnegative integer `byte_length`.
7. Require `byte_length == len(micro_points.bin)`, divisibility by 144, and
   `row_count * 144 == byte_length` before decoding.
8. Decode using the fixed little-endian dtype and copy the `np.frombuffer()`
   result so loaded tables remain writable.
9. Validate every transform and nugget through `MicroPointsTable`.
10. Reject unknown element IDs and ambiguous duplicate element IDs.
11. Redistribute rows to element-owned tables by `element_id`.

Loading an existing version 1 archive produces independent empty micro tables,
so old model files remain loadable. A version 2 archive missing
`micro_points.bin` is invalid and raises a clear error.

### 10.5 What Is Not Serialized

Do not serialize:

- Engine-coordinate transforms.
- Inverse `3x3` anisotropy operators.
- Macro scalar samples or gradients.
- Target scalar values.
- Residuals.
- RBF weights.
- Solver factorizations or condition estimates.

These values depend on the current macro model and are recomputed. If solved
state is cached in the future, it must be a disposable cache keyed by a strong
fingerprint over all inputs and options, not authoritative model data.

## 11. Test Status

### 11.1 Implemented Table Tests

The table tests are implemented in:

```text
gempy/test/test_core/test_micro_points.py
```

Coverage includes:

- Exact dtype names, byte order, offsets, and 144-byte row size.
- Construction, empty defaults, writable `xyz`, and element filtering.
- Independent empty tables for separate structural elements.
- Invalid input shapes and multidimensional structured arrays.
- Invalid affine rows, nonfinite values, zero scales, shear, reflections, and
  excessive condition numbers.
- Invalid nugget values and mismatched input lengths.

### 11.2 Implemented Serialization Tests

Serialization tests are implemented in:

```text
gempy/test/test_modules/test_micro_points_serialization.py
```

Coverage includes:

- Public `save_model()` and `load_model()` round trips.
- Exact matrix, ID, and nugget preservation for multiple elements.
- Structural-order aggregation and per-element/per-group counts.
- Redistribution of shuffled binary rows by element ID.
- Loading the existing version 1 `Greenstone.gempy` fixture with empty tables.
- Exact archive member order, `ZIP_STORED`, and platform metadata.
- Byte-for-byte deterministic repeated saves.
- Rejection of missing members, truncated rows, row-count mismatch,
  unsupported dtype and container versions, unknown IDs, duplicate element
  IDs, and ownership mismatches.
- Verification that support matrices remain outside `header.json` and loaded
  arrays remain writable.

### 11.3 Remaining Persistence Tests

- Moving and removing elements while preserving ownership semantics.
- Explicit basement-association rejection.
- Unsupported manifest format and byte-order values.
- Saving a computed model and proving no derived micro runtime state appears in
  the archive once engine integration exists.

### 11.4 Required Coordinate Conversion Tests

- Identity conversion.
- Translation, rotation, and isotropic model scaling.
- Nonuniform model scaling.
- Grid rotation around a nonzero pivot.
- Combined grid and input transforms.
- Micro center conversion matches the normal point-conversion pipeline.
- Normalized support distance is invariant between world and engine frames.
- Source matrices are not mutated during conversion.

### 11.5 Required Local RBF Tests

- One-point correction.
- Solve/evaluate round trip with different support transforms at every point.
- All supported kernels use the same type during solve and evaluation.
- Zero nugget reproduces residuals within solver tolerance.
- Nonzero nugget has documented smoothing behavior.
- `strength=0` returns the exact macro field.
- Analytic micro gradients match finite differences.
- Scalar output is identical whether gradients are requested or not.
- Duplicate and nearly duplicate points fail or regularize deterministically.
- NumPy and PyTorch preserve dtype, device, and numerical parity.

### 11.6 Required Integration Tests

- Micro points only modify their associated stack.
- Points on two interfaces receive the correct target scalar values.
- Fault and unsupported stack types reject micro correction clearly.
- Macro-preservation policy has measurable, documented behavior.
- Corrected gradients reach dual contouring.
- Contact-driven octree refinement resolves isolated contacts.
- Extracted interfaces approach contacts within the configured tolerance.
- Plotting is opt-in and disabled in automated tests.

## 12. Implementation Sequence

Current implementation progress:

- [x] Add and validate `MicroPointsTable` in the user-facing `gempy` package.
- [x] Attach an empty table to every `StructuralElement` and aggregate it
  through `StructuralFrame`.
- [x] Introduce serialization version 2 and `micro_points.bin` with
  compatibility tests for existing files.
- [ ] Add GemPy APIs for adding, modifying, and deleting micro points.
- [ ] Add the numerical micro data object to `InterpolationInput` and partition
  metadata to `InputDataDescriptor`.
- [ ] Compose support transforms into engine coordinates in the GemPy engine
  factory.
- [ ] Slice micro data per stack and derive residuals from immutable macro
  interface values.
- [ ] Replace the prototype solve with the consistent local RBF system.
- [ ] Add micro gradients, backend parity, and conditioning diagnostics.
- [ ] Add contact-driven octree refinement and mesh-level compliance tests.

Each stage should leave models with no micro points behaviorally identical to
current models.

## 13. Open Decisions

- Whether macro preservation uses all macro surface points, one reference point
  per interface, or nearby weighted anchors.
- Which nonsymmetric solver is used after the dense reference implementation.
- Whether per-point nugget is a direct diagonal regularizer or derived from a
  separately named uncertainty measurement.
- Whether to add Euler/TRS convenience construction for triaxial supports;
  complete matrices with unequal orthogonal scales are already accepted.
- How micro correction is masked across finite-fault domains.
- Whether the existing name-derived fallback element IDs should eventually be
  replaced by stable UUIDs across all observation tables.

These decisions do not change the core contract: a micro observation is owned
by one structural element and persisted as a local-support-to-world `4x4`
transform whose scale defines anisotropic kernel support.
