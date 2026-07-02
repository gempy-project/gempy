# GeoModel Validation Spec

## Summary

Adds a `GeoModel.validate()` method that runs semantic checks post-deserialization, called automatically
from `gp.compute_model()` (with `skip_validation=False` default). `gempy_server` catches any validation
error and returns HTTP 422.

## Rules (all hard — raise `ModelValidationError`)

| ID | Reason code | Condition | Field |
|----|------------|-----------|-------|
| R1 | `empty_model` | `len(sp) == 0 and len(ori) == 0` | `input_data` |
| R2 | `empty_fault_group` | Any fault group with `len(elements) == 0` | `structural_groups[{i}]` |
| R3 | `empty_non_fault_group` | Any non-fault group with `len(elements) == 0` | `structural_groups[{i}]` |
| R4 | `underdetermined_input` | `len(sp) <= 1 and len(ori) == 0` | `input_data` |
| R5 | `basement_relation_on_non_last_group` | Any group at index `i < len(groups)-1` with `structural_relation == StackRelationType.BASEMENT` | `structural_groups[{i}].structural_relation` |

R1 fires first (precedence) — an empty-data model gets `empty_model`, not `empty_fault_group`.

## Files changed

### gempy

| File | Change |
|------|--------|
| `gempy/core/data/validation.py` | **New** — `ModelValidationError(ValueError)` with `field`, `reason`, `context` |
| `gempy/core/data/geo_model.py` | Added `validate()` method (5 rules), import `StackRelationType` + `ModelValidationError`, fixed bare `raise ValidationError` → proper message |
| `gempy/core/data/__init__.py` | Export `ModelValidationError` in `__all__` |
| `gempy/__init__.py` | Top-level `from .core.data.validation import ModelValidationError` |
| `gempy/API/compute_API.py` | Added `skip_validation=False` kwarg to `compute_model` + `compute_model_at`; calls `model.validate()` when not skipped |

### gempy_server

| File | Change |
|------|--------|
| `gempy_server/API/_compute_model_fn.py` | Calls `geomodel.validate()` after `_load_model_from_bytes` |
| `gempy_server/API/server.py` | Catches `ModelValidationError` → 422 JSON, `pydantic.ValidationError` → 422 JSON |

## Error shape (HTTP 422)

```json
{
  "reason": "empty_model",
  "field": "input_data",
  "context": {"surface_points": 0, "orientations": 0}
}
```

```json
{
  "reason": "schema_validation_failed",
  "details": [{"loc": ["interpolation_options", "block_solutions_type"], "msg": "...", "type": "enum"}]
}
```

## Skip validation

```python
gp.compute_model(model, skip_validation=True)  # bypass validate()
```

## Verification

```
tests/test_generated_models.py — 50 passed (plus expected xfail/skip unchanged)
tests/test_server_generic.py — 4 passed
```
