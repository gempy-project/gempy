"""Fixed-weight queries of a computed model in world coordinates."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.transforms import Transform
from gempy_engine.modules.activator.unit_classification import classify_units

if TYPE_CHECKING:
    from ..core.data.geo_model import GeoModel


def _backend_state():
    return (BackendTensor.engine_backend, BackendTensor.dtype, BackendTensor.use_gpu,
            BackendTensor.use_pykeops, BackendTensor.COMPUTE_GRADS)


@dataclass
class _FieldEvaluationContext:
    interpolation_input: InterpolationInput
    descriptor: InputDataDescriptor
    options: InterpolationOptions
    input_transform: Transform
    grid_transform: Transform
    backend: tuple

    @classmethod
    def capture(cls, model, interpolation_input, descriptor, options):
        # Previous-solve weights are not query inputs. The engine evaluator uses
        # the production weights retained in Solutions, including on Torch.
        snapshot = copy.deepcopy(interpolation_input, {id(interpolation_input.weights): []})
        return cls(snapshot, copy.deepcopy(descriptor), copy.deepcopy(options),
                   copy.deepcopy(model.input_transform), copy.deepcopy(model.grid.transform),
                   _backend_state())


@dataclass(frozen=True)
class FieldEvaluator:
    """A fixed-weight engine evaluator accepting world-coordinate points.

    Calls return raw stack scalars of shape (number of selected stacks, M).
    Metadata indices refer to the group/surface order at compute time. The
    evaluator owns snapshots and remains valid after edits to the GeoModel.
    The engine backend must remain unchanged while querying.
    """

    _engine_evaluator: gempy_engine.FieldEvaluator
    _input_transform: Transform
    _grid_transform: Transform

    def _points_to_engine(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite array of shape (M, 3)")
        return self._input_transform.apply(self._grid_transform.apply_with_cached_pivot(points))

    def __call__(self, points, stacks=None, fault_sides=None):
        """Query raw fields; optional fault_sides is an (F, M) array of ±1."""
        return self._engine_evaluator(self._points_to_engine(points), stacks=stacks, fault_sides=fault_sides)

    def fault_values(self, points):
        """Return fault scalars relative to their surface levels, shape (F, M)."""
        return self._engine_evaluator.fault_values(self._points_to_engine(points))

    def classify_units(self, points, fault_sides=None):
        """Return (stack, interval) arrays, not GemPy element IDs."""
        return classify_units(self(points, fault_sides=fault_sides), self.stack_relations, self.stack_isovalues)

    @property
    def fault_stacks(self):
        return self._engine_evaluator.fault_stacks

    @property
    def affected_by(self):
        return self._engine_evaluator.affected_by

    @property
    def stack_isovalues(self):
        return self._engine_evaluator.stack_isovalues

    @property
    def stack_relations(self):
        return self._engine_evaluator.stack_relations

    @property
    def diagnostics(self):
        return self._engine_evaluator.diagnostics


def prepare_field_evaluator(gempy_model: "GeoModel") -> FieldEvaluator:
    """Prepare repeated queries from the most recent successful gp.compute_model.

    No grid changes or new solve. Uses compute-time inputs and coordinate
    transforms, even if the model has since been edited. Requires float64,
    no autograd, and the original backend. The engine currently supports only
    independent infinite faults and no micro corrections or external fields.
    Loaded models must be computed before preparing an evaluator.
    """
    context = gempy_model._field_evaluation_context
    if context is None or gempy_model.solutions is None:
        raise ValueError("Compute the model with gp.compute_model before preparing a field evaluator")
    if context.backend != _backend_state():
        raise RuntimeError("Engine backend changed since compute_model; recompute with the desired backend")
    evaluator = gempy_engine.prepare_field_evaluator(
        gempy_model.solutions, context.interpolation_input, context.descriptor, context.options)
    return FieldEvaluator(evaluator, copy.deepcopy(context.input_transform), copy.deepcopy(context.grid_transform))


def evaluate_fields(gempy_model: "GeoModel", points: np.ndarray, stacks=None, fault_sides=None) -> np.ndarray:
    """One-shot raw field query in world coordinates, without recomputing the model."""
    return prepare_field_evaluator(gempy_model)(points, stacks=stacks, fault_sides=fault_sides)
