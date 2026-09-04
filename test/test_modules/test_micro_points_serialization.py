import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.modules.serialization.save_load import _load_model_from_bytes, model_to_bytes


def _support_transform(translation, scales) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scales)
    transform[:3, 3] = translation
    return transform


def _model_with_micro_points():
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    elements = [element for group in model.structural_frame.structural_groups for element in group.elements]
    name_id_map = model.structural_frame.element_name_id_map

    elements[0].micro_points = gp.data.MicroPointsTable.from_transforms(
        support_transforms=np.stack([
                _support_transform((1.0, 2.0, 3.0), (5.0, 5.0, 0.5)),
                _support_transform((4.0, 5.0, 6.0), (4.0, 4.0, 0.25)),
        ]),
        names=elements[0].name,
        nugget=np.array([0.0, 0.1]),
        name_id_map=name_id_map,
    )
    elements[1].micro_points = gp.data.MicroPointsTable.from_transforms(
        support_transforms=np.stack([
                _support_transform((7.0, 8.0, 9.0), (3.0, 3.0, 0.2)),
        ]),
        names=elements[1].name,
        name_id_map=name_id_map,
    )
    return model


def _rewrite_archive(data: bytes, replacements=None, removed=()) -> bytes:
    replacements = replacements or {}
    source = io.BytesIO(data)
    target = io.BytesIO()
    with zipfile.ZipFile(source, "r") as input_zip, zipfile.ZipFile(target, "w") as output_zip:
        for info in input_zip.infolist():
            if info.filename in removed:
                continue
            value = replacements.get(info.filename, input_zip.read(info.filename))
            output_zip.writestr(info.filename, value, compress_type=zipfile.ZIP_STORED)
    return target.getvalue()


def test_micro_points_round_trip_and_archive_layout():
    model = _model_with_micro_points()
    serialized = model_to_bytes(model)
    restored = _load_model_from_bytes(serialized)

    original_elements = [element for group in model.structural_frame.structural_groups for element in group.elements]
    restored_elements = [element for group in restored.structural_frame.structural_groups for element in group.elements]
    for original, loaded in zip(original_elements, restored_elements):
        np.testing.assert_array_equal(original.micro_points.data, loaded.micro_points.data)
        assert loaded.micro_points.data.flags.writeable

    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        assert archive.namelist() == [
                "header.json",
                "input.bin",
                "micro_points.bin",
                "grid.bin",
                "liquid_earth_meta.json",
        ]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert all(info.create_system == 0 for info in archive.infolist())
        assert all(info.external_attr == 0x20 for info in archive.infolist())
        header = json.loads(archive.read("header.json"))
        assert header["serialization"]["version"] == 2
        assert header["structural_frame"]["binary_meta_data"]["micro_points"] == {
                "dtype_version": 1,
                "row_count"   : 3,
                "byte_length" : 3 * gp.data.MicroPointsTable.dt.itemsize,
        }
        assert "support_transform" not in archive.read("header.json").decode("utf-8")


def test_structural_frame_aggregates_micro_points_in_element_order():
    model = _model_with_micro_points()

    np.testing.assert_array_equal(model.structural_frame.number_of_micro_points_per_element, [2, 1, 0])
    np.testing.assert_array_equal(model.structural_frame.number_of_micro_points_per_group, [3])
    np.testing.assert_array_equal(
        model.structural_frame.micro_points_copy.xyz,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    )


def test_public_save_and_load_model_round_trip(tmp_path):
    model = _model_with_micro_points()
    path = tmp_path / "micro-points.gempy"

    gp.save_model(model, path=str(path), validate_serialization=True)
    restored = gp.load_model(str(path))

    np.testing.assert_array_equal(
        model.structural_frame.micro_points_copy.data,
        restored.structural_frame.micro_points_copy.data,
    )


def test_micro_points_serialization_is_deterministic():
    model = _model_with_micro_points()
    assert model_to_bytes(model) == model_to_bytes(model)


def test_micro_points_are_redistributed_by_element_id():
    model = _model_with_micro_points()
    serialized = model_to_bytes(model)
    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        rows = np.frombuffer(
            archive.read("micro_points.bin"),
            dtype=gp.data.MicroPointsTable.dt,
        ).copy()[::-1]

    restored = _load_model_from_bytes(
        _rewrite_archive(serialized, replacements={"micro_points.bin": rows.tobytes()})
    )
    for element in [element for group in restored.structural_frame.structural_groups for element in group.elements]:
        assert np.all(element.micro_points.ids == element.id)


def test_legacy_archive_loads_with_empty_micro_tables():
    archive_path = Path(__file__).parents[2] / "examples/data/gempy_models/Greenstone.gempy"
    restored = _load_model_from_bytes(archive_path.read_bytes())
    assert all(
        len(element.micro_points) == 0
        for group in restored.structural_frame.structural_groups
        for element in group.elements
    )


def test_version_two_requires_micro_points_member():
    serialized = model_to_bytes(_model_with_micro_points())
    malformed = _rewrite_archive(serialized, removed={"micro_points.bin"})
    with pytest.raises(ValueError, match="missing micro_points.bin"):
        _load_model_from_bytes(malformed)


def test_micro_points_binary_length_is_validated():
    serialized = model_to_bytes(_model_with_micro_points())
    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        truncated = archive.read("micro_points.bin")[:-1]
    malformed = _rewrite_archive(serialized, replacements={"micro_points.bin": truncated})
    with pytest.raises(ValueError, match="binary length"):
        _load_model_from_bytes(malformed)


@pytest.mark.parametrize(
    "metadata_key, value, error",
    [
            ("row_count", 4, "row count"),
            ("dtype_version", 999, "dtype version"),
    ],
)
def test_micro_points_binary_metadata_is_validated(metadata_key, value, error):
    serialized = model_to_bytes(_model_with_micro_points())
    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        header = json.loads(archive.read("header.json"))
    header["structural_frame"]["binary_meta_data"]["micro_points"][metadata_key] = value
    malformed = _rewrite_archive(
        serialized,
        replacements={"header.json": json.dumps(header, indent=4)},
    )
    with pytest.raises(ValueError, match=error):
        _load_model_from_bytes(malformed)


def test_unknown_micro_point_element_id_is_rejected():
    serialized = model_to_bytes(_model_with_micro_points())
    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        rows = np.frombuffer(
            archive.read("micro_points.bin"),
            dtype=gp.data.MicroPointsTable.dt,
        ).copy()
    rows[0]["element_id"] = np.iinfo(np.int64).max
    malformed = _rewrite_archive(serialized, replacements={"micro_points.bin": rows.tobytes()})
    with pytest.raises(ValueError, match="unknown structural element IDs"):
        _load_model_from_bytes(malformed)


def test_mismatched_element_ownership_is_rejected_before_save():
    model = _model_with_micro_points()
    element = model.structural_frame.structural_groups[0].elements[0]
    element.micro_points.data["element_id"] = np.iinfo(np.int64).max

    with pytest.raises(ValueError, match="mismatched element ID"):
        model_to_bytes(model)


def test_duplicate_structural_element_ids_are_rejected_when_micro_points_exist():
    model = _model_with_micro_points()
    elements = model.structural_frame.structural_groups[0].elements
    elements[1]._id = elements[0].id

    with pytest.raises(ValueError, match="must be unique"):
        model_to_bytes(model)


def test_unsupported_serialization_version_is_rejected():
    serialized = model_to_bytes(_model_with_micro_points())
    with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
        header = json.loads(archive.read("header.json"))
    header["serialization"]["version"] = 999
    malformed = _rewrite_archive(
        serialized,
        replacements={"header.json": json.dumps(header, indent=4)},
    )
    with pytest.raises(ValueError, match="Unsupported serialization version"):
        _load_model_from_bytes(malformed)
