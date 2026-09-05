import numpy as np
import pytest

import gempy as gp


def _support_transform(
        translation=(0.0, 0.0, 0.0),
        scales=(2.0, 2.0, 0.25),
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scales)
    transform[:3, 3] = translation
    return transform


def test_micro_points_table_layout_and_views():
    transforms = np.stack([
            _support_transform(translation=(1.0, 2.0, 3.0)),
            _support_transform(translation=(4.0, 5.0, 6.0), scales=(3.0, 1.5, 0.5)),
    ])
    table = gp.data.MicroPointsTable.from_transforms(
        support_transforms=transforms,
        names="layer",
        nugget=np.array([0.0, 0.1]),
    )

    assert table.data.dtype.names == ("support_transform", "element_id", "nugget")
    assert table.data.dtype.fields["support_transform"][1] == 0
    assert table.data.dtype.fields["element_id"][1] == 128
    assert table.data.dtype.fields["nugget"][1] == 136
    assert table.data.dtype.itemsize == 144
    assert table.data.dtype.descr == [
            ("support_transform", "<f8", (4, 4)),
            ("element_id", "<i8"),
            ("nugget", "<f8"),
    ]
    np.testing.assert_array_equal(table.support_transforms, transforms)
    np.testing.assert_array_equal(table.xyz, transforms[:, :3, 3])
    np.testing.assert_array_equal(table.nugget, [0.0, 0.1])

    table.xyz[0] = [7.0, 8.0, 9.0]
    np.testing.assert_array_equal(table.support_transforms[0, :3, 3], [7.0, 8.0, 9.0])


def test_micro_points_table_filters_by_element():
    transforms = np.stack([_support_transform(), _support_transform(translation=(1.0, 0.0, 0.0))])
    table = gp.data.MicroPointsTable.from_transforms(
        support_transforms=transforms,
        names=["a", "b"],
        name_id_map={"a": 10, "b": 20},
    )

    selected = table.get_micro_points_by_name("b")
    assert len(selected) == 1
    np.testing.assert_array_equal(selected.ids, [20])
    np.testing.assert_array_equal(selected.xyz, [[1.0, 0.0, 0.0]])


def test_structural_elements_have_independent_empty_micro_tables():
    empty_surface_points = gp.data.SurfacePointsTable.initialize_empty()
    empty_orientations = gp.data.OrientationsTable.initialize_empty()
    element_a = gp.data.StructuralElement(
        name="a",
        surface_points=empty_surface_points,
        orientations=empty_orientations,
        color="#111111",
    )
    element_b = gp.data.StructuralElement(
        name="b",
        surface_points=gp.data.SurfacePointsTable.initialize_empty(),
        orientations=gp.data.OrientationsTable.initialize_empty(),
        color="#222222",
    )

    assert element_a.micro_points is not element_b.micro_points
    assert len(element_a.micro_points) == len(element_b.micro_points) == 0


@pytest.mark.parametrize(
    "mutate, error",
    [
            (lambda value: value.__setitem__((0, 3, 3), 2.0), "affine last row"),
            (lambda value: value.__setitem__((0, 0, 0), 0.0), "support scales"),
            (lambda value: value.__setitem__((0, 0, 1), 0.5), "orthogonal axes"),
            (lambda value: value.__setitem__((0, 0, 0), -1.0), "right-handed"),
            (lambda value: value.__setitem__((0, 0, 0), np.nan), "must be finite"),
    ],
)
def test_micro_points_table_rejects_invalid_transforms(mutate, error):
    transforms = np.stack([_support_transform(scales=(1.0, 1.0, 1.0))])
    mutate(transforms)

    with pytest.raises(ValueError, match=error):
        gp.data.MicroPointsTable.from_transforms(transforms, names="layer")


def test_micro_points_table_rejects_invalid_input_shapes_and_nuggets():
    with pytest.raises(ValueError, match="shape"):
        gp.data.MicroPointsTable.from_transforms(np.eye(4), names="layer")

    transforms = np.stack([_support_transform(), _support_transform()])
    with pytest.raises(ValueError, match="same length"):
        gp.data.MicroPointsTable.from_transforms(transforms, names=["layer"])
    with pytest.raises(ValueError, match="shape"):
        gp.data.MicroPointsTable.from_transforms(transforms, names="layer", nugget=np.array([0.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        gp.data.MicroPointsTable.from_transforms(
            transforms,
            names="layer",
            nugget=np.array([0.0, -1.0]),
        )


def test_micro_points_table_rejects_multidimensional_structured_data():
    data = np.zeros((0, 2), dtype=gp.data.MicroPointsTable.dt)
    with pytest.raises(ValueError, match="one-dimensional"):
        gp.data.MicroPointsTable(data=data)


def test_micro_points_table_rejects_ill_conditioned_support():
    transforms = np.stack([_support_transform(scales=(1.0, 1.0, 1e-13))])
    with pytest.raises(ValueError, match="condition number"):
        gp.data.MicroPointsTable.from_transforms(transforms, names="layer")
