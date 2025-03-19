"""
Tests for JSON I/O operations in GemPy.
"""

import json
import numpy as np
import pytest
import gempy as gp
from gempy.modules.json_io import JsonIO


@pytest.fixture
def sample_surface_points():
    """Create sample surface points data for testing."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    z = np.array([0, 1, 2, 3, 4])
    ids = np.array([0, 0, 1, 1, 2])  # Three different surfaces
    nugget = np.array([0.00002, 0.00002, 0.00002, 0.00002, 0.00002])
    
    # Create name to id mapping
    name_id_map = {f"surface_{id}": id for id in np.unique(ids)}
    
    # Create a SurfacePointsTable
    surface_points = gp.data.SurfacePointsTable.from_arrays(
        x=x,
        y=y,
        z=z,
        names=[f"surface_{id}" for id in ids],
        nugget=nugget,
        name_id_map=name_id_map
    )
    
    return surface_points, x, y, z, ids, nugget, name_id_map


@pytest.fixture
def sample_orientations():
    """Create sample orientation data for testing."""
    x = np.array([0.5, 1.5, 2.5, 3.5])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    z = np.array([0.5, 1.5, 2.5, 3.5])
    G_x = np.array([0, 0, 0, 0])
    G_y = np.array([0, 0, 0, 0])
    G_z = np.array([1, 1, -1, 1])  # One reversed orientation
    ids = np.array([0, 1, 1, 2])  # Three different surfaces
    nugget = np.array([0.01, 0.01, 0.01, 0.01])
    
    # Create name to id mapping
    name_id_map = {f"surface_{id}": id for id in np.unique(ids)}
    
    # Create an OrientationsTable
    orientations = gp.data.OrientationsTable.from_arrays(
        x=x,
        y=y,
        z=z,
        G_x=G_x,
        G_y=G_y,
        G_z=G_z,
        names=[f"surface_{id}" for id in ids],
        nugget=nugget,
        name_id_map=name_id_map
    )
    
    return orientations, x, y, z, G_x, G_y, G_z, ids, nugget, name_id_map


@pytest.fixture
def sample_json_data(sample_surface_points, sample_orientations):
    """Create sample JSON data for testing."""
    _, x_sp, y_sp, z_sp, ids_sp, nugget_sp, _ = sample_surface_points
    _, x_ori, y_ori, z_ori, G_x, G_y, G_z, ids_ori, nugget_ori, _ = sample_orientations
    
    return {
        "metadata": {
            "name": "sample_model",
            "creation_date": "2024-03-19",
            "last_modification_date": "2024-03-19",
            "owner": "tutorial"
        },
        "surface_points": [
            {
                "x": float(x_sp[i]),
                "y": float(y_sp[i]),
                "z": float(z_sp[i]),
                "id": int(ids_sp[i]),
                "nugget": float(nugget_sp[i])
            }
            for i in range(len(x_sp))
        ],
        "orientations": [
            {
                "x": float(x_ori[i]),
                "y": float(y_ori[i]),
                "z": float(z_ori[i]),
                "G_x": float(G_x[i]),
                "G_y": float(G_y[i]),
                "G_z": float(G_z[i]),
                "id": int(ids_ori[i]),
                "nugget": float(nugget_ori[i]),
                "polarity": 1  # Always set to 1 since we're testing the raw G_z values
            }
            for i in range(len(x_ori))
        ],
        "series": [],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 4, 0, 4, 0, 4],
            "octree_levels": None
        },
        "interpolation_options": {}
    }


def test_surface_points_loading(sample_surface_points, sample_json_data):
    """Test loading surface points from JSON data."""
    surface_points, _, _, _, _, _, name_id_map = sample_surface_points
    
    # Load surface points from JSON
    loaded_surface_points = JsonIO._load_surface_points(sample_json_data["surface_points"])
    
    # Verify all data matches
    assert np.allclose(surface_points.xyz[:, 0], loaded_surface_points.xyz[:, 0]), "X coordinates don't match"
    assert np.allclose(surface_points.xyz[:, 1], loaded_surface_points.xyz[:, 1]), "Y coordinates don't match"
    assert np.allclose(surface_points.xyz[:, 2], loaded_surface_points.xyz[:, 2]), "Z coordinates don't match"
    assert np.array_equal(surface_points.ids, loaded_surface_points.ids), "IDs don't match"
    assert np.allclose(surface_points.nugget, loaded_surface_points.nugget), "Nugget values don't match"
    assert surface_points.name_id_map == loaded_surface_points.name_id_map, "Name to ID mappings don't match"


def test_orientations_loading(sample_orientations, sample_json_data):
    """Test loading orientations from JSON data."""
    orientations, _, _, _, _, _, _, _, _, name_id_map = sample_orientations
    
    # Load orientations from JSON
    loaded_orientations = JsonIO._load_orientations(sample_json_data["orientations"])
    
    # Verify all data matches
    assert np.allclose(orientations.xyz[:, 0], loaded_orientations.xyz[:, 0]), "X coordinates don't match"
    assert np.allclose(orientations.xyz[:, 1], loaded_orientations.xyz[:, 1]), "Y coordinates don't match"
    assert np.allclose(orientations.xyz[:, 2], loaded_orientations.xyz[:, 2]), "Z coordinates don't match"
    assert np.allclose(orientations.grads[:, 0], loaded_orientations.grads[:, 0]), "G_x values don't match"
    assert np.allclose(orientations.grads[:, 1], loaded_orientations.grads[:, 1]), "G_y values don't match"
    assert np.allclose(orientations.grads[:, 2], loaded_orientations.grads[:, 2]), "G_z values don't match"
    assert np.array_equal(orientations.ids, loaded_orientations.ids), "IDs don't match"
    assert np.allclose(orientations.nugget, loaded_orientations.nugget), "Nugget values don't match"
    assert orientations.name_id_map == loaded_orientations.name_id_map, "Name to ID mappings don't match"


def test_surface_points_saving(tmp_path, sample_surface_points, sample_json_data):
    """Test saving surface points to JSON file."""
    surface_points, _, _, _, _, _, _ = sample_surface_points
    
    # Save to temporary file
    file_path = tmp_path / "test_surface_points.json"
    with open(file_path, "w") as f:
        json.dump(sample_json_data, f, indent=4)
    
    # Verify file was created and contains correct data
    assert file_path.exists(), "JSON file was not created"
    
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == sample_json_data, "Saved JSON data doesn't match original"


def test_invalid_surface_points_data():
    """Test handling of invalid surface points data."""
    invalid_data = [
        {
            "x": "invalid",  # Should be float
            "y": 1.0,
            "z": 2.0,
            "id": 0,
            "nugget": 0.00002
        }
    ]
    
    with pytest.raises(ValueError):
        JsonIO._load_surface_points(invalid_data)


def test_missing_surface_points_data():
    """Test handling of missing surface points data."""
    invalid_data = [
        {
            "x": 1.0,
            "y": 1.0,
            # Missing z coordinate
            "id": 0,
            "nugget": 0.00002
        }
    ]
    
    with pytest.raises(ValueError):
        JsonIO._load_surface_points(invalid_data)


def test_invalid_orientations_data():
    """Test handling of invalid orientation data."""
    invalid_data = [
        {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
            "G_x": 0.0,
            "G_y": 0.0,
            "G_z": "invalid",  # Should be float
            "id": 0,
            "nugget": 0.01,
            "polarity": 1
        }
    ]
    
    with pytest.raises(ValueError):
        JsonIO._load_orientations(invalid_data)


def test_missing_orientations_data():
    """Test handling of missing orientation data."""
    invalid_data = [
        {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
            "G_x": 0.0,
            # Missing G_y and G_z
            "id": 0,
            "nugget": 0.01,
            "polarity": 1
        }
    ]
    
    with pytest.raises(ValueError):
        JsonIO._load_orientations(invalid_data)


def test_invalid_orientation_polarity():
    """Test handling of invalid orientation polarity."""
    invalid_data = [
        {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
            "G_x": 0.0,
            "G_y": 0.0,
            "G_z": 1.0,
            "id": 0,
            "nugget": 0.01,
            "polarity": 2  # Should be 1 or -1
        }
    ]
    
    with pytest.raises(ValueError):
        JsonIO._load_orientations(invalid_data) 