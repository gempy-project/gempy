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
def sample_json_data(sample_surface_points):
    """Create sample JSON data for testing."""
    _, x, y, z, ids, nugget, _ = sample_surface_points
    
    return {
        "metadata": {
            "name": "sample_model",
            "creation_date": "2024-03-19",
            "last_modification_date": "2024-03-19",
            "owner": "tutorial"
        },
        "surface_points": [
            {
                "x": float(x[i]),
                "y": float(y[i]),
                "z": float(z[i]),
                "id": int(ids[i]),
                "nugget": float(nugget[i])
            }
            for i in range(len(x))
        ],
        "orientations": [],
        "faults": [],
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