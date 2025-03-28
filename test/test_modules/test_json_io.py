"""
Tests for JSON I/O operations in GemPy.
"""

import json
import numpy as np
import pytest
import gempy as gp
from gempy.modules.json_io import JsonIO
from gempy_engine.core.data.stack_relation_type import StackRelationType


@pytest.fixture
def sample_surface_points():
    """Create sample surface points data for testing."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    z = np.array([0, 1, 2, 3, 4])
    nugget = np.array([0.00002, 0.00002, 0.00002, 0.00002, 0.00002])
    
    # Create a SurfacePointsTable
    surface_points = gp.data.SurfacePointsTable.from_arrays(
        x=x,
        y=y,
        z=z,
        names=["surface_0", "surface_0", "surface_1", "surface_1", "surface_2"],
        nugget=nugget
    )
    
    return surface_points, x, y, z, nugget


@pytest.fixture
def sample_orientations():
    """Create sample orientation data for testing."""
    x = np.array([0.5, 1.5, 2.5, 3.5])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    z = np.array([0.5, 1.5, 2.5, 3.5])
    G_x = np.array([0, 0, 0, 0])
    G_y = np.array([0, 0, 0, 0])
    G_z = np.array([1, 1, -1, 1])  # One reversed orientation
    nugget = np.array([0.01, 0.01, 0.01, 0.01])
    
    # Create an OrientationsTable
    orientations = gp.data.OrientationsTable.from_arrays(
        x=x,
        y=y,
        z=z,
        G_x=G_x,
        G_y=G_y,
        G_z=G_z,
        names=["surface_0", "surface_1", "surface_1", "surface_2"],
        nugget=nugget
    )
    
    return orientations, x, y, z, G_x, G_y, G_z, nugget


@pytest.fixture
def sample_json_data(sample_surface_points, sample_orientations):
    """Create sample JSON data for testing."""
    _, x_sp, y_sp, z_sp, nugget_sp = sample_surface_points
    _, x_ori, y_ori, z_ori, G_x, G_y, G_z, nugget_ori = sample_orientations
    
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
                "id": 0,  # Default ID, not used in tests
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
                "id": 0,  # Default ID, not used in tests
                "nugget": float(nugget_ori[i]),
                "polarity": 1  # Always set to 1 since we're testing the raw G_z values
            }
            for i in range(len(x_ori))
        ],
        "series": [
            {
                "name": "series1",
                "surfaces": ["surface_0", "surface_1"],
                "structural_relation": "erode",
                "colors": ["#FF0000", "#00FF00"]
            },
            {
                "name": "series2",
                "surfaces": ["surface_2"],
                "structural_relation": "erode",
                "colors": ["#0000FF"]
            }
        ],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 4, 0, 4, 0, 4],
            "octree_levels": None
        },
        "interpolation_options": {}
    }


def test_surface_points_loading():
    """Test loading surface points from JSON data."""
    data = {
        "surface_points": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "id": 0,
                "nugget": 0.0
            }
        ]
    }
    surface_points = JsonIO._load_surface_points(data["surface_points"])
    assert len(surface_points.xyz) == 1
    assert surface_points.xyz[0, 0] == 1.0  # x coordinate
    assert surface_points.xyz[0, 1] == 1.0  # y coordinate
    assert surface_points.xyz[0, 2] == 1.0  # z coordinate
    assert surface_points.nugget[0] == 0.0  # nugget


def test_orientations_loading():
    """Test loading orientations from JSON data."""
    data = {
        "orientations": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "id": 0,
                "nugget": 0.01,
                "polarity": 1
            }
        ]
    }
    orientations = JsonIO._load_orientations(data["orientations"])
    assert len(orientations.xyz) == 1
    assert orientations.xyz[0, 0] == 1.0  # x coordinate
    assert orientations.xyz[0, 1] == 1.0  # y coordinate
    assert orientations.xyz[0, 2] == 1.0  # z coordinate
    assert orientations.grads[0, 0] == 0.0  # G_x
    assert orientations.grads[0, 1] == 0.0  # G_y
    assert orientations.grads[0, 2] == 1.0  # G_z
    assert orientations.nugget[0] == 0.01  # nugget


def test_surface_points_saving(tmp_path):
    """Test saving surface points to a JSON file."""
    # Create sample surface points
    surface_points = np.array([[1.0, 1.0, 1.0, 0, 0.0]])
    
    # Create a complete model data structure
    data = {
        "surface_points": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "id": 0,
                "nugget": 0.0
            }
        ],
        "orientations": [],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 10, 0, 10, 0, 10]
        }
    }
    
    # Save to temporary file
    file_path = tmp_path / "test_surface_points.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Load and verify
    loaded_model = JsonIO.load_model_from_json(str(file_path))
    assert len(loaded_model.surface_points_copy.xyz) == 1
    assert loaded_model.surface_points_copy.xyz[0, 0] == 1.0  # x coordinate
    assert loaded_model.surface_points_copy.xyz[0, 1] == 1.0  # y coordinate
    assert loaded_model.surface_points_copy.xyz[0, 2] == 1.0  # z coordinate
    assert loaded_model.surface_points_copy.nugget[0] == 0.0  # nugget


def test_invalid_surface_points_data():
    """Test that invalid surface points data raises appropriate errors."""
    data = {
        'surface_points': [{'x': 0, 'y': 0}],  # Missing z
        'orientations': [],
        'grid_settings': {'regular_grid_resolution': [50, 50, 50], 'regular_grid_extent': [0, 1, 0, 1, 0, 1]}
    }
    with pytest.raises(ValueError, match="Missing required key in surface point: z"):
        JsonIO._validate_json_schema(data)


def test_missing_surface_points_data():
    """Test handling of missing surface points data."""
    data = {
        "orientations": []  # Missing surface_points key
    }
    with pytest.raises(ValueError, match="Missing required key: surface_points"):
        JsonIO._validate_json_schema(data)


def test_invalid_orientations_data():
    """Test that invalid orientations data raises appropriate errors."""
    data = {
        'surface_points': [{'x': 0, 'y': 0, 'z': 0}],
        'orientations': [{'x': 0, 'y': 0, 'z': 0}],  # Missing G_x, G_y, G_z
        'grid_settings': {'regular_grid_resolution': [50, 50, 50], 'regular_grid_extent': [0, 1, 0, 1, 0, 1]}
    }
    with pytest.raises(ValueError, match="Missing required key in orientation: G_x"):
        JsonIO._validate_json_schema(data)


def test_missing_orientations_data():
    """Test handling of missing orientations data."""
    data = {
        "surface_points": []  # Missing orientations key
    }
    with pytest.raises(ValueError, match="Missing required key: orientations"):
        JsonIO._validate_json_schema(data)


def test_invalid_orientation_polarity():
    """Test handling of invalid orientation polarity."""
    data = {
        "orientations": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "id": 0,
                "nugget": 0.01,
                "polarity": 2  # Invalid polarity value
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid polarity in orientation"):
        JsonIO._load_orientations(data["orientations"])


def test_default_nugget_values(tmp_path):
    """Test that default nugget values are correctly applied."""
    # Create minimal data without nugget values
    data = {
        "surface_points": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "id": 0
            }
        ],
        "orientations": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "id": 0,
                "polarity": 1
            }
        ],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 10, 0, 10, 0, 10]
        }
    }
    
    # Save data to temporary file
    file_path = tmp_path / "test_default_nugget.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Load model from JSON
    model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify default nugget values
    assert model.surface_points_copy.nugget[0] == 0.0  # Default surface point nugget
    assert model.orientations_copy.nugget[0] == 0.01  # Default orientation nugget


def test_default_series_values(tmp_path):
    """Test that default series values are correctly applied."""
    # Create minimal data without series
    data = {
        "surface_points": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "id": 0,
                "nugget": 0.0
            }
        ],
        "orientations": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "id": 0,
                "nugget": 0.01,
                "polarity": 1
            }
        ],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 10, 0, 10, 0, 10]
        }
    }
    
    # Save data to temporary file
    file_path = tmp_path / "test_default_series.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Load model from JSON (this will create default series)
    model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify default series
    assert len(model.structural_frame.structural_groups) == 1
    assert model.structural_frame.structural_groups[0].name == "Strat_Series"
    assert model.structural_frame.structural_groups[0].structural_relation == StackRelationType.ERODE


def test_default_interpolation_options():
    """Test that default interpolation options are correctly applied."""
    data = {
        'surface_points': [{'x': 0, 'y': 0, 'z': 0}],
        'orientations': [{'x': 0, 'y': 0, 'z': 0, 'G_x': 0, 'G_y': 0, 'G_z': 1}],
        'grid_settings': {'regular_grid_resolution': [50, 50, 50], 'regular_grid_extent': [0, 1, 0, 1, 0, 1]},
        'interpolation_options': {}
    }
    JsonIO._validate_json_schema(data)
    assert data['interpolation_options'].get('number_octree_levels', 4) == 4  # Default is now 4
    assert data['interpolation_options'].get('mesh_extraction', True) is True


def test_default_metadata(tmp_path):
    """Test that default metadata is correctly applied."""
    # Create minimal data without metadata
    data = {
        "surface_points": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "id": 0,
                "nugget": 0.0
            }
        ],
        "orientations": [
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "id": 0,
                "nugget": 0.01,
                "polarity": 1
            }
        ],
        "grid_settings": {
            "regular_grid_resolution": [10, 10, 10],
            "regular_grid_extent": [0, 10, 0, 10, 0, 10]
        }
    }
    
    # Save data to temporary file
    file_path = tmp_path / "test_default_metadata.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Load model from JSON
    model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify default metadata
    assert model.meta.name == "GemPy Model"  # Default name
    assert model.meta.owner == "GemPy Modeller"  # Default owner
    assert model.meta.creation_date is not None  # Should be set to current date
    assert model.meta.last_modification_date is not None  # Should be set to current date


def test_optional_series_colors():
    """Test that series colors are optional and properly validated."""
    data = {
        'surface_points': [{'x': 0, 'y': 0, 'z': 0}],
        'orientations': [{'x': 0, 'y': 0, 'z': 0, 'G_x': 0, 'G_y': 0, 'G_z': 1}],
        'grid_settings': {'regular_grid_resolution': [50, 50, 50], 'regular_grid_extent': [0, 1, 0, 1, 0, 1]},
        'series': [
            {'name': 'Series1', 'surfaces': ['Surface1'], 'colors': ['#015482']},  # Valid color
            {'name': 'Series2', 'surfaces': ['Surface2']}  # No colors specified
        ]
    }
    JsonIO._validate_json_schema(data)
    assert data['series'][0]['colors'] == ['#015482']  # Color should be preserved
    assert 'colors' not in data['series'][1]  # No colors should be added if not specified


@pytest.fixture
def sample_model_with_series():
    """Create a sample model with series and fault relations."""
    # Create a simple model with two series and a fault
    from gempy.core.data.structural_frame import StructuralFrame
    from gempy.core.data import StructuralElement, StructuralGroup
    from gempy_engine.core.data.stack_relation_type import StackRelationType
    
    # Create structural frame first
    structural_frame = StructuralFrame.initialize_default_structure()
    
    # Create model with structural frame
    model = gp.create_geomodel(
        project_name="test_model",
        extent=[0, 10, 0, 10, 0, 10],
        resolution=[10, 10, 10],
        structural_frame=structural_frame
    )
    
    # Create structural elements
    fault_element = StructuralElement(
        name="fault",
        color=next(model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.initialize_empty(),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    rock1_element = StructuralElement(
        name="rock1",
        color=next(model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.initialize_empty(),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    rock2_element = StructuralElement(
        name="rock2",
        color=next(model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.initialize_empty(),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    # Create structural groups
    fault_group = StructuralGroup(
        name="fault_series",
        elements=[fault_element],
        structural_relation=StackRelationType.FAULT
    )
    
    strat_group = StructuralGroup(
        name="strat_series",
        elements=[rock1_element, rock2_element],
        structural_relation=StackRelationType.ERODE
    )
    
    # Set up the structural frame
    model.structural_frame.structural_groups = [fault_group, strat_group]
    
    # Set fault relations (2x2 matrix: fault affects strat_series)
    model.structural_frame.fault_relations = np.array([
        [0, 1],  # fault_series affects strat_series
        [0, 0]   # strat_series doesn't affect any series
    ])
    
    # Add surface points
    gp.add_surface_points(
        geo_model=model,
        x=[1, 2, 3, 4, 5],
        y=[1, 2, 3, 4, 5],
        z=[1, 2, 3, 4, 5],
        elements_names=["fault", "rock1", "rock1", "rock2", "rock2"]
    )
    
    # Add orientations
    gp.add_orientations(
        geo_model=model,
        x=[1.5, 2.5, 3.5, 4.5],
        y=[1.5, 2.5, 3.5, 4.5],
        z=[1.5, 2.5, 3.5, 4.5],
        elements_names=["fault", "rock1", "rock1", "rock2"],
        pole_vector=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    
    return model


def test_metadata_handling(tmp_path, sample_model_with_series):
    """Test metadata handling in JSON I/O."""
    model = sample_model_with_series
    
    # Add custom metadata
    model.meta.owner = "test_value"
    
    # Save model
    file_path = tmp_path / "test_metadata.json"
    JsonIO.save_model_to_json(model, str(file_path))
    
    # Load model
    loaded_model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify metadata
    assert loaded_model.meta.owner == "test_value"
    assert loaded_model.meta.creation_date is not None
    assert loaded_model.meta.last_modification_date is not None


def test_grid_settings_handling(tmp_path, sample_model_with_series):
    """Test grid settings handling in JSON I/O."""
    model = sample_model_with_series
    
    # Modify grid settings
    model.grid.regular_grid.resolution = [20, 20, 20]
    model.grid.regular_grid.extent = [0, 20, 0, 20, 0, 20]
    
    # Save model
    file_path = tmp_path / "test_grid_settings.json"
    JsonIO.save_model_to_json(model, str(file_path))
    
    # Load model
    loaded_model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify grid settings
    np.testing.assert_array_equal(
        model.grid.regular_grid.resolution,
        loaded_model.grid.regular_grid.resolution
    )
    np.testing.assert_array_equal(
        model.grid.regular_grid.extent,
        loaded_model.grid.regular_grid.extent
    )


def test_interpolation_options_handling(tmp_path, sample_model_with_series):
    """Test interpolation options handling in JSON I/O."""
    model = sample_model_with_series
    
    # Set custom interpolation options
    model.interpolation_options.kernel_options.range = 10.0
    model.interpolation_options.kernel_options.c_o = 15.0
    model.interpolation_options.mesh_extraction = False
    model.interpolation_options.number_octree_levels = 2
    
    # Save model
    file_path = tmp_path / "test_interpolation_options.json"
    JsonIO.save_model_to_json(model, str(file_path))
    
    # Load model
    loaded_model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify interpolation options
    assert loaded_model.interpolation_options.kernel_options.range == 10.0
    assert loaded_model.interpolation_options.kernel_options.c_o == 15.0
    assert loaded_model.interpolation_options.mesh_extraction is False
    assert loaded_model.interpolation_options.number_octree_levels == 2 