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


def test_surface_points_loading(sample_surface_points, sample_json_data):
    """Test loading surface points from JSON data."""
    surface_points, _, _, _, _ = sample_surface_points
    
    # Load surface points from JSON
    loaded_surface_points = JsonIO._load_surface_points(sample_json_data["surface_points"], {})
    
    # Verify all data matches
    assert np.allclose(surface_points.xyz[:, 0], loaded_surface_points.xyz[:, 0]), "X coordinates don't match"
    assert np.allclose(surface_points.xyz[:, 1], loaded_surface_points.xyz[:, 1]), "Y coordinates don't match"
    assert np.allclose(surface_points.xyz[:, 2], loaded_surface_points.xyz[:, 2]), "Z coordinates don't match"
    assert np.allclose(surface_points.nugget, loaded_surface_points.nugget), "Nugget values don't match"


def test_orientations_loading(sample_orientations, sample_json_data):
    """Test loading orientations from JSON data."""
    orientations, _, _, _, _, _, _, _ = sample_orientations
    
    # Load orientations from JSON
    loaded_orientations = JsonIO._load_orientations(sample_json_data["orientations"], {})
    
    # Verify all data matches
    assert np.allclose(orientations.xyz[:, 0], loaded_orientations.xyz[:, 0]), "X coordinates don't match"
    assert np.allclose(orientations.xyz[:, 1], loaded_orientations.xyz[:, 1]), "Y coordinates don't match"
    assert np.allclose(orientations.xyz[:, 2], loaded_orientations.xyz[:, 2]), "Z coordinates don't match"
    assert np.allclose(orientations.grads[:, 0], loaded_orientations.grads[:, 0]), "G_x values don't match"
    assert np.allclose(orientations.grads[:, 1], loaded_orientations.grads[:, 1]), "G_y values don't match"
    assert np.allclose(orientations.grads[:, 2], loaded_orientations.grads[:, 2]), "G_z values don't match"
    assert np.allclose(orientations.nugget, loaded_orientations.nugget), "Nugget values don't match"


def test_surface_points_saving(tmp_path, sample_surface_points, sample_json_data):
    """Test saving surface points to JSON file."""
    surface_points, _, _, _, _ = sample_surface_points
    
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
        JsonIO._load_surface_points(invalid_data, {})


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
        JsonIO._load_surface_points(invalid_data, {})


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
        JsonIO._load_orientations(invalid_data, {})


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
        JsonIO._load_orientations(invalid_data, {})


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
        JsonIO._load_orientations(invalid_data, {})


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


def test_complete_model_saving_loading(tmp_path, sample_model_with_series):
    """Test saving and loading a complete model with series and fault relations."""
    model = sample_model_with_series
    
    # Save model to temporary file
    file_path = tmp_path / "test_complete_model.json"
    JsonIO.save_model_to_json(model, str(file_path))
    
    # Load model from file
    loaded_model = JsonIO.load_model_from_json(str(file_path))
    
    # Verify model structure
    assert len(loaded_model.structural_frame.structural_groups) == 2, "Wrong number of structural groups"
    assert loaded_model.structural_frame.structural_groups[0].name == "fault_series", "First group should be fault_series"
    assert loaded_model.structural_frame.structural_groups[1].name == "strat_series", "Second group should be strat_series"
    
    # Verify fault relations
    assert loaded_model.structural_frame.structural_groups[0].structural_relation == StackRelationType.FAULT, "First group should be a fault"
    assert loaded_model.structural_frame.structural_groups[1].structural_relation == StackRelationType.ERODE, "Second group should be erode"
    
    # Verify data points (only coordinates)
    assert np.allclose(model.surface_points_copy.xyz, loaded_model.surface_points_copy.xyz), "Surface points don't match"
    assert np.allclose(model.orientations_copy.xyz, loaded_model.orientations_copy.xyz), "Orientations don't match"


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
    assert loaded_model.meta.owner == "test_value", "Custom metadata not preserved"
    assert loaded_model.meta.creation_date is not None, "Creation date missing"
    assert loaded_model.meta.last_modification_date is not None, "Last modification date missing"


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
    assert np.array_equal(loaded_model.grid.regular_grid.resolution, [20, 20, 20]), "Grid resolution not preserved"
    assert np.array_equal(loaded_model.grid.regular_grid.extent, [0, 20, 0, 20, 0, 20]), "Grid extent not preserved"


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
    assert loaded_model.interpolation_options.kernel_options.range == 10.0, "Range not preserved"
    assert loaded_model.interpolation_options.kernel_options.c_o == 15.0, "C_o not preserved"
    assert loaded_model.interpolation_options.mesh_extraction is False, "Mesh extraction not preserved"
    assert loaded_model.interpolation_options.number_octree_levels == 2, "Number of octree levels not preserved" 