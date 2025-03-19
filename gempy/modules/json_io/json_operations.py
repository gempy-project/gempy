"""
Module for JSON I/O operations in GemPy.
This module provides functionality to load and save GemPy models to/from JSON files.
"""

import json
from typing import Dict, Any, Optional, List
import numpy as np

from gempy.core.data.surface_points import SurfacePointsTable
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.structural_frame import StructuralFrame
from gempy.core.data.grid import Grid
from gempy.core.data.geo_model import GeoModel
from .schema import SurfacePoint, GemPyModelJson


class JsonIO:
    """Class for handling JSON I/O operations for GemPy models."""
    
    @staticmethod
    def load_model_from_json(file_path: str) -> GeoModel:
        """
        Load a GemPy model from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            GeoModel: A new GemPy model instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate the JSON data against our schema
        if not JsonIO._validate_json_schema(data):
            raise ValueError("Invalid JSON schema")
            
        # Load surface points
        surface_points = JsonIO._load_surface_points(data['surface_points'])
        
        # TODO: Load other components
        raise NotImplementedError("Only surface points loading is implemented")
    
    @staticmethod
    def _load_surface_points(surface_points_data: List[SurfacePoint]) -> SurfacePointsTable:
        """
        Load surface points from JSON data.
        
        Args:
            surface_points_data (List[SurfacePoint]): List of surface point dictionaries
            
        Returns:
            SurfacePointsTable: A new SurfacePointsTable instance
            
        Raises:
            ValueError: If the data is invalid or missing required fields
        """
        # Validate data structure
        required_fields = {'x', 'y', 'z', 'id', 'nugget'}
        for i, sp in enumerate(surface_points_data):
            missing_fields = required_fields - set(sp.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields in surface point {i}: {missing_fields}")
            
            # Validate data types
            if not all(isinstance(sp[field], (int, float)) for field in ['x', 'y', 'z', 'nugget']):
                raise ValueError(f"Invalid data type in surface point {i}. All coordinates and nugget must be numeric.")
            if not isinstance(sp['id'], int):
                raise ValueError(f"Invalid data type in surface point {i}. ID must be an integer.")
        
        # Extract coordinates and other data
        x = np.array([sp['x'] for sp in surface_points_data])
        y = np.array([sp['y'] for sp in surface_points_data])
        z = np.array([sp['z'] for sp in surface_points_data])
        ids = np.array([sp['id'] for sp in surface_points_data])
        nugget = np.array([sp['nugget'] for sp in surface_points_data])
        
        # Create name_id_map from unique IDs
        unique_ids = np.unique(ids)
        name_id_map = {f"surface_{id}": id for id in unique_ids}
        
        # Create SurfacePointsTable
        return SurfacePointsTable.from_arrays(
            x=x,
            y=y,
            z=z,
            names=[f"surface_{id}" for id in ids],
            nugget=nugget,
            name_id_map=name_id_map
        )
    
    @staticmethod
    def save_model_to_json(model: GeoModel, file_path: str) -> None:
        """
        Save a GemPy model to a JSON file.
        
        Args:
            model (GeoModel): The GemPy model to save
            file_path (str): Path where to save the JSON file
        """
        # TODO: Implement saving logic
        raise NotImplementedError("JSON saving not yet implemented")
    
    @staticmethod
    def _validate_json_schema(data: Dict[str, Any]) -> bool:
        """
        Validate the JSON data against the expected schema.
        
        Args:
            data (Dict[str, Any]): The JSON data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check required top-level keys
        required_keys = {'metadata', 'surface_points', 'orientations', 'faults', 
                        'series', 'grid_settings', 'interpolation_options'}
        if not all(key in data for key in required_keys):
            return False
            
        # Validate surface points
        if not isinstance(data['surface_points'], list):
            return False
            
        for sp in data['surface_points']:
            required_sp_keys = {'x', 'y', 'z', 'id', 'nugget'}
            if not all(key in sp for key in required_sp_keys):
                return False
            if not all(isinstance(sp[key], (int, float)) for key in ['x', 'y', 'z', 'nugget']):
                return False
            if not isinstance(sp['id'], int):
                return False
                
        return True 