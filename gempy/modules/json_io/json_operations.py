"""
Module for JSON I/O operations in GemPy.
This module provides functionality to load and save GemPy models to/from JSON files.
"""

import json
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

from .schema import SurfacePoint, Orientation, GemPyModelJson
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.core.data.structural_element import StructuralElement


class JsonIO:
    """Class for handling JSON I/O operations for GemPy models."""
    
    @staticmethod
    def _numpy_to_list(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    @staticmethod
    def load_model_from_json(file_path: str):
        """
        Load a GemPy model from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            GeoModel: A new GemPy model instance
        """
        # Import here to avoid circular imports
        from gempy.core.data.geo_model import GeoModel, GeoModelMeta
        from gempy.core.data.grid import Grid
        from gempy.core.data.structural_frame import StructuralFrame
        from gempy_engine.core.data import InterpolationOptions
        from gempy.API.map_stack_to_surfaces_API import map_stack_to_surfaces
        from gempy_engine.core.data.stack_relation_type import StackRelationType
        from gempy.core.data.structural_group import StructuralGroup
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate the JSON data against our schema
        if not JsonIO._validate_json_schema(data):
            raise ValueError("Invalid JSON schema")
            
        # Create a mapping from surface IDs to names
        id_to_name = {}
        for series in data['series']:
            for surface_name in series['surfaces']:
                # Find the ID for this surface name from surface points
                for sp in data['surface_points']:
                    if sp['id'] not in id_to_name:
                        id_to_name[sp['id']] = surface_name
                        break
        
        # Create GeoModel with default structural frame
        model = GeoModel(
            name=data['metadata']['name'],
            structural_frame=StructuralFrame.initialize_default_structure(),
            grid=Grid(
                extent=data['grid_settings']['regular_grid_extent'],
                resolution=data['grid_settings']['regular_grid_resolution']
            ),
            interpolation_options=InterpolationOptions(
                range=data['interpolation_options'].get('kernel_options', {}).get('range', 1.7),
                c_o=data['interpolation_options'].get('kernel_options', {}).get('c_o', 10),
                mesh_extraction=data['interpolation_options'].get('mesh_extraction', True),
                number_octree_levels=data['interpolation_options'].get('number_octree_levels', 1)
            )
        )
        
        # Set the metadata with proper dates
        model_meta = GeoModelMeta(
            name=data['metadata']['name'],
            creation_date=data['metadata'].get('creation_date', datetime.now().isoformat()),
            last_modification_date=data['metadata'].get('last_modification_date', datetime.now().isoformat()),
            owner=data['metadata'].get('owner', None)
        )
        model.meta = model_meta
        
        # Load surface points and orientations with the ID to name mapping
        surface_points = JsonIO._load_surface_points(data['surface_points'], id_to_name)
        orientations = JsonIO._load_orientations(data['orientations'], id_to_name)
        
        # Create structural groups based on series data
        model.structural_frame.structural_groups = []
        for i, series in enumerate(data['series']):
            group = StructuralGroup(
                name=series['name'],
                elements=[],
                structural_relation=StackRelationType[series.get('structural_relation', 'ERODE')]
            )
            model.structural_frame.structural_groups.append(group)
            
            # Add elements to the group
            for surface_name in series['surfaces']:
                element = StructuralElement(
                    name=surface_name,
                    id=len(group.elements),
                    surface_points=surface_points.get_surface_points_by_name(surface_name),
                    orientations=orientations.get_orientations_by_name(surface_name),
                    color=next(model.structural_frame.color_generator)
                )
                group.append_element(element)
        
        # Ensure the last group has the correct structural relation for the basement
        if model.structural_frame.structural_groups:
            model.structural_frame.structural_groups[-1].structural_relation = StackRelationType.BASEMENT
        
        return model
    
    @staticmethod
    def _load_surface_points(surface_points_data: List[SurfacePoint], id_to_name: Optional[Dict[int, str]] = None):
        """
        Load surface points from JSON data.
        
        Args:
            surface_points_data (List[SurfacePoint]): List of surface point dictionaries
            id_to_name (Optional[Dict[int, str]]): Optional mapping from surface IDs to names
            
        Returns:
            SurfacePointsTable: A new SurfacePointsTable instance
        """
        # Import here to avoid circular imports
        from gempy.core.data.surface_points import SurfacePointsTable
        
        # Extract coordinates and other data
        x = np.array([sp['x'] for sp in surface_points_data])
        y = np.array([sp['y'] for sp in surface_points_data])
        z = np.array([sp['z'] for sp in surface_points_data])
        nugget = np.array([sp['nugget'] for sp in surface_points_data])
        
        # Handle names based on whether id_to_name mapping is provided
        if id_to_name is not None:
            names = [id_to_name.get(sp['id'], f"surface_{sp['id']}") for sp in surface_points_data]
        else:
            # If no mapping provided, use surface IDs as names
            names = [f"surface_{sp['id']}" for sp in surface_points_data]
        
        # Create SurfacePointsTable
        return SurfacePointsTable.from_arrays(
            x=x,
            y=y,
            z=z,
            names=names,
            nugget=nugget
        )

    @staticmethod
    def _load_orientations(orientations_data: List[Orientation], id_to_name: Optional[Dict[int, str]] = None):
        """
        Load orientations from JSON data.
        
        Args:
            orientations_data (List[Orientation]): List of orientation dictionaries
            id_to_name (Optional[Dict[int, str]]): Optional mapping from surface IDs to names
            
        Returns:
            OrientationsTable: A new OrientationsTable instance
        """
        # Import here to avoid circular imports
        from gempy.core.data.orientations import OrientationsTable
        
        # Extract coordinates and other data
        x = np.array([ori['x'] for ori in orientations_data])
        y = np.array([ori['y'] for ori in orientations_data])
        z = np.array([ori['z'] for ori in orientations_data])
        G_x = np.array([ori['G_x'] for ori in orientations_data])
        G_y = np.array([ori['G_y'] for ori in orientations_data])
        G_z = np.array([ori['G_z'] for ori in orientations_data])
        nugget = np.array([ori['nugget'] for ori in orientations_data])
        
        # Handle names based on whether id_to_name mapping is provided
        if id_to_name is not None:
            names = [id_to_name.get(ori['id'], f"surface_{ori['id']}") for ori in orientations_data]
        else:
            # If no mapping provided, use surface IDs as names
            names = [f"surface_{ori['id']}" for ori in orientations_data]
        
        # Apply polarity to gradients
        for i, ori in enumerate(orientations_data):
            if ori.get('polarity', 1) == -1:
                G_x[i] *= -1
                G_y[i] *= -1
                G_z[i] *= -1
        
        # Create OrientationsTable
        return OrientationsTable.from_arrays(
            x=x,
            y=y,
            z=z,
            G_x=G_x,
            G_y=G_y,
            G_z=G_z,
            names=names,
            nugget=nugget
        )
    
    @staticmethod
    def save_model_to_json(model, file_path: str) -> None:
        """
        Save a GemPy model to a JSON file.
        
        Args:
            model: The GemPy model to save
            file_path (str): Path where to save the JSON file
        """
        # Create JSON structure
        json_data = {
            "metadata": {
                "name": model.meta.name,
                "creation_date": model.meta.creation_date,
                "last_modification_date": model.meta.last_modification_date,
                "owner": model.meta.owner
            },
            "surface_points": [],
            "orientations": [],
            "series": [],
            "grid_settings": {
                "regular_grid_resolution": JsonIO._numpy_to_list(model.grid._dense_grid.resolution),
                "regular_grid_extent": JsonIO._numpy_to_list(model.grid._dense_grid.extent),
                "octree_levels": None  # TODO: Add octree levels if needed
            },
            "interpolation_options": {
                "kernel_options": {
                    "range": float(model.interpolation_options.kernel_options.range),
                    "c_o": float(model.interpolation_options.kernel_options.c_o)
                },
                "mesh_extraction": bool(model.interpolation_options.mesh_extraction),
                "number_octree_levels": int(model.interpolation_options.number_octree_levels)
            },
            "fault_relations": JsonIO._numpy_to_list(model.structural_frame.fault_relations) if hasattr(model.structural_frame, 'fault_relations') else None
        }
        
        # Get series and surface information
        for group in model.structural_frame.structural_groups:
            series_entry = {
                "name": group.name,
                "surfaces": [element.name for element in group.elements],
                "structural_relation": group.structural_relation.name,
                "colors": [element.color for element in group.elements]
            }
            json_data["series"].append(series_entry)
        
        # Get surface points
        surface_points_table = model.surface_points_copy
        xyz_values = surface_points_table.xyz
        ids = surface_points_table.ids
        nugget_values = surface_points_table.nugget
        
        for i in range(len(xyz_values)):
            point = {
                "x": float(xyz_values[i, 0]),
                "y": float(xyz_values[i, 1]),
                "z": float(xyz_values[i, 2]),
                "id": int(ids[i]),
                "nugget": float(nugget_values[i])
            }
            json_data["surface_points"].append(point)
        
        # Get orientations
        orientations_table = model.orientations_copy
        ori_xyz_values = orientations_table.xyz
        ori_grads_values = orientations_table.grads
        ori_ids = orientations_table.ids
        ori_nugget_values = orientations_table.nugget
        
        for i in range(len(ori_xyz_values)):
            orientation = {
                "x": float(ori_xyz_values[i, 0]),
                "y": float(ori_xyz_values[i, 1]),
                "z": float(ori_xyz_values[i, 2]),
                "G_x": float(ori_grads_values[i, 0]),
                "G_y": float(ori_grads_values[i, 1]),
                "G_z": float(ori_grads_values[i, 2]),
                "id": int(ori_ids[i]),
                "nugget": float(ori_nugget_values[i]),
                "polarity": 1  # Default value, update if available
            }
            json_data["orientations"].append(orientation)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    
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
        required_keys = {'metadata', 'surface_points', 'orientations', 'series', 
                        'grid_settings', 'interpolation_options'}
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
                
        # Validate orientations
        if not isinstance(data['orientations'], list):
            return False
            
        for ori in data['orientations']:
            required_ori_keys = {'x', 'y', 'z', 'G_x', 'G_y', 'G_z', 'id', 'nugget', 'polarity'}
            if not all(key in ori for key in required_ori_keys):
                return False
            if not all(isinstance(ori[key], (int, float)) for key in ['x', 'y', 'z', 'G_x', 'G_y', 'G_z', 'nugget']):
                return False
            if not isinstance(ori['id'], int):
                return False
            if not isinstance(ori['polarity'], int) or ori['polarity'] not in {-1, 1}:
                return False
                
        return True 