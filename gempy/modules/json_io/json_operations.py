"""
Module for JSON I/O operations in GemPy.
This module provides functionality to load and save GemPy models to/from JSON files.
"""

import json
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

from .schema import SurfacePoint, Orientation, GemPyModelJson, IdNameMapping
from gempy_engine.core.data.stack_relation_type import StackRelationType


class JsonIO:
    """Class for handling JSON I/O operations for GemPy models."""
    
    @staticmethod
    def _numpy_to_list(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return obj

    @staticmethod
    def _create_id_to_name(name_to_id: Dict[str, int]) -> Dict[int, str]:
        """Create an id_to_name mapping from a name_to_id mapping."""
        return {id: name for name, id in name_to_id.items()}

    @staticmethod
    def _calculate_default_range(grid_extent: List[float]) -> float:
        """Calculate the default range based on the model extent (room diagonal)."""
        # Extract min and max coordinates
        x_min, x_max = grid_extent[0], grid_extent[1]
        y_min, y_max = grid_extent[2], grid_extent[3]
        z_min, z_max = grid_extent[4], grid_extent[5]
        
        # Calculate the room diagonal (Euclidean distance)
        return np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)

    @staticmethod
    def _calculate_default_grid_settings(surface_points: List[SurfacePoint], orientations: List[Orientation]) -> Dict[str, Any]:
        """Calculate default grid settings based on data points.
        
        Args:
            surface_points: List of surface points
            orientations: List of orientations
            
        Returns:
            Dict containing grid settings with default values
        """
        # Collect all x, y, z coordinates
        all_x = [sp['x'] for sp in surface_points] + [ori['x'] for ori in orientations]
        all_y = [sp['y'] for sp in surface_points] + [ori['y'] for ori in orientations]
        all_z = [sp['z'] for sp in surface_points] + [ori['z'] for ori in orientations]
        
        # Calculate extents
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Add 10% padding to each dimension
        x_padding = x_range * 0.1
        y_padding = y_range * 0.1
        z_padding = z_range * 0.1
        
        return {
            "regular_grid_resolution": [20, 20, 20],
            "regular_grid_extent": [
                x_min - x_padding, x_max + x_padding,
                y_min - y_padding, y_max + y_padding,
                z_min - z_padding, z_max + z_padding
            ]
        }

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
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate the JSON data against our schema
        if not JsonIO._validate_json_schema(data):
            raise ValueError("Invalid JSON schema")
            
        # Get unique surface IDs from surface points and orientations
        surface_ids = set(sp['id'] for sp in data['surface_points'])
        surface_ids.update(ori['id'] for ori in data['orientations'])
        
        # Create default series if not provided
        if 'series' not in data:
            data['series'] = [{
                'name': 'Strat_Series',
                'surfaces': [f'surface_{id}' for id in sorted(surface_ids)],
                'structural_relation': 'ERODE'
            }]
            
        # Get surface names from series data
        surface_names = []
        for series in data['series']:
            surface_names.extend(series['surfaces'])
            
        # Create ID to name mapping if not provided
        if 'id_name_mapping' in data:
            id_to_name = JsonIO._create_id_to_name(data['id_name_mapping']['name_to_id'])
        else:
            # Create mapping from series data
            id_to_name = {i: name for i, name in enumerate(surface_names)}
            
        # Create surface points table
        surface_points_table = JsonIO._load_surface_points(data['surface_points'], id_to_name)
        
        # Create orientations table
        orientations_table = JsonIO._load_orientations(data['orientations'], id_to_name)
        
        # Create structural frame
        structural_frame = StructuralFrame.from_data_tables(surface_points_table, orientations_table)
        
        # Get grid settings with defaults if not provided
        grid_settings = data.get('grid_settings', JsonIO._calculate_default_grid_settings(data['surface_points'], data['orientations']))
        
        # Create grid
        grid = Grid(
            resolution=grid_settings['regular_grid_resolution'],
            extent=grid_settings['regular_grid_extent']
        )
        
        # Calculate default range based on model extent
        # default_range = JsonIO._calculate_default_range(grid_settings['regular_grid_extent'])
        # set as fixed value
        default_range = 1.7  # Match standard GemPy default
        
        # Create interpolation options with defaults if not provided
        interpolation_options = InterpolationOptions.from_args(
            range=data.get('interpolation_options', {}).get('kernel_options', {}).get('range', default_range),
            c_o=data.get('interpolation_options', {}).get('kernel_options', {}).get('c_o', 10),
            mesh_extraction=data.get('interpolation_options', {}).get('mesh_extraction', True),
            number_octree_levels=data.get('interpolation_options', {}).get('number_octree_levels', 1)
        )
        
        # Create GeoModel with default metadata if not provided
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_name = data.get('metadata', {}).get('name', "GemPy Model")
        
        model = GeoModel.from_args(
            name=model_name,
            structural_frame=structural_frame,
            grid=grid,
            interpolation_options=interpolation_options
        )
        
        # Set the metadata with proper dates and defaults
        metadata = data.get('metadata', {})
        model_meta = GeoModelMeta(
            name=metadata.get('name', model.meta.name),  # Use model's name if available
            creation_date=metadata.get('creation_date', current_date),  # Set current date as default
            last_modification_date=metadata.get('last_modification_date', current_date),  # Set current date as default
            owner=metadata.get('owner', "GemPy Modeller")  # Set default owner
        )
        model.meta = model_meta
        
        # Map series to surfaces with structural relations
        mapping_object = {series['name']: series['surfaces'] for series in data['series']}
        # Ensure each series has structural_relation set to ERODE by default
        series_data = []
        for series in data['series']:
            series_copy = series.copy()
            if 'structural_relation' not in series_copy:
                series_copy['structural_relation'] = 'ERODE'
            series_data.append(series_copy)
        map_stack_to_surfaces(model, mapping_object, series_data=series_data)
        
        # Set fault relations after structural groups are set up
        if 'fault_relations' in data and data['fault_relations'] is not None:
            fault_relations = np.array(data['fault_relations'])
            if fault_relations.shape == (len(model.structural_frame.structural_groups), len(model.structural_frame.structural_groups)):
                model.structural_frame.fault_relations = fault_relations
        
        # Set colors for each element
        for series in data['series']:
            if 'colors' in series:
                for surface_name, color in zip(series['surfaces'], series['colors']):
                    element = model.structural_frame.get_element_by_name(surface_name)
                    if element is not None:
                        element.color = color
        
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
            
        Raises:
            ValueError: If the data is invalid or missing required fields
        """
        # Import here to avoid circular imports
        from gempy.core.data.surface_points import SurfacePointsTable
        
        # Validate data structure
        required_fields = {'x', 'y', 'z', 'nugget', 'id'}
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
            
        Raises:
            ValueError: If the data is invalid or missing required fields
        """
        # Import here to avoid circular imports
        from gempy.core.data.orientations import OrientationsTable
        
        # Validate data structure
        required_fields = {'x', 'y', 'z', 'G_x', 'G_y', 'G_z', 'nugget', 'polarity', 'id'}
        for i, ori in enumerate(orientations_data):
            missing_fields = required_fields - set(ori.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields in orientation {i}: {missing_fields}")
            
            # Validate data types
            if not all(isinstance(ori[field], (int, float)) for field in ['x', 'y', 'z', 'G_x', 'G_y', 'G_z', 'nugget']):
                raise ValueError(f"Invalid data type in orientation {i}. All coordinates, gradients, and nugget must be numeric.")
            if not isinstance(ori.get('polarity', 1), int) or ori.get('polarity', 1) not in {-1, 1}:
                raise ValueError(f"Invalid polarity in orientation {i}. Must be 1 (normal) or -1 (reverse).")
            if not isinstance(ori['id'], int):
                raise ValueError(f"Invalid data type in orientation {i}. ID must be an integer.")
        
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
        # Get current date for default values
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create JSON structure with metadata handling
        json_data = {
            "metadata": {
                "name": model.meta.name if model.meta.name is not None else "GemPy Model",
                "creation_date": model.meta.creation_date if model.meta.creation_date is not None else current_date,
                "last_modification_date": model.meta.last_modification_date if model.meta.last_modification_date is not None else current_date,
                "owner": model.meta.owner if model.meta.owner is not None else "GemPy Modeller"
            },
            "surface_points": [],
            "orientations": [],
            "series": [],
            "grid_settings": {
                "regular_grid_resolution": [int(x) for x in model.grid._dense_grid.resolution],
                "regular_grid_extent": [float(x) for x in model.grid._dense_grid.extent],
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
            "fault_relations": [[int(x) for x in row] for row in model.structural_frame.fault_relations] if hasattr(model.structural_frame, 'fault_relations') else None,
            "id_name_mapping": {
                "name_to_id": {k: int(v) for k, v in model.structural_frame.element_name_id_map.items()}
            }
        }
        
        # Get series and surface information
        for group in model.structural_frame.structural_groups:
            series_entry = {
                "name": str(group.name),
                "surfaces": [str(element.name) for element in group.elements],
                "structural_relation": str(group.structural_relation.name),
                "colors": [str(element.color) for element in group.elements]
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
    def _validate_json_schema(data: Dict) -> None:
        """Validate the JSON schema and set default values."""
        # Required top-level keys
        required_keys = ['surface_points', 'orientations', 'grid_settings']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Validate surface points
        for point in data['surface_points']:
            required_point_keys = ['x', 'y', 'z']
            for key in required_point_keys:
                if key not in point:
                    raise ValueError(f"Missing required key in surface point: {key}")
            # Set default nugget if not provided
            if 'nugget' not in point:
                point['nugget'] = 0.0  # Default nugget for surface points
            # Set default id if not provided
            if 'id' not in point:
                point['id'] = 0  # Default id

        # Validate orientations
        for orientation in data['orientations']:
            required_orientation_keys = ['x', 'y', 'z', 'G_x', 'G_y', 'G_z']
            for key in required_orientation_keys:
                if key not in orientation:
                    raise ValueError(f"Missing required key in orientation: {key}")
            # Set default nugget if not provided
            if 'nugget' not in orientation:
                orientation['nugget'] = 0.01  # Default nugget for orientations
            # Set default polarity if not provided
            if 'polarity' not in orientation:
                orientation['polarity'] = 1
            # Set default id if not provided
            if 'id' not in orientation:
                orientation['id'] = 0  # Default id

        # Validate grid settings
        grid_settings = data['grid_settings']
        required_grid_keys = ['regular_grid_resolution', 'regular_grid_extent']
        for key in required_grid_keys:
            if key not in grid_settings:
                raise ValueError(f"Missing required key in grid_settings: {key}")

        # Validate series if provided
        if 'series' in data:
            for series in data['series']:
                required_series_keys = ['name', 'surfaces']
                for key in required_series_keys:
                    if key not in series:
                        raise ValueError(f"Missing required key in series: {key}")
                # Set default structural relation if not provided
                if 'structural_relation' not in series:
                    series['structural_relation'] = "ERODE"
                # Validate colors if provided
                if 'colors' in series:
                    if not isinstance(series['colors'], list):
                        raise ValueError("Colors must be a list")
                    if not all(isinstance(color, str) and color.startswith('#') for color in series['colors']):
                        raise ValueError("Colors must be hex color codes starting with #")
                
        # Validate interpolation options if present
        if 'interpolation_options' in data:
            if not isinstance(data['interpolation_options'], dict):
                raise ValueError("Interpolation options must be a dictionary")
            if 'kernel_options' in data['interpolation_options']:
                kernel_options = data['interpolation_options']['kernel_options']
                if not isinstance(kernel_options, dict):
                    raise ValueError("Kernel options must be a dictionary")
                if 'range' in kernel_options and not isinstance(kernel_options['range'], (int, float)):
                    raise ValueError("Kernel range must be a number")
                if 'c_o' in kernel_options and not isinstance(kernel_options['c_o'], (int, float)):
                    raise ValueError("Kernel c_o must be a number")
            if 'mesh_extraction' in data['interpolation_options'] and not isinstance(data['interpolation_options']['mesh_extraction'], bool):
                raise ValueError("Mesh extraction must be a boolean")
            if 'number_octree_levels' in data['interpolation_options'] and not isinstance(data['interpolation_options']['number_octree_levels'], int):
                raise ValueError("Number of octree levels must be an integer")
                
        # Validate and set default metadata
        if 'metadata' not in data:
            data['metadata'] = {}

        metadata = data['metadata']
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Set default values for metadata
        current_date = datetime.now().strftime("%Y-%m-%d")
        if 'name' not in metadata or metadata['name'] is None:
            metadata['name'] = "GemPy Model"
        elif not isinstance(metadata['name'], str):
            raise ValueError("Metadata name must be a string")

        if 'creation_date' not in metadata or metadata['creation_date'] is None:
            metadata['creation_date'] = current_date
        elif not isinstance(metadata['creation_date'], str):
            raise ValueError("Metadata creation_date must be a string")

        if 'last_modification_date' not in metadata or metadata['last_modification_date'] is None:
            metadata['last_modification_date'] = current_date
        elif not isinstance(metadata['last_modification_date'], str):
            raise ValueError("Metadata last_modification_date must be a string")

        if 'owner' not in metadata or metadata['owner'] is None:
            metadata['owner'] = "GemPy Modeller"
        elif not isinstance(metadata['owner'], str):
            raise ValueError("Metadata owner must be a string")

        return True 