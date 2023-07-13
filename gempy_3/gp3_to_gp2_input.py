from gempy import GeoModel
from gempy.optional_dependencies import require_gempy_legacy


def gempy3_to_gempy2(geo_model: GeoModel) -> "gempy_legacy.Project":
    gl = require_gempy_legacy()
    legacy_model: "gempy_legacy.Project" = gl.create_model(project_name=geo_model.meta.name)
    
    # * Set data
    gl.init_data(
        geo_model=legacy_model,
        extent=geo_model.grid.extent,
        resolution=geo_model.grid.resolution,
        surface_points_df=
        orientations_df=
    )
    
    
    # * Map StructuralFrame
    gl.map_stack_to_surfaces(
        geo_model=legacy_model,
        mapping_object=
    )
    
    
    return legacy_model
    