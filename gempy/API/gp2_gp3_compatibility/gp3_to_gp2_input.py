from gempy.core.data import GeoModel
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable
from gempy.optional_dependencies import require_gempy_legacy


def gempy3_to_gempy2(geo_model: GeoModel) -> "gempy_legacy.Project":
    gl = require_gempy_legacy()
    legacy_model: "gempy_legacy.Project" = gl.create_model(project_name=geo_model.meta.name)

    surface_points: SurfacePointsTable = geo_model.structural_frame.surface_points
    surface_points_df = surface_points.df  # This is a property
    surface_points_df['surfaces'] = surface_points_df['id'].map(geo_model.structural_frame.element_id_name_map)
    
    orientations: OrientationsTable = geo_model.structural_frame.orientations
    orientations_df = orientations.df
    orientations_df['surfaces'] = orientations_df['id'].map(geo_model.structural_frame.element_id_name_map)
    
    # * Set data
    gl.init_data(
        geo_model=legacy_model,
        extent=geo_model.grid.regular_grid.extent,
        resolution=geo_model.grid.regular_grid.resolution,
        surface_points_df=surface_points_df,
        orientations_df=orientations_df
    )


    # # * Map StructuralFrame
    mapper: dict[str, list[str]] = geo_model.structural_frame.groups_to_mapper
    gl.map_stack_to_surfaces(
        geo_model=legacy_model,
        mapping_object=mapper
    )
    
    legacy_model.add_surfaces("basement")

    
    return legacy_model
    