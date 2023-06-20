from gempy.core.data.orientations import Orientations
from gempy.core.data.surface_points import SurfacePoints
from gempy.optional_dependencies import require_pandas


def read_surface_points(path: str,
                        coord_x_name="X",
                        coord_y_name="Y",
                        coord_z_name="Z",
                        surface_name="formation",
                        **pandas_kwargs) -> SurfacePoints:
    pd = require_pandas()
    csv = pd.read_csv(path, **pandas_kwargs)

    if 'sep' not in pandas_kwargs:
        pandas_kwargs['sep'] = ','

    surface_points: SurfacePoints = SurfacePoints.from_arrays(
        x=csv[coord_x_name].values,
        y=csv[coord_y_name].values,
        z=csv[coord_z_name].values,
        id=csv[surface_name].values  # TODO: This we will have to map it with StructuralFrame
    )

    return surface_points


def read_orientations(
        path: str,
        coord_x_name="X",
        coord_y_name="Y",
        coord_z_name="Z",
        gx_name="G_x",
        gy_name="G_y",
        gz_name="G_z",
        surface_name="formation",
        **pandas_kwargs) -> Orientations:
    pd = require_pandas()
    csv = pd.read_csv(path, **pandas_kwargs)

    if 'sep' not in pandas_kwargs:
        pandas_kwargs['sep'] = ','

    orientations: Orientations = Orientations.from_arrays(
        x=csv[coord_x_name].values,
        y=csv[coord_y_name].values,
        z=csv[coord_z_name].values,
        G_x=csv[gx_name].values,
        G_y=csv[gy_name].values,
        G_z=csv[gz_name].values,
        id=csv[surface_name].values  # TODO: This we will have to map it with StructuralFrame
    )

    return orientations
