import numpy as np

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
    csv_standardized = _standardize(csv)
    csv_with_gradient = _add_gradient_columns(csv_standardized)

    if 'sep' not in pandas_kwargs:
        pandas_kwargs['sep'] = ','

    orientations: Orientations = Orientations.from_arrays(
        x=csv_with_gradient[coord_x_name].values,
        y=csv_with_gradient[coord_y_name].values,
        z=csv_with_gradient[coord_z_name].values,
        G_x=csv_with_gradient[gx_name].values,
        G_y=csv_with_gradient[gy_name].values,
        G_z=csv_with_gradient[gz_name].values,
        id=csv_with_gradient[surface_name].values  # TODO: This we will have to map it with StructuralFrame
    )

    return orientations


COLUMN_NAME_MAPPING = {
    "X"        : ["X", "x"],
    "Y"        : ["Y", "y"],
    "Z"        : ["Z", "z"],
    "azimuth"  : ["azimuth", "Azimuth"],
    "dip"      : ["dip", "Dip"],
    "polarity" : ["polarity", "Polarity"],
    "formation": ["formation", "Formation", "surface"],
    "G_x"      : ["G_x", "gradient_x"],
    "G_y"      : ["G_y", "gradient_y"],
    "G_z"      : ["G_z", "gradient_z"],
}


def _standardize(df: 'pandas.DataFrame'):
    for column in df.columns:
        for standard_name, possible_names in COLUMN_NAME_MAPPING.items():
            if column in possible_names:
                df.rename(columns={column: standard_name}, inplace=True)

    return df


def _add_gradient_columns(df):
    if "azimuth" in df.columns and "dip" in df.columns and "polarity" in df.columns:
        # Convert azimuth, dip, polarity to gradient
        df['G_x'] = np.sin(np.deg2rad(df['dip'])) * np.sin(np.deg2rad(df['azimuth'])) * df['polarity']
        df['G_y'] = np.sin(np.deg2rad(df['dip'])) * np.cos(np.deg2rad(df['azimuth'])) * df['polarity']
        df['G_z'] = -np.cos(np.deg2rad(df['dip'])) * df['polarity']

    return df
