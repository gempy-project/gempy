"""
This scrip is to add test for specific issues that the community has found. Use the the following template:

def test_issue_XXX():
    ...

"""
import numpy as np
import pytest

import gempy as gp
from gempy.core.data.importer_helper import ImporterHelper
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable
from gempy.optional_dependencies import require_pandas


def test_issue_904_misspelled_formation_in_orientations_file(tmp_path):
    """A formation name present in the orientations file but not in the surface points file
    used to raise a bare ``KeyError`` with just the name. It must explain what is wrong."""
    require_pandas()
    surface_points_path = tmp_path / "surface_points.csv"
    surface_points_path.write_text(
        "X,Y,Z,formation\n"
        "0,0,0,Sandstone\n10,0,0,Sandstone\n0,10,0,Sandstone\n"
        "0,0,-10,Shale\n10,0,-10,Shale\n0,10,-10,Shale\n"
    )
    orientations_path = tmp_path / "orientations.csv"
    orientations_path.write_text(
        "X,Y,Z,azimuth,dip,polarity,formation\n"
        "5,5,0,90,10,1,Sandstone\n"
        "5,5,-10,90,10,1,Shael\n"
    )

    with pytest.raises(ValueError, match=r"\['Shael'\].*Available names: \['Sandstone', 'Shale'\]"):
        gp.create_geomodel(
            project_name="issue_904",
            extent=[0, 10, 0, 10, -10, 0],
            refinement=1,
            importer_helper=ImporterHelper(
                path_to_surface_points=str(surface_points_path),
                path_to_orientations=str(orientations_path),
            ),
        )


def test_issue_904_unknown_name_in_from_arrays():
    name_id_map = {"Sandstone": 1, "Shale": 2}
    coords = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match=r"\['Shael', 'Basement'\]"):
        SurfacePointsTable.from_arrays(
            x=coords, y=coords, z=coords,
            names=np.array(["Sandstone", "Shael", "Basement"]),
            name_id_map=name_id_map,
        )

    with pytest.raises(ValueError, match=r"\['Shael'\]"):
        OrientationsTable.from_arrays(
            x=coords, y=coords, z=coords,
            G_x=coords, G_y=coords, G_z=coords,
            names=["Sandstone", "Shael", "Shael"],
            name_id_map=name_id_map,
        )

    surface_points = SurfacePointsTable.from_arrays(
        x=coords, y=coords, z=coords,
        names=["Shale", "Sandstone", "Shale"],
        name_id_map=name_id_map,
    )
    np.testing.assert_array_equal(surface_points.data["id"], [2, 1, 2])
