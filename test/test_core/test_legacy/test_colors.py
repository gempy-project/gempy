import gempy as gp
import numpy as np


def test_colors_101_surfaces():
    """Tests if GemPy Colors class works with at least 101 surfaces."""
    geomodel = gp.create_model("ColorfulModel")
    for n in range(101):
        geomodel.add_surfaces(f"Surface {n}")
    assert np.all(geomodel.surfaces.df.color.values != np.nan)