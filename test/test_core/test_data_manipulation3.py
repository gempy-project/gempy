

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp


def test_add_point():
    extend = [0.0, 1.0, 0.0, 1.0, 0.0, 1.1]
    discretization = [5, 20, 20]

    x, y, z, f = 0.0, 0.0, 0.5, 'surface2'

    # %%
    geo_model = gp.create_model('test')
    gp.init_data(geo_model, extend, discretization)

    geo_model.set_default_surfaces()
    geo_model.set_default_orientation()

    strats = ['surface1', 'surface2', 'basement']

    gp.map_series_to_surfaces(geo_model, {'Strat_Series': strats})

    geo_model.add_surface_points(x, y, z, f)
    geo_model.add_orientations(x, y, z, f, pole_vector=(1,0,0))