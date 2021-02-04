import gempy as gp



def test_issue_566():
    from pyvista import set_plot_theme
    set_plot_theme('document')


    geo_model = gp.create_model('Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 791, 0, 200, -582, 0],
                             resolution=[100, 10, 100])

    geo_model.set_default_surfaces()
    geo_model.add_surface_points(X=223, Y=0.01, Z=-94, surface='surface1')

