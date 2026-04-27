import pyvista
import numpy as np

import gempy as gp
import gempy_viewer as gpv


def test_generate_fold_model():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # Compute the geological model
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64',
            compute_grads=True
        )
    )

    # --- Backward Pass ---
    # We choose a specific vertex to compute the gradient with respect to
    vertex_idx = 14
    vertices_tensor = geo_data.solutions.dc_meshes[0].vertices_tensor
    # Backward on the Z-coordinate of the chosen vertex
    vertices_tensor[vertex_idx, 2].backward(retain_graph=True, create_graph=True)

    # --- Visualization ---
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords
    p3d = gpv.plot_3d(geo_data, show_surfaces=True, show_data=True, show=False, show_lith=False)
    
    # 1. Highlight the specific triangle(s) associated with the vertex
    mesh = geo_data.solutions.dc_meshes[0]
    triangles = mesh.edges
    
    # Transform vertices from normalized (engine) space to world space
    vertices_world = geo_data.input_transform.apply_inverse(mesh.vertices)
    
    # Find triangles that contain the vertex_idx
    mask_triangles = np.any(triangles == vertex_idx, axis=1)
    highlight_faces = triangles[mask_triangles]
    
    if len(highlight_faces) > 0:
        # Create a pyvista mesh for the highlighted triangles in world coordinates
        faces_pv = np.column_stack((np.full(len(highlight_faces), 3), highlight_faces)).flatten()
        highlight_mesh = pyvista.PolyData(vertices_world, faces_pv)
        # Small Z-offset to avoid Z-fighting
        highlight_mesh.points += np.array([0, 0, 2.0]) 
        p3d.p.add_mesh(
            highlight_mesh, 
            color='red', 
            style='surface',
            opacity=1.0,
            label=f'Triangles sharing vertex {vertex_idx}',
            line_width=10, 
            render_lines_as_tubes=True
        )

    # 2. Mark the vertex itself
    vertex_pos = vertices_world[vertex_idx].reshape(1, 3) + np.array([0, 0, 2.0])
    p3d.p.add_mesh(
        pyvista.PolyData(vertex_pos),
        color='yellow',
        point_size=20,
        render_points_as_spheres=True,
        label=f'Vertex {vertex_idx}'
    )
    
    # 3. Show the gradient at every surface point
    grad = sp_coords.grad.detach().numpy()
    
    # Use world-space positions for the arrow locations
    sp_pos = geo_data.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()
    
    # Scale arrows for better visibility relative to the model extent
    grad_norms = np.linalg.norm(grad, axis=1)
    
    # Use logarithmic scaling for magnitudes to handle large ranges
    # We use log10(norm + epsilon) to avoid log(0)
    log_grad_norms = np.log10(grad_norms + 1e-15)
    # Normalize log norms to a reasonable range for arrow sizes
    min_log = log_grad_norms.min()
    max_log = log_grad_norms.max()
    if max_log > min_log:
        scaled_mag = (log_grad_norms - min_log) / (max_log - min_log) * 50 + 10
    else:
        scaled_mag = np.full_like(grad_norms, 30)

    # Create arrows with color mapping
    # We need to create a PolyData object for the arrows to use scalars for coloring
    arrows_poly = pyvista.PolyData(sp_pos)
    arrows_poly['gradient_norm'] = grad_norms
    
    # Normalize directions for arrows (magnitudes are handled by 'mag' parameter or glyph)
    grad_dir = grad / (grad_norms[:, np.newaxis] + 1e-15)
    arrows_poly['vectors'] = grad_dir
    
    # Create glyphs (arrows)
    glyphs = arrows_poly.glyph(orient='vectors', scale=False, factor=1.0)
    
    # Scale each arrow glyph according to its corresponding scaled_mag
    # Since we can't easily scale points of individual glyphs in one go without a loop 
    # if they are already combined into one PolyData, we can use the 'scale' parameter in glyph() 
    # by adding the scaled magnitudes as a scalar array to the arrows_poly.
    arrows_poly['scaled_mag'] = scaled_mag
    glyphs = arrows_poly.glyph(orient='vectors', scale='scaled_mag', factor=1.0)
    
    p3d.p.add_mesh(
        glyphs, 
        scalars='gradient_norm',
        cmap='plasma',
        scalar_bar_args={'title': 'Gradient Norm'},
        label='Gradient (Z-vertex wrt SP)'
    )

    p3d.p.add_legend()
    p3d.p.show()
