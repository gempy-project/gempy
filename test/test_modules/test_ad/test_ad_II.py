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
    vertex_idx = 40
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
    
    # 2. Show the gradient at every surface point
    grad = sp_coords.grad.detach().numpy()
    
    # Use world-space positions for the arrow locations
    sp_pos = geo_data.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()
    
    # Scale arrows for better visibility relative to the model extent
    grad_norms = np.linalg.norm(grad, axis=1)
    max_grad = grad_norms.max()
    scale_factor = 100.0 / max_grad if max_grad > 0 else 1.0
    
    p3d.p.add_arrows(
        cent=sp_pos, 
        direction=grad, 
        mag=scale_factor, 
        color='blue', 
        label='Gradient (Z-vertex wrt SP)'
    )

    p3d.p.add_legend()
    p3d.p.show()
