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
    vertex_idx = 0
    vertices_tensor = geo_data.solutions.dc_meshes[0].vertices_tensor
    # Backward on the Z-coordinate of the chosen vertex
    vertices_tensor[vertex_idx, 2].backward(retain_graph=True, create_graph=True)

    # --- Visualization ---
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords
    p3d = gpv.plot_3d(geo_data, show_surfaces=True, show_data=True, show=False)
    
    # 1. Highlight the specific triangle(s) associated with the vertex
    mesh = geo_data.solutions.dc_meshes[0]
    vertices = mesh.vertices
    triangles = mesh.edges
    
    # Find triangles that contain the vertex_idx
    # In GemPy dc_meshes, edges are actually the faces (triangles)
    mask_triangles = np.any(triangles == vertex_idx, axis=1)
    highlight_faces = triangles[mask_triangles]
    
    if len(highlight_faces) > 0:
        # Create a pyvista mesh for the highlighted triangles
        # pv.PolyData faces format: [n_points, p1_idx, p2_idx, ..., pn_idx, n_points, ...]
        faces_pv = np.column_stack((np.full(len(highlight_faces), 3), highlight_faces)).flatten()
        highlight_mesh = pyvista.PolyData(vertices, faces_pv)
        p3d.p.add_mesh(highlight_mesh, color='red', label=f'Triangles sharing vertex {vertex_idx}', line_width=5)
    
    # 2. Show the gradient at surface points
    grad = sp_coords.grad.detach().numpy()
    sp_pos = geo_data.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()
    
    # Normalize gradients for visualization if they are too small or too large
    grad_norm = np.linalg.norm(grad, axis=1)
    mask = grad_norm > 1e-10
    
    if np.any(mask):
        grad_filtered = grad[mask]
        sp_pos_filtered = sp_pos[mask]
        
        # Scale arrows for better visibility
        max_grad = grad_norm.max()
        scale_factor = 50 / max_grad if max_grad > 0 else 1
        
        p3d.p.add_arrows(
            cent=sp_pos_filtered, 
            direction=grad_filtered, 
            mag=scale_factor, 
            color='blue', 
            label='Gradient (Z-vertex wrt SP)'
        )

    p3d.p.add_legend()
    p3d.p.show()
