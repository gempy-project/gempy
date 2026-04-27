import pyvista
import numpy as np

import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_borehole(plotter, vertex_pos, extent_z, radius=8.0, n_segments=40,
                  color="white", opacity=0.25):
    """Add a mocked vertical borehole cylinder passing through *vertex_pos*.

    The borehole spans the full vertical extent of the model so it looks like
    a real well trajectory.
    """
    center = np.array([vertex_pos[0], vertex_pos[1],
                       (extent_z[0] + extent_z[1]) / 2.0])
    height = extent_z[1] - extent_z[0]
    borehole = pyvista.Cylinder(
        center=center,
        direction=(0, 0, 1),
        radius=radius,
        height=height,
        resolution=n_segments,
    )
    plotter.add_mesh(
        borehole,
        color=color,
        opacity=opacity,
        label="Borehole",
    )


def _add_gradient_glyphs(plotter, sp_coords, geo_data, arrow_scale=0.5):
    """Create gradient arrows at every surface-point location."""
    grad = sp_coords.grad.detach().numpy()
    sp_pos = geo_data.surface_points_copy.df[["X", "Y", "Z"]].to_numpy()

    grad_norms = np.linalg.norm(grad, axis=1)

    # Logarithmic scaling so that very different magnitudes are still visible
    log_norms = np.log10(grad_norms + 1e-15)
    lo, hi = log_norms.min(), log_norms.max()
    if hi > lo:
        scaled_mag = (log_norms - lo) / (hi - lo) * 50 + 10
    else:
        scaled_mag = np.full_like(grad_norms, 30)

    arrows_poly = pyvista.PolyData(sp_pos)
    arrows_poly["gradient_norm"] = grad_norms
    grad_dir = grad / (grad_norms[:, np.newaxis] + 1e-15)
    arrows_poly["vectors"] = grad_dir
    arrows_poly["scaled_mag"] = scaled_mag

    glyphs = arrows_poly.glyph(orient="vectors", scale="scaled_mag",
                               factor=arrow_scale)
    plotter.add_mesh(
        glyphs,
        scalars="gradient_norm",
        cmap="plasma",
        scalar_bar_args={"title"          : "‖∇Z‖  (vertex → surface pts)",
                         "title_font_size": 10,
                         "label_font_size": 9, "n_labels": 3,
                         "fmt"            : "%.1e",
                         "position_x"     : 0.75, "position_y": 0.02,
                         "width"          : 0.22, "height": 0.06,
                         "color"          : "black",
                         "vertical"       : False},
        label="Gradient (Z-vertex w.r.t. SP)",
    )


def _style_plotter(plotter, title=""):
    """Apply a clean, talk-friendly style to the plotter."""
    plotter.set_background("white", top="aliceblue")
    plotter.add_text(title, font_size=14, color="black",
                     position="upper_left")
    plotter.add_legend(bcolor=(1, 1, 1, 0.6), border=True,
                       size=(0.18, 0.18))
    plotter.camera.zoom(1.1)


def _highlight_vertex_and_triangles(plotter, geo_data, mesh, vertex_idx):
    """Highlight the triangles sharing *vertex_idx* and the vertex itself."""
    triangles = mesh.edges
    vertices_world = geo_data.input_transform.apply_inverse(mesh.vertices)

    mask = np.any(triangles == vertex_idx, axis=1)
    highlight_faces = triangles[mask]

    z_offset = np.array([0, 0, 2.0])

    if len(highlight_faces) > 0:
        faces_pv = np.column_stack(
            (np.full(len(highlight_faces), 3), highlight_faces)
        ).flatten()
        hmesh = pyvista.PolyData(vertices_world, faces_pv)
        hmesh.points += z_offset
        plotter.add_mesh(
            hmesh,
            color="white",
            style="surface",
            opacity=0.85,
            label=f"Triangles @ vertex {vertex_idx}",
            line_width=4,
            render_lines_as_tubes=True,
            edge_color="black",
            show_edges=True,
        )

    vertex_pos = vertices_world[vertex_idx].reshape(1, 3) + z_offset
    cone_height = 20.0
    tip_pos = vertex_pos.flatten()
    cone_center = tip_pos + np.array([0, 0, cone_height / 2.0])
    marker = pyvista.Cone(
        center=cone_center,
        direction=(0, 0, -1),
        height=cone_height,
        radius=8.0,
        resolution=30,
    )
    plotter.add_mesh(
        marker,
        color="gold",
        label=f"Vertex {vertex_idx}",
    )
    return vertices_world[vertex_idx]


# ---------------------------------------------------------------------------
# Test 1 – Fold model (original, now prettier + borehole)
# ---------------------------------------------------------------------------

def test_generate_fold_model():
    data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name="fold",
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv",
        ),
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ("rock2", "rock1")},
    )

    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype="float64",
            compute_grads=True,
        ),
    )

    # --- Backward pass ---
    vertex_idx = 14
    vertices_tensor = geo_data.solutions.dc_meshes[0].vertices_tensor
    vertices_tensor[vertex_idx, 2].backward(retain_graph=True,
                                            create_graph=True)

    # --- Visualisation ---
    image = True
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords
    p3d = gpv.plot_3d(geo_data, show_surfaces=True, show_data=True,
                      show=False, show_lith=False, image=image,
                      kwargs_plot_surfaces={"opacity": 0.7})
    plotter = p3d.p

    mesh = geo_data.solutions.dc_meshes[0]
    vtx_world = _highlight_vertex_and_triangles(plotter, geo_data, mesh,
                                                vertex_idx)

    # Mocked borehole through the chosen vertex
    _add_borehole(plotter, vtx_world, extent_z=(0, 1000))

    # Gradient arrows
    _add_gradient_glyphs(plotter, sp_coords, geo_data)

    _style_plotter(plotter, title="Fold model – AD gradients")
    if not image:
        plotter.show()


# ---------------------------------------------------------------------------
# Test 2 – Combination model (fold + unconformity + fault)
# ---------------------------------------------------------------------------

def test_generate_combination_model():
    from gempy.API.examples_generator import generate_example_model

    geo_data = generate_example_model(ExampleModel.COMBINATION,
                                      compute_model=False)
    geo_data.interpolation_options.number_octree_levels = 5
    geo_data.interpolation_options.number_octree_levels_surface = 5
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype="float32",
            compute_grads=True,
        ),
    )

    # --- Backward pass ---
    vertex_idx = 1_000
    mesh_id = 2
    vertices_tensor = geo_data.solutions.dc_meshes[mesh_id].vertices_tensor
    vertices_tensor[vertex_idx, 2].backward(retain_graph=True,
                                            create_graph=True)

    # --- Visualisation ---
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords

    image = True
    p3d = gpv.plot_3d(geo_data, show_surfaces=True, show_data=True,
                      show=False, show_lith=False, image=image,
                      kwargs_plot_surfaces={"opacity": 0.7})
    plotter = p3d.p

    mesh = geo_data.solutions.dc_meshes[mesh_id]
    vtx_world = _highlight_vertex_and_triangles(plotter, geo_data, mesh,
                                                vertex_idx)

    # Mocked borehole through the chosen vertex
    extent = geo_data.grid.regular_grid.extent
    _add_borehole(plotter, vtx_world, extent_z=(extent[4], extent[5]))

    # Gradient arrows
    _add_gradient_glyphs(plotter, sp_coords, geo_data)

    _style_plotter(plotter, title="Combination model – AD gradients")
    if not image:
        plotter.show()
