"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    gempy is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to visualized structural geology data and
    potential fields of the regional modelling based on the potential field
    method. Tested on Ubuntu 18, MacOS 12

    Created on 16.10.2019

    @author: Miguel de la Varga, Bane Sullivan, Alexander Schaaf
"""

import warnings
# insys.path.append("../../pyvista")
from copy import deepcopy
from typing import Union, Dict, List, Iterable, Set, Tuple

import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pn
import pyvista as pv
from pyvista.plotting.theme import parse_color

import gempy as gp

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False

from nptyping import Array
from logging import debug


class Vista:
    def __init__(
            self,
            model: gp.Model,
            extent: List[float] = None,
            color_lot: pn.DataFrame = None,
            real_time: bool = False,
            plotter_type: Union["basic", "background"] = 'basic',
            **kwargs
    ):
        """GemPy 3-D visualization using pyVista.
        
        Args:
            model (gp.Model): Geomodel instance with solutions.
            extent (List[float], optional): Custom extent. Defaults to None.
            color_lot (pn.DataFrame, optional): Custom color scheme in the form
                of a look-up table. Defaults to None.
            real_time (bool, optional): Toggles real-time updating of the plot. 
                Defaults to False.
            plotter_type (Union["basic", "background"], optional): Set the 
                plotter type. Defaults to 'basic'.
        """
        self.model = model
        if extent:
            self.extent = list(extent)
        else:
            self.extent = list(model.grid.regular_grid.extent)

        if color_lot:
            self._color_lot = color_lot
        else:
            self._color_lot = model.surfaces.df.set_index('surface')['color']
        self._color_id_lot = model.surfaces.df.set_index('id')['color']

        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
        elif plotter_type == 'background':
            self.p = pv.BackgroundPlotter(**kwargs)

        self._surface_actors = {}
        self._surface_points_actors = {}
        self._orientations_actors = {}

        self._actors = []
        self._live_updating = False

        self.topo_edges = None
        self.topo_ctrs = None

    def show(self):
        self.p.show()

    def _actor_exists(self, new_actor):
        if not hasattr(new_actor, "points"):
            return False
        for actor in self._actors:
            actor_str = actor.points.tostring()
            if new_actor.points.tostring() == actor_str:
                debug("Actor already exists.")
                return True
        return False

    def set_bounds(
            self,
            extent: list = None,
            grid: bool = False,
            location: str = 'furthest',
            **kwargs
    ):
        """Set and toggle display of bounds of geomodel.
        
        Args:
            extent (list, optional): [description]. Defaults to None.
            grid (bool, optional): [description]. Defaults to False.
            location (str, optional): [description]. Defaults to 'furthest'.
        """
        if extent is None:
            extent = self.extent
        self.p.show_bounds(
            bounds=extent, location=location, grid=grid, **kwargs
        )

    def plot_surface_points(self, fmt: str, **kwargs):
        i = self.model.surface_points.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        mesh = pv.PolyData(
            self.model.surface_points.df.loc[i][["X", "Y", "Z"]].values
        )
        if self._actor_exists(mesh):
            return []

        self.p.add_mesh(
            mesh,
            color=self._color_lot[fmt],
            **kwargs
        )
        self._actors.append(mesh)
        return [mesh]

    def plot_orientations(self, fmt: str, length: float = None, **kwargs):
        meshes = []
        i = self.model.orientations.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return meshes
        if not length:
            length = abs(
                np.min(
                    [
                        np.diff(self.extent[:2]),
                        np.diff(self.extent[2:4]),
                        np.diff(self.extent[4:])
                    ]
                )
            ) / 10

        pts = self.model.orientations.df.loc[i][["X", "Y", "Z"]].values
        nrms = self.model.orientations.df.loc[i][["G_x", "G_y", "G_z"]].values

        line_kwargs = dict(
            color=self._color_lot[fmt],
            line_width=3,
        )
        line_kwargs.update(kwargs)

        for pt, nrm in zip(pts, nrms):
            mesh = pv.Line(
                pointa=pt,
                pointb=pt + length * nrm,
            )
            if self._actor_exists(mesh):
                continue
            self.p.add_mesh(
                mesh,
                **line_kwargs
            )
            self._actors.append(mesh)
            meshes.append(mesh)
        return meshes

    def plot_surface_points_all(self, **kwargs):
        meshes = []
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            new_meshes = self.plot_surface_points(fmt, **kwargs)
            for mesh in new_meshes:
                if mesh is not None:
                    meshes.append(mesh)
        return meshes

    def plot_orientations_all(self, **kwargs):
        meshes = []
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            orient_meshes = self.plot_orientations(fmt, **kwargs)
            for orient_mesh in orient_meshes:
                if orient_mesh is not None:
                    meshes.append(orient_mesh)
        return meshes

    def get_surface(self, fmt: str) -> pv.PolyData:
        i = np.where(self.model.surfaces.df.surface == fmt)[0][0]
        ver = self.model.solutions.vertices[
            i]  # TODO: BUG surfaces within series are flipped in order !!!!!!!

        sim = self._simplices_to_pv_tri_simplices(
            self.model.solutions.edges[i]
        )
        mesh = pv.PolyData(ver, sim)
        return mesh

    def plot_surface(self, fmt: str, **kwargs):
        mesh = self.get_surface(fmt)
        if self._actor_exists(mesh):
            return []

        mesh_kwargs = dict(color=self._color_lot[fmt])
        mesh_kwargs.update(kwargs)

        self.p.add_mesh(mesh, **mesh_kwargs)
        self._actors.append(mesh)
        self._surface_actors[fmt] = mesh
        return [mesh]

    def clip_horizon_with_faults(
            self,
            horizon: pv.PolyData,
            faults: Iterable[pv.PolyData],
            value: float = None
    ) -> List[pv.PolyData]:
        """Clip given horizon surface with given list of fault surfaces. The
        given value represents the distance to clip away from the fault 
        surfaces.
        
        Args:
            horizon (pv.PolyData): The horizon surface to be clipped.
            faults (Iterable[pv.PolyData]): Fault(s) surface(s) to clip with.
            value (float, optional): Set the clipping value of the implicit 
                function (clipping distance from faults). Defaults to 50.
        
        Returns:
            List[pv.PolyData]: Individual clipped horizon surfaces.
        """
        if hasattr(faults, "next"):
            if type(faults[0]) == str:
                faults = [self.get_surface(f) for f in faults]

        horizons = []
        if not value:
            value = np.mean(self.model.grid.regular_grid.get_dx_dy_dz()[:2])

        # TODO: this somehow doesn't work properly with Gullfaks model
        horizons.append(
            horizon.clip_surface(faults[0], value=-value)
        )

        horizons.append(
            horizon.clip_surface(faults[-1], invert=False, value=-value)
        )

        if len(faults) == 1:
            print("Returning after 1")
            return horizons

        for f1, f2 in zip(faults[:-1], faults[1:]):
            horizons.append(
                horizon.clip_surface(
                    f1, invert=False, value=value
                ).clip_surface(
                    f2, value=-value
                )
            )

        return horizons

    def plot_surfaces_all(self, fmts: Iterable[str] = None, **kwargs):
        """Plot all geomodel surfaces. If given an iterable containing surface
        strings, it will plot all surfaces specified in it.
        
        Args:
            fmts (List[str], optional): Names of surfaces to plot. 
                Defaults to None.
        """
        meshes = []
        if not fmts:
            fmts = self.model.surfaces.df.surface[:-1].values
        for fmt in fmts:
            m = self.plot_surface(fmt, **kwargs)
            for mesh in m:
                meshes.append(mesh)
        return meshes

    @staticmethod
    def _simplices_to_pv_tri_simplices(sim: Array[int, ..., 3]) -> Array[
        int, ..., 4]:
        """Convert triangle simplices (n, 3) to pyvista-compatible
        simplices (n, 4)."""
        n_edges = np.ones(sim.shape[0]) * 3
        return np.append(n_edges[:, None], sim, axis=1)

    def plot_structured_grid(self, name: str, series: str = None, render_topography: bool = False,
                             **kwargs) -> list:
        """Plot a structured grid of the geomodel.

        Args:
            name (str): Can be either one of the following

                'lith' - Lithology id block.
                'scalar' - Scalar field block.
                'values' - Values matrix block.
        """
        regular_grid = self.model.grid.regular_grid

        grid_values = regular_grid.values
        grid_3d = grid_values.reshape(*regular_grid.resolution, 3).T
        mesh = pv.StructuredGrid(*grid_3d)

        if name == "lith":
            vals = self.model.solutions.lith_block.copy()
            n_faults = self.model.faults.df['isFault'].sum()
            cmap = mcolors.ListedColormap(list(self._color_id_lot[n_faults:]))
            kwargs['cmap'] = kwargs.get('cmap', cmap)
        elif name == "scalar":
            if series == None:
                # default to oldest series above basement
                series = self.model.series.df.iloc[-2].name
            vals = self.model.solutions.scalar_field_matrix.copy()[
                self.model.series.df.index.get_loc(series)]
        elif name == "values":
            vals = self.model.solutions.values_matrix.copy().T
            if vals.shape[1] == 0:
                print("No scalar values matrix found in given geomodel.")
                return

        mesh.point_arrays[name] = vals

        if render_topography == True:
            mesh[name][regular_grid.mask_topo.T.ravel(order='F')] = -100
            mesh = mesh.threshold(-99)

        if self._actor_exists(mesh):
            return []
        self._actors.append(mesh)
        self.p.add_mesh(mesh, **kwargs)
        return [mesh]


    def plot_structured_grid_interactive(
            self,
            name: str,
            render_topography: bool = False,
            **kwargs,
    ):
        """Plot interactive 3-D geomodel with three cross sections in subplot.

        Args:
            geo_model: Geomodel object with solutions.
            name (str): Can be either one of the following
                'lith' - Lithology id block.
                'scalar' - Scalar field block.
                'values' - Values matrix block.
            render_topography: Render topography. Defaults to False.
            **kwargs:

        Returns:
            (Vista) GemPy Vista object for plotting.
        """
        mesh = self.plot_structured_grid(name=name, render_topography=render_topography, **kwargs)[0]

        # define colormaps
        if name == "lith":
            cmap = mcolors.ListedColormap(list(self._color_id_lot[self.model.series.faults.n_faults:]))
        elif name == "scalar":
            cmap = cm.viridis

        # callback functions for subplots
        def xcallback(normal, origin):
            self.p.subplot(1)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin), name="xslc", cmap=cmap)

        def ycallback(normal, origin):
            self.p.subplot(2)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin), name="yslc", cmap=cmap)

        def zcallback(normal, origin):
            self.p.subplot(3)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin), name="zslc", cmap=cmap)

        # cross section widgets
        self.p.subplot(0)
        self.p.add_plane_widget(xcallback, normal="x")
        self.p.subplot(0)
        self.p.add_plane_widget(ycallback, normal="y")
        self.p.subplot(0)
        self.p.add_plane_widget(zcallback, normal="z")

        # Lock other three views in place
        self.p.subplot(1)
        self.p.view_yz()
        self.p.disable()
        self.p.subplot(2)
        self.p.view_xz()
        self.p.disable()
        self.p.subplot(3)
        self.p.view_xy()
        self.p.disable()


    def _callback_surface_points(self, pos, index, widget):
        i = index
        x, y, z = pos

        self.model.modify_surface_points(i, X=x, Y=y, Z=z)

        if self._live_updating:
            self._recompute()
            self._update_surface_polydata()

    def _callback_orientations(self, normal, loc, widget):
        i = widget.WIDGET_INDEX
        x, y, z = loc
        gx, gy, gz = normal

        self.model.modify_orientations(
            i,
            X=x, Y=y, Z=z,
            G_x=gx, G_y=gy, G_z=gz
        )

        if self._live_updating:
            self._recompute()
            self._update_surface_polydata()

    def _recompute(self, **kwargs):
        gp.compute_model(self.model, compute_mesh=True, **kwargs)
        # self.topo_edges, self.topo_ctrs = tp.topology.compute_topology(
        #     self.model
        # )q

    def _update_surface_polydata(self):
        surfaces = self.model.surfaces.df
        for surf, (idx, val) in zip(
                surfaces.surface,
                surfaces[['vertices', 'edges']].dropna().iterrows()
        ):
            polydata = self._surface_actors.get(surf, False)
            if polydata:
                polydata.points = val["vertices"]
                polydata.faces = np.insert(
                    val['edges'], 0, 3, axis=1
                ).ravel()
                self._surface_actors[surf] = polydata

    def plot_surface_points_interactive(self, fmt: str, **kwargs):
        self._live_updating = True
        i = self.model.surface_points.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        pts = self.model.surface_points.df.loc[i][["X", "Y", "Z"]].values

        self.p.add_sphere_widget(
            self._callback_surface_points,
            center=pts,
            radius=np.mean(self.extent) / 20,
            color=self._color_lot[fmt],
            indices=i,
            test_callback=False,
            phi_resolution=6,
            theta_resolution=6,
            pass_widget=True,
            **kwargs
        )

    def plot_surface_points_interactive_all(self, **kwargs):
        self._live_updating = True
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_surface_points_interactive(fmt, **kwargs)

    def plot_orientations_interactive(self, fmt: str):
        self._live_updating = True
        i = self.model.orientations.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        pts = self.model.orientations.df.loc[i][["X", "Y", "Z"]].values
        nrms = self.model.orientations.df.loc[i][["G_x", "G_y", "G_z"]].values

        for index, pt, nrm in zip(i, pts, nrms):
            widget = self.p.add_plane_widget(
                self._callback_orientations,
                normal=nrm,
                origin=pt,
                bounds=self.extent,
                factor=0.15,
                implicit=False,
                pass_widget=True,
                test_callback=False,
                color=self._color_lot[fmt]
            )
            widget.WIDGET_INDEX = index

    def plot_orientations_interactive_all(self):
        self._live_updating = True
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_orientations_interactive(fmt)

    def _scale_topology_centroids(
            self,
            centroids: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Scale topology centroid coordinates from grid coordinates to 
        physical coordinates.
        
        Args:
            centroids (Dict[int, Array[float, 3]]): Centroid dictionary.
        
        Returns:
            Dict[int, Array[float, 3]]: Rescaled centroid dictionary.
        """
        res = self.model.grid.regular_grid.resolution
        scaling = np.diff(self.extent)[::2] / res

        scaled_centroids = {}
        for n, pos in centroids.items():
            pos_scaled = pos * scaling
            pos_scaled[0] += np.min(self.extent[:2])
            pos_scaled[1] += np.min(self.extent[2:4])
            pos_scaled[2] += np.min(self.extent[4:])
            scaled_centroids[n] = pos_scaled

        return scaled_centroids

    def plot_topology(
            self,
            edges: Set[Tuple[int, int]],
            centroids: Dict[int, np.ndarray],
            node_kwargs: dict = {},
            edge_kwargs: dict = {}
    ):
        """Plot geomodel topology graph based on given set of topology edges 
        and node centroids.
        
        Args:
            edges (Set[Tuple[int, int]]): Topology edges.
            centroids (Dict[int, Array[float, 3]]): Topology node centroids
            node_kwargs (dict, optional): Node plotting options. Defaults to {}.
            edge_kwargs (dict, optional): Edge plotting options. Defaults to {}.
        """
        lot = gp.assets.topology.get_lot_node_to_lith_id(self.model, centroids)
        centroids_scaled = self._scale_topology_centroids(centroids)

        for node, pos in centroids_scaled.items():
            mesh = pv.Sphere(
                center=pos,
                radius=np.average(self.extent) / 15
            )
            # * Requires topo id to lith id lot
            self.p.add_mesh(
                mesh,
                color=self._color_id_lot[lot[node]],
                **node_kwargs
            )

        ekwargs = dict(
            line_width=3
        )
        ekwargs.update(edge_kwargs)

        for e1, e2 in edges:
            pos1, pos2 = centroids_scaled[e1], centroids_scaled[e2]

            x1, y1, z1 = pos1
            x2, y2, z2 = pos2
            x, y, z = (x1, x2), (y1, y2), (z1, z2)
            pos_mid = (
                min(x) + (max(x) - min(x)) / 2,
                min(y) + (max(y) - min(y)) / 2,
                min(z) + (max(z) - min(z)) / 2
            )
            mesh = pv.Line(
                pointa=pos1,
                pointb=pos_mid,
            )
            self.p.add_mesh(mesh, color=self._color_id_lot[lot[e1]], **ekwargs)

            mesh = pv.Line(
                pointa=pos_mid,
                pointb=pos2,
            )
            self.p.add_mesh(mesh, color=self._color_id_lot[lot[e2]], **ekwargs)

    def plot_topography(
            self,
            topography = None,
            scalars="geomap",
            **kwargs
    ):
        if not topography:
            try:
                topography = self.model.grid.topography.values
            except AttributeError:
                print("Unable to plot topography: Given geomodel instance "
                      "does not contain topography grid.")
                return

        polydata = pv.PolyData(topography)

        rgb = False
        if scalars == "geomap":
            arr_ = np.empty((0, 3), dtype=int)
            # convert hex colors to rgb
            for val in list(self._color_lot):
                rgb = (255 * np.array(mcolors.hex2color(val)))
                arr_ = np.vstack((arr_, rgb))

            sel = np.round(self.model.solutions.geological_map).astype(int)[0]

            scalars_val = numpy_to_vtk(arr_[sel - 1], array_type=3)
            cm = None
            rgb = True
        elif scalars == "topography":
            scalars_val = topography[:, 2]
            cm = 'terrain'
        elif type(scalars) is np.ndarray:
            scalars_val = scalars
            scalars = 'custom'
            cm = 'terrain'
        else:
            raise AttributeError("Parameter scalars needs to be either \
                'geomap', 'topography' or a np.ndarray with scalar values")

        topography_actor = self.p.add_mesh(
            polydata.delaunay_2d(),
            scalars=scalars_val,
            cmap=cm,
            rgb=rgb,
            **kwargs
        )
        self._surface_actors["topography"] = topography_actor
        return topography_actor


    def plot_scalar_surfaces_3D(self, surfaces_nr: int = 10):
        """Plot scalar field as surfaces

        Args:
            surfaces_nr: Number of plotted scalar field isosurfaces

        Returns:

        """
        regular_grid = self.model.grid.regular_grid

        grid_values = regular_grid.values
        grid_3d = grid_values.reshape(*regular_grid.resolution, 3).T
        mesh = pv.StructuredGrid(*grid_3d)

        values = self.model.solutions.scalar_field_matrix.reshape(self.model.grid.regular_grid.resolution)
        mesh["vol"] = values.flatten()
        contours = mesh.contour(np.linspace(values.min(), values.max(), surfaces_nr + 2))
        self.p.add_mesh(contours, show_scalar_bar=True, label="scalar_field_main")

