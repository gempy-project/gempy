"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


Module with classes for convenient stochastic perturbation of gempy data points
with flexible back-ends.

    _StochasticSurfaceAbstract provides base functionality shared by all classes
    inheriting from it. It provides a set of abstract methods that need to be
    filled in by the subclasses.

Tested on Windows 10

@author: Alexander Schaaf
"""

import scipy.stats as ss
import numpy as np
from nptyping import Array
from gstools import Gaussian, SRF
from copy import deepcopy
import pandas as pd
from abc import ABC, abstractmethod
from gempy.core.model import Model


class _StochasticSurface(ABC):
    """Abstract base class for stochastic surfaces."""
    stochastic_surfaces = {}

    def __init__(self, geo_model: Model, surface:str, grouping: str = "surface"):
        # store independant copy of initial dataframe for reference/resets
        self.__class__.surface_points_init = deepcopy(geo_model.surface_points.df)
        self.__class__.orientations_init = deepcopy(geo_model.orientations.df)
        # store Model instance
        self.__class__.geo_model = geo_model

        self.surface = surface
        # class-spanning list of all instantiated stochastic surfaces for easy
        # access
        self.stochastic_surfaces[surface] = self

        # indices and boolean arrays for easy access to relevant data in Model
        # dataframes
        self.fsurf_bool = geo_model.surface_points.df[grouping] == surface
        self.isurf = geo_model.surface_points.df[self.fsurf_bool].index
        self.forient_bool = geo_model.orientations.df[grouping] == surface
        self.iorient = geo_model.orientations.df[self.forient_bool].index

        # number of surface points and orientations associated with this surface
        self.nsurf = len(self.isurf)
        self.norient = len(self.iorient)

        # sample storage dataframes
        self.surfpts_sample = pd.DataFrame(columns=["i", "col", "val"])
        self.orients_sample = pd.DataFrame(columns=["i", "col", "val"])

    @property
    def surface_points(self) -> pd.DataFrame:
        """Access geomodel surface points."""
        return self.geo_model.surface_points.df.loc[self.isurf]

    @property
    def orientations(self) -> pd.DataFrame:
        """Access geomodel orientations."""
        return self.geo_model.orientations.df.loc[self.iorient]

    @abstractmethod
    def sample(self, random_state=None):
        pass

    def modify_surfpts(self):
        "Inplace modification of interface dataframe."
        for col, i in self.surfpts_sample.groupby("col").groups.items():
            i_init = self.surfpts_sample.loc[i, "i"]
            self.geo_model.modify_surface_points(
                i_init,
                **{
                    col: self.surface_points_init.loc[i_init, col].values \
                         + self.surfpts_sample.loc[i, "val"].values
                }
            )

    def modifiy_orient(self):
        """Inplace modification of orientation dataframe."""
        for col, i in self.orients_sample.groupby("col").groups.items():
            i_init = self.orients_sample.loc[i, "i"]
            self.geo_model.modify_orientations(
                i_init,
                **{
                    col: self.orientations_init.loc[i_init, col].values \
                         + self.orients_sample[i, "val"].values
                }
            )

    def modify(self):
        """Modify geomodel parameters based on sample."""
        self.modify_surfpts()
        self.modifiy_orient()


    def reset(self):
        """Reset geomodel parameters to initial values."""
        self.geo_model.modify_surface_points(
            self.isurf,
            **{
                "Z": deepcopy(self.surface_points_init.loc[self.isurf, "Z"].values),
                "Y": deepcopy(self.surface_points_init.loc[self.isurf, "Y"].values),
                "X": deepcopy(self.surface_points_init.loc[self.isurf, "X"].values)
            }
        )

        self.geo_model.modify_orientations(
            self.iorient,
            **{
                "X": deepcopy(self.orientations_init.loc[self.isurf, "X"].values),
                "Y": deepcopy(self.orientations_init.loc[self.isurf, "Y"].values),
                "Z": deepcopy(self.orientations_init.loc[self.isurf, "Z"].values),
                "G_x": deepcopy(self.orientations_init.loc[self.isurf, "G_x"].values),
                "G_y": deepcopy(self.orientations_init.loc[self.isurf, "G_y"].values),
                "G_z": deepcopy(self.orientations_init.loc[self.isurf, "G_z"].values),
                "dip": deepcopy(self.orientations_init.loc[self.isurf, "dip"].values),
                "azimuth": deepcopy(self.orientations_init.loc[self.isurf, "azimuth"].values),
                "polarity": deepcopy(self.orientations_init.loc[self.isurf, "polarity"].values),
            }
        )

    def recalculate_orients(self):
        pass


class StochasticSurfaceScipy(_StochasticSurface):
    def __init__(self, geo_model: Model,
                 surface: str,
                 grouping: str = "surface"):
        """Easy-to-use class to sample and modify a GemPy surface using
        SciPy-based stochastic parametrization (scipy.stats).

        Args:
            geo_model: GemPy Model object.
            surface (str): Name of the surface.
            grouping (str): Specifies which DataFrame column will be used to
                identify the surface. Default: "surface".

                For example, if another column is specified in the GemPy
                DataFrames to identify subgroups of surfaces (e.g. per fault
                block).
        """
        super().__init__(geo_model, surface, grouping=grouping)
        self._i = {"Z": 5, "X": 1, "Y": 3}
        self._extent = self.geo_model.grid.regular_grid.extent
        self.surfpts_parametrization = None
        self.orients_parametrization = None

    def sample(self, random_state: int=None):
        """Draw a sample from stochastic parametrization of the surface. Stores
        it in self.stoch_param dataframe. Calling self.modifiy() will then
        modifiy the actual GemPy Model dataframe values (init + sample).

        Args:
            random_state (int): Random state to be passed to the distributions.
        """
        if not self.surfpts_parametrization and not self.orients_parametrization:
            raise AssertionError("No parametrization for either surface "
                                 "points or orientations found.")

        if self.surfpts_parametrization:
            self.surfpts_sample = pd.DataFrame(columns=["i", "col", "val"])
            for entry in self.surfpts_parametrization:
                val = entry[2].rvs(random_state=random_state)
                for i in entry[0]:
                    self.surfpts_sample = self.surfpts_sample.append(
                        {'i': i, 'col': entry[1], 'val': val},
                        ignore_index=True
                    )
        if self.orients_parametrization:
            self.orients_sample = pd.DataFrame(columns=["i", "col", "val"])
            for entry in self.orients_parametrization:
                val = entry[2].rvs(random_state=random_state)
                for i in entry[0]:
                    self.orients_sample = self.orients_sample.append(
                        {'i': i, 'col': entry[1], 'val': val},
                        ignore_index=True
                    )

    def parametrize_surfpts_single(self,
                                   stdev: float,
                                   direction: str = "Z"):
        """Naive parametrization, associating the entirety of surface points of
        the surface with a single Normal distribution (μ=0) with given standard
        deviation along given coordinate axis.

        Args:
            stdev (float): Standard deviation of Normal distribution.
            direction (str): Coordinate axis along which to perturbate (X,Y,Z)
                Default: "Z".
        """
        dist = ss.norm(loc=0, scale=stdev)
        self.surfpts_parametrization = [(self.isurf, direction, dist)]

    def parametrize_surfpts_individual(self,
                                       stdev: float,
                                       direction: str = "Z"):
        """Naive parametrization, associating an individual Normal distribution
        (μ=0) with given standard deviation with each individual surface point
        of the surface.

        Args:
            stdev (float): Standard deviation of Normal distributions
            direction (str): Coordinate axis along which to perturbate (X,Y,Z)
                Default: "Z".
        """
        self.surfpts_parametrization = [
            ([i], direction, ss.norm(loc=0, scale=stdev))
            for i in self.isurf
        ]


# class StochasticSurfaceGRF(_StochasticSurface):
#     def __init__(self, geo_model: object, surface: str):
#         super().__init__(geo_model, surface)
#
#     def parametrize_surfpts_naive(self,
#                                   factor: float = 800) -> None:
#         # create a simple GRF across domain to sample from
#         self.Gaussian = Gaussian(dim=2, var=1, len_scale=factor)
#
#     def sample(self, seed=None) -> dict:
#         # draw from GRF at surface point locations
#         srf = SRF(self.Gaussian, seed=seed)
#         pos = self.surface_points[["X", "Y"]].values
#         sample = srf((pos[:, 0], pos[:, 1])) * 30
#         return {"Z": sample}


def _trifacenormals_from_pts(points: Array[float, ..., 3]) -> pd.DataFrame:
    """Robust 2D Delaunay triangulation along main axes of given point cloud.

    Args:
        points(np.ndarray): x,y,z coordinates of points making up the surface.

    Returns:
        pd.DataFrame of face normals compatible with GemPy orientation
        dataframes.
    """
    import pyvista as pv
    from mplstereonet import vector2pole

    pointcloud = pv.PolyData(points)
    trisurf = pointcloud.delaunay_2d()

    simplices = trisurf.faces.reshape((trisurf.n_faces, 4))[:, 1:]
    centroids = trisurf.points[simplices].mean(axis=1)
    normals = trisurf.face_normals

    columns = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
    orients = pd.DataFrame(columns=columns)

    orients["X"], orients["Y"], orients["Z"] = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    orients["G_x"], orients["G_y"], orients["G_z"] = normals[:, 0], normals[:, 1], normals[:, 2]
    orients["azimuth"], orients["dip"] = vector2pole(normals[:, 0], normals[:, 1], normals[:, 2])
    orients["polarity"] = 1

    return orients