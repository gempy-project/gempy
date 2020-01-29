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

    You should have received a copy of the GNU General Public License along
    with gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes for convenient stochastic perturbation of gempy data
    points with flexible back-ends.

    _StochasticSurfaceAbstract provides base functionality shared by all
    classes inheriting from it. It provides a set of abstract methods that need
    to be filled in by the subclasses.

    Tested on Windows 10

    @author: Alexander Schaaf
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, Sequence, Union

import numpy as np
import pandas as pd
import scipy.stats as ss
from gstools import SRF, Gaussian
from nptyping import Array

from gempy.core.api_modules.data_mutation import modify_surface_points
from gempy.core.model import Model


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


class StochasticModel:
    def __init__(self, geo_model):
        self.geo_model = geo_model

        self.surface_points_init = deepcopy(geo_model.surface_points.df)
        self.orientations_init = deepcopy(geo_model.orientations.df)

        self.priors = {}

    def prior_surface_single(
            self,
            surface:str,
            dist:object,
            column="Z",
            grouping:str="surface",
            name:str=None,  
        ):
        self._create_prior(surface, "surfpts", dist, column, grouping, name)

    def _create_prior(
            self,
            surface:str,
            type_:str,
            dist:object,
            column:str,
            grouping:str,
            name:str=None,
        ):
        if type_.lower() not in ("surfpts"):
            raise NotImplementedError
        name = name if name else f"{surface}_{column}_{type_}"
        self.priors[name] = dict(
            surface=surface,
            type=type_.lower(),
            dist=dist,
            column=column,
            grouping=grouping,
            idx=self.geo_model.surface_points.df[
                self.geo_model.surface_points.df[grouping] == surface
                ].index
        )

    def sample(self) -> Dict[str, float]:
        surfpts_samples, orients_samples = {}, {}
        for name, prior in self.priors.items():
            try:
                sample = prior["dist"].rvs()
            except:
                sample = prior["dist"].resample(1)[0, 0]
            if prior["type"] == "surfpts":
                surfpts_samples[name] = sample
            elif prior["type"] == "orient":
                orients_samples[name] = sample
        return surfpts_samples, orients_samples

    def modify(self, surfpts_samples, orients_samples):
        self._modify_surface_points(surfpts_samples)

    def _modify_surface_points(self, samples:Dict[str, float]):
        samples_df = pd.DataFrame(columns=["i", "col", "val"])
        for name, sample in samples.items():
            prior = self.priors.get(name)
            idx = prior.get("idx")
            samples_df = samples_df.append(
                pd.DataFrame(
                    {
                        "i": idx,
                        "col": prior.get("column"),
                        "val": np.repeat(sample, len(idx))
                    }
                )                
            )

        for col, i in samples_df.groupby("col").groups.items():
            i_init = samples_df.loc[i, "i"]  # get initial indices
            self.geo_model.modify_surface_points(
                i_init,
                **{
                    col: self.surface_points_init.loc[i_init, col].values \
                         + samples_df.loc[i, "val"].values
                }
            )
    
    def _modify_orientations(self, samples:Dict[str, float]):
        raise NotImplementedError

    def reset(self):
        """Reset geomodel parameters to initial values."""
        i = self.surface_points_init.index
        self.geo_model.modify_surface_points(
            i,
            **{
                "X": self.surface_points_init.loc[i, "X"].values,
                "Y": self.surface_points_init.loc[i, "Y"].values,
                "Z": self.surface_points_init.loc[i, "Z"].values
            }
        )

        i = self.orientations_init.index
        self.geo_model.modify_orientations(
            i,
            **{
                "X": self.orientations_init.loc[i, "X"].values,
                "Y": self.orientations_init.loc[i, "Y"].values,
                "Z": self.orientations_init.loc[i, "Z"].values,
                "G_x": self.orientations_init.loc[i, "G_x"].values,
                "G_y": self.orientations_init.loc[i, "G_y"].values,
                "dip": self.orientations_init.loc[i, "dip"].values,
                "G_z": self.orientations_init.loc[i, "G_z"].values,
                "azimuth": self.orientations_init.loc[i, "azimuth"].values,
                "polarity": self.orientations_init.loc[i, "polarity"].values,
            }
        )


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

    orients["X"], orients["Y"], orients["Z"] = tuple(centroids.T)
    orients["G_x"], orients["G_y"], orients["G_z"] = tuple(normals.T)
    orients["azimuth"], orients["dip"] = vector2pole(*tuple(normals.T))
    orients["polarity"] = 1  # TODO actual polarity

    return orients