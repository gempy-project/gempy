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
from typing import Any
from gstools import Gaussian, SRF
from copy import deepcopy
import pandas as pd
from abc import ABC, abstractmethod
from gempy.core.model import Model


class _StochasticSurface(ABC):
    """Abstract base class"""
    stochastic_surfaces = {}

    def __init__(self, geo_model: Model, surface:str, grouping: str = "surface"):
        self.__class__.surface_points_init = deepcopy(geo_model.surface_points.df)
        self.__class__.orientations_init = deepcopy(geo_model.orientations.df)
        self.__class__.geo_model = geo_model

        self.surface = surface
        self.stochastic_surfaces[surface] = self

        self.fsurf_bool = geo_model.surface_points.df[grouping] == surface
        self.isurf = geo_model.surface_points.df[self.fsurf_bool].index
        self.forient_bool = geo_model.orientations.df[grouping] == surface
        self.iorient = geo_model.orientations.df[self.forient_bool].index

        self.nsurf = len(self.isurf)
        self.norient = len(self.iorient)

        self.stoch_param = pd.DataFrame(columns=["i", "col", "val"])

    @property
    def surface_points(self) -> pd.DataFrame:
        """Access geomodel surface points."""
        return self.geo_model.surface_points.df.loc[self.isurf]

    @property
    def orientations(self) -> pd.DataFrame:
        """Access geomodel orientations."""
        return self.geo_model.orientations.df.loc[self.iorient]

    @abstractmethod
    def sample(self):
        pass

    def modify(self):
        for col, i in self.stoch_param.groupby("col").groups.items():
            self.geo_model.modify_surface_points(
                self.stoch_param.loc[i, "i"],
                **{
                    col:
                        self.surface_points_init.loc[self.stoch_param.loc[i, "i"], col].values + self.stoch_param.loc[i, "val"].values
                }
            )

    def reset(self) -> None:
        """Reset geomodel parameters."""
        self.geo_model.modify_surface_points(
            self.isurf,
            **{
                "Z":
                    deepcopy(self.surface_points_init.loc[self.isurf, "Z"].values),
                "Y":
                    deepcopy(self.surface_points_init.loc[self.isurf, "Y"].values),
                "X":
                    deepcopy(self.surface_points_init.loc[self.isurf, "X"].values)
            }
        )


class StochasticSurfaceScipy(_StochasticSurface):
    def __init__(self, geo_model: Model,
                 surface: str,
                 grouping: str = "surface"):
        super().__init__(geo_model, surface, grouping=grouping)
        self._i = {"Z": 5, "X": 1, "Y": 3}
        self._extent = self.geo_model.grid.regular_grid.extent
        self.parametrization = None

    def sample(self):
        if not self.parametrization:
            raise AssertionError("No stochastic parametrization found.")

        self.stoch_param = pd.DataFrame(columns=["i", "col", "val"])

        for entry in self.parametrization:
            val = entry[2].rvs()
            for i in entry[0]:
                self.stoch_param = self.stoch_param.append(
                    {'i': i, 'col': entry[1], 'val': val},
                    ignore_index=True
                )


    def parametrize_surfpts_single(self,
                                   stdev: float,
                                   direction: str = "Z") -> None:
        dist = ss.norm(loc=0, scale=stdev)
        self.parametrization = [(self.isurf, direction, dist)]

    def parametrize_surfpts_individual(self,
                                       factor: float = 0.01,
                                       direction: str = "Z") -> None:
        scale = self._extent[self._i[direction.capitalize()]] * factor
        self.parametrization = [
            (i, direction, ss.norm(loc=0, scale=abs(scale)))
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
