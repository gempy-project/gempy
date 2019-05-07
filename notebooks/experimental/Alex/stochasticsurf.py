import scipy.stats as ss
import numpy as np
from nptyping import Array
from copy import deepcopy
import pandas as pd
from abc import ABC, abstractmethod


class _StochasticSurfaceAbstract(ABC):
    # TODO: Store samples
    """Abstract StochasticSurface class that contains functionality independant
    from individual parametrization or sampling libraries."""
    stochastic_surfaces = {}

    def __init__(self, geo_model: object, surface: str):
        # class attributes, shared across all instances to allow access to the
        # original parametrization and the geo_model instance.
        self.__class__.surface_points_init = deepcopy(
            geo_model.surface_points.df)
        self.__class__.orientations_init = deepcopy(geo_model.orientations.df)
        self.__class__.geo_model = geo_model

        # instance attributes
        self.surface = surface
        self.stochastic_surfaces[surface] = self

        self.fsurf_bool = geo_model.surface_points.df.surface == surface
        self.isurf = geo_model.surface_points.df[self.fsurf_bool].index
        self.forient_bool = geo_model.orientations.df.surface == surface
        self.iorient = geo_model.orientations.df[self.forient_bool].index

        self.nsurf = len(self.isurf)
        self.norient = len(self.iorient)

        self.stoch_param = {}

    @property
    def surface_points(self) -> pd.DataFrame:
        """Access geomodel surface points."""
        return self.geo_model.surface_points.df.loc[self.isurf]

    @property
    def orientations(self) -> pd.DataFrame:
        """Access geomodel orientations."""
        return self.geo_model.orientations.df.loc[self.iorient]

    @abstractmethod
    def parametrize_surfpts_naive(self, factor: float = 0.01) -> None:
        pass

    @abstractmethod
    def draw_surfpts(self) -> Array:
        pass

    @classmethod
    def modify_surface_points_all(self) -> None:
        for stochastic_surface in self.stochastic_surfaces.values():
            stochastic_surface.modify_surface_points()

    def modify_surface_points(self) -> None:
        """Modify geomodel dataframe surface point Z-values."""
        sample = self.draw_surfpts()
        for key, value in sample.items():
            self.geo_model.modify_surface_points(
                self.isurf, **{
                    key:
                    self.surface_points_init.loc[self.isurf, key].values +
                    value
                })

    def reset(self) -> None:
        """Reset geomodel parameters."""
        self.geo_model.modify_surface_points(
            self.isurf, **{
                "Z":
                deepcopy(self.surface_points_init.loc[self.isurf, "Z"].values),
                "Y":
                deepcopy(self.surface_points_init.loc[self.isurf, "Y"].values),
                "X":
                deepcopy(self.surface_points_init.loc[self.isurf, "X"].values)
            })


class StochasticSurfaceScipy(_StochasticSurfaceAbstract):
    """StochasticSurface subclass providing stochastic parametrization via the
    scipy.stats ecosystem."""

    def __init__(self, geo_model: object, surface: str):
        super().__init__(geo_model, surface)

    def parametrize_surfpts_naive(self,
                                  factor: float = 0.01,
                                  direction: str = "Z") -> None:
        """Naive stochastic parametrization of surface point Z-values.

        Args:
            factor (float, optional): Scaling factor for uncertainty based on
                maximum z-extent. Defaults to 0.01.
        """
        direction = direction.capitalize()
        i = {"Z": 5, "X": 1, "Y": 3}
        scale = self.geo_model.grid.extent[i[direction]] * factor

        params = [
            ss.norm(loc=0, scale=scale)
            for param in self.surface_points.loc[self.isurf, direction]
        ]

        self.stoch_param.update({direction: params})

    def draw_surfpts(self) -> Array:
        """Random draw from stochastic parametrization."""
        if not self.stoch_param:
            raise AttributeError("No stochastic parametrization present.")

        # TODO: add None possibility, for non-stochastic points

        sample = {
            key: np.array([dist.rvs() for dist in value])
            for key, value in self.stoch_param.items()
        }

        return sample
