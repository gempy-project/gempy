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

Tested on Ubuntu 16

Created on 23/06/2018

@author: Miguel de la Varga, Alexander Schaaf
"""
import numpy as np
import pandas as pn


def find_surface_points_from_block(block, value):
    """
    Find the voxel at an interface. We shift left since gempy is based on bottoms

    Args:
        block (ndarray):
        value:

    Returns:

    """
    A = block > value
    # Matrix shifting along axis
    B = A  #
    x_shift = B[:-1, :, :] ^ B[1:, :, :]

    # Matrix shifting along axis
    y_shift = B[:, :-1, :] ^ B[:, 1:, :]

    # Matrix shifting along axis
    z_shift = B[:, :, :-1] ^ B[:, :, 1:]

    final_bool = np.zeros_like(block, dtype=bool)
    final_bool[:-1, :-1, :-1] = x_shift[:, :-1, :-1] + y_shift[:-1, :, :-1] + z_shift[-1:, -1:, :]

    return final_bool


def surface_points_from_surface_points_block(block_bool, block_grid, formation='default_formation', series='Default_series',
                                     formation_number=1, order_series=1, n_points=20):

    assert np.ravel(block_bool).shape[0] == block_grid.shape[0], 'Grid and block block must have the same size. If you' \
                                                           'are importing a model from noddy make sure that the' \
                                                           'resolution is the same'
    coord_select = block_grid[np.ravel(block_bool)]

    loc_points = np.linspace(0, coord_select.shape[0]-1, n_points, dtype=int)

    # Init dataframe
    p = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series', 'formation_number',
                              'order_series', 'isFault'])

    p[['X', 'Y', 'Z']] = pn.DataFrame(coord_select[loc_points])
    p['formation'] = formation
    p['series'] = series
    p['formation_number'] = formation_number
    p['order_series'] = order_series

    return p


def set_surface_points_from_block(geo_data, block, block_grid=None, n_points=20, reset_index=False):
    values = np.unique(np.round(block))
    values.sort()
    values = values[:-1]

    if block_grid is None:
        block_grid = geo_data.grid.values

    for e, value in enumerate(values):
        block_bool = find_surface_points_from_block(block, value)

        geo_data.set_interface_object(surface_points_from_surface_points_block(block_bool, block_grid,
                                                                       formation='formation_'+str(e), series='Default_series',
                                                                       formation_number=e, order_series=1,
                                                                       n_points=n_points), append=True)
        if reset_index:
            geo_data.surface_points.reset_index(drop=True, inplace=True)

    return geo_data


class VanMisesFisher:
    def __init__(self, mu, kappa, dim=3):
        """van Mises-Fisher distribution for sampling vector components from n-dimensional spheres.

        Adapted from source: https://github.com/pymc-devs/pymc3/issues/2458

        Args:
            mu (np.ndarray): Mean direction of vector [Gx, Gy, Gz]
            kappa (float): Concentration parameter (the lower the higher the spread on the sphere)
            dim (int, optional): Dimensionality of the Sphere
        """
        self.mu = mu
        self.kappa = kappa
        self.dim = dim

    def rvs(self, n=1):
        """Obtain n samples from van Mises-Fisher distribution.

        Args:
            n (int): Number of samples to draw

        Returns:
            np.ndarray with shape (n, 3) containing samples.

        """
        result = np.zeros((n, self.dim))
        for nn in range(n):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight()
            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to()
            # compute new point
            result[nn, :] = v * np.sqrt(1. - w** 2) + w * self.mu
        return result

    def _sample_weight(self):
        """Who likes documentation anyways. This is totally intuitive and trivial."""
        dim = self.dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * self.kappa ** 2 + dim ** 2) + 2 * self.kappa)
        x = (1. - b) / (1. + b)
        c = self.kappa * x + dim * np.log(1 - x ** 2)

        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if self.kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                # print(w)
                return w

    def _sample_orthonormal_to(self):
        """Who likes documentation anyways. This is totally intuitive and trivial."""
        v = np.random.randn(self.mu.shape[0])
        proj_mu_v = self.mu * np.dot(self.mu, v) / np.linalg.norm(self.mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)

    def stats(self):
        return self.mu, self.kappa


def change_data(interp_data, geo_data_stoch, priors):
    """Changes input data with prior distributions (scipy.stats distributions) given in list.
    Prior distribution objects must contain .rvs() method for drawing samples.

    Args:
        interp_data:
        geo_data_stoch:
        priors:
        verbose:

    Returns:

    """
    prior_draws = []
    for prior in priors:
        if hasattr(prior, "gradient"):
            value = prior.rvs()
        else:
            value = prior.rvs() / interp_data.rescaling_factor
        prior_draws.append(value)

        if prior.index_interf is not None:
            if prior.replace:  # replace the value
                # geo_data.interfaces.set_value(prior.index_interf, prior.column, prior.rvs() / rf)
                interp_data.geo_data_res.interfaces.loc[prior.index_interf, prior.column] = value
            else:  # add value
                interp_data.geo_data_res.interfaces.loc[prior.index_interf, prior.column] = geo_data_stoch.interfaces.loc[
                                                                                prior.index_interf, prior.column] + value
        if prior.index_orient is not None:
            if prior.replace:  # replace the value
                interp_data.geo_data_res.orientations.loc[prior.index_orient, prior.column] = value
            else:  # add value
                interp_data.geo_data_res.orientations.loc[prior.index_orient, prior.column] = geo_data_stoch.orientations.loc[
                                                                                  prior.index_orient, prior.column] + value
    return prior_draws