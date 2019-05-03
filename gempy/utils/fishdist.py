import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mplstereonet

'''The solution for vMF sampling originally appeared here: 
- https://github.com/pymc-devs/pymc3/issues/2458 and
- https://github.com/jasonlaska/spherecluster/blob/master/spherecluster/util.py'''
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

    Parts of this solution for vMF sampling originally appeared here (12.12.2018): 
    - https://github.com/pymc-devs/pymc3/issues/2458 and
    - https://github.com/jasonlaska/spherecluster/blob/master/spherecluster/util.py

    @author: Elisa Heim
"""




class vMF():
    '''draws and visualizes samples from a von mises fisher distribution based on mean, concentration and number of
    samples for stochastic simulations with orientation uncertainty'''

    def __init__(self, mu, kappa, num_samples):
        self.mu = mu
        self.kappa = kappa
        self.num_samples = num_samples
        # print(self.mu, self.kappa, self.num_samples)
        self.points = self.sample_vMF(self.mu, self.kappa, self.num_samples)
        self.pointssph = self.xyz_to_spherical_coordinates(self.points)

    def sample_vMF(self, mu, kappa, num_samples):
        """Generate num_samples N-dimensional samples from von Mises Fisher
        distribution around center mu \in R^N with concentration kappa.
        """
        dim = len(mu)
        result = np.zeros((num_samples, dim))
        for nn in range(num_samples):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(kappa, dim)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(mu)

            # compute new point
            result[nn, :] = v * np.sqrt(1. - w ** 2) + w * mu

        return result

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)

        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu):
        """Sample point on sphere orthogonal to mu."""
        v = np.random.randn(mu.shape[0])
        proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)

    def plot(self):
        self.plot_vMF_3D()
        self.plot_vMF_2D()

    def plot_vMF_3D(self):
        # this code is partially from somewhere (stackoverflow)
        fig = plt.figure(figsize=[5, 5])
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        ax.view_init(azim=30)

        # render the sphere mesh
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        # print(u,v)
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="lightgray")
        plt.axis('on')

        # coordinate system in centre of sphere
        origin = [0, 0, 0]
        X, Y, Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        O, O, O = zip(origin, origin, origin)
        X, Y, Z = zip(X, Y, Z)
        ax.quiver(O, O, O, X, Y, Z, arrow_length_ratio=0.1, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)

        # Plot arrows
        for i in self.points:
            ax.add_artist(Arrow3D([0, i[0]], [0, i[1]], [0, i[2]], mutation_scale=20, lw=1, arrowstyle="-|>",
                                  color="darkgreen"))  # samples
        ax.add_artist(
            Arrow3D([0, self.mu[0]], [0, self.mu[1]], [0, self.mu[2]], mutation_scale=20, lw=1, arrowstyle="-|>",
                    color="darkorange"))  # mean
        plt.show()
        return fig

    def plot_vMF_2D(self, poles=True):
        mean = self.xyz_to_spherical_coordinates(self.mu)
        points_sph = self.pointssph
        fig, ax = mplstereonet.subplots(figsize=(5, 5))
        if poles is True:
            for point in points_sph:
                ax.pole(point[0] - 90, point[1], color='k', linewidth=1, marker='v', markersize=6,label=('samples'))
            ax.pole(mean[0] - 90, mean[1], color='r', markersize=6, label='mean')
        ax.grid()
        ax.density_contourf(points_sph[:, 0] - 90, points_sph[:, 1], measurement='poles', cmap='inferno', alpha=0.7)
        ax.set_title('kappa = '+str(self.kappa), y=1.2)
        #return fig

    def xyz_to_spherical_coordinates(self, gamma1):
        '''conversion of cartesian to spherical coordinates'''
        if gamma1.ndim == 1:
            theta = np.rad2deg(np.nan_to_num(np.arccos(gamma1[2])))
            phi = np.round(np.rad2deg(np.nan_to_num(np.arctan2(gamma1[0], gamma1[1]))), 0)
            if phi < 0:
                phi += 360
            return np.array([phi, theta])
        else:
            a = np.empty(shape=(gamma1.shape[0], 2))
            theta = np.rad2deg(np.nan_to_num(np.arccos(gamma1[:, 2])))
            # theta = theta*(-1)
            phi = np.round(np.rad2deg(np.nan_to_num(np.arctan2(gamma1[:, 0], gamma1[:, 1]))), 0)
            phi[phi < 0] += 360
            a[:, 0] = phi
            a[:, 1] = theta
            return a