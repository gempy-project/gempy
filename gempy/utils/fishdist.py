
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
try:
    import mplstereonet
except ImportError:
    print('mplstereonet package required for visualization in stereonets')

try:
    from spherecluster import VonMisesFisherMixture
except ImportError:
    print('for some purposes spherecluster package would be good (or I copy the respective file?!')

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

    def __init__(self, mean=None, kappa=None):
        """
        Class to generate and/or load orientation data (azimuth and dip or pole vectors) based on the von-Mises-Fisher
        distribution. Contains methods for visualization and parameter estimation.
        Args:
            mean:
            kappa:
        """

        if kappa:
            self.kappa = kappa

        if mean is not None:
            self.mean = mean



    def sample(self, mean=None, kappa=None, num_samples=100, direct_output = False):
        """
        Generates num_samples N-dimensional samples from von Mises Fisher
        distribution around center mu in R^N with concentration kappa.
        Args:
            mean: mean direction, as np.
            kappa: concentration parameter
            num_samples:number of samples
            direct_output: whether the sampled orientations should be returned directly as np.ndarray

        Returns: self.samples_xyz, self.samples_sph

        """
        if mean is not None:
            if kappa is not None:
                self.mean = mean
                self.kappa = kappa

        self.num_samples = num_samples

        try:
            self.samples_xyz = self._generate_samples()
            self.samples_azdip = self._cartesian2spherical(self.samples_xyz)

            if direct_output is True:
                return self.samples_xyz

        except AttributeError:
            print('mean and kappa must be defined')


    def add_orientation_data(self, orient):
        """
        Method to load manually orientation measurements (e.g. to plot stereonets or estimate concentration)
        Args:
            orient: np.ndarray with orientations, can be either azimuth and dip or pole vectors (normalized)

        Returns: self.kappa, self.num_samples

        """
        try: # check if there are already samples loaded to not overwrite them
            getattr(self, 'samples_xyz')
            old_samples = self.samples_xyz
        except AttributeError:
            old_samples = None

        assert type(orient) == np.ndarray

        if orient.shape[1] == 2:
            self.samples_xyz = self._spherical2cartesian(orient)
            self.samples_azdip = orient

        elif orient.shape[1] == 3:
            self.samples_xyz = orient
            self.samples_azdip = self._cartesian2spherical(orient)

        else:
            print('No. Something is wrong with the orientation data')

        self.num_samples = orient.shape[0]

        if old_samples is not None: #append new samples to old samples
            self.samples_xyz = np.concatenate((old_samples,self.samples_xyz))
            self.samples_azdip = self._cartesian2spherical(self.samples_xyz)

    def estimate_vMF_params(self):
        """

        Returns:

        """
        vmf_soft = VonMisesFisherMixture(n_clusters=1, posterior_type='soft')
        try:
            vmf_soft.fit(self.samples_xyz)
            self.kappa = vmf_soft.concentrations_[0]
            self.mean = vmf_soft.cluster_centers_[0]
            print('concentration parameter ', self.kappa, 'mean direction ', self.mean)
        except AttributeError:
            print('object has no orientations. Use add_orientations to load orientation data manually or sample from a vMF distribution with the .sample method')

    def plot_samples_3D(self):
        """

        Returns:

        """
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
        for i in self.samples_xyz:
            ax.add_artist(Arrow3D([0, i[0]], [0, i[1]], [0, i[2]], mutation_scale=20, lw=1, arrowstyle="-|>",
                                  color="darkgreen"))  # samples
        try:
            ax.add_artist(
                Arrow3D([0, self.mean[0]], [0, self.mean[1]], [0, self.mean[2]], mutation_scale=20, lw=1, arrowstyle="-|>",
                        color="darkorange"))  # mean
        except AttributeError:
            pass
        plt.show()
        return fig

    def plot_stereonet(self, poles=True):
        """

        Args:
            poles:

        Returns:

        """
        fig, ax = mplstereonet.subplots(figsize=(5, 5))
        if poles is True:
            for point in self.samples_azdip:
                ax.pole(point[0] - 90, point[1], color='k', linewidth=1, marker='v', markersize=6,label=('samples'))
            try:
                mean_sph = self._cartesian2spherical(self.mean)
                ax.pole(mean_sph[0] - 90, mean_sph[1], color='r', markersize=6, label='mean')
            except AttributeError:
                pass
        ax.grid()
        ax.density_contourf(self.samples_azdip[:, 0] - 90, self.samples_azdip[:, 1], measurement='poles', cmap='inferno', alpha=0.7)
        try:
            ax.set_title('kappa = '+str(self.kappa), y=1.2)
        except AttributeError:
            pass
        #return fig

    def _generate_samples(self):

        dim = len(self.mean)
        result = np.zeros((self.num_samples, dim))
        for nn in range(self.num_samples):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(dim=dim)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to()

            # compute new point
            result[nn, :] = v * np.sqrt(1. - w ** 2) + w * self.mean

        return result

    def _sample_weight(self, dim):
        """
        Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * self.kappa ** 2 + dim ** 2) + 2 * self.kappa)
        x = (1. - b) / (1. + b)
        c = self.kappa * x + dim * np.log(1 - x ** 2)

        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if self.kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self):
        """
        Sample point on sphere orthogonal to mu.
        """
        v = np.random.randn(self.mean.shape[0])
        proj_mu_v = self.mean * np.dot(self.mean, v) / np.linalg.norm(self.mean)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)

    #def plot(self):
        #self.plot_vectors()
        #self.plot_stereonet()

    def _cartesian2spherical(self, xyz):
        """

        Args:
            xyz:

        Returns:

        """
        if xyz.ndim == 1:
            theta = np.rad2deg(np.nan_to_num(np.arccos(xyz[2])))
            phi = np.round(np.rad2deg(np.nan_to_num(np.arctan2(xyz[0], xyz[1]))), 0)
            if phi < 0:
                phi += 360
            return np.array([phi, theta])
        else:
            a = np.empty((xyz.shape[0], 2))
            theta = np.rad2deg(np.nan_to_num(np.arccos(xyz[:, 2])))
            # theta = theta*(-1)
            phi = np.round(np.rad2deg(np.nan_to_num(np.arctan2(xyz[:, 0], xyz[:, 1]))), 0)
            phi[phi < 0] += 360
            a[:, 0] = phi
            a[:, 1] = theta
            return a

    def _spherical2cartesian(self, orient):
        """

        Args:
            orient:

        Returns:

        """
        azimuth = orient[:, 0]
        dip = orient[:, 1]
        xyz = np.empty((orient.shape[0], 3))
        xyz[:, 0] = np.sin(np.deg2rad(dip.astype('float'))) * np.sin(np.deg2rad(azimuth.astype('float')))
        xyz[:, 1] = np.sin(np.deg2rad(dip.astype('float'))) * np.cos(np.deg2rad(azimuth.astype('float')))
        xyz[:, 2] = np.cos(np.deg2rad(dip.astype('float')))
        return xyz





