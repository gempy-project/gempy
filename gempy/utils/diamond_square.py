"""Implementation of Diamond-Square algorithm

This algorithm is often used for random topography generation, following Fournier et al., 1982
see https://en.wikipedia.org/wiki/Diamond-square_algorithm.

Fournier, Alain; Fussell, Don; Carpenter, Loren (June 1982). "Computer rendering of stochastic models".
Communications of the ACM. 25 (6): 371–384.

Here the description from the wikipedia page:

+++ begin Wikipedia +++

The diamond-square algorithm begins with a 2D square array of width and
height 2n + 1. The four corner points of the array must first be set to initial values. The diamond and square steps
are then performed alternately until all array values have been set.

The diamond step: For each square in the array, set the midpoint of that square to be the average of the four corner
points plus a random value.

The square step: For each diamond in the array, set the midpoint of that diamond to be the average of the four corner
points plus a random value.

At each iteration, the magnitude of the random value should be reduced.

During the square steps, points located on the edges of the array will have only three adjacent values set rather
than four. There are a number of ways to handle this complication - the simplest being to take the average of just
the three adjacent values. Another option is to 'wrap around', taking the fourth value from the other side of the
array. When used with consistent initial corner values this method also allows generated fractals to be stitched
together without discontinuities.

+++ end Wikipedia +++

The implementations (in Python) I could find online were either difficult to understand or had many case
selections, especially at edges. Here is a fully vectorized implementation, using padding at edges, to avoid
all these case selections for a more straight-forward implementation (I hope).

This impoementaion is also adjusted to work on non-square start matrices, and on reduced hierarchies with
more initial (internal) points.

Created on 10.04.2020

@author: Florian Wellmann

"""
import numpy as np
import random
import matplotlib.pyplot as plt


class DiaomondSquare(object):

    def __init__(self, size=(16, 16), roughness=0.5, z_min=0, z_max=1, seed=87243):
        """Implementation of vectorized Diaomnd-Square algorithm for random topography generation

        Args:
            size (int, int): shape of grid to interpolate; note: the standard diamond-square algorithm
                operates on a square grid with side length 2**n+1. This implementation is adjusted to non-square
                grids with (2**n+1, 2**m+1). If the input size (int, int) is not matching to the ideal dimension,
                the next bigger size is taken and the grid finally cut (lower left corner is kept);
            roughness: roughness parameter, [0,1]: 0: deterministic interpolation, 1: very rough and bumpy
            z_min: minimum height of surface
            z_max: maximum height of surface
            seed: seed for random function to enable reproducibility and testing
        """
        self.size = size
        # Create mesh with optimal size (2**n+1, 2**m+1)
        # If the input size `self.size` does not match to these dimensions, then the next larger suitable
        # size is chosen.
        self.n = np.ceil(np.log2(self.size[0] - 1)).astype('int8')
        self.m = np.ceil(np.log2(self.size[1] - 1)).astype('int8')
        self.grid = np.empty((2 ** self.n + 1, 2 ** self.m + 1))
        self.roughness = roughness
        self.z_min = z_min
        self.z_max = z_max
        self.seed = seed

    def get_selection_diamond(self, m_pow):
        """get selected points for diamond step on grid z on hierarchy m

        This method is mostly implemented for testing and visualization purposes.
        """

        m = int(2 ** m_pow)

        z = np.zeros_like(self.grid)

        # points to interpolate
        z[m::2 * m, m::2 * m] = 1

        # top left
        z[:-2 * m:2 * m, :-2 * m:2 * m] = 2

        # top right
        z[:-2 * m:2 * m, 2 * m::2 * m] = 2

        # bottom left
        z[2 * m::2 * m, :-2 * m:2 * m] = 2

        # bottom right
        z[2 * m::2 * m, 2 * m::2 * m] = 2

        return z

    def get_selection_square(self, m_pow):
        """Plot selected points for square step on grid z on hierarchy m

        This method is mostly implemented for testing and visualization purposes.
        """
        z = np.zeros_like(self.grid)
        m = int(2 ** m_pow)

        # pad cells with zero value
        z_pad = np.pad(z, m)

        # Checkerboard odd
        # ----------------

        # check-odd, values to interpolate:
        z_pad[m::2 * m, 2 * m:-2 * m:2 * m] = 1

        # check-odd, left
        z_pad[m::2 * m, m:-2 * m:2 * m] = 2

        # check-odd, right
        z_pad[m::2 * m, 3 * m:-m:2 * m] = 2

        # check-odd, top
        z_pad[:-m:2 * m, 2 * m:-2 * m:2 * m] = 2

        # check-odd, bottom
        z_pad[2 * m::2 * m, 2 * m:-2 * m:2 * m] = 2

        # Checkerboard even
        # -----------------

        # check-even, values to interpolate:
        z_pad[2 * m:-2 * m:2 * m, m:-m:2 * m] = 1

        # check-even, left:
        z_pad[2 * m:-2 * m:2 * m, :-2 * m:2 * m] = 2

        # check-even, right:
        z_pad[2 * m:-2 * m:2 * m, 2 * m::2 * m] = 2

        # check-even, top:
        z_pad[m:-2 * m:2 * m, m:-m:2 * m] = 2

        # check-even, bottom:
        z_pad[3 * m::2 * m, m:-m:2 * m] = 2

        return z_pad

    def plot_diamond_and_square(self, pad=False):
        """Plot selected points for diamond and square step for all hierarchies side by side"""

        m_pow_max = min(self.n, self.m)

        shape_ratio = self.n / self.m

        f, axes = plt.subplots(2, m_pow_max, figsize=(12, 12 * shape_ratio / m_pow_max * 2))

        for i, m_pow in enumerate(np.arange(m_pow_max)[::-1]):
            m = 2 ** m_pow
            # z_zero = np.zeros_like(self.grid)
            z_diamond = self.get_selection_diamond(m_pow)
            z_square = self.get_selection_square(m_pow)
            if pad:
                z_pad = np.pad(z_diamond, m)
                axes[0, i].imshow(z_pad, cmap='viridis', vmin=0, vmax=2)
                axes[1, i].imshow(z_square, cmap='viridis', vmin=0, vmax=2)
            else:
                axes[0, i].imshow(z_diamond, cmap='viridis', vmin=0, vmax=2)
                axes[1, i].imshow(z_square[m:-m, m:-m], cmap='viridis', vmin=0, vmax=2)
