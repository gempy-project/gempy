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


class DiaomondSquare(object):

    def __init__(self):
        """Implementation of vectorized Diaomnd-Square algorithm for random topography generation"""
        pass

    def get_selection_diamond(self, z, m_pow):
        """get selected points for diamond step on grid z on hierarchy m
        """

        m = int(2 ** m_pow)

        # points to interpolate
        z[m::2 * m, m::2 * m] = 1

        # top left
        z[:-2 * m:2 * m, :-2 * m:2 * m] = 2

        # top right
        z[:m * -(n - 1):m * (n - 1), m * (n - 1)::m * (n - 1)] = 2

        # bottom left
        z[m * (n - 1)::m * (n - 1), :-m * (n - 1):m * (n - 1)] = 2

        # bottom right
        z[m * (n - 1)::m * (n - 1), m * (n - 1)::m * (n - 1)] = 2

        return z

    def get_selection_square(self, z, m_pow):
        """Plot selected points for square step on grid z on hierarchy m
        """
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
