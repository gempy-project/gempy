# Source: http://docs.enthought.com/mayavi/mayavi/auto/example_canyon_decimation.html#example-canyon-decimation

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2008-2015, Enthought, Inc.
# License: BSD Style.

import numpy as np
import sys
from math import *

def table2grid(table):
    nPts= table.shape[0]
    xmin, ymin, zmin = table.min(0)
    xmax, ymax, zmax = table.max(0)
    xvals = np.unique(table[:, 0])
    yvals = np.unique(table[:, 1])
    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]
    nx = len(xvals)
    ny = len(yvals)
    arr = np.empty([nx, ny])
    print "Grid info:"
    print "x-spacing:", dx
    print "y-spacing:", dy
    print "x-range:", xmin, "-", xmax
    print "y-range:", ymin, "-", ymax
    print "z-range:", zmin, "-", zmax
    for row in range(nPts):
        xcoord = table[row, 0]
        ycoord = table[row, 1]
        xArr = (xcoord - xmin)/dx
        yArr = (ycoord - ymin)/dy
        if xArr % 1 == 0 and yArr % 1 == 0:
            xArr = int(xArr)
            yArr = int(yArr)
        else:
            print "Error. Exiting."
            sys.exit()
        arr[xArr, yArr] = table[row, 2]
    return arr

# Specify input xyz grid
FileIn='Crust.xyz'
RawData = np.loadtxt(FileIn, skiprows=0)
data = table2grid(RawData)
data = data.astype(np.float32)
xmin, ymin, zmin = RawData.min(0)
xmax, ymax, zmax = RawData.max(0)

# Plot an interecting section ##################################################
from mayavi import mlab
mlab.figure(1, size=(450, 390))
mlab.clf()
data = mlab.pipeline.array2d_source(data)

# Use a greedy_terrain_decimation to created a decimated mesh
terrain = mlab.pipeline.greedy_terrain_decimation(data)
terrain.filter.error_measure = 'number_of_triangles'
# terrain.filter.error_measure = 'absolute_error'
# terrain.filter.absolute_error = 10.
terrain.filter.number_of_triangles = RawData.shape[0]/2
terrain.filter.compute_normals = True

scale=0.01

# Plot it black the lines of the mesh
lines = mlab.pipeline.surface(terrain, color=(0, 0, 0),
                                      representation='wireframe')
# The terrain decimator has done the warping. We control the warping
# scale via the actor's scale.
lines.actor.actor.scale = [1, 1,scale]

# Display the surface itself.
surf = mlab.pipeline.surface(terrain, colormap='gist_earth',
                                      vmin=zmin, vmax=zmax)
surf.actor.actor.scale = [1, 1, scale]

# Display the original regular grid. This time we have to use a
# warp_scalar filter.
warp = mlab.pipeline.warp_scalar(data, warp_scale=scale)
grid = mlab.pipeline.surface(warp, color=(1, 1, 1),
                                      representation='wireframe')

fig = mlab.view(azimuth=200, elevation=100, distance='auto', focalpoint='auto')

n1 = RawData.shape[0]
n2 = terrain.filter.number_of_triangles
perc = float(n2)/n1*100
print "Number of points in initial grid:", n1
print "Number of points in triangulated grid:", n2
print "Relative:", perc

mlab.show()
