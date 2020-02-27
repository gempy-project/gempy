""" Method to handle geographic information

General set of methods to handle geographic inforamtion and
additional data sets (e.g. GoogleEarth .kml files, GeoTiffs, etc.)

2020 Florian Wellmann
"""

try:
    from osgeo import osr
    # import gdal

    gdal_installed = True
except ModuleNotFoundError:
    print("Geopgraphic libraries (osgeo, gdal) not (correctly) installed")
    print("Continuing... but some functionality may not work!")
    gdal_installed = False

import numpy as np


class GeographicPoint(object):
    """Geographic point in 2-D (on surface) or 3-D (with z-coordinate)

    General class to handle points in geographic reference systems.

    Note: so far limited to lat/long and UTM projection

    Args:
        x (float) : x or longitude
        y (float) : y or latitude

    Optional Args:
        z (float) : z or altitude
        type ('utm', 'latlong', 'nongeo'): coordinate system # use nongeo for non-geographic projection;
            default: 'nongeo'
        zone (int): utm zone (needs to be defined for type=utm!)
    """

    def __init__(self, x, y, *z, **kwds):
        """3-D point in space

        """
        self.x = x
        self.y = y
        self.type = kwds.get("type", "nongeo")
        if len(z) == 1:
            self.z = z[0]
        if 'zone' in kwds:
            self.zone = kwds['zone']
        if 'type' in kwds and kwds['type'] == 'utm' and 'zone' not in kwds:
            raise AttributeError("Please provide utm zone")
        # for the case that self.type == 'latlong': determine UTM zone:
        if self.type == 'latlong':
            self.zone = int(np.floor(np.mod((self.x + 180) / 6, 60)) + 1)

    def __repr__(self):
        if hasattr(self, 'z'):
            return "p(%f, %f, %f) in %s" % (self.x, self.y, self.z, self.type)
        else:
            return "p(%f, %f) in %s" % (self.x, self.y, self.type)

    def latlong_to_utm(self):
        """Convert point from lat long to utm for given zone"""
        if self.type == 'utm':
            # Points already in utm coordinates, nothing to do...
            return
        if not gdal_installed:
            print("gdal not imported, conversion not possible.")
            print("Please check gdal installation and import.")
            return None

        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        if self.zone == 40:
            utm = osr.SpatialReference()
            utm.ImportFromEPSG(32640)
        elif self.zone == 33:
            # !!! NOTE: this is 33S (for Namibia example)!
            utm = osr.SpatialReference()
            utm.ImportFromEPSG(32733)
        else:
            raise AttributeError("Sorry, zone %d not yet implemented\
             (to fix: check EPSG code on http://spatialreference.org/ref/epsg/ and include in code!)" % self.zone)
        ct = osr.CoordinateTransformation(wgs, utm)
        self.x, self.y = ct.TransformPoint(self.x, self.y)[:2]
        self.type = 'utm'

    def utm_to_latlong(self):
        """Convert point from utm to lat long for given zone"""
        if self.type == 'latlong':
            # Points already in latlong coordinates, nothing to do...
            return
        if not gdal_installed:
            print("gdal not imported, conversion not possible.")
            print("Please check gdal installation and import.")
            return None

        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        if self.zone == 40:
            utm = osr.SpatialReference()
            utm.ImportFromEPSG(32640)
        else:
            raise AttributeError(
                "Sorry, zone %d not yet implemented (check EPSG code and include in code!)" % self.zone)
        ct = osr.CoordinateTransformation(utm, wgs)
        self.x, self.y = ct.TransformPoint(self.x, self.y)[:2]
        self.type = 'latlong'


class GeopgraphicPointSet(object):
    """Set of geographic points in 2-D (on surface) or 3-D (with z-coordinate)

    General class to handle point sets in in geographic reference systems. The main
    purpose is to combine sets of points of similar features, and to perform joint
    operations (e.g. fitting planes to sets of points).

    Note: so far limited to lat/long and UTM projection

    Optional Args:
        type ('utm', 'latlong', 'nongeo'): coordinate system # use nongeo for non-geographic projection;
            default: 'nongeo'
    """

    def __init__(self, **kwds):
        """Point set as a collection of points (picks on one line)

        **Optional keywords**:
            - *type* = 'utm', 'latlong': coordinate type (default: 'latlong')
        """
        self.points = []
        self.type = kwds.get('type', 'latlong')
        self.normal = None
        self.ctr = None
        self.dip_direction = None
        self.dip = None
        self.min = None
        self.max = None

    def __repr__(self):
        """Print out information about point set"""
        out_str = "Point set with %d points" % len(self.points)
        out_str += "; " + self.type
        if hasattr(self, 'ctr'):
            out_str += "; Centroid: at (%.2f, %.2f, %.2f)" % (self.ctr.x, self.ctr.y, self.ctr.z)

        if hasattr(self, 'dip'):
            out_str += "; Orientation: (%03d/%02d)" % (self.dip_direction, self.dip)
        return out_str

    def add_point(self, point):
        self.points.append(point)

    def latlong_to_utm(self):
        """Convert all points from lat long to utm"""
        if self.type == 'latlong':  # else not required...
            for point in self.points:
                point.latlong_to_utm()
            self.type = 'utm'
        # convert plane centroid, if already calculated:
        if hasattr(self, 'ctr'):
            self.ctr.latlong_to_utm()

    def utm_to_latlong(self):
        """Convert all points from utm to lat long"""
        if self.type == 'utm':  # else not required...
            for point in self.points:
                point.utm_to_latlong()
            self.type = 'latlong'
        # convert plane centroid, if already calculated:
        if hasattr(self, 'ctr'):
            self.ctr.utm_to_latlong()

    def get_z_values_from_geotiff(self, filename):
        """Open GeoTiff file and get z-value for all points in set

        Args:
            filename: filename of GeoTiff file
        Note: requires gdal installed!
        """

        # check if points in latlong, else: convert
        if self.type == 'utm':
            self.utm_to_latlong()

        # initialise lookup for entire point set
        l = looker(filename)

        for point in self.points:
            point.z = l.lookup(point.x, point.y)

    def plane_fit(self):
        """Fit plane to points in PointSet

        Fit an d-dimensional plane to the points in a point set
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.

        adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
        """
        if self.type == 'latlong':
            self.latlong_to_utm()

        points = np.empty((3, len(self.points)))
        for i, point in enumerate(self.points):
            points[0, i] = point.x
            points[1, i] = point.y
            points[2, i] = point.z

        from numpy.linalg import svd
        points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                       points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        m = np.dot(x, x.T)  # Could also use np.cov(x) here.

        self.ctr = GeographicPoint(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
        self.normal = svd(m)[0][:, -1]
        # return ctr, svd(M)[0][:, -1]
        if self.normal[2] < 0:
            self.normal = - self.normal

    def get_orientation(self):
        """Get orientation (dip_direction, dip) for points in all point set"""
        if "normal" not in dir(self):
            self.plane_fit()

        # calculate dip
        self.dip = np.arccos(self.normal[2]) / np.pi * 180.

        # calculate dip direction
        # +/+
        if self.normal[0] >= 0 and self.normal[1] > 0:
            self.dip_direction = np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
        # border cases where arctan not defined:
        elif self.normal[0] > 0 and self.normal[1] == 0:
            self.dip_direction = 90
        elif self.normal[0] < 0 and self.normal[1] == 0:
            self.dip_direction = 270
        # +-/-
        elif self.normal[1] < 0:
            self.dip_direction = 180 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.
        # -/-
        elif self.normal[0] < 0 <= self.normal[1]:
            self.dip_direction = 360 + np.arctan(self.normal[0] / self.normal[1]) / np.pi * 180.

    #    elif normal_vec[1] == 0:
    #        return 90

    def stereonet(self):
        """Create stereonet plot of plane pole and half circle for this point set"""
        import matplotlib.pyplot as plt
        import mplstereonet

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='stereonet')

        # ax.plane(dip_dirs, dips, 'g-', linewidth=0.5)
        ax.pole(self.dip_direction - 90, self.dip, 'gs', markersize=4)
        ax.plane(self.dip_direction - 90, self.dip, 'g', markersize=4)

        # ax.rake(strike, dip, -25)
        ax.grid()

    def minmax(self):
        """Get minimum and maximum values of points in point set (e.g. to determine surrounding box)"""
        point_array = np.empty((len(self.points), 2))

        for i, p in enumerate(self.points):
            point_array[i] = (p.x, p.y)

        self.min = np.min(point_array, axis=0)
        self.max = np.max(point_array, axis=0)

