""" Method to handle geographic information

General set of methods to handle geographic inforamtion and
additional data sets (e.g. GoogleEarth .kml files, GeoTiffs, etc.)

2020 Florian Wellmann
"""

try:
    from osgeo import ogr, osr
    import gdal
except ImportError:
    print("Geopgraphic libraries (osgeo, gdal) not (correctly) installed")
    print("Continuing... but some functionality may not work!")

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

    def __init__(self, x, y, **kwds):
        """3-D point in space

        """
        self.type = kwds.get("type", "nongeo")
        self.type = kwds['type']
        if 'z' in kwds:
            self.z = kwds['z']
        # self.z = z
        if 'zone' in kwds:
            self.zone = kwds['zone']
        if 'type' in kwds and kwds['type'] == 'utm' and not 'zone' in kwds:
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


def broken_function():
    raise Exception('This is broken')

