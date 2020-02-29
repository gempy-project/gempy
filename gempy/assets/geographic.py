""" Method to handle geographic information

General set of methods to handle geographic inforamtion and
additional data sets (e.g. GoogleEarth .kml files, GeoTiffs, etc.)

2020 Florian Wellmann
"""

try:
    from osgeo import ogr, osr
    import pyproj
    import gdal

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

        # Determine UTM zone automatically - see Wikipedia page
        # Note: works in most cases (except extreme North and South)
        # For generality, keep option to add zone manually!

        if hasattr(self, 'utm'):
            zone = self.zone
        else:
            zone = np.int((np.mod(np.floor((self.x + 180)/6), 60))) + 1

        # create projection
        p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')

        # convert points
        self.x, self.y = p(self.x, self.y)

        # wgs = osr.SpatialReference()
        # wgs.ImportFromEPSG(4326)
        # if self.zone == 40:
        #     utm = osr.SpatialReference()
        #     #utm.ImportFromEPSG(32640)
        #     utm.ImportFromEPSG(2975)
        #     print(utm)
        # elif self.zone == 33:
        #     # !!! NOTE: this is 33S (for Namibia example)!
        #     utm = osr.SpatialReference()
        #     utm.ImportFromEPSG(32733)
        # else:
        #     raise AttributeError("Sorry, zone %d not yet implemented\
        #      (to fix: check EPSG code on http://spatialreference.org/ref/epsg/ and include in code!)" % self.zone)
        # ct = osr.CoordinateTransformation(wgs, utm)
        # self.x, self.y = ct.TransformPoint(self.x, self.y)[:2]

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


class GeographicPointSet(object):
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
        if hasattr(self, 'ctr') and self.ctr is not None:
            out_str += "; Centroid: at (%.2f, %.2f, %.2f)" % (self.ctr.x, self.ctr.y, self.ctr.z)

        if hasattr(self, 'dip') and self.dip is not None:
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
        if hasattr(self, 'ctr') and self.ctr is not None:
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
        # if self.type == 'utm':
        #     self.utm_to_latlong()
        #     print("converted to utm")

        # initialise lookup for entire point set
        geotiff_file = GeoTiffgetValue(filename)

        for point in self.points:
            point.z = geotiff_file.lookup(point.x, point.y)

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


class GeoTiffgetValue(object):
    """Determin a value from a GeoTiff

    Required, for example, to determine z-values in GeoTiffs of DEMs.

    Note: at this stage, the value is taken from Raster Band 1.

    Credits to entry on stackoverflow:
    http://stackoverflow.com/questions/13439357/extract-point-from-raster-in-gdal

    Args:
        tifname (filename): path and filename to GeoTiff file
    """

    def __init__(self, tifname='test.tif'):
        # open the GeoTiff raster file and its spatial reference
        self.ds = gdal.Open(tifname)
        sr_raster = osr.SpatialReference(self.ds.GetProjection())

        # get the WGS84 spatial reference
        sr_point = osr.SpatialReference()
        sr_point.ImportFromEPSG(4326)  # WGS84

        # coordinate transformation
        self.ct = osr.CoordinateTransformation(sr_point, sr_raster)

        # geotranformation and its inverse
        gt = self.ds.GetGeoTransform()
        dev = (gt[1] * gt[5] - gt[2] * gt[4])
        gtinv = (gt[0], gt[5] / dev, -gt[2] / dev,
                 gt[3], -gt[4] / dev, gt[1] / dev)
        self.gt = gt
        self.gtinv = gtinv

        # band as array
        b = self.ds.GetRasterBand(1)
        self.arr = b.ReadAsArray()

    def lookup(self, lon, lat):
        """look up value at lon, lat"""

        # get coordinate of the raster
        xgeo, ygeo, zgeo = self.ct.TransformPoint(lon, lat, 0)

        # convert it to pixel/line on band
        u = xgeo - self.gtinv[0]
        v = ygeo - self.gtinv[3]
        # FIXME this int() is probably bad idea, there should be
        # half cell size thing needed
        xpix = int(self.gtinv[1] * u + self.gtinv[2] * v)
        ylin = int(self.gtinv[4] * u + self.gtinv[5] * v)

        # look the value up
        return self.arr[ylin, xpix]


class KmlPoints(object):
    """Get point sets from KML file

    Optional Args:

        filename (string): filename of kml file
        debug (bool): provide debug output (Default: false)
        auto_remove (bool): automatically remove unsuitable points (e.g. outside Geotiffs)
            and point sets (e.g. too few points, too close on a line)
        type ('utm', 'latlong') : coordinate system of points (default: latlong)
    """

    def __init__(self, **kwds):
        self.debug = kwds.get("debug", False)
        self.auto_remove = kwds.get("auto_remove", True)
        self.type = kwds.get("type", 'latlong')
        self.geotiffs = []
        self.points = []
        self.point_sets = []
        # if kwds.has_key('filename'):
        if 'filename' in kwds:
            if self.debug:
                print("read kml")
            self.read_kml(kwds['filename'])

    def read_kml(self, filename):
        """Read kml file and extract points"""

        ds = ogr.Open(filename)
        # point_sets = []

        for lyr in ds:
            for j, feat in enumerate(lyr):
                geom = feat.GetGeometryRef()
                ps = GeographicPointSet()
                if geom is not None:
                    for i in range(0, geom.GetPointCount()):
                        print (geom.GetPoint(i))
                        point = GeographicPoint(x=geom.GetPoint(i)[0],
                                                y=geom.GetPoint(i)[1],
                                                type='latlong')
                        print(point)
                        ps.add_point(point)
                        # points.append([geom.GetPoint(i)[0], geom.GetPoint(i)[1], j])

                self.point_sets.append(ps)

        if self.debug:
            print("%d point set(s) added" % len(self.point_sets))

    def test_point_sets(self):
        """Test if point sets contain at least three points; if not: remove"""
        # test if all point sets have at least three points:
        for ps in self.point_sets:
            if len(ps.points) < 3:
                self.point_sets.remove(ps)
                if self.debug:
                    print("Removed point set")

        if self.debug:
            print("%d point set(s) remaining" % len(self.point_sets))

    def determine_z_values(self):
        """Determine z values for all points in point sets

        Approach: test all geotiffs in given order, stored in self.geotiffs list
        """
        if len(self.geotiffs) == 0:
            raise AttributeError("Please define geotiffs first (self.add_geotiff())")

        # check that coordinates are in latlong, if not: convert
        if self.type == 'utm':
            self.utm_to_latlong()

        for ps in self.point_sets:
            fail = True
            for geotiff in self.geotiffs:
                try:
                    ps.get_z_values_from_geotiff(geotiff)
                except IndexError:
                    continue
                fail = False

            # if point can not be detected: remove (default) or raise error
            # if self.auto_remove = False

            if fail:
                if self.auto_remove:
                    if self.debug:
                        print("Point outside geotiff, drop")
                    self.point_sets.remove(ps)
                else:
                    raise IndexError("Point outside of defined geotiffs!\nPlease define\
                                     suitable geotiff or remove point (set self.auto_remove = True)")

    def fit_plane_to_all_sets(self):
        """Fit plane to all point sets

        Results are stored in point set object (self.ctr, self.normal)
        """
        if self.type == 'latlong':
            self.latlong_to_utm()

        for ps in self.point_sets:
            ps.plane_fit()
            ps.get_orientation()

    def stereonet(self):
        """Create stereonet plot of all plane pole and half circle for all planes"""
        import matplotlib.pyplot as plt
        import mplstereonet

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='stereonet')

        # ax.plane(dip_dirs, dips, 'g-', linewidth=0.5)
        for ps in self.point_sets:
            ax.pole(ps.dip_direction - 90, ps.dip, 'gs', markersize=4)
            ax.plane(ps.dip_direction - 90, ps.dip, 'g', markersize=4)

        # ax.rake(strike, dip, -25)
        ax.grid()

    def add_geotiff(self, geotiff):
        """Add geotiff to list of geotiffs (self.geotiffs)

        Args:
            geotiff (filename) : filename (with complete path) to geotiff
        """
        if self.debug:
            print("Note: for efficiency reasons, add the most important geotiff first!")
        self.geotiffs.append(geotiff)

    def latlong_to_utm(self):
        """Convert all points from lat long to utm"""
        if self.type == 'latlong':  # else not required...
            if self.debug:
                print("Convert Lat/Long to UTM")
            for ps in self.point_sets:
                ps.latlong_to_utm()
            self.type = 'utm'

    def utm_to_latlong(self):
        """Convert all points from utm to lat long"""
        if self.type == 'utm':  # else not required...
            if self.debug:
                print("Convert UTM to Lat/Long")
            for ps in self.point_sets:
                ps.utm_to_latlong()
            self.type = 'latlong'
