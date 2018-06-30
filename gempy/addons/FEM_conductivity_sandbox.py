"""
placeholder file for thermal conductivity modelling module
"""
import scipy
import numpy
import matplotlib.pyplot as plt
from gempy.addons.sandbox import Calibration


class Thermal_conductivity:
    def __init__(self, associated_calibration=None):
        self.kmin = 0.1 #bogus
        self.kmax = 10.0
        self.scaling = 'linear' # log

        self.legend = None
        self.cmap = 'viridis'

        self.contours = True
        self.main_contours = numpy.arange(0, 2000, 50)
        self.sub_contours = numpy.arange(0, 2000, 10)

        self.stop_threat = False

        if associated_calibration == None:
            try:
                self.associated_calibration = Calibration._instances[-1]
                print("no calibration specified, using last calibration instance created: ",self.associated_calibration)
            except:
                print("ERROR: no calibration instance found. please create a calibration")

    def setup(self):
        self.render_legend()
        pass

    def render_frame(self,depth):
        depth_rotated = scipy.ndimage.rotate(depth, self.associated_calibration.calibration_data['rot_angle'], reshape=False)
        depth_cropped = depth_rotated[self.associated_calibration.calibration_data['y_lim'][0]:self.associated_calibration.calibration_data['y_lim'][1],
                        self.associated_calibration.calibration_data['x_lim'][0]:self.associated_calibration.calibration_data['x_lim'][1]]
        depth_masked = numpy.ma.masked_outside(depth_cropped, self.associated_calibration.calibration_data['z_range'][0],
                                               self.associated_calibration.calibration_data['z_range'][
                                                   1])

        h = self.associated_calibration.calibration_data['scale_factor'] * (
                self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0]) / 100.0
        w = self.associated_calibration.calibration_data['scale_factor'] * (
                self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0]) / 100.0

        fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        if self.contours is True:
            x = range(numpy.shape(depth_cropped)[1])
            y = range(numpy.shape(depth_cropped)[0])
            z = depth_cropped
            sub_contours = plt.contour(x, y, z, levels=self.sub_levels, linewidths=0.5, colors=[(0, 0, 0, 0.8)])
            main_contours = plt.contour(x, y, z, levels=self.main_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
            plt.clabel(main_contours, inline=0, fontsize=15, fmt='%3.0f')
        ax.pcolormesh(depth_masked, vmin=self.associated_calibration.calibration_data['z_range'][0],
                      vmax=self.associated_calibration.calibration_data['z_range'][1], cmap=self.cmap)
        plt.savefig('current_frame.png', pad_inches=0)
        plt.close(fig)

def render_legend(self):
    ...




#please put Simulation code here