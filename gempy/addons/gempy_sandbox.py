import weakref
import numpy
import scipy
import gempy
import matplotlib.pyplot as plt
from itertools import count
from gempy.addons.sandbox import Calibration

class Model:
    _ids = count(0)
    _instances = []

    def __init__(self, model, extent=None, associated_calibration=None, xy_isometric=True, lock=None):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))
        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.output_res = None

        self.legend = True
        self.model = model
        gempy.compute_model(self.model)
        self.empty_depth_grid = None
        self.depth_grid = None
        self.cmap = None
        self.norm = None
        self.lot = None

        self.contours = True
        self.main_contours = numpy.arange(0, 2000, 50)
        self.sub_contours = numpy.arange(0, 2000, 10)

        self.scalar_contours = False
        self.scalar_main_contours = numpy.arange(0.0, 1.0, 0.1)
        self.scalar_sub_contours = numpy.arange(0.0, 1.0, 0.02)

        self.show_faults=True
        self.fault_contours= numpy.arange(0.0, 50, 0.5)

        self.stop_threat = False
        self.lock = lock

        if associated_calibration is None:
            try:
                self.associated_calibration = Calibration._instances[-1]
                print("no calibration specified, using last calibration instance created: ",self.associated_calibration)
            except:
                print("ERROR: no calibration instance found. please create a calibration")
                # parameters from the model:
        else:
            self.associated_calibration = associated_calibration
        if extent == None:  # extent should be array with shape (6,) or convert to list?
            self.extent = self.model._geo_data.extent

        else:
            self.extent = extent  # check: array with 6 entries!

    def calculate_scales(self):
        self.output_res = (self.associated_calibration.calibration_data['x_lim'][1] -
                           self.associated_calibration.calibration_data['x_lim'][0],
                           self.associated_calibration.calibration_data['y_lim'][1] -
                           self.associated_calibration.calibration_data['y_lim'][0])
        self.pixel_size[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_size[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])

        if self.xy_isometric == True:  # model is scaled to fit into box
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_size[0] >= self.pixel_size[1]:
                self.pixel_size[1] = self.pixel_size[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_size[0] = self.pixel_size[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_size[0]
        self.scale[1] = self.pixel_size[1]
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (
                    self.associated_calibration.calibration_data['z_range'][1] -
                    self.associated_calibration.calibration_data['z_range'][0])
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.

    def create_empty_depth_grid(self):
        grid_list = []
        self.output_res = (self.associated_calibration.calibration_data['x_lim'][1] -
                           self.associated_calibration.calibration_data['x_lim'][0],
                           self.associated_calibration.calibration_data['y_lim'][1] -
                           self.associated_calibration.calibration_data['y_lim'][0])
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.pixel_size[1] + self.extent[2], x * self.pixel_size[0] + self.extent[0]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid

    # return self.empty_depth_grid

    def update_grid(self, depth):
        filtered_depth = numpy.ma.masked_outside(depth, self.associated_calibration.calibration_data['z_range'][0],
                                                 self.associated_calibration.calibration_data['z_range'][1])
        scaled_depth = self.extent[5] - (
                    (filtered_depth - self.associated_calibration.calibration_data['z_range'][0]) / (
                        self.associated_calibration.calibration_data['z_range'][1] -
                        self.associated_calibration.calibration_data['z_range'][0]) * (self.extent[5] - self.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth, self.associated_calibration.calibration_data['rot_angle'],
                                             reshape=False)
        cropped_depth = rotated_depth[self.associated_calibration.calibration_data['y_lim'][0]:
                                      self.associated_calibration.calibration_data['y_lim'][1],
                        self.associated_calibration.calibration_data['x_lim'][0]:
                        self.associated_calibration.calibration_data['x_lim'][1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)
        self.depth_grid = depth_grid

    def set_cmap(self, cmap=None, norm=None):
        if cmap is None:
            plotter = gempy.PlotData2D(self.model._geo_data)
            self.cmap = plotter._cmap
        if norm is None:
            self.norm = plotter._norm
            self.lot = plotter._color_lot

    def render_frame(self, outfile=None):

        if self.cmap is None:
            self.set_cmap()
        if self.lock is not None:
            self.lock.acquire()
            lith_block, fault_block = gempy.compute_model_at(self.depth_grid, self.model)
            self.lock.release()
        else:
            lith_block, fault_block = gempy.compute_model_at(self.depth_grid, self.model)
        scalar_field = lith_block[1].reshape((self.output_res[1], self.output_res[0]))
        block = lith_block[0].reshape((self.output_res[1], self.output_res[0]))
        h = self.associated_calibration.calibration_data['scale_factor'] * (self.output_res[1]) / 100.0
        w = self.associated_calibration.calibration_data['scale_factor'] * (self.output_res[0]) / 100.0

        fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.pcolormesh(block, cmap=self.cmap, norm=self.norm)

        if self.show_faults is True:
            plt.contour(fault_block[0].reshape((self.output_res[1], self.output_res[0])), levels=self.fault_contours, linewidths=3.0, colors=[(1.0, 1.0, 1.0, 1.0)])

        if self.contours is True:
            x = range(self.output_res[0])
            y = range(self.output_res[1])
            z = self.depth_grid.reshape((self.output_res[1], self.output_res[0], 3))[:, :, 2]
            sub_contours = plt.contour(x, y, z, levels=self.sub_contours, linewidths=0.5, colors=[(0, 0, 0, 0.8)])
            main_contours = plt.contour(x, y, z, levels=self.main_contours, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
            # plt.contour(lith_block[1].reshape((self.output_res[1], self.output_res[0])) levels=main_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
            plt.clabel(main_contours, inline=0, fontsize=15, fmt='%3.0f')

        if self.scalar_contours is True:
            x = range(self.output_res[0])
            y = range(self.output_res[1])
            z = scalar_field
            sub_contours = plt.contour(x, y, z, levels=self.scalar_sub_contours, linewidths=0.5, colors=[(0, 0, 0, 0.8)])
            main_contours = plt.contour(x, y, z, levels=self.scalar_main_contours, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
           # plt.contour(lith_block[1].reshape((self.output_res[1], self.output_res[0])) levels=main_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
            plt.clabel(main_contours, inline=0, fontsize=15, fmt='%3.0f')


        if outfile == None:
            plt.show()
            plt.close()
        else:
            plt.savefig(outfile, pad_inches=0)
            plt.close(fig)

    def create_legend(self):
        # ...
        pass

    def setup(self, start_stream=False):
        if start_stream == True:
            self.associated_calibration.associated_projector.start_stream()
        self.calculate_scales()
        self.create_empty_depth_grid()

    def run(self):
        run_model(self)

    def convert_coordinates(self, coords):
        converted_coords = []
        for point in coords:
            y = point[0] * self.pixel_size[1] + self.extent[2]
            x = point[1] * self.pixel_size[0] + self.extent[0]
            converted_coords.append([x, y])
        return converted_coords

    #### from hackathon:
    def get_arbitrary_2d_grid(self, px, py, s):
        """Creates arbitrary 2d grid given two input points.

        Args:

            px (list): x coordinates of the two input points (e.g. [0, 2000])
            py (list: y coordinates of the two input points (e.g. [0, 2000])
            s (int): pixel/voxel edge length

        Returns:
            numpy.ndarray: grid (n, 3) for use with gempy.compute_model_at function
            tuple: shape information to reshape into 2d array

        """
        px = numpy.array(px)
        py = numpy.array(py)

        gradient, *_ = scipy.stats.linregress(px, py)

        theta = numpy.arctan(gradient)
        dy = numpy.sin(theta) * s
        dx = numpy.cos(theta) * s

        if px[1] - px[0] == 0:
            ys = numpy.arange(py[0], py[1], s)
            xs = numpy.repeat(px[0], len(ys))
        elif py[1] - py[0] == 0:
            xs = numpy.arange(px[0], px[1], s)
            ys = numpy.repeat(py[0], len(xs))
        else:
            xs = numpy.arange(px[0], px[1], dx)
            ys = numpy.arange(py[0], py[1], dy)

        zs = numpy.arange(self.model._geo_data.extent[4], self.model._geo_data.extent[5] + 1, s)

        a = numpy.tile([xs, ys], len(zs)).T
        b = numpy.repeat(zs, len(xs))
        grid = numpy.concatenate((a, b[:, numpy.newaxis]), axis=1)

        return grid, (len(zs), len(ys))

    def drill(self, x, y, s=1):
        """Creates 1d vertical grid at given x,y location.

        Args:
            geo_data: gempy geo_data object
            x (int): x coordinate of the drill location
            y (int): y coordinate of the drill location
            s (int, optional): pixel/voxel edge length (default: 1)

        Returns:
            numpy.ndarray: grid (n, 3) for use with gempy.compute_model_at

        """
        zs = numpy.arange(self.model._geo_data.extent[4], self.model._geo_data.extent[5], s)
        grid = numpy.array([numpy.repeat(x, len(zs)), numpy.repeat(y, len(zs)), zs]).T
        return grid

    def drill_image(self,lb, n_rep=100, fp="well.png"): #is lb lego brick?
        p = numpy.repeat(lb[0, numpy.newaxis].astype(int), n_rep, axis=0)
        plt.figure(figsize=(2, 6))
        im = plt.imshow(p.T, origin="lower", cmap=self.cmap, norm=self.norm)
        plt.ylabel("z")
        im.axes.get_xaxis().set_ticks([])
        plt.tight_layout()
        plt.title("Lego well")
        plt.savefig(fp, dpi=100)
        return im


## global functions to run the model in loop.
def run_model(model, calibration=None, kinect=None, projector=None, filter_depth=True, n_frames=5,
              sigma_gauss=4):  # continous run functions with exit handling
    if calibration == None:
        calibration = model.associated_calibration
    if kinect == None:
        kinect = calibration.associated_kinect
    if projector == None:
        projector = calibration.associated_projector

    while True:
        if filter_depth == True:
            depth = kinect.get_filtered_frame(n_frames=n_frames, sigma_gauss=sigma_gauss)
        else:
            depth = kinect.get_frame()

        model.update_grid(depth)
        model.render_frame(outfile="current_frame.png")
        projector.show(input="current_frame.png", rescale=False)
        if model.stop_threat is True:
            raise Exception('Threat stopped')
