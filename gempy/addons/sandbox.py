
import os
from warnings import warn
try:
    import freenect
except ImportError:
    warn('Freenect is not installed. Sandbox wont work. Good luck')
try:
    import cv2
except ImportError:
    warn('opencv is not installed. Object detection will not work')

import webbrowser
import pickle
import weakref
import numpy
import scipy

from itertools import count
from PIL import Image, ImageDraw
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
#import gempy.hackathon as hackathon
import IPython
import threading


class Kinect:  # add dummy
    _ids = count(0)
    _instances = []

    def __init__(self, dummy=False, mirror=True):
        self.__class__._instances.append(weakref.proxy(self))
        self.id = next(self._ids)
        self.resolution = (640, 480)
        self.dummy = dummy
        self.mirror = mirror
        self.rgb_frame = None

        #TODO: include filter self.-filter parameters as function defaults
        self.n_frames = 5 #filter parameters
        self.sigma_gauss = 3
        self.filter = 'gaussian' #TODO: deprecate get_filtered_frame, make it switchable in runtime


        if self.dummy == False:
            print("looking for kinect...")
            self.ctx = freenect.init()
            self.dev = freenect.open_device(self.ctx, self.id)
            print(self.id)
            freenect.close_device(self.dev)  # TODO Test if this has to be done!

            self.angle = None
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[
                0]  # get the first Depth frame already (the first one takes much longer than the following)
            self.filtered_depth = None
            print("kinect initialized")
        else:
            self.angle = None
            self.filtered_depth = None
            self.depth = self.get_frame()
            print("dummy mode. get_frame() will return a synthetic depth frame, other functions may not work")

    def set_angle(self, angle):
        self.angle = angle
        freenect.set_tilt_degs(self.dev, self.angle)

    def get_frame(self, horizontal_slice=None):
        if self.dummy is False:
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            self.depth = numpy.fliplr(self.depth)
            return self.depth
        else:
            synth_depth = numpy.zeros((480, 640))
            for x in range(640):
                for y in range(480):
                    if horizontal_slice == None:
                        synth_depth[y, x] = int(800 + 200 * (numpy.sin(2 * numpy.pi * x / 320)))
                    else:
                        synth_depth[y, x] = horizontal_slice
            self.depth = synth_depth
            return self.depth

    def get_filtered_frame(self, n_frames=None, sigma_gauss=None): #TODO: deprecate?
        if n_frames==None:
            n_frames=self.n_frames
        if sigma_gauss==None:
            sigma_gauss=self.sigma_gauss

        if self.dummy == True:
            self.get_frame()
            return self.depth
        elif self.filter=='gaussian':

            depth_array = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            for i in range(n_frames - 1):
                depth_array = numpy.dstack([depth_array, freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]])
            depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)
            self.depth = numpy.ma.mean(depth_array_masked, axis=2)
            self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, sigma_gauss)
            return self.depth


    def get_rgb_frame(self):
        if self.dummy == False:
            self.rgb_frame = freenect.sync_get_video(index=self.id)[0]
            self.rgb_frame = numpy.fliplr(self.rgb_frame)

            return self.rgb_frame
        else:
            pass

    def calibrate_frame(self, frame, calibration=None):
        if calibration is None:
            try:
                calibration = Calibration._instances[-1]
                print("using last calibration instance created: ",calibration)
            except:
                print("no calibration found")
        rotated = scipy.ndimage.rotate(frame, calibration.calibration_data['rot_angle'], reshape=False)
        cropped = rotated[calibration.calibration_data['y_lim'][0]: calibration.calibration_data['y_lim'][1],
                  calibration.calibration_data['x_lim'][0]: calibration.calibration_data['x_lim'][1]]
        cropped = numpy.flipud(cropped)
        return cropped


def Beamer(*args, **kwargs):
    warn("'Projector' class is deprecated due to the stupid german name. Use 'Projector' instead.")
    return Projector(*args, **kwargs)

class Projector:
    _ids = count(0)
    _instances = []

    def __init__(self, calibration=None, resolution=None):
        self.__class__._instances.append(weakref.proxy(self))
        self.id = next(self._ids)
        self.html_filename = "projector" + str(self.id) + ".html"
        self.frame_filenamne = "frame" + str(self.id) + ".png"
        self.work_directory = None
        self.html_file = None
        self.html_text = None
        self.frame_file = None
        self.drawdate = "false"  # Boolean as string for html, only used for testing.
        self.refresh = 100  # wait time in ms for html file to load image
        if resolution is None:
            resolution = (800, 600)
        self.resolution = resolution
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            self.calibration = Calibration(associated_projector=self)
            print("calibration not provided or invalid. a new calibration was created:", self.calibration)

    def calibrate(self):
        self.calibration.create()

    def start_stream(self):
        # def start_stream(self, html_file=self.html_file, frame_file=self.frame_file):
        if self.work_directory is None:
            self.work_directory = os.getcwd()
        self.html_file = open(os.path.join(self.work_directory, self.html_filename), "w")

        self.html_text = """
            <html>
            <head>
                <style>
                    body {{ margin: 0px 0px 0px 0px; padding: 0px; }} 
                </style>
            <script type="text/JavaScript">
            var url = "output.png"; //url to load image from
            var refreshInterval = {0} ; //in ms
            var drawDate = {1}; //draw date string
            var img;

            function init() {{
                var canvas = document.getElementById("canvas");
                var context = canvas.getContext("2d");
                img = new Image();
                img.onload = function() {{
                    canvas.setAttribute("width", img.width)
                    canvas.setAttribute("height", img.height)
                    context.drawImage(this, 0, 0);
                    if(drawDate) {{
                        var now = new Date();
                        var text = now.toLocaleDateString() + " " + now.toLocaleTimeString();
                        var maxWidth = 100;
                        var x = img.width-10-maxWidth;
                        var y = img.height-10;
                        context.strokeStyle = 'black';
                        context.lineWidth = 2;
                        context.strokeText(text, x, y, maxWidth);
                        context.fillStyle = 'white';
                        context.fillText(text, x, y, maxWidth);
                    }}
                }};
                refresh();
            }}
            function refresh()
            {{
                img.src = url + "?t=" + new Date().getTime();
                setTimeout("refresh()",refreshInterval);
            }}

            </script>
            <title>AR Sandbox output</title>
            </head>

            <body onload="JavaScript:init();">
            <canvas id="canvas"/>
            </body>
            </html>

            """
        self.html_text = self.html_text.format(self.refresh, self.drawdate)
        self.html_file.write(self.html_text)
        self.html_file.close()

        webbrowser.open_new('file://' + str(os.path.join(self.work_directory, self.html_filename)))

    def show(self, input='current_frame.png', legend_frame='legend.png', profile_frame='profile.png',
             hot_frame='hot.png', rescale=True):

        projector_output = Image.new('RGB', self.resolution)
        frame = Image.open(input)
        if rescale is True:
            projector_output.paste(frame.resize((int(frame.width * self.calibration.calibration_data['scale_factor']),
                                              int(frame.height * self.calibration.calibration_data['scale_factor']))),
                                (
                                self.calibration.calibration_data['x_pos'], self.calibration.calibration_data['y_pos']))
        else:
            projector_output.paste(frame, (self.calibration.calibration_data['x_pos'], self.calibration.calibration_data['y_pos']))

        if self.calibration.calibration_data['legend_area'] is not False:
            legend = Image.open(legend_frame)
            projector_output.paste(legend, (
            self.calibration.calibration_data['legend_x_lim'][0], self.calibration.calibration_data['legend_y_lim'][0]))
        if self.calibration.calibration_data['profile_area'] is not False:
            profile = Image.open(profile_frame)
            projector_output.paste(profile, (self.calibration.calibration_data['profile_x_lim'][0],
                                          self.calibration.calibration_data['profile_y_lim'][0]))
        if self.calibration.calibration_data['hot_area'] is not False:
            hot = Image.open(hot_frame)
            projector_output.paste(hot, (
            self.calibration.calibration_data['hot_x_lim'][0], self.calibration.calibration_data['hot_y_lim'][0]))

        projector_output.save('output.png')  # TODO: Projector specific outputs

    # TODO: threaded runloop exporting filtered and unfiltered depth

    def draw_markers(self, coords,image=None):
        """
        Draw markers onto an image at the given coordinates
        if image is a filename, the file will be overwritten. if image is an cv2 image object (numpy.ndarray), function will return an image object
        :param image:
        :param coords:
        :return:
        """
        if image is None:
            image=self.frame_file
        if type(image) is str:
            img = cv2.imread(image)
        if type(image) is numpy.ndarray:
            img=image
        for point in coords:
            cv2.circle(img, tuple(point), 6, (255, 255, 255), -1)
        if type(image) is str:
            cv2.imwrite(self.frame_file, img)
        else:
            return img

    def draw_line(self, coords, image=None):  # takes list of exactly 2 coordinate pairs
        if image is None:
            image=self.frame_file
        if type(image) is str:
            img = cv2.imread(image)
        if type(image) is numpy.ndarray:
            img=image
        lineThickness = 2
        cv2.line(img, tuple(coords[0]), tuple(coords[1]), (255, 255, 255), lineThickness)
        if type(image) is str:
            cv2.imwrite(self.frame_file, img)
        else:
            return img




class Calibration:  # TODO: add legend position; add rotation; add z_range!!!!
    _ids = count(0)
    _instances = []

    def __init__(self, associated_projector=None, associated_kinect=None):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))
        self.associated_projector = associated_projector
        self.projector_resolution = associated_projector.resolution
        self.associated_kinect = associated_kinect
        self.calibration_file = "calibration" + str(self.id) + ".dat"
        self.calibration_data = {'rot_angle': 0,  # TODO: refactor calibration_data as an inner class for type safety
                                 'x_lim': (0, 640),
                                 'y_lim': (0, 480),
                                 'x_pos': 0,
                                 'y_pos': 0,
                                 'scale_factor': 1.0,
                                 'z_range': (800, 1400),
                                 'box_dim': (400, 300),
                                 'legend_area': False,
                                 'legend_x_lim': (self.projector_resolution[1] - 50, self.projector_resolution[0] - 1),
                                 'legend_y_lim': (self.projector_resolution[1] - 100, self.projector_resolution[1] - 50),
                                 'profile_area': False,
                                 'profile_x_lim': (self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
                                 'profile_y_lim': (self.projector_resolution[1] - 100, self.projector_resolution[1] - 1),
                                 'hot_area': False,
                                 'hot_x_lim': (self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
                                 'hot_y_lim': (self.projector_resolution[1] - 100, self.projector_resolution[1] - 1)
                                 }

        self.cmap = None

    # ...

    def load(self, calibration_file=None):
        if calibration_file == None:
            calibration_file = self.calibration_file
        try:
            self.calibration_data = pickle.load(open(calibration_file, 'rb'))
        except OSError:
            print("calibration data file not found")

    def save(self, calibration_file=None):
        if calibration_file == None:
            calibration_file = self.calibration_file
        pickle.dump(self.calibration_data, open(calibration_file, 'wb'))
        print("calibration saved to " + str(calibration_file))

    def create(self):
        if self.associated_projector == None:
            try:
                self.associated_projector = Projector._instances[-1]
                print("no associated projector specified, using last projector instance created")
            except:
                print("Error: no Projector instance found.")

        if self.associated_kinect == None:
            try:
                self.associated_kinect = Kinect._instances[-1]
                print("no associated kinect specified, using last kinect instance created")
            except:
                print("Error: no kinect instance found.")

        def calibrate(rot_angle, x_lim, y_lim, x_pos, y_pos, scale_factor, z_range, box_width, box_height, legend_area,
                      legend_x_lim, legend_y_lim, profile_area, profile_x_lim, profile_y_lim, hot_area, hot_x_lim,
                      hot_y_lim, close_click):
            depth = self.associated_kinect.get_frame()
            depth_rotated = scipy.ndimage.rotate(depth, rot_angle, reshape=False)
            depth_cropped = depth_rotated[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            depth_masked = numpy.ma.masked_outside(depth_cropped, self.calibration_data['z_range'][0],
                                                   self.calibration_data['z_range'][
                                                       1])  # depth pixels outside of range are white, no data pixe;ls are black.

            self.cmap = matplotlib.colors.Colormap('viridis')
            self.cmap.set_bad('white', 800)
            plt.set_cmap(self.cmap)
            h = (y_lim[1] - y_lim[0]) / 100.0
            w = (x_lim[1] - x_lim[0]) / 100.0

            fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.pcolormesh(depth_masked, vmin=self.calibration_data['z_range'][0],
                          vmax=self.calibration_data['z_range'][1])
            plt.contour(depth_masked, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
            plt.savefig('current_frame.png', pad_inches=0)
            plt.close(fig)

            self.calibration_data = {'rot_angle': rot_angle,
                                     'x_lim': x_lim,
                                     # TODO: refactor calibration_data as an inner class for type safety
                                     'y_lim': y_lim,
                                     'x_pos': x_pos,
                                     'y_pos': y_pos,
                                     'scale_factor': scale_factor,
                                     'z_range': z_range,
                                     'box_dim': (box_width, box_height),
                                     'legend_area': legend_area,
                                     'legend_x_lim': legend_x_lim,
                                     'legend_y_lim': legend_y_lim,
                                     'profile_area': profile_area,
                                     'profile_x_lim': profile_x_lim,
                                     'profile_y_lim': profile_y_lim,
                                     'hot_area': hot_area,
                                     'hot_x_lim': hot_x_lim,
                                     'hot_y_lim': hot_y_lim
                                     }

            if self.calibration_data['legend_area'] is not False:
                legend = Image.new('RGB', (
                self.calibration_data['legend_x_lim'][1] - self.calibration_data['legend_x_lim'][0],
                self.calibration_data['legend_y_lim'][1] - self.calibration_data['legend_y_lim'][0]), color='white')
                ImageDraw.Draw(legend).text((10, 10), "Legend", fill=(255, 255, 0))
                legend.save('legend.png')
            if self.calibration_data['profile_area'] is not False:
                profile = Image.new('RGB', (
                self.calibration_data['profile_x_lim'][1] - self.calibration_data['profile_x_lim'][0],
                self.calibration_data['profile_y_lim'][1] - self.calibration_data['profile_y_lim'][0]), color='blue')
                ImageDraw.Draw(profile).text((10, 10), "Profile", fill=(255, 255, 0))
                profile.save('profile.png')
            if self.calibration_data['hot_area'] is not False:
                hot = Image.new('RGB', (self.calibration_data['hot_x_lim'][1] - self.calibration_data['hot_x_lim'][0],
                                        self.calibration_data['hot_y_lim'][1] - self.calibration_data['hot_y_lim'][0]),
                                color='red')
                ImageDraw.Draw(hot).text((10, 10), "Hot Area", fill=(255, 255, 0))
                hot.save('hot.png')
            self.associated_projector.show()
            if close_click == True:
                calibration_widget.close()

        calibration_widget = widgets.interactive(calibrate,
                                                 rot_angle=widgets.IntSlider(
                                                     value=self.calibration_data['rot_angle'], min=-180, max=180,
                                                     step=1, continuous_update=False),
                                                 x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['x_lim'][0],
                                                            self.calibration_data['x_lim'][1]],
                                                     min=0, max=640, step=1, continuous_update=False),
                                                 y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['y_lim'][0],
                                                            self.calibration_data['y_lim'][1]],
                                                     min=0, max=480, step=1, continuous_update=False),
                                                 x_pos=widgets.IntSlider(value=self.calibration_data['x_pos'], min=0,
                                                                         max=self.projector_resolution[0]),
                                                 y_pos=widgets.IntSlider(value=self.calibration_data['y_pos'], min=0,
                                                                         max=self.projector_resolution[1]),
                                                 scale_factor=widgets.FloatSlider(
                                                     value=self.calibration_data['scale_factor'], min=0.1, max=4.0,
                                                     step=0.01, continuous_update=False),
                                                 z_range=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['z_range'][0],
                                                            self.calibration_data['z_range'][1]],
                                                     min=500, max=2000, step=1, continuous_update=False),
                                                 box_width=widgets.IntSlider(value=self.calibration_data['box_dim'][0],
                                                                             min=0,
                                                                             max=2000, continuous_update=False),
                                                 box_height=widgets.IntSlider(value=self.calibration_data['box_dim'][1],
                                                                              min=0,
                                                                              max=2000, continuous_update=False),
                                                 legend_area=widgets.ToggleButton(
                                                     value=self.calibration_data['legend_area'],
                                                     description='display a legend',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'),
                                                 legend_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['legend_x_lim'][0],
                                                            self.calibration_data['legend_x_lim'][1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 legend_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['legend_y_lim'][0],
                                                            self.calibration_data['legend_y_lim'][1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 profile_area=widgets.ToggleButton(
                                                     value=self.calibration_data['profile_area'],
                                                     description='display a profile area',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'),
                                                 profile_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['profile_x_lim'][0],
                                                            self.calibration_data['profile_x_lim'][1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 profile_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['profile_y_lim'][0],
                                                            self.calibration_data['profile_y_lim'][1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 hot_area=widgets.ToggleButton(
                                                     value=self.calibration_data['hot_area'],
                                                     description='display a hot area for qr codes',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'),
                                                 hot_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['hot_x_lim'][0],
                                                            self.calibration_data['hot_x_lim'][1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 hot_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['hot_y_lim'][0],
                                                            self.calibration_data['hot_y_lim'][1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 close_click=widgets.ToggleButton(
                                                     value=False,
                                                     description='Close calibration',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'
                                                 )

                                                 )
        IPython.display.display(calibration_widget)

class Detector:
    """
    Detector for Objects or Markers in a specified Region, based on the RGB image from a kinect
    """
    #TODO: implement area of interest!
    def __init__(self):

        self.shapes=None
        self.circles=None
        self.circle_coords=None
        self.shape_coords=None

        #default parameters for the detection function:
        self.thresh_value=80
        self.min_area=30


    def where_shapes(self,image, thresh_value=None, min_area=None):
        """Get the coordinates for all detected shapes.

                Args:
                    image (image file): Image input.
                    min_area (int, float): Minimal area for a shape to be detected.
                Returns:
                    x- and y- coordinates for all detected shapes as a 2D array.

            """
        if thresh_value is None:
            thresh_value = self.thresh_value
        if min_area is None:
            min_area=self.min_area

        bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
        gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
        edge_detected_image = cv2.Canny(thresh, 75, 200)

        _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        contour_coords = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) & (len(approx) < 23) & (area > min_area)):
                contour_list.append(contour)
                contour_coords.append([cX, cY])
        self.shapes=numpy.array(contour_coords)


    def where_circles(self, image, thresh_value=None):
        """Get the coordinates for all detected circles.

                    Args:
                        image (image file): Image input.
                        thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                    Returns:
                        x- and y- coordinates for all detected circles as a 2D array.

                """
        if thresh_value is None:
            thresh_value = self.thresh_value
        #output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2 100)
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 2, numpy.array([]), 200, 8, 4, 8)

        if circles != [] and circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = numpy.round(circles[0, :]).astype("int")
            # print(circles)
            circle_coords = numpy.array(circles)[:, :2]
            dist = scipy.spatial.distance.cdist(circle_coords, circle_coords, 'euclidean')
            #minima = np.min(dist, axis=1)
            dist_bool = (dist > 0) & (dist < 5)
            pos = numpy.where(dist_bool == True)[0]
            grouped = circle_coords[pos]
            mean_grouped = (numpy.sum(grouped, axis=0) / 2).astype(int)
            circle_coords = numpy.delete(circle_coords, list(pos), axis=0)
            circle_coords = numpy.vstack((circle_coords, mean_grouped))

            self.circles = circle_coords.tolist()



    def filter_circles(self, shape_coords, circle_coords):
        dist = scipy.spatial.distance.cdist(shape_coords, circle_coords, 'euclidean')
        minima = numpy.min(dist, axis=1)
        non_circle_pos = numpy.where(minima > 10)
        return non_circle_pos

    def where_non_circles(self, image, thresh_value=None, min_area=None):
        if thresh_value is None:
            thresh_value = self.thresh_value
        if min_area is None:
            min_area=self.min_area
        shape_coords = self.where_shapes(image, thresh_value, min_area)
        circle_coords = self.where_circles(image, thresh_value)
        if len(circle_coords)>0:
            non_circles = self.filter_circles(shape_coords, circle_coords)
            return shape_coords[non_circles].tolist()  #ToDo: what is this output?
        else:
            return shape_coords.tolist()

    def get_shape_coords(self, image, thresh_value=None, min_area=None):
        """Get the coordinates for all shapes, classified as circles and non-circles.

                        Args:
                            image (image file): Image input.
                            thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                            min_area (int, float): Minimal area for a non-circle shape to be detected.
                        Returns:
                            x- and y- coordinates for all detected shapes as 2D arrays.
                            [0]: non-circle shapes
                            [1]: circle shapes

                    """
        if thresh_value is None:
            thresh_value = self.thresh_value
        if min_area is None:
            min_area=self.min_area
        non_circles = self.where_non_circles(image, thresh_value, min_area)
        circles = self.where_circles(image, thresh_value)

        return non_circles, circles


    def plot_all_shapes(self, image, thresh_value=None, min_area=None):
        """Plot detected shapes onto image.

                            Args:
                                image (image file): Image input.
                                thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
                                min_area (int, float): Minimal area for a non-circle shape to be detected.

                        """
        if thresh_value is None:
            thresh_value = self.thresh_value
        if min_area is None:
            min_area=self.min_area

        output = image.copy()
        non_circles, circles = self.get_shape_coords(image, thresh_value, min_area)
        for (x, y) in circles:
            cv2.circle(output, (x, y), 5, (0, 255, 0), 3)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        for (x, y) in non_circles:
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        out_image = numpy.hstack([image, output])
        plt.imshow(out_image)

    def non_circles_fillmask(self, image, th1=60, th2=80):   #TODO: what is this function?
        bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
        gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, th1, 1, cv2.THRESH_BINARY)[1]
        circle_coords = self.where_circles(image, th2)
        for (x, y) in circle_coords:           cv2.circle(thresh, (x, y), 20, 1, -1)
        return numpy.invert(thresh.astype(bool))


class Terrain:
    """
    simple module to visualize the topography in the sandbox with contours and a colormap.
    """
    def __init__(self,calibration=None, cmap='terrain', contours=True):
        """
        :type contours: boolean
        :type cmap: matplotlib colormap object or keyword
        :type calibration: Calibration object. By default the last created calibration is used.

        """
        if calibration is None:
            try:
                self.calibration = Calibration._instances[-1]
                print("using last calibration instance created: ", calibration)
            except:
                print("no calibration found")
                self.calibration = calibration

        self.cmap = cmap
        self.contours = contours
        self.main_levels = numpy.arange(0, 2000, 50)
        self.sub_levels = numpy.arange(0, 2000, 10)


    def setup(self):
        pass

    def render_frame(self,depth):
        depth_rotated = scipy.ndimage.rotate(depth, self.calibration.calibration_data['rot_angle'], reshape=False)
        depth_cropped = depth_rotated[self.calibration.calibration_data['y_lim'][0]:self.calibration.calibration_data['y_lim'][1],
                        self.calibration.calibration_data['x_lim'][0]:self.calibration.calibration_data['x_lim'][1]]
        depth_masked = numpy.ma.masked_outside(depth_cropped, self.calibration.calibration_data['z_range'][0],
                                               self.calibration.calibration_data['z_range'][
                                                   1])  # depth pixels outside of range are white, no data pixe;ls are black.

        h = self.calibration.calibration_data['scale_factor'] * (
                    self.calibration.calibration_data['y_lim'][1] - self.calibration.calibration_data['y_lim'][0]) / 100.0
        w = self.calibration.calibration_data['scale_factor'] * (
                self.calibration.calibration_data['x_lim'][1] - self.calibration.calibration_data['x_lim'][0]) / 100.0

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
        ax.pcolormesh(depth_masked, vmin=self.calibration.calibration_data['z_range'][0],
                      vmax=self.calibration.calibration_data['z_range'][1], cmap=self.cmap)
        plt.savefig('current_frame.png', pad_inches=0)
        plt.close(fig)

class Module:
    """
    container for modules that handles threading. any kind of module can be loaded, as long as it contains a 'setup' and 'render_frame" method!
    """
    _ids = count(0)
    _instances = []

    def __init__(self, module, kinect=None, calibration=None, projector=None):

        if kinect is None:
            try:
                self.kinect = Kinect._instances[-1]
                print("using last kinect instance created: ", self.kinect)
            except:
                print("no kinect found")
                self.kinect = kinect


        if calibration is None:
            try:
                self.calibration = Calibration._instances[-1]
                print("using last calibration instance created: ", self.calibration)
            except:
                print("no calibration found")
                self.calibration = calibration

        if projector is None:
            try:
                self.projector = Projector._instances[-1]
                print("using last projector instance created: ", self.projector)
            except:
                print("no projector found")
                self.projector=projector

        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))

        self.module = module
        self.thread = None
        self.lock = threading.Lock()
        self.stop_thread = False

        #controlParameters:


    def loop(self):
        while self.stop_thread is False:
            depth=self.kinect.get_filtered_frame()
            self.module.render_frame(depth)
            self.beamer.show()

    def run(self):
        self.stop_threat= False
        self.module.setup()
        self.lock.acquire()
        self.thread = threading.Thread(target=self.loop, daemon=None)
        self.thread.start()
        # with thread and thread lock move these to main sandbox


    def pause(self):
        self.lock.release()

    def resume(self):
        self.lock.acquire()

    def kill(self):
        self.stop_threat=True
        try:
            self.lock.release()
        except:
            pass



def detect_shapes(kinect, model, calibration, frame=None):
    if frame is None:
        frame = kinect.get_RGB_frame()
    rotated_frame = scipy.ndimage.rotate(frame, calibration.calibration_data['rot_angle'], reshape=False)
    cropped_frame = rotated_frame[calibration.calibration_data['y_lim'][0]:calibration.calibration_data['y_lim'][1],
                    calibration.calibration_data['x_lim'][0]:calibration.calibration_data['x_lim'][1]]
    squares, circles = Detector.get_shape_coords(cropped_frame)

    for square in squares:
        print(square)






def render_depth_frame(calibration=None, kinect=None, projector=None, filter_depth=True, n_frames=5, sigma_gauss=4,
                       cmap='terrain'):  ##TODO:remove duplicates in run_depth
    pass


def render_depth_diff_frame(target_depth, kinect, projector):
    pass


def run_depth_diff(target_depth, kinect, projector):
    pass


"""

not yet finished functions:

def array_lookup(self, output_array=numpy.zeros((480, 640))):
    for index, x in numpy.ndenumerate(depth):  # can we find a solution with slicing? Takes almost a second!
        if output_z_range[0] < depth[index] < output_z_range[1]:
            output_array[index] = lith_block_reshaped[int(index[0] / output_res_y * model_res_y), int(
                index[1] / output_res_x * model_res_x), model_res_z - int(
                (depth[index] - output_z_range[0]) / (output_z_range[1] - output_z_range[0]) * model_res_z)]
        else:
            output_array[index] = 0
    return output_array

##deprecated:
def run():
    freenect.runloop(depth=display_depth,
                     video=display_rgb,
                     body=body)
                        def test(self):
        gempy.plot_section(self.model._geo_data, gempy.compute_model(self.model)[0], cell_number=0, direction='y', plot_data=False)

    def render_frame_old(self, outfile=None):
        lith_block, fault_block = gempy.compute_model_at(self.depth_grid, self.model)
        #gp.plot_section(geo_data, lith_block[0], cell_number=0, direction='z', ar_output='current_out.png')
        cell_number = 0
        direction = 'z'
        plotter=gempy.PlotData2D(self.model._geo_data)

        block=lith_block[0]
        print(block)
        plot_block = block.reshape(plotter._data.resolution[0], plotter._data.resolution[1], plotter._data.resolution[2])
        #plot_block = block.reshape((self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0],self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0], 1)) ##check ihere first when sequence is wrong
        print(numpy.shape(plot_block))
        _a, _b, _c, extent_val, x, y = plotter._slice(direction, cell_number)[:-2]


        h = (self.associated_calibration.calibration_data['y_lim'][1] -
             self.associated_calibration.calibration_data['y_lim'][0]) / 100.0
        w = (self.associated_calibration.calibration_data['x_lim'][1] -
             self.associated_calibration.calibration_data['x_lim'][0]) / 100.0
        print(h,w)
        fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(plot_block[_a, _b, _c].T, origin="bottom", cmap=plotter._cmap, norm=plotter._norm )
        if outfile==None:
           # plt.show(fig )
            ...
        #plt.close(fig)
           # return fig
        else:
            plt.savefig(outfile, pad_inches=0)
            plt.close(fig)

"""
