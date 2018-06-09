import sys
import os
#sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import freenect
import webbrowser
import pickle
import weakref
import numpy
import scipy
import gempy
#from gempy.plotting.colors import color_lot, cmap, norm
from itertools import count
from PIL import Image
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib


# TODO: Superclass or not? methods: run sandbox with runnable, height map only, diff height...
class Kinect: # add dummy
    _ids = count(0)
    _instances = []

    def __init__(self, dummy=False, mirror=True):
        self.__class__._instances.append(weakref.proxy(self))
        self.id = next(self._ids)
        self.resolution = (640, 480)
        self.dummy=dummy
        self.mirror=mirror

        if self.dummy==False:
            print("looking for kinect...")
            self.ctx = freenect.init()
            self.dev = freenect.open_device(self.ctx, self.id)
            print(self.id)
            freenect.close_device(self.dev)  # TODO Test if this has to be done!

            self.angle = None
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]  # get the first Depth frame already (the first one takes much longer than the following)
            self.filtered_depth = None
            print("kinect initialized")
        else:
            print ("dummy mode. get_frame() will return a synthetic depth frame, other functions may not work")

    def set_angle(self, angle):
        self.angle = angle
        freenect.set_tilt_degs(self.dev, self.angle)

    def get_frame(self, horizontal_slice=None):
        if self.dummy==False:
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            self.depth=numpy.fliplr(self.depth)
            return self.depth
        else:
            synth_depth = numpy.zeros((480, 640))
            for x in range(640):
                for y in range(480):
                    if horizontal_slice==None:
                        synth_depth[y, x] = int(800 + 200 * (numpy.sin(2 * numpy.pi * x / 320)))
                    else:
                        synth_depth[y, x] = horizontal_slice
            self.depth=synth_depth
            return self.depth

    def get_filtered_frame(self, n_frames=5, sigma_gauss= None):
        if self.dummy==True:
            self.get_frame()
            return self.depth
        else:
            depth_array = self.get_frame()
            for i in range(n_frames - 1):
                depth_array = numpy.dstack([depth_array, self.get_frame()])
            depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)
            self.depth=numpy.ma.mean(depth_array_masked, axis=2)
            if sigma_gauss:
                self.depth=scipy.ndimage.filters.gaussian_filter(self.depth, sigma_gauss)
            return self.depth


class Beamer:
    _ids = count(0)
    _instances = []

    def __init__(self, calibration=None, resolution=(800,600)):
        self.__class__._instances.append(weakref.proxy(self))
        self.id = next(self._ids)
        self.html_filename = "beamer" + str(self.id) + ".html"
        self.frame_filenamne = "frame" + str(self.id) + ".png"
        self.work_directory=None
        self.html_file = None
        self.html_text = None
        self.frame_file = None
        self.drawdate = "false"  # Boolean as string for html, only used for testing.
        self.refresh = 100  # wait time in ms for html file to load image
        self.resolution = resolution
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            self.calibration = Calibration(associated_beamer=self)
            print("calibration not provided or invalid. a new calibration was created.")

    def calibrate(self):
        self.calibration.create()

    def start_stream(self):
        # def start_stream(self, html_file=self.html_file, frame_file=self.frame_file):
        if self.work_directory==None:
            self.work_directory=os.getcwd()
        self.html_file = open(os.path.join(self.work_directory,self.html_filename), "w")

        self.html_text ="""
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
        self.html_text=self.html_text.format(self.refresh, self.drawdate)
        self.html_file.write(self.html_text)
        self.html_file.close()

        webbrowser.open_new('file://'+str(os.path.join(self.work_directory,self.html_filename)))

    def show(self, input='current_frame.png'):

        beamer_output = Image.new('RGB', self.resolution)
        frame = Image.open(input)
        beamer_output.paste(frame.resize((int(frame.width * self.calibration.calibration_data['scale_factor']), int(frame.height * self.calibration.calibration_data['scale_factor']))),
                            (self.calibration.calibration_data['x_pos'], self.calibration.calibration_data['y_pos']))
        beamer_output.save('output.png') #TODO: Beamer specific outputs

    # TODO: threaded runloop exporting filtered and unfiltered depth


class Calibration:  # TODO: add legend position; add rotation; add z_range!!!!
    _ids = count(0)
    _instances = []

    def __init__(self, associated_beamer=None,associated_kinect=None):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))
        self.associated_beamer = associated_beamer
        self.beamer_resolution = associated_beamer.resolution
        self.associated_kinect = associated_kinect
        self.calibration_file = "calibration" + str(self.id) + ".dat"
        self.calibration_data = {'rot_angle': 0, # TODO: refactor calibration_data as an inner class for type safety
                                 'x_lim': (0, 640),
                                 'y_lim': (0, 480),
                                 'x_pos': 0,
                                 'y_pos': 0,
                                 'scale_factor': 1.0,
                                 'z_range':(800,1400),
                                 'box_dim':(400,300)}
        self.cmap=None
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
        if self.associated_beamer==None:
            try:
                self.associated_beamer = Beamer._instances[-1]
                print("no associated beamer specified, using last beamer instance created")
            except:
                print("Error: no Beamer instance found.")

        if self.associated_kinect==None:
            try:
                self.associated_kinect = Kinect._instances[-1]
                print("no associated kinect specified, using last kinect instance created")
            except:
                print("Error: no kinect instance found.")

        def calibrate(rot_angle, x_lim, y_lim, x_pos, y_pos, scale_factor, z_range, box_width,box_height, close_click):
            depth = self.associated_kinect.get_frame()
            depth_rotated = scipy.ndimage.rotate(depth,rot_angle, reshape=False )
            depth_cropped = depth_rotated[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            depth_masked=numpy.ma.masked_outside(depth_cropped,self.calibration_data['z_range'][0],self.calibration_data['z_range'][1]) #depth pixels outside of range are white, no data pixe;ls are black.

            self.cmap=matplotlib.colors.Colormap('hsv')
            self.cmap.set_bad('white',800)
            plt.set_cmap(self.cmap)
            h = (y_lim[1]-y_lim[0])  / 100.0
            w = (x_lim[1]-x_lim[0]) / 100.0

            fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.pcolormesh(depth_masked, vmin=self.calibration_data['z_range'][0], vmax=self.calibration_data['z_range'][1])
            plt.savefig('current_frame.png', pad_inches=0)
            plt.close(fig)

            self.calibration_data = {'rot_angle': rot_angle,
                                 'x_lim': x_lim,  # TODO: refactor calibration_data as an inner class for type safety
                                 'y_lim': y_lim,
                                 'x_pos': x_pos,
                                 'y_pos': y_pos,
                                 'scale_factor': scale_factor,
                                 'z_range':z_range,
                                 'box_dim': (box_width,box_height)}
            self.associated_beamer.show()
            if close_click==True:
                calibration_widget.close()

        calibration_widget = widgets.interactive(calibrate,
                                                 rot_angle=widgets.IntSlider(
                                                     value=self.calibration_data['rot_angle'], min=-180, max=180, step=1,continuous_update=False),
                                                 x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['x_lim'][0], self.calibration_data['x_lim'][1]],
                                                     min=0, max=640, step=1,continuous_update=False),
                                                 y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['y_lim'][0], self.calibration_data['y_lim'][1]],
                                                     min=0, max=480, step=1,continuous_update=False),
                                                 x_pos=widgets.IntSlider(value=self.calibration_data['x_pos'], min=0,
                                                                         max=self.beamer_resolution[0]),
                                                 y_pos=widgets.IntSlider(value=self.calibration_data['y_pos'], min=0,
                                                                         max=self.beamer_resolution[1]),
                                                 scale_factor=widgets.FloatSlider(
                                                     value=self.calibration_data['scale_factor'], min=0.1, max=4.0,
                                                     step=0.01,continuous_update=False),
                                                 z_range=widgets.IntRangeSlider(
                                                     value=[self.calibration_data['z_range'][0], self.calibration_data['z_range'][1]],
                                                     min=500, max=2000, step=1, continuous_update=False),
                                                 box_width=widgets.IntSlider(value=self.calibration_data['box_dim'][0], min=0,
                                                                         max=2000,continuous_update=False),
                                                 box_height=widgets.IntSlider(value=self.calibration_data['box_dim'][1], min=0,
                                                                         max=2000,continuous_update=False),
                                                 close_click=widgets.ToggleButton(
                                                    value=False,
                                                    description='Close calibration',
                                                    disabled=False,
                                                    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                    tooltip='Description',
                                                    icon='check'
                                                    )

                                                 )
        display(calibration_widget)


class Model:
    _ids = count(0)
    _instances = []

    def __init__(self, model, extent=None, associated_calibration=None, xy_isometric=True):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))
        self.xy_isometric = xy_isometric
        self.scale = [None,None,None]
        self.pixel_size = [None, None]

        self.legend = True
        self.model = model
        gempy.compute_model(self.model)
        self.empty_depth_grid = None
        self.depth_grid = None
        self.cmap = None
        self.norm = None
        self.lot = None

        if associated_calibration==None:
            try:
                self.associated_calibration = Calibration._instances[-1]
                print("no calibration specified, using last calibration instance created")
            except:
                print("ERROR: no calibration instance found. please create a calibration")
                # parameters from the model:
        if extent == None:  # extent should be array with shape (6,) or convert to list?
            self.extent = self.model._geo_data.extent

        else:
            self.extent = extent #check: array with 6 entries!

    def calculate_scales(self):
        output_res = (self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0],
                      self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0])
        self.pixel_size[0] = float(self.extent[1] - self.extent[0]) / float(output_res[0])
        self.pixel_size[1] = float(self.extent[3] - self.extent[2]) / float(output_res[1])

        if self.xy_isometric==True: #model is scaled to fit into box
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_size[0] >= self.pixel_size[1]:
                self.pixel_size[1] = self.pixel_size[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_size[0] = self.pixel_size[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_size[0]
        self.scale[1] = self.pixel_size[1]
        self.scale[2] = float(self.extent[5]-self.extent[4]) / (self.associated_calibration.calibration_data['z_range'][1]-self.associated_calibration.calibration_data['z_range'][0])
        print("scale in Model units/ mm (X,Y,Z): "+str(self.scale))


    #TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.

    def create_empty_depth_grid(self):
        grid_list = []
        output_res = (self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0],
                      self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0])
        for x in range(output_res[1]):
            for y in range(output_res[0]):
                grid_list.append([ y * self.pixel_size[1]+self.extent[2],x * self.pixel_size[0]+self.extent[0]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid
       # return self.empty_depth_grid

    def update_grid(self,depth):
        filtered_depth = numpy.ma.masked_outside(depth,self.associated_calibration.calibration_data['z_range'][0],self.associated_calibration.calibration_data['z_range'][1])
        scaled_depth = self.extent[5] - ((filtered_depth - self.associated_calibration.calibration_data['z_range'][0]) / (self.associated_calibration.calibration_data['z_range'][1] - self.associated_calibration.calibration_data['z_range'][0]) * (self.extent[5] - self.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth,  self.associated_calibration.calibration_data['rot_angle'], reshape=False)
        cropped_depth = rotated_depth[self.associated_calibration.calibration_data['y_lim'][0] : self.associated_calibration.calibration_data['y_lim'][1], self.associated_calibration.calibration_data['x_lim'][0] : self.associated_calibration.calibration_data['x_lim'][1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)
        self.depth_grid=depth_grid

    def render_frame(self, outfile=None):
        if self.cmap == None:
            plotter = gempy.PlotData2D(self.model._geo_data)
            self.cmap = plotter._cmap
            self.norm = plotter._norm
            self.lot = plotter._color_lot

        lith_block, fault_block = gempy.compute_model_at(self.depth_grid, self.model)
        block=lith_block[0].reshape((self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0],self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0]))
        h = (self.associated_calibration.calibration_data['y_lim'][1] - self.associated_calibration.calibration_data['y_lim'][0]) / 100.0
        w = (self.associated_calibration.calibration_data['x_lim'][1] - self.associated_calibration.calibration_data['x_lim'][0]) / 100.0

        fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.pcolormesh(block, cmap=self.cmap, norm=self.norm )

        if outfile==None:
                plt.show()
                plt.close()
        else:
            plt.savefig(outfile, pad_inches=0)
            plt.close(fig)

    def create_legend(self):
        #...
        pass

    def setup(self, start_stream=True):
        if start_stream==True:
            self.associated_calibration.associated_beamer.start_stream()
        self.calculate_scales()
        self.create_empty_depth_grid()

    def run(self):
        run_model(self)


def run_model(model, calibration=None, kinect=None, beamer=None, filter_depth=True, n_frames=5, sigma_gauss=4 ):  # continous run functions with exit handling
    if calibration == None:
        calibration = model.associated_calibration
    if kinect == None:
        kinect = calibration.associated_kinect
    if beamer == None:
        beamer = calibration.associated_beamer

    while True:
        if filter_depth == True:
            depth = kinect.get_filtered_frame(n_frames=n_frames, sigma_gauss=sigma_gauss)
        else:
            depth = kinect.get_frame()

        model.update_grid(depth)
        model.render_frame(outfile="current_frame.png")
        beamer.show(input="current_frame.png")


def run_depth(calibration=None, kinect=None, beamer=None, filter_depth=True, n_frames=5, sigma_gauss=4,  cmap='terrain'):
    if calibration is None:
        try:
            calibration=Calibration._instances[-1]
            print("using last calibration instance created.")
        except:
            print("no calibration found")
    if kinect is None:
        kinect = calibration.associated_kinect
    if beamer is None:
        beamer = calibration.associated_beamer

    while True:
        if filter_depth == True:
            depth = kinect.get_filtered_frame(n_frames=n_frames, sigma_gauss=sigma_gauss)
        else:
            depth = kinect.get_frame()

        depth_rotated = scipy.ndimage.rotate(depth, calibration.calibration_data['rot_angle'], reshape=False)
        depth_cropped = depth_rotated[calibration.calibration_data['y_lim'][0]:calibration.calibration_data['y_lim'][1], calibration.calibration_data['x_lim'][0]:calibration.calibration_data['x_lim'][1]]
        depth_masked = numpy.ma.masked_outside(depth_cropped, calibration.calibration_data['z_range'][0],
                                               calibration.calibration_data['z_range'][
                                                   1])  # depth pixels outside of range are white, no data pixe;ls are black.

        h = (calibration.calibration_data['y_lim'][1] - calibration.calibration_data['y_lim'][0]) / 100.0
        w = (calibration.calibration_data['x_lim'][1] - calibration.calibration_data['x_lim'][0]) / 100.0

        fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.pcolormesh(depth_masked, vmin=calibration.calibration_data['z_range'][0], vmax=calibration.calibration_data['z_range'][1],cmap=cmap)
        plt.savefig('current_frame.png', pad_inches=0)
        plt.close(fig)
        beamer.show(input='current_frame.png')



def render_depth_frame(kinect, beamer, cmap='viridis'):
    pass


def render_depth_diff_frame(target_depth, kinect, beamer):
    pass

def run_depth_diff(target_depth, kinect, beamer):
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