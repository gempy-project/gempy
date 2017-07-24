# draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations
import numpy as np
import matplotlib.pyplot as plt


class DraggableRectangle:
    lock = None  # only one can be animated at a time

    def __init__(self, rect, parent=None):
        self.rect = rect
        self.parent = parent
        self.press = None
        self.background = None

        self.o_x = None
        self.o_y = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes:
            return
        if DraggableRectangle.lock is not None:
            return

        # save the original position for snapping back
        self.o_y = self.rect.get_y()
        self.o_x = self.rect.get_x()

        contains, attrd = self.rect.contains(event)
        if not contains: return
        print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        DraggableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes:
            return

        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        # self.rect.set_x(x0 + dx)
        self.rect.set_y(y0 + dy)

        if self.rect.get_y() > self.parent.bars[self.i + 1].rect.get_y() + self.rect.get_height() / 2:
            # reduce i for above
            self.parent.bars[self.i + 1].i -= 1
            y_bot = self.get_y()
            self.rect.set_y(self.parent.bars[self.i + 1].rect.get_y())
            self.parent.bars[self.i + 1].set_y(y_bot)
            # swap positions
            self.parent.bars[self.i], self.parent.bars[self.i + 1] = self.parent.bars[self.i + 1], self.parent.bars[self.i]
            # increase i for self
            self.i += 1


        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        # check if it is released past half of the above or below
        # get y of above and below
        if self.i == 0:
            y_below = None
        else:
            y_below = self.parent.bars[self.i - 1].rect.get_y()

        if self.i == len(self.parent.bars) - 1:
            y_above = None
        else:
            y_above = self.parent.bars[self.i + 1].rect.get_y()

        # if the released rect is not above
        if y_above is None or y_below is None:
            self.rect.set_y(self.o_y)

        if self.rect.get_y() <= (y_above + self.rect.get_height() / 2):
            self.rect.set_y(self.o_y)
        elif self.rect.get_y() >= (y_below + self.rect.get_height() / 2):
            self.rect.set_y(self.o_y)

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


class DaPlot:
    def __init__(self, formations, series):
        self.formations = formations
        self.series = series

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.barplot = self.ax.barh(range(len(self.formations)), [1 for i in range(len(self.formations))])
        self.bars = []

        self.ax.xaxis.set_visible(False)
        self.ax.set_yticklabels(self.formations)

        for i, entry in enumerate(self.barplot):
            bar = DraggableRectangle(entry, parent=self)
            bar.i = i
            bar.connect()
            self.bars.append(bar)

        plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #formations = ax.barh(range(10), 20 * np.random.rand(10))
    # list of DraggableRectangle instances
    #drs = []

    #for i, fmt in enumerate(formations):
    #    # create draggable rectangle from all created bars
    #    dr = DraggableRectangle(fmt)
    #    dr.i = i
    #    dr.connect()
    #    drs.append(dr)  # add instance to list of all rectangles
    #print(drs)
    #plt.show()

import pandas as pn

formations = np.array(['Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Fault'], dtype=object)
series = pn.Index(['fault', 'Rest'], dtype='object')

plot = DaPlot(formations, series)
