"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt


def series_anchor_point(series):
    n_series = series.columns.shape[0]
    anch_series_pos = np.linspace(0, 10, n_series, endpoint=False)

    return pn.DataFrame(anch_series_pos.reshape(1, -1), columns=series.columns)

class StratigraphicPile(object):
    def __init__(self, geo_data):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(-5,15)
        self.anch_series = series_anchor_point(geo_data.series)
        global series_rect
        series_rect = []
        # for series in geo_data.series.columns:
        #     rect2 = ax.barh(self.anch_series[series],
        #                    2,
        #                    10/len(geo_data.series.columns)-1)
        print(self.anch_series.as_matrix())
        pos_anch = np.squeeze(self.anch_series.as_matrix())
        rects = ax.barh(pos_anch, np.ones_like(pos_anch)*2, 9/(len(geo_data.series.columns)))

        for e, series in enumerate(geo_data.series.columns):
            rects[0].set_color('red')
            dr = DraggableRectangle(rects[e], geo_data, series)
            dr.connect()
            series_rect.append(dr)

        plt.show()

class DraggableRectangle:
    def __init__(self, rect, geo_data, s):
        self.geo_data = geo_data

        self.rect = rect
        self.rect.s = s
        self.press = None

        self.anch_series = series_anchor_point(self.geo_data.series)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidpick = self.rect.figure.canvas.mpl_connect(
            'pick_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
       # print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        print(self.rect, self.rect.s)

    def pick_handler(event):
        mouseevent = event.mouseevent
        artist = event.artist
        print(artist)


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
       # dx = event.xdata - xpress
        dy = event.ydata - ypress
        # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
      #  self.rect.set_x(x0 + dx)
        self.rect.set_y(y0 + dy)

        self.rect.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.rect.figure.canvas.draw()
        print(self.rect, self.rect.s)

        old_arch = np.copy(self.anch_series[self.rect.s].values)
        new_arch, old_pos = self.compute_new_arch()

        self.anch_series[self.rect.s] = new_arch
        print(old_pos, 'old pos')
        self.anch_series.iloc[0, old_pos] = old_arch
        print(self.anch_series, 'anch series')


       # self.rect.set_y(self.anch_series[self.s].values-2)
        for r in series_rect:
            r.rect.set_y(self.anch_series[r.rect.s].values-2)
            r.rect.set_animated(False)
            r.background = None

            # redraw the full figure
            r.rect.figure.canvas.draw()
        # Reset figure
        # self.rect.set_animated(False)
        # self.background = None
        #
        # # redraw the full figure
        # self.rect.figure.canvas.draw()

    def compute_new_arch(self):

        dist = np.abs(self.anch_series.as_matrix() - self.rect.get_y())

        arg_min = np.argmin(dist)

        new_arch = self.anch_series.iloc[0, arg_min]
        return new_arch, arg_min

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../")

# Importing gempy
import gempy as gp


# Aux imports
import numpy as np

geo_data = gp.read_pickle('../input_data/NoFault.pickle')

StratigraphicPile(geo_data)


