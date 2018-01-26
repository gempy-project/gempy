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
from gempy.colors import *
import matplotlib.cm as cm
from gempy.colors import color_lot, cmap, norm


def _create_color_lot(geo_data, cd_rgb):
    """Returns color [r,g,b] LOT for formation numbers."""
    if "formation number" not in geo_data.interfaces or "formation number" not in geo_data.foliations:
        geo_data.set_formation_number()  # if not, set formation numbers

    c_names = ["indigo", "red", "yellow", "brown", "orange",
                "green", "blue", "amber", "pink", "light-blue",
                "lime", "blue-grey", "deep-orange", "grey", "cyan",
                "deep-purple", "purple", "teal", "light-green"]

    lot = {}
    ci = 0  # use as an independent running variable because of fault formations
    # get unique formation numbers
    fmt_numbers = np.unique([val for val in geo_data.interfaces['formation number'].unique()])
    # get unique fault formation numbers
    fault_fmt_numbers = np.unique(geo_data.interfaces[geo_data.interfaces["isFault"] == True]["formation number"])
    # iterate over all unique formation numbers
    for i, n in enumerate(fmt_numbers):
        # if its a fault formation set it to black by default
        if n in fault_fmt_numbers:
            lot[n] = cd_rgb["black"]["400"]
        # if not, just go through
        else:
            lot[n] = cd_rgb[c_names[ci]]["400"]
            ci += 1

    return lot


def set_anchor_points(geo_data):
    """
    Compute the location of each series and formation depending on the number of those

    Args:
        geo_data (gempy.data_management.InputData):

    Returns:
        list:
        - DataFrame: location of the series

        - DataFrame: location of the formations

        - float: thickness of the series

        - list, floats:

    """
    # Formations per serie
    for_ser = geo_data.interfaces.groupby('series')
    series_names = geo_data.series.columns
    # Get number of series
    n_series = len(geo_data.series.columns)

    # Make anchor points for each serie
    anch_series_pos_aux = np.linspace(10, 0, n_series, endpoint=True)
    anch_series_pos = pn.DataFrame(anch_series_pos_aux.reshape(1, -1),
                                   columns=series_names)
    # Thicknes of series. We just make sure we have white space in between
    thick_series = 11.5 / n_series

    # Setting formations anchor
    anch_formations_pos = pn.DataFrame()
    thick_formations = []
    for series in series_names:
        try:
            formations = for_ser.formation.unique()[series]
        except KeyError:
            formations = np.empty(0, dtype='object')
        formations = np.insert(formations, 0, '0_aux' + series)
        formations = np.append(formations, '1_aux' + series)
        anch_for_df = pn.DataFrame(
            np.linspace(anch_series_pos[series][0] + thick_series / 2,
                        anch_series_pos[series][0] - thick_series / 2,
                        formations.shape[0],
                        endpoint=True).reshape(1, -1),
            columns=formations)

        anch_formations_pos = pn.concat([anch_formations_pos, anch_for_df],
                                        axis=1)
        thick_formations = np.append(thick_formations,
                                     (np.tile((thick_series - 2) / formations.shape[0], formations.shape[0])))

    return anch_series_pos, anch_formations_pos, thick_series, thick_formations


plt.style.use(['seaborn-white', 'seaborn-talk'])


class StratigraphicPile(object):
    """
    Class to create the interactive stratigraphic pile
    """
    def __init__(self, geo_data):

        # Set the values of matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 7)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.axis('off')

        # Compute the anchor values for the number of series. This is a DataFrame
        self.anch_series, self.anch_formations, self.thick_series, self.thick_formations = set_anchor_points(geo_data)

        # We create the list that contains rectangles that represent our series ang are global
        global series_rect
        series_rect = {}

        # Define the initial value of each rectangle
        pos_anch = np.squeeze(self.anch_series.as_matrix())
        rects = ax.barh(pos_anch, np.ones_like(pos_anch)*2, self.thick_series, )

        # We connect each rectangle
        for e, series in enumerate(geo_data.series.columns):
            # TODO Alex set the colors of the series accordingly

            rects[e].set_color(cm.Dark2(e))
            rects[e].set_label(series)
            dr = DraggableRectangle(rects[e], geo_data, series)
            dr.connect()
            dr.rect.f = None
            dr.rect.s = series

            series_rect[series] = dr

        global formation_rect
        formation_rect = {}

        # Define the initial value of each rectangle
        pos_anch = np.squeeze(self.anch_formations.as_matrix())
        rects = ax.barh(pos_anch, np.ones_like(pos_anch)*2, .5, left=3.)

        color = 1
        # We connect each rectangle
        for e, formation in enumerate(self.anch_formations.columns):

            if 'aux' in formation:
                rects[e].set_alpha(.1)
                rects[e].set_color('gray')
            else:

                rects[e].set_color(cmap(color))
                rects[e].set_label(formation)
                color += 1

            dr = DraggableRectangle(rects[e], geo_data, formation)
            dr.connect()
            dr.rect.f = formation
            dr.rect.s = None

            formation_rect[formation] = dr
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.ion()
        ax.text(1, self.anch_series.max().values.max() + self.thick_series/2 + 2, r'Series', fontsize=15,
                fontweight='bold', bbox={'facecolor':'gray', 'alpha':0.5, 'pad':10}, horizontalalignment='center')
        ax.text(4, self.anch_series.max().values.max() + self.thick_series/2 + 2, r'Faults/Formations', fontsize=15,
                fontweight='bold', bbox={'facecolor':'gray', 'alpha':0.5, 'pad':10}, horizontalalignment='center')

        self.figure = plt.gcf()


class DraggableRectangle:
    def __init__(self, rect, geo_data, s):
        # The idea of passing geodata is to update the dataframes in place
        self.geo_data = geo_data

        self.rect = rect

        # We add the name of the series as attribute of the rectangle
        self.press = None

        # We initalize the placement of the anchors
        self.anch_series, self.anch_formations, self.thick_series, self.thick_formations = set_anchor_points(self.geo_data)

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
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)

        if not contains: return

        x0, y0 = self.rect.xy

        # We detect the series that has been touched.
        self.selected_rectangle_s = self.rect.s
        self.selected_rectangle_f = self.rect.f

        # We pass all the important attributes through this property
        self.press = x0, y0, event.xdata, event.ydata, self.selected_rectangle_s, self.selected_rectangle_f

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press[:-2]

        # We forbid movement in x
        dy = event.ydata - ypress
        self.rect.set_y(y0 + dy)
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'

        # We extract the rectangle that was clicked
        if self.press is None: return
        selected_rectangle_s = self.press[-2]
        selected_rectangle_f = self.press[-1]
        self.press = None
        self.rect.figure.canvas.draw()

        if self.selected_rectangle_s is not None:
            # Make a copy of the position where the selected rectangle was
            selected_arch_position_s = np.copy(self.anch_series[selected_rectangle_s].values)

            # Compute the position of the closes anchor point and the argument of the rectangle which was there
            dropping_arch_position, dropping_arg = self.compute_new_arch_series()

            # Here we change the selected rectangle to the position to the dropping position
            self.anch_series[selected_rectangle_s] = dropping_arch_position

            # Here we change the rectangle that was on the dropping position to the original position of the selected
            # rectangle
            self.anch_series.iloc[0, dropping_arg] = selected_arch_position_s

            self.update_data_frame()
            self.anch_series, self.anch_formations, self.thick_series, self.thick_formations = set_anchor_points(self.geo_data)

            # We update the visualization of all the rectangles
            for series_name in self.anch_series.columns:
                series_rect[series_name].rect.set_y(self.anch_series[series_rect[series_name].rect.s].values-self.thick_series/2)
                series_rect[series_name].rect.set_animated(False)
                series_rect[series_name].rect.background = None

                # redraw the full figure
                series_rect[series_name].rect.figure.canvas.draw()

        if self.selected_rectangle_f is not None:

            # Make a copy of the position where the selected rectangle was
            selected_arch_position_f = np.copy(self.anch_formations[selected_rectangle_f].values)
            # selected_arch_position_f = np.copy(self.anch_formations[selected_rectangle].values)

            # Compute the position of the closes anchor point and the argument of the rectangle which was there
            dropping_arch_position, dropping_arg = self.compute_new_arch_formation()

            # Here we change the selected rectangle to the position to the dropping position
            self.anch_formations[selected_rectangle_f] = dropping_arch_position

            # Here we change the rectangle that was on the dropping position to the original position of the selected
            # rectangle
            self.anch_formations.iloc[0, dropping_arg] = selected_arch_position_f

            self.update_data_frame()
            self.anch_series, self.anch_formations, self.thick_series, self.thick_formations = set_anchor_points(self.geo_data)

        # We update the visualization of all the rectangles
        for formations_name in self.anch_formations.columns:
            formation_rect[formations_name].anch_formations = self.anch_formations
            new_pos = self.anch_formations[formation_rect[formations_name].rect.f].values
            formation_rect[formations_name].rect.set_y(new_pos - 0.5/2)
            formation_rect[formations_name].rect.set_animated(False)
            formation_rect[formations_name].rect.background = None

            # redraw the full figure
            formation_rect[formations_name].rect.figure.canvas.draw()

        self.selected_rectangle_s = None
        self.selected_rectangle_r = None

    def compute_new_arch_series(self):

        dist = np.abs(self.anch_series.as_matrix() - self.rect.get_y())
        arg_min = np.argmin(dist)
        new_arch = self.anch_series.iloc[0, arg_min]
        return new_arch, arg_min

    def compute_new_arch_formation(self):

        dist = np.abs(self.anch_formations.as_matrix() - self.rect.get_y())
        arg_min = np.argmin(dist)
        new_arch = self.anch_formations.iloc[0, arg_min]
        return new_arch, arg_min

    def update_data_frame(self):

        order_series = self.anch_series.sort_values(by=0, axis=1, ascending=False).columns.values
        order_formations = self.anch_formations.sort_values(by=0, axis=1, ascending=False)

        # drop aux
        aux_columns = ['aux' in i for i in order_formations.columns]
        order_formations.drop(order_formations.columns[aux_columns], axis=1, inplace=True)

        series_dict = {}
        # divide formations to their series
        for name, value in self.anch_series.iteritems():
             cond = ((order_formations <= value.get_values()+self.thick_series/2) & \
                    (order_formations >= value.get_values()-self.thick_series/2))
             format_in_series = order_formations.columns[cond.values[0, :]]
             # Passing from array to tuple what a pain
             series_dict[name] = format_in_series.values

        self.geo_data.set_series(series_dict, order=order_series)
        self.geo_data.set_formation_number(order_formations.columns.values)
        self.geo_data.order_table()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


