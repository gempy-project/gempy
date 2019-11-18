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

    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/11/2019

    @author: Elisa Heim, Miguel de la Varga
"""

from os import path
import sys

# This is for sphenix to find the packages
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from .visualization_2d_pro import Plot2D
from .vista import Vista
import gempy as _gempy
import pandas as pn
import matplotlib.pyplot as plt
from typing import Union


try:
    import mplstereonet
    mplstereonet_import = True
except ImportError:
    mplstereonet_import = False


def plot_2d(model, n_axis = None, section_names:list = None, cell_number: list = 'mid', direction: list = 'y',
            show_data: Union[bool, list] = True, show_lith: Union[bool, list] = True,
            show_scalar: Union[bool, list] = False, show_boundaries: Union[bool, list] = True, **kwargs):

    section_names = [] if section_names is None else section_names
    if cell_number is None:
        cell_number = []
    elif cell_number == 'mid':
        cell_number = ['mid']
    direction = [] if direction is None else direction

    if n_axis is None:
        n_axis = len(section_names) + len(cell_number)

    if type(show_data) is bool:
        show_data = [show_data] * n_axis
    if type(show_lith) is bool:
        show_lith = [show_lith] * n_axis
    if type(show_scalar) is bool:
        show_scalar = [show_scalar] * n_axis
    if type(show_boundaries) is bool:
        show_boundaries = [show_boundaries] * n_axis

    p = Plot2D(model, **kwargs)
    p.create_figure(**kwargs)
    # init e
    e = 0

    for e, sn in section_names:
        assert e<10, 'Reached maximum of axes'
        temp_ax = p.add_section(section_name=sn, ax_pos=(int(n_axis/2)+1)*100+20+e+1, **kwargs)
        if show_data[e] is True:
            p.plot_data(temp_ax, section_name=sn, **kwargs)
        if show_lith[e] is True:
            p.plot_lith(temp_ax, section_name=sn, **kwargs)
        if show_scalar[e] is True:
            p.plot_scalar_field(temp_ax, section_name=sn, **kwargs)
        if show_boundaries[e] is True:
            p.plot_contacts(temp_ax, section_name=sn, **kwargs)

    for e2 in range(len(cell_number)):
        assert (e+e2)<10, 'Reached maximum of axes'
        temp_ax = p.add_section(cell_number=cell_number[e2], direction=direction[e2], ax_pos=(int(n_axis/2)+1)*100+20+e+e2+1)
        if show_data[e+e2] is True:
            p.plot_data(temp_ax, cell_number=cell_number[e2], direction=direction[e2], **kwargs)
        if show_lith[e+e2] is True:
            p.plot_lith(temp_ax, cell_number=cell_number[e2], direction=direction[e2], **kwargs)
        if show_scalar[e+e2] is True:
            p.plot_scalar_field(temp_ax, cell_number=cell_number[e2], direction=direction[e2], **kwargs)
        if show_boundaries[e+e2] is True:
            p.plot_contacts(temp_ax, cell_number=cell_number[e2], direction=direction[e2], **kwargs)

    return p


def plot_stereonet(self, litho=None, planes=True, poles=True, single_plots=False,
                   show_density=False):
    if mplstereonet_import is False:
        raise ImportError('mplstereonet package is not installed. No stereographic projection available.')

    from collections import OrderedDict

    if litho is None:
        litho = self.model.orientations.df['surface'].unique()

    if single_plots is False:
        fig, ax = mplstereonet.subplots(figsize=(5, 5))
        df_sub2 = pn.DataFrame()
        for i in litho:
            df_sub2 = df_sub2.append(self.model.orientations.df[self.model.orientations.df['surface'] == i])

    for formation in litho:
        if single_plots:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='stereonet')
            ax.set_title(formation, y=1.1)

        # if series_only:
        # df_sub = self.model.orientations.df[self.model.orientations.df['series'] == formation]
        # else:
        df_sub = self.model.orientations.df[self.model.orientations.df['surface'] == formation]

        if poles:
            ax.pole(df_sub['azimuth'] - 90, df_sub['dip'], marker='o', markersize=7,
                    markerfacecolor=self._color_lot[formation],
                    markeredgewidth=1.1, markeredgecolor='gray', label=formation + ': ' + 'pole point')
        if planes:
            ax.plane(df_sub['azimuth'] - 90, df_sub['dip'], color=self._color_lot[formation],
                     linewidth=1.5, label=formation + ': ' + 'azimuth/dip')
        if show_density:
            if single_plots:
                ax.density_contourf(df_sub['azimuth'] - 90, df_sub['dip'],
                                    measurement='poles', cmap='viridis', alpha=.5)
            else:
                ax.density_contourf(df_sub2['azimuth'] - 90, df_sub2['dip'], measurement='poles', cmap='viridis',
                                    alpha=.5)

        fig.subplots_adjust(top=0.8)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.9, 1.1))
        ax.grid(True, color='black', alpha=0.25)
