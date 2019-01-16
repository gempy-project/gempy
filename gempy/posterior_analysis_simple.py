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


    @author: Elisa Heim, Alexander Schaaf, Miguel de la Varga
    (I guess, copied some code from posterior_analysis.py)
"""

import warnings
try:
    import pymc
except ImportError:
    warnings.warn("pymc (v2) package is not installed. No support for stochastic simulation posterior analysis.")
import numpy as np
import pandas as pn
import gempy as gp
try:
    import tqdm
except ImportError:
    warnings.warn("tqdm package not installed. No support for dynamic progress bars.")
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib.colors


class Posterior():
    def __init__(self, dbname, model_type='map', entropy=False, topography=None, interpdata=None, geodata=None):

        if entropy:
            print('All post models are calculated. Based on the model complexity and the number of iterations, '
                  'this could take a while')
        # if topography:
        self.topography = topography
        # else:
        # print('no topography defined. Methods that contain the word _map_ are not available')

        self.interp_data = interpdata

        self.geo_data = geodata
        # self.verbose = verbose

        self.db = pymc.database.hdf5.load(dbname)  # load database
        self.n_iter = self.db.getstate()['sampler']['_iter'] - self.db.getstate()["sampler"]["_burn"]
        self.trace_names = self.db.trace_names[0]
        self.input_data = self.db.input_data.gettrace()

        if entropy:
            if topography and model_type == 'map':  # better resolution
                self.all_maps = self.all_post_maps()
                self.map_prob = self.compute_prob(np.round(self.all_maps).astype(int))
                self.map_ie = self.calculate_ie_masked(self.map_prob)

            elif model_type == 'model':
                self.lbs, self.fbs = self.all_post_models()

                if len(self.lbs) != 0:
                    self.lith_prob = self.compute_prob(np.round(self.lbs).astype(int))
                    self.lb_ie = self.calculate_ie_masked(self.lith_prob)

                if len(self.fbs) != 0:
                    self.fault_prob = self.compute_prob(np.round(self.fbs).astype(int))
                    self.fb_ie = self.calculate_ie_masked(self.fault_prob)
            else:
                print('if there is no topography defined, model_type must be set to model')
            # self.ie_total = self.calculate_ie_total()

    def _change_input_data(self, i):
        i = int(i)
        # replace interface data
        self.interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = self.input_data[i][0]
        # replace foliation data
        self.interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", "dip", "azimuth", "polarity"]] = \
            self.input_data[i][1]
        self.interp_data.update_interpolator()
        # if self.verbose:
        # print("interp_data parameters changed.")
        return self.interp_data

    def all_post_maps(self):
        all_maps = []
        for i in range(0, self.n_iter):
            # print(i)
            self._change_input_data(i)
            # geomap = self.topography.calculate_geomap(interpdata = self.interp_data, plot=True)
            geomap, faultmap = gp.compute_model_at(self.topography.surface_coordinates[0], self.interp_data)
            all_maps.insert(i, geomap[0])
        return all_maps

    def all_post_models(self):
        lbs = []
        fbs = []
        for i in range(0, self.n_iter):
            # print(i)
            self._change_input_data(i)
            lith_block, fault_block = gp.compute_model(self.interp_data)
            if lith_block.shape[0] != 0:
                lbs.insert(i, lith_block[0])
            if fault_block.shape[0] != 0:
                n = 0
                while n < fault_block.shape[0]:
                    # print(fault_block.shape[0])
                    fbs.insert(i, fault_block[n])
                    n += 2
        return lbs, fbs

    def compute_prob(self, blocks):
        lith_id = np.unique(blocks)
        # lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
        count = np.zeros((len(np.unique(blocks)), blocks.shape[1]))
        for i, l_id in enumerate(lith_id):
            count[i] = np.sum(blocks == l_id, axis=0)
        prob = count / len(blocks)
        # print(lith_prob)
        return prob

    def calculate_ie_masked(self, prob):
        ie = np.zeros_like(prob[0])
        for l in prob:
            pm = np.ma.masked_equal(l, 0)  # mask where prob is 0
            ie -= (pm * np.ma.log2(pm)).filled(0)
        return ie

    def calculate_ie_total(self, ie, absolute=False):
        if absolute:
            return np.sum(ie)
        else:
            return np.sum(ie) / np.size(ie)

        ##### plotting methods #####

    def plot_section(self, iteration=1, block='lith', cell_number=3, **kwargs):
        '''kwargs: gempy.plotting.plot_section keyword arguments'''
        self._change_input_data(iteration)
        lith_block, fault_block = gp.compute_model(self.interp_data)

        if 'topography' not in kwargs:
            if self.topography:
                topography = self.topography
            else:
                topography = None
            if block == 'lith':
                gp.plot_section(self.geo_data, lith_block[0], cell_number=cell_number, topography=topography, **kwargs)
            else:
                gp.plot_section(self.geo_data, block, cell_number=cell_number, topography=topography, **kwargs)

        else:
            if block == 'lith':
                gp.plot_section(self.geo_data, lith_block[0], cell_number=cell_number, **kwargs)
            else:
                gp.plot_section(self.geo_data, block, cell_number=cell_number, **kwargs)

    def plot_map(self, iteration=1, **kwargs):
        self._change_input_data(iteration)
        # geomap = self.topography.calculate_geomap(interpdata = self.interp_data, plot=True)
        geomap, faultmap = gp.compute_model_at(self.topography.surface_coordinates[0], self.interp_data)
        # gp.plotting.plot_map(geomap)
        gp.plotting.plot_map(self.geo_data, geomap=geomap[0].reshape(self.topography.dem_zval.shape), **kwargs)

    def plot_map_ie(self, plot_data=False):
        if plot_data:
            gp.plotting.plot_data(geo_data, direction='z')
            dist = 12
        else:
            dist = 1

        im = plt.imshow(self.map_ie.reshape(self.topography.dem_zval.shape), extent=self.geo_data.extent[:4],
                        cmap='viridis')
        self.add_colorbar(im, pad_fraction=dist)
        plt.title('Cell entropy of geological map')

    def plot_section_ie(self, block='lith', cell_number=10, direction='y', **kwargs):
        # for lithblock
        if block == 'lith':
            norm = matplotlib.colors.Normalize(self.lb_ie.min(), self.lb_ie.max())
            gp.plotting.plot_section(geo_data, self.lb_ie, cell_number=cell_number, direction=direction, cmap='viridis',
                                     norm=norm, **kwargs)
            # self.add_colorbar(im)
        elif block == 'fault':
            norm = matplotlib.colors.Normalize(self.fb_ie.min(), self.fb_ie.max())
            gp.plotting.plot_section(geo_data, self.fb_ie, cell_number=cell_number, direction=direction, cmap='viridis',
                                     norm=norm, **kwargs)
            # self.add_colorbar(im)

    def add_colorbar(self, im, aspect=20, pad_fraction=1, **kwargs):
        """Add a vertical color bar to an image plot. Source: stackoverflow"""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=2. / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)
