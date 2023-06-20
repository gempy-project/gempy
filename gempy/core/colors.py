import re
from typing import Union, List

import numpy as np
import seaborn as sns
from IPython.core.display_functions import display

try:
    import ipywidgets as widgets

    ipywidgets_import = True
except ModuleNotFoundError:
    VTK_IMPORT = False


class Colors:
    """
    Object that handles the color management in the model.
    """
    def __init__(self, surfaces):
        self.surfaces = surfaces
        self.colordict = None
        self._hexcolors_soft = [
            '#015482', '#9f0052', '#ffbe00', '#728f02', '#443988',
            '#ff3f20', '#5DA629', '#b271d0', '#72e54a', '#583bd1',
            '#d0e63d', '#b949e2', '#95ce4b', '#6d2b9f', '#60eb91',
            '#d746be', '#52a22e', '#5e63d8', '#e5c339', '#371970',
            '#d3dc76', '#4d478e', '#43b665', '#d14897', '#59e5b8',
            '#e5421d', '#62dedb', '#df344e', '#9ce4a9', '#d94077',
            '#99c573', '#842f74', '#578131', '#708de7', '#df872f',
            '#5a73b1', '#ab912b', '#321f4d', '#e4bd7c', '#142932',
            '#cd4f30', '#69aedd', '#892a23', '#aad6de', '#5c1a34',
            '#cfddb4', '#381d29', '#5da37c', '#d8676e', '#52a2a3',
            '#9b405c', '#346542', '#de91c9', '#555719', '#bbaed6',
            '#945624', '#517c91', '#de8a68', '#3c4b64', '#9d8a4d',
            '#825f7e', '#2c3821', '#ddadaa', '#5e3524', '#a3a68e',
            '#a2706b', '#686d56'
        ]  # source: https://medialab.github.io/iwanthue/

    def generate_colordict(
            self,
            hex_colors: Union[List[str], str] = 'palettes',
            palettes: List[str] = 'default',
    ):
        """Generates and sets color dictionary.

        Args:
           hex_colors (list[str], str): List of hex color values. In the future this could
           accommodate the actual geological palettes. For example striplog has a quite
           good set of palettes.
                * palettes: If hexcolors='palettes' the colors will be chosen from the
                   palettes arg
                * soft: https://medialab.github.io/iwanthue/
           palettes (list[str], optional): list with name of seaborn palettes. Defaults to 'default'.
        """
        if hex_colors == 'palettes':
            hex_colors = []
            if palettes == 'default':
                # we predefine some 7 colors manually
                hex_colors = ['#015482', '#9f0052', '#ffbe00', '#728f02', '#443988', '#ff3f20', '#5DA629']
                # then we create a list of seaborn color palette names, as the user didn't provide any
                palettes = ['muted', 'pastel', 'deep', 'bright', 'dark', 'colorblind']
            for palette in palettes:  # for each palette
                hex_colors += sns.color_palette(palette).as_hex()  # get all colors in palette and add to list
                if len(hex_colors) >= len(self.surfaces.df):
                    break
        elif hex_colors == 'soft':
            hex_colors = self._hexcolors_soft

        surface_names = self.surfaces.df['surface'].values
        n_surfaces = len(surface_names)

        while n_surfaces > len(hex_colors):
            hex_colors.append(self._random_hexcolor())

        self.colordict = dict(
            zip(surface_names, hex_colors[:n_surfaces])
        )

    @staticmethod
    def _random_hexcolor() -> str:
        """Generates a random hex color string."""
        return "#"+str(hex(np.random.randint(0, 16777215))).lstrip("0x")

    def change_colors(self, colordict: dict = None):
        """Change the model colors either by providing a color dictionary or,
        if not, by using a color pick widget.

        Args:
            colordict (dict, optional): dict with surface names mapped to hex color codes, e.g. {'layer1':'#6b0318'}
            if None: opens jupyter widget to change colors interactively. Defaults to None.
        """
        assert ipywidgets_import, 'ipywidgets not imported. Make sure the library is installed.'

        if colordict:
            self.update_colors(colordict)
        else:
            items = [
                widgets.ColorPicker(description=surface, value=color)
                for surface, color in self.colordict.items()
            ]
            colbox = widgets.VBox(items)
            print('Click to select new colors.')
            display(colbox)

            def on_change(v):
                self.colordict[v['owner'].description] = v['new']  # update colordict
                self._set_colors()

            for cols in colbox.children:
                cols.observe(on_change, 'value')

    def update_colors(self, colordict: dict = None):
        """ Updates the colors in self.colordict and in surfaces_df.

        Args:
            colordict (dict, optional): dict with surface names mapped to hex
                color codes, e.g. {'layer1':'#6b0318'}. Defaults to None.
        """
        if colordict is None:
            self.generate_colordict()
        else:
            for surf, color in colordict.items():  # map new colors to surfaces
                # assert this because user can set it manually
                assert surf in list(self.surfaces.df['surface']), str(surf) + ' is not a model surface'
                assert re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color), str(color) + ' is not a HEX color code'
                self.colordict[surf] = color

        self._set_colors()

    def _add_colors(self):
        """Assign a color to the last entry of surfaces df or check isnull and assign color there"""
        self.generate_colordict()

    def _set_colors(self):
        """sets colordict in surfaces dataframe"""
        for surf, color in self.colordict.items():
            self.surfaces.df.loc[self.surfaces.df['surface'] == surf, 'color'] = color

    def set_default_colors(self, surfaces=None):
        if surfaces is not None:
            self.colordict[surfaces] = self.colordict[surfaces]
        self._set_colors()

    def delete_colors(self, surfaces):
        for surface in surfaces:
            self.colordict.pop(surface, None)
        self._set_colors()

    def make_faults_black(self, series_fault):
        faults_list = list(self.surfaces.df[self.surfaces.df.series.isin(series_fault)]['surface'])
        for fault in faults_list:
            if self.colordict[fault] == '#527682':
                self.set_default_colors(fault)
            else:
                self.colordict[fault] = '#527682'
                self._set_colors()

    def reset_default_colors(self):
        self.generate_colordict()
        self._set_colors()
        return self.surfaces
