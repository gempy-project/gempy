from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import Field


@dataclass
class ColorsGenerator:
    """
    Object that handles the color management.
    """

    _gempy_default_colors = [
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

    hex_colors: list[str] = Field(
        default=_gempy_default_colors,
        exclude=True
    )
    _index: int = 0

    def __post_init__(self):
        self.regenerate_color_palette()

    @staticmethod
    def _random_hexcolor() -> str:
        """Generates a random hex color string."""
        return f"#{np.random.randint(0, 0xffffff):06x}"

    def regenerate_color_palette(self, seaborn_palettes: Optional[list[str]] = None):
        try:
            import seaborn as sns
            seaborn_installed = True
        except ImportError:
            seaborn_installed = False

        if seaborn_palettes and seaborn_installed:
            hex_colors = []
            for palette in seaborn_palettes:  # for each palette
                hex_colors += sns.color_palette(palette).as_hex()  # get all colors in palette and add to list

        elif seaborn_palettes and not seaborn_installed:
            raise ImportError("Seaborn is not installed. Please install it to use color palettes.")
        else:
            hex_colors = list(self._gempy_default_colors)

        self.hex_colors = hex_colors

    def __iter__(self) -> 'ColorsGenerator':
        """Returns the object itself as an iterator."""
        return self

    def __next__(self) -> str:
        """Generator that yields the next color."""
        color = self.up_next()
        self._index += 1
        del self._next_color
        return color

    def up_next(self) -> str:
        """Returns the next color without incrementing the index."""
        return self._next_color

    @cached_property
    def _next_color(self) -> str:
        if self._index < len(self.hex_colors):
            return self.hex_colors[self._index]

        return self._random_hexcolor()
