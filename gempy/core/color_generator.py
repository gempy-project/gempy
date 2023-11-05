from dataclasses import dataclass
from typing import Optional

import numpy as np


class ChronoStrat:
    """
    Color schemes according to the chronostratigraphic table
    """

    def __init__(self):
        """
        Color schemes according to the chronostratigraphic table
        """

        self.eons_colors = {
            'Hadean': '#AE2A7E',      # RGB: (174, 42, 126)
            'Archean': '#F4439E',     # RGB: (244, 68, 159)
            'Proterozoic': '#FBA9BB', # RGB: (251, 169, 187)
            'Phanerozoic': '#7FB273', # RGB: (127, 178, 115)
        }

        self.eras_colors = {
            'Paleozoic': '#7FB273',       # RGB: (127, 178, 115)
            'Mesozoic': '#9AD979',        # RGB: (154, 217, 121)
            'Cenozoic': '#54FF9F',        # RGB: (84, 255, 159)
            'Paleoproterozoic': '#F4439E', # Same as Archean
            'Mesoproterozoic': '#FBA9BB',  # Same as Proterozoic
            'Neoproterozoic': '#FBA9BB',   # Same as Proterozoic
        }

        self.periods_colors = {
            'Cambrian': '#8080C0',
            'Ordovician': '#74A8D4',
            'Silurian': '#5AA0CF',
            'Devonian': '#70BD9F',
            'Carboniferous': '#76C691',
            'Permian': '#99CEA3',
            'Triassic': '#FFF7CC',
            'Jurassic': '#FEFEC7',
            'Cretaceous': '#CCFF99',
            'Paleogene': '#E3FFB0',
            'Neogene': '#D2FFB5',
            'Quaternary': '#CCFFE6',
        }

        self.epochs_colors = {
            # Cenozoic
            'Paleocene': '#E3FFB0',  # RGB: (227, 255, 176)
            'Eocene': '#D2FFB5',     # RGB: (210, 255, 181)
            'Oligocene': '#CCFFE6',  # RGB: (204, 255, 230)
            'Miocene': '#FFFFCC',    # RGB: (255, 255, 204)
            'Pliocene': '#FFFF99',   # RGB: (255, 255, 153)    
            'Pleistocene': '#E6FFCC', # RGB: (230, 255, 204)
            'Holocene': '#CCFFCC',    # RGB: (204, 255, 204)
            # Mesosoic
            'Early Triassic': '#FFE4C4',  # RGB: (255, 228, 196)
            'Middle Triassic': '#FFEBB5', # RGB: (255, 235, 181)
            'Late Triassic': '#FFFFB5',   # RGB: (255, 255, 181)
            'Early Jurassic': '#FFFF99',  # RGB: (255, 255, 153)
            'Middle Jurassic': '#FFFF7F', # RGB: (255, 255, 127)
            'Late Jurassic': '#FFFF66',   # RGB: (255, 255, 102)
            'Early Cretaceous': '#FFFF00',  # RGB: (255, 255, 0)
            'Late Cretaceous': '#E6FF33',   # RGB: (230, 255, 51)
        }

        self.stages_colors = {
            # Cambrian
            'Terreneuvian': '#8585AD',   # RGB: (133, 133, 173)
            'Stage 2': '#8C8CB8',        # RGB: (140, 140, 184)
            'Stage 3': '#9494C2',        # RGB: (148, 148, 194)
            'Miaolingian': '#9B9BCD',    # RGB: (155, 155, 205)
            'Wuliuan': '#A3A3D7',        # RGB: (163, 163, 215)
            'Drumian': '#AAAAE1',        # RGB: (170, 170, 225)
            'Guzhangian': '#B2B2EB',     # RGB: (178, 178, 235)
            'Furongian': '#B9B9F5',      # RGB: (185, 185, 245)
            'Paibian': '#C1C1FF',        # RGB: (193, 193, 255)
            'Jiangshanian': '#C9C9FF',   # RGB: (201, 201, 255)
            'Stage 10': '#D1D1FF',       # RGB: (209, 209, 255)
            # Lower Ordovician Stages
            'Tremadocian': '#A3D6F5',  # RGB: (163, 214, 245)
            'Floian': '#A8DBF5',       # RGB: (168, 219, 245)

            # Middle Ordovician Stages
            'Dapingian': '#ADDEF5',    # RGB: (173, 222, 245)
            'Darriwilian': '#B2E1F5',  # RGB: (178, 225, 245)

            # Upper Ordovician Stages
            'Sandbian': '#B8E4F5',     # RGB: (184, 228, 245)
            'Katian': '#BDE7F5',       # RGB: (189, 231, 245)
            'Hirnantian': '#C2EAF5',   # RGB: (194, 234, 245)

            ## Silurian
            # Llandovery Epoch Stages
            'Rhuddanian': '#00BFFF',   # RGB: (0, 191, 255)
            'Aeronian': '#00B2EE',     # RGB: (0, 178, 238)
            'Telychian': '#009ACD',    # RGB: (0, 154, 205)

            # Wenlock Epoch Stages
            'Sheinwoodian': '#008B8B', # RGB: (0, 139, 139)
            'Homerian': '#00688B',     # RGB: (0, 104, 139)

            # Ludlow Epoch Stages
            'Gorstian': '#006400',     # RGB: (0, 100, 0)
            'Ludfordian': '#2E8B57',   # RGB: (46, 139, 87)

            # Pridoli Epoch (undivided in the standard global stratigraphic scheme)
            'Pridoli': '#228B22',      # RGB: (34, 139, 34)

            ## Devonian
            # Early (Lower) Devonian Stages
            'Lochkovian': '#556B2F',  # RGB: (85, 107, 47)
            'Pragian': '#6B8E23',     # RGB: (107, 142, 35)
            'Emsian': '#7CFC00',      # RGB: (124, 252, 0)

            # Middle Devonian Stages
            'Eifelian': '#32CD32',    # RGB: (50, 205, 50)
            'Givetian': '#00FF00',    # RGB: (0, 255, 0)

            # Late (Upper) Devonian Stages
            'Frasnian': '#ADFF2F',    # RGB: (173, 255, 47)
            'Famennian': '#FFFF00',   # RGB: (255, 255, 0)        

            ## Carboniferous
            # Mississippian (Early Carboniferous) - Not universally separated in the geological time scale
            'Tournaisian': '#8B814C',    # RGB: (139, 129, 76)
            'Visean': '#9ACD32',         # RGB: (154, 205, 50)
            'Serpukhovian': '#6E8B3D',   # RGB: (110, 139, 61)

            # Pennsylvanian (Late Carboniferous) - Not universally separated in the geological time scale
            'Bashkirian': '#556B2F',     # RGB: (85, 107, 47)
            'Moscovian': '#8FBC8F',      # RGB: (143, 188, 143)
            'Kasimovian': '#66CDAA',     # RGB: (102, 205, 170)
            'Gzhelian': '#458B74',       # RGB: (69, 139, 116)        

            ## Permian
            # Cisuralian Epoch (Early Permian)
            'Asselian': '#FFD700',  # RGB: (255, 215, 0)
            'Sakmarian': '#EEC900',  # RGB: (238, 201, 0)
            'Artinskian': '#CDAD00',  # RGB: (205, 173, 0)
            'Kungurian': '#8B7500',  # RGB: (139, 117, 0)

            # Guadalupian Epoch (Middle Permian)
            'Roadian': '#FFA500',  # RGB: (255, 165, 0)
            'Wordian': '#EE9A00',  # RGB: (238, 154, 0)
            'Capitanian': '#CD8500',  # RGB: (205, 133, 0)

            # Lopingian Epoch (Late Permian)
            'Wuchiapingian': '#FF4500',  # RGB: (255, 69, 0)
            'Changhsingian': '#CD3700',  # RGB: (205, 55, 0)        
            
            ## Triassic
            # Early (Lower) Triassic
            'Induan': '#FF6347',    # RGB: (255, 99, 71)
            'Olenekian': '#EE5C42',  # RGB: (238, 92, 66)

            # Middle Triassic
            'Anisian': '#CD4F39',   # RGB: (205, 79, 57)
            'Ladinian': '#8B3626',  # RGB: (139, 54, 38)

            # Late (Upper) Triassic
            'Carnian': '#A0522D',   # RGB: (160, 82, 45)
            'Norian': '#8B4513',    # RGB: (139, 69, 19)
            'Rhaetian': '#5E2612',  # RGB: (94, 38, 18)

            ## Jurassic
            # Early (Lower) Jurassic
            'Hettangian': '#FF8C00',  # RGB: (255, 140, 0)
            'Sinemurian': '#EE7600',  # RGB: (238, 118, 0)
            'Pliensbachian': '#CD6600',  # RGB: (205, 102, 0)
            'Toarcian': '#8B4500',  # RGB: (139, 69, 0)

            # Middle Jurassic
            'Aalenian': '#FF7F24',  # RGB: (255, 127, 36)
            'Bajocian': '#EE7621',  # RGB: (238, 118, 33)
            'Bathonian': '#CD661D',  # RGB: (205, 102, 29)
            'Callovian': '#8B4513',  # RGB: (139, 69, 19)

            # Late (Upper) Jurassic
            'Oxfordian': '#FF7256',  # RGB: (255, 114, 86)
            'Kimmeridgian': '#EE6A50',  # RGB: (238, 106, 80)
            'Tithonian': '#CD5B45',  # RGB: (205, 91, 69)

            ## Cretaceous
            # Early (Lower) Cretaceous
            'Berriasian': '#98F5FF',  # RGB: (152, 245, 255)
            'Valanginian': '#8EE5EE',  # RGB: (142, 229, 238)
            'Hauterivian': '#7AC5CD',  # RGB: (122, 197, 205)
            'Barremian': '#68838B',  # RGB: (104, 131, 139)
            'Aptian': '#00CED1',  # RGB: (0, 206, 209)
            'Albian': '#48D1CC',  # RGB: (72, 209, 204)

            # Late (Upper) Cretaceous
            'Cenomanian': '#40E0D0',  # RGB: (64, 224, 208)
            'Turonian': '#00CED1',  # RGB: (0, 206, 209)
            'Coniacian': '#48D1CC',  # RGB: (72, 209, 204)
            'Santonian': '#20B2AA',  # RGB: (32, 178, 170)
            'Campanian': '#5F9EA0',  # RGB: (95, 158, 160)
            'Maastrichtian': '#B0E0E6',  # RGB: (176, 224, 230)

            # Paleogene Period
            # Paleocene Epoch
            'Danian': '#00FF7F',  # RGB: (0, 255, 127)
            'Selandian': '#00EE76',  # RGB: (0, 238, 118)
            'Thanetian': '#00CD66',  # RGB: (0, 205, 102)

            # Eocene Epoch
            'Ypresian': '#00B45A',  # RGB: (0, 180, 90)
            'Lutetian': '#008B3E',  # RGB: (0, 139, 62)
            'Bartonian': '#FFD700',  # RGB: (255, 215, 0)
            'Priabonian': '#EEC900',  # RGB: (238, 201, 0)

            # Oligocene Epoch
            'Rupelian': '#CDAD00',  # RGB: (205, 173, 0)
            'Chattian': '#8B7500',  # RGB: (139, 117, 0)

            # Neogene Period
            # Miocene Epoch
            'Aquitanian': '#FFA500',  # RGB: (255, 165, 0)
            'Burdigalian': '#EE9A00',  # RGB: (238, 154, 0)
            'Langhian': '#CD8500',  # RGB: (205, 133, 0)
            'Serravallian': '#8B5A00',  # RGB: (139, 90, 0)
            'Tortonian': '#FFEBCD',  # RGB: (255, 235, 205)
            'Messinian': '#FFE4C4',  # RGB: (255, 228, 196)

            # Pliocene Epoch
            'Zanclean': '#FFDAB9',  # RGB: (255, 218, 185)
            'Piacenzian': '#CD853F',  # RGB: (205, 133, 63)

            # Quaternary Period
            # Pleistocene Epoch
            'Gelasian': '#A0522D',  # RGB: (160, 82, 45)
            'Calabrian': '#8B4513',  # RGB: (139, 69, 19)
            'Chibanian': '#A52A2A',  # RGB: (165, 42, 42)
            'Tarantian': '#800000',  # RGB: (128, 0, 0)

            # Holocene Epoch
            'Greenlandian': '#008000',  # RGB: (0, 128, 0)
            'Northgrippian': '#006400',  # RGB: (0, 100, 0)
            'Meghalayan': '#2E8B57',  # RGB: (46, 139, 87)        
        }




@dataclass
class ColorsGenerator:
    """
    Object that handles the color management.
    """
    hex_colors: list[str]
    _index: int = 0
    
    def __init__(self):
        self._gempy_default_colors = [
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
        
        self.regenerate_color_palette()


    @staticmethod
    def _random_hexcolor() -> str:
        """Generates a random hex color string."""
        return "#"+str(hex(np.random.randint(0, 16777215))).lstrip("0x")

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
            hex_colors = self._gempy_default_colors

        self.hex_colors = hex_colors


    def __iter__(self) -> 'ColorsGenerator':
        """Returns the object itself as an iterator."""
        return self
    
    def __next__(self) -> str:
        """Generator that yields the next color."""
        for color in self.hex_colors:
            result = self.hex_colors[self._index]
            self._index += 1
            return result

        while True:
            return self._random_hexcolor()
    
    def up_next(self) -> str:
        """Returns the next color without incrementing the index."""
        return self.hex_colors[self._index]