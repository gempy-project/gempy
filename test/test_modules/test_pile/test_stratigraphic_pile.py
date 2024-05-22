import numpy as np
import os
import pandas as pd

import pytest

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, pv_plot

import gempy as gp


# @pytest.mark.skip(reason="Not implemented yet")
class TestStratigraphicPile:
    @pytest.fixture(autouse=True)
    def borehole_set(self):
        reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY"),
            columns_map={
                    'hole_id'   : 'id',
                    'depth_from': 'top',
                    'depth_to'  : 'base',
                    'lit_code'  : 'component lith'
            }
        )

        lith: pd.DataFrame = read_lith(reader)
        reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
            columns_map={
                    'depth'  : 'md',
                    'dip'    : 'dip',
                    'azimuth': 'azi'
            },
        )
        df = read_survey(reader)

        survey: Survey = Survey.from_df(df)
        survey.update_survey_with_lith(lith)

        reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
            header=0,
            usecols=[0, 1, 2, 4],
            columns_map={
                    "hole_id"            : "id",  # ? Index name is not mapped
                    "X_GK5_incl_inserted": "x",
                    "Y__incl_inserted"   : "y",
                    "Z_GK"               : "z"
            }
        )
        df_collar = read_collar(reader_collar)
        collar = Collars.from_df(df_collar)

        borehole_set = BoreholeSet(
            collars=collar,
            survey=survey,
            merge_option=MergeOptions.INTERSECT
        )

        return borehole_set

    def test_structural_elements(self, borehole_set: BoreholeSet):
        from subsurface import LineSet
        borehole_trajectory: LineSet = borehole_set.combined_trajectory
        if PLOT := False:
            s = to_pyvista_line(
                line_set=borehole_trajectory,
                radius=10,
                active_scalar="lith_ids"
            )
            pv_plot([s], image_2d=False, cmap="tab20c")

        elements = gp.structural_elements_from_borehole_set(
            borehole_set=borehole_set,
            elements_dict={
                # "Pleistozen": {"id": 10_000, "color": "#f9f97f", "top_lith": 10_000},
                # "Kreide": {"id": 30_000, "color": "#a6d84a", "top_lith": 30_000},
                # "Trias": {"id": 50_000, "color": "#a4469f", "top_lith": 50_000},
                # "Perm": {"id": 60_000, "color": "#f4a142", "top_lith": 60_000},
                "Rotliegend": {"id": 62_000, "color": "#bb825b", "top_lith": 62_000},
                # "Devon": {"id": 80_000, "color": "#969594", "top_lith": 80_000}
            }
        )
        
        
        group = gp.data.StructuralGroup(
            name="Stratigraphic Pile",
            elements=elements,
            structural_relation=gp.data.StackRelationType.ERODE
        )
        structural_frame = gp.data.StructuralFrame(
            structural_groups=[group],
            color_gen=gp.data.ColorsGenerator()
        )
        print(group)


        component_lith = borehole_set.get_top_coords_for_each_lith()
        rotliegend_xyz = component_lith[62_000]
        extent_from_data = rotliegend_xyz.min(axis=0), rotliegend_xyz.max(axis=0)

        geo_model = gp.data.GeoModel(
            name="Stratigraphic Pile",
            structural_frame=structural_frame,
            grid=gp.data.Grid(
                extent=[extent_from_data[0][0], extent_from_data[1][0], extent_from_data[0][1], extent_from_data[1][1], extent_from_data[0][2], extent_from_data[1][2]],
                resolution=(50, 50, 50)
            ),
            interpolation_options=gp.data.InterpolationOptions(
                range=5,
                c_o=10,
                mesh_extraction=True,
                number_octree_levels=3,
            ),

        )
        
        import gempy_viewer as gpv
        gpv.plot_3d(geo_model)
        pass
