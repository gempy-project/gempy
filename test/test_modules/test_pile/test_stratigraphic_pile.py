import os
import pandas as pd

import pytest

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, pv_plot


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
        if PLOT := True:
            s = to_pyvista_line(
                line_set=borehole_set.combined_trajectory,
                radius=10,
                active_scalar="lith"
            )
            pv_plot([s], image_2d=False, cmap="tab20c")
