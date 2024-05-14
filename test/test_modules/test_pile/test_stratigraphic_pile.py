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
        if PLOT := False:
            s = to_pyvista_line(
                line_set=borehole_set.combined_trajectory,
                radius=10,
                active_scalar="lith_ids"
            )
            pv_plot([s], image_2d=False, cmap="tab20c")

        vertex_attributes: pd.DataFrame = borehole_set.combined_trajectory.data.points_attributes
        unique_lith_codes = vertex_attributes['component lith'].unique()
        
        pleistozen = gp.data.StructuralElement(
            name= "Pleistozen",
            id=10_000,
            color="#f9f97f",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        
        kreide = gp.data.StructuralElement(
            name= "Kreide",
            id=30_000,
            color="#a6d84a",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        
        trias = gp.data.StructuralElement(
            name= "Trias",
            id=50_000,
            color="#a4469f",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        
        perm = gp.data.StructuralElement(
            name= "Perm",
            id=60_000,
            color="#f4a142",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        
        rotliegend = gp.data.StructuralElement(
            name= "Rotliegend",
            id=62_000,
            color="#bb825b",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        
        devon = gp.data.StructuralElement(
            name= "Devon",
            id=80_000,
            color="#969594",
            surface_points=gp.data.SurfacePointsTable(),
            orientations=gp.data.OrientationsTable()
        )
        

        pass
