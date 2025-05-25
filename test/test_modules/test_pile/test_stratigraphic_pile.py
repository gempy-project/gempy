import numpy as np
import os
import pandas as pd

import pytest

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, pv_plot, to_pyvista_points

import gempy as gp
import gempy_viewer as gpv


# Check if PATH_TO_SPREMBERG_STRATIGRAPHY is set if not skip the test
@pytest.mark.skipif(
    os.getenv("PATH_TO_SPREMBERG") is None,
    reason="PATH_TO_SPREMBERG_STRATIGRAPHY is not set"
)

class TestStratigraphicPile:
    @pytest.fixture(autouse=True)
    def borehole_set(self):
        reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG") + "Spremberg_stratigraphy.csv",
            columns_map={
                    'hole_id'   : 'id',
                    'depth_from': 'top',
                    'depth_to'  : 'base',
                    'lit_code'  : 'component lith'
            }
        )

        lith: pd.DataFrame = read_lith(reader)
        reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG") + "Spremberg_survey.csv",
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
            file_or_buffer=os.getenv("PATH_TO_SPREMBERG") + "Spremberg_collar_updated.csv",
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

    # TODO: Rename this to test structural elements from borehole set
    def test_structural_elements_from_borehole_set(self, borehole_set: BoreholeSet):
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
                    "Buntsandstein"       : {
                            "id"   : 53_300,
                            "color": "#983999"
                    },
                    "Werra-Anhydrit"      : {
                            "id"   : 61_730,
                            "color": "#00923f"
                    },
                    "Kupfershiefer"       : {
                            "id"   : 61_760,
                            "color": "#da251d"
                    },
                    "Zechsteinkonglomerat": {
                            "id"   : 61_770,
                            "color": "#f8c300"
                    },
                    "Rotliegend"          : {
                            "id"   : 62_000,
                            "color": "#bb825b"
                    }
            },
            group_by="component lith"
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

        all_surface_points_coords: gp.data.SurfacePointsTable = structural_frame.surface_points_copy
        extent_from_data = all_surface_points_coords.xyz.min(axis=0), all_surface_points_coords.xyz.max(axis=0)
        
        geo_model = gp.data.GeoModel.from_args(
            name="Stratigraphic Pile",
            structural_frame=structural_frame,
            grid=gp.data.Grid(
                extent=[extent_from_data[0][0], extent_from_data[1][0], extent_from_data[0][1], extent_from_data[1][1], extent_from_data[0][2], extent_from_data[1][2]],
                resolution=(50, 50, 50)
            ),
            interpolation_options=gp.data.InterpolationOptions.from_args(
                range=5,
                c_o=10,
                mesh_extraction=True,
                number_octree_levels=3,
            ),
        )
        gempy_plot = gpv.plot_3d(
            model=geo_model,
            # ve=10,
            kwargs_pyvista_bounds={
                    'show_xlabels': False,
                    'show_ylabels': False,
                    # 'show_zlabels': True,
            },
            show=True,
            image=True
        )
        