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

        vertex_attributes: pd.DataFrame = borehole_trajectory.data.points_attributes
        unique_lith_codes = vertex_attributes['component lith'].unique()

        component_lith = borehole_set.compute_tops()

        pleistozen = gp.data.StructuralElement(
            name="Pleistozen",
            id=10_000,
            color="#f9f97f",
            surface_points=gp.data.SurfacePointsTable(np.empty(0, dtype=gp.data.SurfacePointsTable.dt)),
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        kreide = gp.data.StructuralElement(
            name="Kreide",
            id=30_000,
            color="#a6d84a",
            surface_points=gp.data.SurfacePointsTable(np.empty(0, dtype=gp.data.SurfacePointsTable.dt)),
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        trias = gp.data.StructuralElement(
            name="Trias",
            id=50_000,
            color="#a4469f",
            surface_points=gp.data.SurfacePointsTable(np.empty(0, dtype=gp.data.SurfacePointsTable.dt)),
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        perm = gp.data.StructuralElement(
            name="Perm",
            id=60_000,
            color="#f4a142",
            surface_points=gp.data.SurfacePointsTable(np.empty(0, dtype=gp.data.SurfacePointsTable.dt)),
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        rotliegend_id = 62_000
        rotliegend_xyz = component_lith[rotliegend_id]

        # Add the id 
        rotliegend_surface_points = gp.data.SurfacePointsTable.from_arrays(
            x=rotliegend_xyz[:, 0],
            y=rotliegend_xyz[:, 1],
            z=rotliegend_xyz[:, 2],
            names=["Rotliegend"],
            name_id_map={"Rotliegend": rotliegend_id}
        )

        rotliegend = gp.data.StructuralElement(
            name="Rotliegend",
            id=rotliegend_id,
            color="#bb825b",
            surface_points=rotliegend_surface_points,
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        devon = gp.data.StructuralElement(
            name="Devon",
            id=80_000,
            color="#969594",
            surface_points=gp.data.SurfacePointsTable(np.empty(0, dtype=gp.data.SurfacePointsTable.dt)),
            orientations=gp.data.OrientationsTable(np.zeros(0, dtype=gp.data.OrientationsTable.dt))
        )

        group = gp.data.StructuralGroup(
            name="Stratigraphic Pile",
            elements=[rotliegend],
            structural_relation=gp.data.StackRelationType.ERODE
        )
        structural_frame = gp.data.StructuralFrame(
            structural_groups=[group],
            color_gen=gp.data.ColorsGenerator()
        )
        print(group)

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
