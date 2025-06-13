import os
import dotenv

import gempy as gp
from gempy.API.io_API import read_surface_points
from gempy.core.data.surface_points import SurfacePointsTable
import gempy_viewer as gpv

dotenv.load_dotenv()


def test_2025_2():
    # * Here I just leave as variable both, the manual that
    # * you did and "the proper" that injects it to gempy 
    # * before interpolation without modifying anything else
    manual_rescale = 1
    proper_rescale = 20.
    
    range_ = 2.4
    orientation_loc = -690 * manual_rescale
    path_to_data = os.getenv("TEST_DATA")

    data = {
            "a": read_surface_points(f"{path_to_data}/a.dat"),
            "b": read_surface_points(f"{path_to_data}/b.dat"),
            "c": read_surface_points(f"{path_to_data}/c.dat"),
            "d": read_surface_points(f"{path_to_data}/d.dat"),
            "e": read_surface_points(f"{path_to_data}/e.dat"),
            "f": read_surface_points(f"{path_to_data}/f.dat"),
    }

    # rescale the Z values
    data = {
        k: SurfacePointsTable.from_arrays(
            x=v.data["X"],
            y=v.data["Y"],
            z=manual_rescale * v.data["Z"],  # rescaling the z values
            names=[k] * len(v.data),
            nugget=v.data["nugget"]
        )
        for k, v in data.items()
    }

    color_generator = gp.data.ColorsGenerator()
    elements = []
    for event, pts in data.items():
        orientations = gp.data.OrientationsTable.initialize_empty()
        element = gp.data.StructuralElement(
            name=event,
            color=next(color_generator),
            surface_points=pts,
            orientations=orientations,
        )
        elements.append(element)

    group = gp.data.StructuralGroup(
        name="Series1",
        elements=elements,
        structural_relation=gp.data.StackRelationType.ERODE,
        fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_FORMATIONS,
    )
    structural_frame = gp.data.StructuralFrame(
        structural_groups=[group], color_gen=color_generator
    )

    xmin = 525816
    xmax = 543233
    ymin = 5652470
    ymax = 5657860
    zmin = -780 * manual_rescale
    zmax = -636 * manual_rescale

    # * Add 20% to extent
    xmin -= 0.2 * (xmax - xmin)
    xmax += 0.2 * (xmax - xmin)
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)
    zmin -= 0.2 * (zmax - zmin)
    zmax += 1 * (zmax - zmin)

    geo_model = gp.create_geomodel(
        project_name="test",
        extent=[xmin, xmax, ymin, ymax, zmin, zmax],
        refinement=5,
        structural_frame=structural_frame,
    )
   
    # * Here it is the way of rescaling one of the axis. Input transform
    # * is used (by default) to rescale data into a unit cube but it accepts any transformation matrix.  
    geo_model.input_transform.scale[2] *= proper_rescale

    if True:
        gpv.plot_3d(
            model=geo_model,
            ve=1,
            image=True,
            kwargs_pyvista_bounds={
                    'show_xlabels': False,
                    'show_ylabels': False,
            },
            transformed_data=True # * This is interesting, transformed data shows the data as it goes to the interpolation (after applying the transform)
        )
        

    geo_model.interpolation_options.evaluation_options.number_octree_levels_surface = 4
    geo_model.interpolation_options.kernel_options.range = range_
    gp.modify_surface_points(geo_model, nugget=1e-5)
    gp.add_orientations(
        geo_model=geo_model,
        x=[525825],
        y=[5651315],
        z=[orientation_loc],  # * Moving the orientation further
        pole_vector=[[0, 0, 1]],
        elements_names=["a"]
    )
    solution = gp.compute_model(
        geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy
        ),
    )
    
    gpv.plot_3d(
        model=geo_model,
        ve=proper_rescale,
        show_lith=True,
        image=True,
        kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
            'show_zlabels': False,
        },
    )
    
    
    # region Exporting scalar field
    gpv.plot_2d(
        geo_model,
        show_scalar=True,
        series_n=0
    )
    
    # * The scalar fields can be found for dense and octree grids:
    print(geo_model.solutions.raw_arrays.scalar_field_matrix)
    
    # * For custom grids so far we do not have a property that gives it directly, but it can be accessed here
    
    octree_lvl = 0  # * All the grids that are not octree are computed on octree level 0
    stack_number = -1  # * Here we choose the stack that we need. At the moment boolean operations--for erosion-- are not calculated on the scalar field
    gempy_output = geo_model.solutions.octrees_output[octree_lvl].outputs_centers[stack_number]
    slice_ = gempy_output.grid.custom_grid_slice
    scalar_field = gempy_output.scalar_fields.exported_fields.scalar_field[slice_]
    print(scalar_field)
    # endregion
    

if __name__ == "__main__":
    test_2025_2()
