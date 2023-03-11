# These two lines are necessary only if GemPy is not installed
import os
import sys

# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pytest

import gempy_engine
from gempy_3.api.gp2_to_gp3_input import gempy_project_to_interpolation_options, gempy_project_to_input_data_descriptor, gempy_project_to_interpolation_input
from gempy_3.api.gp3_to_gp2_output import set_gp3_solutions_to_gp2_solution
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions

sys.path.append("../..")
input_path = os.path.dirname(__file__) + '/../../../test/input_data'

save = False


@pytest.fixture(scope="module")
def geo_model():
    geo_model = gp.create_model('Test_uncomformities')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 10., 0, 2., 0, 5.], [100, 3, 100],
                 path_o=input_path + '/05_toy_fold_unconformity_orientations.csv',
                 path_i=input_path + '/05_toy_fold_unconformity_interfaces.csv',
                 default_values=True)

    gp.map_stack_to_surfaces(geo_model,
                             {"Flat_Series"    : 'Flat',
                              "Inclined_Series": 'Inclined',
                              "Fold_Series"    : ('Basefold', 'Topfold', 'basement')})

    return geo_model


def test_all_erosion(geo_model):
    # @off
    interpolation_input  : InterpolationInput   = gempy_project_to_interpolation_input(geo_model)
    input_data_descriptor: InputDataDescriptor  = gempy_project_to_input_data_descriptor(geo_model)
    options              : InterpolationOptions = gempy_project_to_interpolation_options(geo_model)
    # @on

    solutions: Solutions = gempy_engine.compute_model(
        # @off
        interpolation_input = interpolation_input,
        options             = options,
        data_descriptor     = input_data_descriptor
        # @on
    )

    set_gp3_solutions_to_gp2_solution(gp3_solution=solutions, geo_model=geo_model)

    # sol = gp.compute_model(geo_model, compute_mesh=True)
    sol = geo_model.solutions

    # TODO: find matrix pad equivalent
    mask_lith_0: np.ndarray = solutions.octrees_output[0].outputs_centers[0].squeezed_mask_array
    mask_lith_1: np.ndarray = solutions.octrees_output[0].outputs_centers[1].squeezed_mask_array
    mask_lith_2: np.ndarray = solutions.octrees_output[0].outputs_centers[2].squeezed_mask_array

    gp.plot.plot_2d(geo_model, cell_number=2)

    if True:
        gp.plot_2d(geo_model, cell_number=[2],
                   regular_grid=mask_lith_0,
                   show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})
        gp.plot_2d(geo_model, cell_number=[2],
                   regular_grid=mask_lith_1,
                   show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

        gp.plot_2d(geo_model, cell_number=[2],
                   regular_grid=mask_lith_2,
                   show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})

    print(sol)


def test_one_onlap(geo_model):
    geo_model.set_bottom_relation('Inclined_Series', bottom_relation='Onlap')
    geo_model.set_bottom_relation('Flat_Series', bottom_relation='Erosion')
    # @off
    interpolation_input  : InterpolationInput   = gempy_project_to_interpolation_input(geo_model)
    input_data_descriptor: InputDataDescriptor  = gempy_project_to_input_data_descriptor(geo_model)
    options              : InterpolationOptions = gempy_project_to_interpolation_options(geo_model)
    # @on

    solutions: Solutions = gempy_engine.compute_model(
        # @off
        interpolation_input = interpolation_input,
        options             = options,
        data_descriptor     = input_data_descriptor
        # @on
    )

    set_gp3_solutions_to_gp2_solution(gp3_solution=solutions, geo_model=geo_model)

    sol = geo_model.solutions

    # TODO: find matrix pad equivalent
    mask_lith_0: np.ndarray = solutions.octrees_output[0].outputs_centers[0].squeezed_mask_array
    mask_lith_1: np.ndarray = solutions.octrees_output[0].outputs_centers[1].squeezed_mask_array
    mask_lith_2: np.ndarray = solutions.octrees_output[0].outputs_centers[2].squeezed_mask_array

    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2], regular_grid=mask_lith_0,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2], regular_grid=mask_lith_1,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2], regular_grid=mask_lith_2,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    if plot_3d := False:
        p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                         image=True, kwargs_plot_structured_grid={'opacity': .2})


def test_two_onlap(geo_model):
    geo_model.set_bottom_relation(['Flat_Series', 'Inclined_Series'], bottom_relation='Onlap')

    # @off
    interpolation_input  : InterpolationInput   = gempy_project_to_interpolation_input(geo_model)
    input_data_descriptor: InputDataDescriptor  = gempy_project_to_input_data_descriptor(geo_model)
    options              : InterpolationOptions = gempy_project_to_interpolation_options(geo_model)
    # @on

    solutions: Solutions = gempy_engine.compute_model(
        # @off
        interpolation_input = interpolation_input,
        options             = options,
        data_descriptor     = input_data_descriptor
        # @on
    )

    set_gp3_solutions_to_gp2_solution(gp3_solution=solutions, geo_model=geo_model)

    sol = geo_model.solutions

    # TODO: find matrix pad equivalent
    mask_lith_0: np.ndarray = solutions.octrees_output[0].outputs_centers[0].squeezed_mask_array
    mask_lith_1: np.ndarray = solutions.octrees_output[0].outputs_centers[1].squeezed_mask_array
    mask_lith_2: np.ndarray = solutions.octrees_output[0].outputs_centers[2].squeezed_mask_array

    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_0,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_1,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_2,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})


def test_masked_marching_cubes():
    cwd = os.path.dirname(__file__)
    data_path = cwd + '/../../../examples/'
    geo_model = gp.load_model(
        name=r'Tutorial_ch1-8_Onlap_relations',
        path=data_path + 'data/gempy_models/Tutorial_ch1-8_Onlap_relations',
        recompile=False
    )

    geo_model.set_regular_grid([-200, 1000, -500, 500, -1000, 0], [50, 50, 50])

    # @off
    interpolation_input  : InterpolationInput   = gempy_project_to_interpolation_input(geo_model)
    input_data_descriptor: InputDataDescriptor  = gempy_project_to_input_data_descriptor(geo_model)
    options              : InterpolationOptions = gempy_project_to_interpolation_options(geo_model)
    # @on

    solutions: Solutions = gempy_engine.compute_model(
        # @off
        interpolation_input = interpolation_input,
        options             = options,
        data_descriptor     = input_data_descriptor
        # @on
    )

    set_gp3_solutions_to_gp2_solution(gp3_solution=solutions, geo_model=geo_model)

    sol = geo_model.solutions

    # TODO: find matrix pad equivalent
    mask_lith_0: np.ndarray = solutions.octrees_output[0].outputs_centers[0].squeezed_mask_array
    mask_lith_1: np.ndarray = solutions.octrees_output[0].outputs_centers[1].squeezed_mask_array
    mask_lith_2: np.ndarray = solutions.octrees_output[0].outputs_centers[2].squeezed_mask_array
    mask_lith_3: np.ndarray = solutions.octrees_output[0].outputs_centers[3].squeezed_mask_array

    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_0,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_1,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_2,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=mask_lith_3,
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})
