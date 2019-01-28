import pytest
import numpy as np
import sys, os
sys.path.append("../..")
import gempy as gp
import gempy.utils.fault_analysis as fa
import matplotlib.pyplot as plt

input_path = os.path.dirname(__file__)+'/../notebooks'

@pytest.fixture
def fault_geodata():
    # initialize geo_data object

    sys.path.insert(0, input_path)

    geo_data = gp.create_data([0,2000,0,2000,0,2000],[30,30,30],
                         path_o = input_path+"/input_data/NormalFault_AnticlinalTrap_O.csv",
                         path_i = input_path+"/input_data/NormalFault_AnticlinalTrap_P.csv")

    geo_data.n_faults = 1

    # Assigning series to formations and orders
    gp.set_series(geo_data, {"fault":'NormalFault',
                      "Rest":('Overlying', 'Sandstone2', 'Shale', 'Sandstone')},
                       order_series = ["fault","Rest",], order_formations=['NormalFault',
                                         'Overlying', 'Sandstone2', 'Shale', 'Sandstone',
                                         ])
    return geo_data

@pytest.fixture
def fault_interp(fault_geodata):
    interp_data = gp.InterpolatorData(fault_geodata,u_grade=[3],
                                  output='gradients', dtype='float64', compile_theano=True)
    return interp_data

@pytest.fixture
def fault_model(fault_interp):
    geo_model = gp.compute_model(fault_interp)
    return geo_model

def test_get_fault_mask(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    fault_sol = fault_model[1]
    mask_test = np.load("input_data/utils_testdata/fa_test_mask.npy")
    fa_mask = fa.get_fault_mask(fault_geodata, fault_sol, fault_n, fault_side='both')
    assert np.array_equal(fa_mask, mask_test), "Mismatch in fault mask"

@pytest.fixture
def test_get_LF_contact_VOX(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    contact0 = fa.get_LF_contact_VOX(fault_geodata, lith_sol, fault_sol, lith_n=sst_n, fault_n=fault_n, fault_side='footwall')
    return contact0

def test_contact(test_get_LF_contact_VOX):
    contact_test = np.load("input_data/utils_testdata/fa_test_contact.npy")
    assert np.array_equal(test_get_LF_contact_VOX, contact_test), "Mismatch in lithology-fault contact in voxels."

def test_get_extrema_line_voxels(test_get_LF_contact_VOX):
    maxline = fa.get_extrema_line_voxels(test_get_LF_contact_VOX, extrema_type='max')
    maxline_test = np.load("input_data/utils_testdata/fa_test_maxline.npy")
    assert np.array_equal(maxline, maxline_test), "Mismatch in lithology-fault contact extrema line (max) in voxels."

def test_project_voxels(test_get_LF_contact_VOX):
    proj_test = np.load("input_data/utils_testdata/fa_test_contact_proj.npy")
    proj = fa.project_voxels(test_get_LF_contact_VOX)
    assert np.array_equal(proj, proj_test), "Mismatch in the projection of fault-contact voxels voxels."

def test_FA_plotting(fault_geodata, fault_model):
    fault_plot = fa.PlotFault2D(fault_geodata)
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    shale_n = fault_geodata.formations.loc['Shale', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]

    plt.subplot(321)
    plt.title('B1- Full lithology-fault contact')
    fault_plot.plot_lith_fault_contact_full(lith_sol, fault_sol, fault_n, fault_side='footwall')

    plt.subplot(322)
    plt.title('B2 - Contact of chosen lithologies')
    fault_plot.plot_lith_fault_contact(lith_sol, fault_sol, fault_n, lith_n=[shale_n, sst_n], fault_side='footwall')

    plt.subplot(323)
    plt.title('B3 - Conceptual Allan Diagram')
    fault_plot.plot_AllanDiagram(lith_sol, fault_sol, fault_n, lith_target=[sst_n], lith_jux=[sst_n], target_side='fw')

    plt.subplot(324)
    plt.title('B4 - Juxtapositions')
    fault_plot.plot_juxtapositions(lith_sol, fault_sol, fault_n, lith_target=[sst_n], lith_jux=[sst_n],
                                   target_side='fw');

def test_arg_contact_peaks_VOX(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    arg_test = np.array([12,13,14,15,16])
    arg = fa.arg_contact_peaks_VOX(fault_geodata, lith_sol, fault_sol, lith_n=sst_n, fault_n=fault_n)
    assert np.array_equal(arg, arg_test), "Mismatch in contact peak locations (voxel positions)."

def test_get_full_LFcontact_projected(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    full_contact_test = np.load("input_data/utils_testdata/fa_test_contact_full.npy")
    full_contact = fa.get_full_LFcontact_projected(fault_geodata, lith_sol, fault_sol, \
                                    fault_n, fault_side='footwall',
                                    projection='automatic')
    assert np.array_equal(full_contact, full_contact_test), "Mismatch in full fault contact (projected)."

def test_get_contact_peaks_VOX(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    c_peaks_test = np.array([[12,14],[13,14], [14,14], [15,14], [16,14]])
    c_peaks = fa.get_contact_peaks_VOX(fault_geodata, lith_sol, fault_sol, lith_n=sst_n, \
                          fault_n=fault_n, projection='automatic', fault_side='fw', \
                          order='automatic')
    assert np.array_equal(c_peaks, c_peaks_test), "Mismatch in contact peaks (voxel positions X/Y and Z)."

def test_get_faultthrow_at(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    faultthrow_test = 133.0
    faultthrow = np.round(fa.get_faultthrow_at(fault_geodata, lith_sol, fault_sol, lith_n=sst_n, fault_n=fault_n, position=15))
    assert faultthrow == faultthrow_test, "Mismatch in determined fault throw."

def test_get_lithcontact_thickness_at(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    lc_thickness_test = 200.0
    lc_thickness = fa.get_lithcontact_thickness_at(fault_geodata, lith_sol, fault_sol, lith_n=sst_n, fault_n=fault_n, fault_side='fw', position=15)
    assert lc_thickness == lc_thickness_test, "Mismatch in determined thickness of lithology at contact with fault."

def test_get_juxtapositions(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    juxta_test = np.load("input_data/utils_testdata/fa_test_juxta.npy")
    juxta = fa.get_juxtapositions(fault_geodata, lith_sol, fault_sol, fault_n, \
                          lith_target=sst_n, lith_jux=sst_n, target_side='fw', \
                          projection='automatic')
    assert np.array_equal(juxta, juxta_test), "Mismatch between determined juxtapositions."

def test_get_juxtapositions_at(fault_geodata, fault_model):
    fault_n = fault_geodata.formations.loc['NormalFault', 'formation_number']
    sst_n = fault_geodata.formations.loc['Sandstone', 'formation_number']
    lith_sol = fault_model[0]
    fault_sol = fault_model[1]
    juxta_at_test = np.array([False, False, False, False, False, False, False, False, False,
       False, False,  True,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False])
    juxta_at = fa.get_juxtapositions_at(fault_geodata, lith_sol, fault_sol, position=15, fault_n=fault_n, \
                          lith_target=sst_n, lith_jux=sst_n, target_side='fw', \
                          projection='automatic')
    assert np.array_equal(juxta_at, juxta_at_test), "Mismatch between determined juxtaposition columns at positions."



