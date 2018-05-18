
# coding: utf-8

import sys, os
sys.path.append("notebooks/tutorials")


# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

#from ..conftest import theano_f_1f
input_path = os.path.dirname(__file__)+'/../../notebooks'

def test_ch1(theano_f_1f):
    # Importing the data from CSV-files and setting extent and resolution
    geo_data = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                              path_o=input_path+'/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                              path_i=input_path+'/input_data/tut_chapter1/simple_fault_model_points.csv')


    gp.get_data(geo_data)

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"Fault_Series":'Main_Fault',
                             "Strat_Series": ('Sandstone_2','Siltstone',
                                              'Shale', 'Sandstone_1')},
                           order_series = ["Fault_Series", 'Strat_Series'],
                           order_formations=['Main_Fault',
                                             'Sandstone_2','Siltstone',
                                             'Shale', 'Sandstone_1',
                                             ], verbose=0)


    gp.get_sequential_pile(geo_data)

    print(gp.get_grid(geo_data))

    gp.get_data(geo_data, 'interfaces').head()

    gp.get_data(geo_data, 'orientations')

    gp.plot_data(geo_data, direction='y')

    # interp_data = gp.InterpolatorData(geo_data, u_grade=[1,1],
    #                                   output='geology', compile_theano=True,
    #                                   theano_optimizer='fast_compile',
    #                                   verbose=[])

    interp_data = theano_f_1f
    interp_data.update_interpolator(geo_data)

    gp.get_kriging_parameters(interp_data) # Maybe move this to an extra part?

    lith_block, fault_block = gp.compute_model(interp_data)


    gp.plot_section(geo_data, lith_block[0], cell_number=25,  direction='y', plot_data=True)


    gp.plot_scalar_field(geo_data, lith_block[1], cell_number=25, N=15,
                            direction='y', plot_data=False)


    gp.plot_scalar_field(geo_data, lith_block[1], cell_number=25, N=15,
                            direction='z', plot_data=False)

    gp.plot_section(geo_data, fault_block[0], cell_number=25, plot_data=True, direction='y')

    gp.plot_scalar_field(geo_data, fault_block[1], cell_number=25, N=20,
                            direction='y', plot_data=False)


    ver, sim = gp.get_surfaces(interp_data,lith_block[1], fault_block[1], original_scale=True)

    # Cropping a cross-section to visualize in 2D #REDO this part?
    bool_b = np.array(ver[1][:,1] > 999)* np.array(ver[1][:,1] < 1001)
    bool_r = np.array(ver[1][:,1] > 1039)* np.array(ver[1][:,1] < 1041)

    # Plotting section
    gp.plot_section(geo_data, lith_block[0], 25, plot_data=True)
    ax = plt.gca()

    # Adding grid
    ax.set_xticks(np.linspace(0, 2000, 100, endpoint=False))
    ax.set_yticks(np.linspace(0, 2000, 100, endpoint=False))
    plt.grid()

    plt.ylim(1000,1600)
    plt.xlim(500,1100)
    # Plotting vertices
    ax.plot(ver[1][bool_r][:, 0], ver[1][bool_r][:, 2], '.', color='b', alpha=.9)
    ax.get_xaxis().set_ticklabels([])



    ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
                                   fault_block[1],
                                   original_scale=True)

  #  gp.plotting.plot_surfaces_3D_real_time(geo_data, interp_data, ver_s, sim_s)




