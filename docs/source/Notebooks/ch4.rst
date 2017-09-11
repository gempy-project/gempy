
Chapter 4: Bayesian Statistics in pymc3 (Working in progress proof of concept)
==============================================================================

.. code:: ipython3

    # These two lines are necessary only if gempy is not installed
    import sys, os
    sys.path.append("../")
    
    # Importing gempy
    import gempy as gp
    
    # Embedding matplotlib figures into the notebooks
    %matplotlib inline
    
    # Aux imports
    import numpy as np

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000-10000,747000 + 20600,6863000 - 20600,6950000 + 20600,-20000, 600],[50, 50, 50],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")

.. code:: ipython3

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite', 
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'}, 
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2', 'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)




.. parsed-literal::

    <gempy.strat_pile.StratigraphicPile at 0x7f76e43129e8>




.. image:: ch4_files/ch4_3_1.png


Setting uncertainties adding the values to the Dataframe.

.. code:: ipython3

    geo_data.interfaces['X_std'] = None
    geo_data.interfaces['Y_std'] = 0
    geo_data.interfaces['Z_std'] = 800
    
    geo_data.foliations['X_std'] = None
    geo_data.foliations['Y_std'] = 0
    geo_data.foliations['Z_std'] = 0
    
    geo_data.foliations['dip_std'] = 10
    geo_data.foliations['azimuth_std'] = 10
    geo_data.foliations.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
          <th>order_series</th>
          <th>isFault</th>
          <th>formation number</th>
          <th>annotations</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>X_std</th>
          <th>Y_std</th>
          <th>Z_std</th>
          <th>dip_std</th>
          <th>azimuth_std</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>722403.8130</td>
          <td>6880913.25</td>
          <td>470.707065</td>
          <td>123.702047</td>
          <td>80.0</td>
          <td>1</td>
          <td>EarlyGranite</td>
          <td>EarlyGranite_Series</td>
          <td>1</td>
          <td>False</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
          <td>0.819295</td>
          <td>-0.546444</td>
          <td>0.173648</td>
          <td>None</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
          <td>10</td>
        </tr>
        <tr>
          <th>1</th>
          <td>718928.3440</td>
          <td>6883605.50</td>
          <td>509.462245</td>
          <td>176.274084</td>
          <td>80.0</td>
          <td>1</td>
          <td>EarlyGranite</td>
          <td>EarlyGranite_Series</td>
          <td>1</td>
          <td>False</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},1}$</td>
          <td>0.063996</td>
          <td>-0.982726</td>
          <td>0.173648</td>
          <td>None</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
          <td>10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>720690.5630</td>
          <td>6882822.25</td>
          <td>489.909423</td>
          <td>254.444427</td>
          <td>80.0</td>
          <td>1</td>
          <td>EarlyGranite</td>
          <td>EarlyGranite_Series</td>
          <td>1</td>
          <td>False</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},2}$</td>
          <td>-0.948735</td>
          <td>-0.264099</td>
          <td>0.173648</td>
          <td>None</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
          <td>10</td>
        </tr>
        <tr>
          <th>3</th>
          <td>721229.0005</td>
          <td>6880766.25</td>
          <td>477.680894</td>
          <td>255.876557</td>
          <td>80.0</td>
          <td>1</td>
          <td>EarlyGranite</td>
          <td>EarlyGranite_Series</td>
          <td>1</td>
          <td>False</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},3}$</td>
          <td>-0.955039</td>
          <td>-0.240305</td>
          <td>0.173648</td>
          <td>None</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
          <td>10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>710459.8440</td>
          <td>6880521.50</td>
          <td>511.839758</td>
          <td>232.658556</td>
          <td>80.0</td>
          <td>1</td>
          <td>EarlyGranite</td>
          <td>EarlyGranite_Series</td>
          <td>1</td>
          <td>False</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},4}$</td>
          <td>-0.782957</td>
          <td>-0.597349</td>
          <td>0.173648</td>
          <td>None</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
          <td>10</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # input_data_T = interp_data.interpolator.tg.input_parameters_list()
    # input_data_P = interp_data.get_input_data(u_grade=[3, 3])
    # select = interp_data.interpolator.pandas_rest_layer_points['formation'] == 'Reservoir'

.. code:: ipython3

    interp_data_grav = gp.InterpolatorInput(geo_data, output='gravity', compile_theano=False,
                                       u_grade=[3, 3, 3])

.. code:: ipython3

    gp.set_geophysics_obj(interp_data_grav,  [7.050000e+05,747000,6863000,6925000,-20000, 200],
                                                 [50,50], )




.. parsed-literal::

    <gempy.GeoPhysics.GeoPhysicsPreprocessing_pro at 0x7f7651566b70>



.. code:: ipython3

    gp.precomputations_gravity(interp_data_grav, 25, [2.92, 3.1, 2.92, 2.61, 2.61])




.. parsed-literal::

    (array([[  2.32206772e-05,   1.38317570e-05,   4.37779836e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32206772e-05,   1.38317570e-05,   4.37779837e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32206772e-05,   1.38317570e-05,   4.37779837e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            ..., 
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06]]),
     array([False, False, False, ..., False, False, False], dtype=bool))



Now the generation of the geomodel will be an operation embedded in a
larger tree.

.. code:: ipython3

    import theano
    import theano.tensor as T
    geomodel = theano.OpFromGraph(interp_data_grav.interpolator.tg.input_parameters_list(),
                                  interp_data_grav.interpolator.tg.compute_geological_model(n_faults=0, compute_all=True),
                                  on_unused_input='ignore',
                                )

.. code:: ipython3

    import theano
    import theano.tensor as T
    geomodel = theano.OpFromGraph(interp_data_grav.interpolator.tg.input_parameters_list(),
                                  [interp_data_grav.interpolator.tg.compute_forward_gravity(n_faults=0, compute_all=True)],
                                  on_unused_input='ignore',
                                )

Because now the GeMpy model is a theano operation and not a theano
function, to call it we need to use theano variables (with theano
functions we call them with python variables). This is very easy to
modify, we just need to use theano shared to convert our python input
data into theano variables.

The pymc3 objects are already theano variables (pm.Normal and so on).
Now the trick is that using the theano function T.set\_subtensor, we can
change one deterministic value of the input arrays(the ones printed in
the cell above) by a stochastic pymc3 object. Then with the new arrays
we just have to call the theano operation and pymc will do the rest

.. code:: ipython3

    # This is the creation of the model
    import pymc3 as pm
    theano.config.compute_test_value = 'off'
    #theano.config.warn_float64 = 'warn'
    model = pm.Model()
    with model:
        # We create the Stochastic parameters. In this case only the Z position
        # of the interfaces
        Z_rest = pm.Normal('Z_unc_rest',
           interp_data_grav.interpolator.pandas_rest_layer_points['Z'].as_matrix().astype('float32'),
           interp_data_grav.interpolator.pandas_rest_layer_points['Z_std'].as_matrix().astype('float32'),
                      dtype='float32', shape = (66))
        
        Z_ref = pm.Normal('Z_unc_ref', interp_data_grav.interpolator.pandas_ref_layer_points_rep['Z'].astype('float32'),
                  interp_data_grav.interpolator.pandas_ref_layer_points_rep['Z_std'].astype('float32'),
                  dtype='float32', shape = (66))
        
    #     Z_unc = pm.Normal('Z_unc', interp_data_grav.geo_data_res.interfaces['Z'].astype('float32'),
    #                       interp_data_grav.geo_data_res.interfaces['Z_std'].astype('float32'), dtype='float32', shape= (70))
        
    #     interp_data_grav.geo_data_res.interfaces['Z'] = Z_unc
        
        # We convert a python variable to theano.shared
        input_sh = []
        for i in interp_data_grav.get_input_data():
            input_sh.append(theano.shared(i))
        
        # We add the stochastic value to the correspondant array. rest array is
        # a n_points*3 (XYZ) array. We only want to change Z in this case.
        input_sh[4] = T.set_subtensor(
        input_sh[4][:, 2], Z_ref)
    
        input_sh[5] = T.set_subtensor(
        input_sh[5][:, 2], Z_rest)
        
        # With the stochastic parameters we create the geomodel result:
        geo_model = pm.Deterministic('GemPy', geomodel(input_sh[0], input_sh[1], input_sh[2],
                                                       input_sh[3], input_sh[4], input_sh[5]))

.. code:: ipython3

    theano.config.compute_test_value = 'ignore'
    # This is the sampling
    # BEFORE RUN THIS FOR LONG CHECK IN THE MODULE THEANOGRAF THAT THE FLAG 
    # THEANO OPTIMIZER IS IN 'fast_run'!!
    with model:
       # backend = pm.backends.ndarray.NDArray('geomodels')
        step = pm.NUTS()
        trace = pm.sample(30, tune=10, init=None, step=step, )


.. parsed-literal::

    100%|██████████| 40/40 [01:36<00:00,  2.25s/it]/home/miguel/anaconda3/lib/python3.6/site-packages/pymc3/step_methods/hmc/nuts.py:418: UserWarning: Chain 0 contains only 30 samples.
      % (self._chain_id, n))
    /home/miguel/anaconda3/lib/python3.6/site-packages/pymc3/step_methods/hmc/nuts.py:440: UserWarning: The acceptance probability in chain 0 does not match the target. It is 0.955144655704, but should be close to 0.8. Try to increase the number of tuning steps.
      % (self._chain_id, mean_accept, target_accept))
    


.. code:: ipython3

    trace.get_values('GemPy')[5] -  trace.get_values('GemPy')[15]




.. parsed-literal::

    array([-0.00171852, -0.00116444, -0.00116444, ...,  0.        ,
            0.        ,  0.        ], dtype=float32)



.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.imshow(trace.get_values('GemPy')[-10].reshape(50,50), cmap='viridis', origin='lower', extent=[7.050000e+05,747000,6863000,6950000] )
    plt.colorbar()




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7f75f9def2e8>




.. image:: ch4_files/ch4_17_1.png


.. code:: ipython3

    import matplotlib.pyplot as plt
    for i in range(100):
        gp.plot_section(geo_data, trace.get_values('GemPy')[i][0, :], 18,
                           direction='y', plot_data=False)
        plt.show()



.. image:: ch4_files/ch4_18_0.png



.. image:: ch4_files/ch4_18_1.png



.. image:: ch4_files/ch4_18_2.png



.. image:: ch4_files/ch4_18_3.png



.. image:: ch4_files/ch4_18_4.png



.. image:: ch4_files/ch4_18_5.png



.. image:: ch4_files/ch4_18_6.png



.. image:: ch4_files/ch4_18_7.png



.. image:: ch4_files/ch4_18_8.png



.. image:: ch4_files/ch4_18_9.png



.. image:: ch4_files/ch4_18_10.png



.. image:: ch4_files/ch4_18_11.png



.. image:: ch4_files/ch4_18_12.png



.. image:: ch4_files/ch4_18_13.png



.. image:: ch4_files/ch4_18_14.png



.. image:: ch4_files/ch4_18_15.png



.. image:: ch4_files/ch4_18_16.png



.. image:: ch4_files/ch4_18_17.png



.. image:: ch4_files/ch4_18_18.png



.. image:: ch4_files/ch4_18_19.png



.. image:: ch4_files/ch4_18_20.png



.. image:: ch4_files/ch4_18_21.png



.. image:: ch4_files/ch4_18_22.png



.. image:: ch4_files/ch4_18_23.png



.. image:: ch4_files/ch4_18_24.png



.. image:: ch4_files/ch4_18_25.png



.. image:: ch4_files/ch4_18_26.png



.. image:: ch4_files/ch4_18_27.png



.. image:: ch4_files/ch4_18_28.png



.. image:: ch4_files/ch4_18_29.png


::


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-13-2e039459f740> in <module>()
          1 import matplotlib.pyplot as plt
          2 for i in range(100):
    ----> 3     gp.plot_section(geo_data, trace.get_values('GemPy')[i][0, :], 18,
          4                        direction='y', plot_data=False)
          5     plt.show()


    IndexError: index 30 is out of bounds for axis 0 with size 30


.. code:: ipython3

    from theano.printing import pydotprint
    
    pydotprint(model.logpt)


.. parsed-literal::

    The output file is available at /home/miguel/.theano/compiledir_Linux-4.10--generic-x86_64-with-debian-stretch-sid-x86_64-3.6.1-64/theano.pydotprint.cpu.png


