import os
import gempy as gp
from gempy.bayesian.theano_op import GemPyThOp
import pytest
pm = pytest.importorskip("rgeomod")
# import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import theano
np.random.seed(4003)


def test_basic():
    y_obs_list = [2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                  2.19, 2.07, 2.16, 2.11, 2.13, 1.92]

    with pm.Model() as model:
        mu = pm.Normal('$\mu$', 2.08, .07)
        sigma = pm.Gamma('$\sigma$', 0.3, 3)
        y = pm.Normal('$y$', mu, sigma, observed=y_obs_list)

        prior = pm.sample_prior_predictive(1000)
        trace = pm.sample(1000, discard_tuned_samples=False, cores=1)
        post = pm.sample_posterior_predictive(trace)

    pm.plot_posterior(trace)
    plt.show()


def test_gempy_th_op_test():
    path_dir = os.getcwd() + '/../../examples/tutorials/ch5_probabilistic_modeling'
    geo_model = gp.load_model(r'2-layers', path=path_dir, recompile=False)
    gto = GemPyThOp(geo_model)
    sol = gto.test_gradient('lith', 'surface_points')
    print(sol)


def test_gempy_th_op_set_grav():
    path_dir = os.getcwd() + '/../../examples/tutorials/ch5_probabilistic_modeling'
    geo_model = gp.load_model(r'2-layers', path=path_dir, recompile=False)
    gp.set_interpolator(geo_model, output='grav')

    gto = GemPyThOp(geo_model)
    th_op_grav = gto.set_th_op('gravity')
    i = geo_model.interpolator.get_python_input_block()
    th_f = theano.function([], th_op_grav(*i), on_unused_input='warn')