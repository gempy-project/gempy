import aesara
import aesara.tensor  as tt
import gempy as gp
import copy


class GemPyThOp:
    def __init__(self, model: gp.Project):
        model = copy.deepcopy(model)
        gp.set_interpolator(model, compile_aesara=False,
                            output=['geology', 'gravity', 'magnetics'],
                            gradient=True)
        self.model = model
        self.th_op = None

    def get_output(self, output):
        if output == 'gravity':
            out = self.model._interpolator.aesara_graph.aesara_output()[12][0]

        elif output == 'lith':
            out = self.model._interpolator.aesara_graph.aesara_output()[0][1]
        else:
            raise AttributeError()

        return out

    def get_wrt(self, wrt: str):
        if wrt == 'surface_points':
            wrt_ = self.model._interpolator.aesara_graph.input_parameters_loop[4]
        else:
            raise AttributeError

        return wrt_

    def set_th_op(self, output):
        interpolator = self.model._interpolator
        out = self.get_output(output)

        i = interpolator.get_python_input_block()
        aesara.config.compute_test_value = 'ignore'
        self.th_op = aesara.OpFromGraph(interpolator.aesara_graph.input_parameters_loop,
                                        [out],
                                        inline=False,
                                        on_unused_input='ignore',
                                        name=output)
        return self.th_op

    def test_gradient(self, output: str, wrt: str):
        aesara.config.compute_test_value = 'ignore'
        interpolator = self.model._interpolator
        out = self.get_output(output)
        wrt_ = self.get_wrt(wrt)

        geo_model_T = aesara.OpFromGraph(interpolator.aesara_graph.input_parameters_loop,
                                         [aesara.grad(out[0], wrt_)],
                                         inline=True,
                                         on_unused_input='ignore',
                                         name='test_'+output)

        i = interpolator.get_python_input_block()
        th_f = aesara.function([], geo_model_T(*i), on_unused_input='warn')

        interpolator.aesara_graph.sig_slope.set_value(20)

        return th_f()

    @staticmethod
    def set_shared(python_input):
        input_sh = []
        i = python_input
        for ii in i:
            input_sh.append(aesara.shared(ii))

        return input_sh


def gempy_th_op(geo_model):

    aesara.config.compute_test_value = 'ignore'
    geo_model_T = aesara.OpFromGraph(geo_model.interpolator.aesara_graph.input_parameters_loop,
                                    [aesara.grad(geo_model.interpolator.aesara_graph.aesara_output()[12],
                                                 geo_model.interpolator.aesara_graph.input_parameters_loop[4])],
                                     inline=True,
                                     on_unused_input='ignore',
                                     name='forw_grav')

    # %%
    i = geo_model.interpolator.get_python_input_block()
    th_f = aesara.function([], geo_model_T(*i), on_unused_input='warn')

    # %%
    geo_model.interpolator.aesara_graph.sig_slope.set_value(20)

    # %%
    th_f()


    # %%
    # Setup Bayesian model
    # --------------------
    #

    # %%
    i = geo_model.interpolator.get_python_input_block()
    aesara.config.compute_test_value = 'ignore'
    geo_model_T_grav = aesara.OpFromGraph(geo_model.interpolator.aesara_graph.input_parameters_loop,
                                    [geo_model.interpolator.aesara_graph.aesara_output()[12]],
                                     inline=False,
                                     on_unused_input='ignore',
                                     name='forw_grav')

    # %%
    geo_model_T_thick = aesara.OpFromGraph(geo_model.interpolator.aesara_graph.input_parameters_loop,
                                    [geo_model.interpolator.aesara_graph.compute_series()[0][1][0:250000]], inline=True,
                                     on_unused_input='ignore',
                                     name='geo_model')

    # %%
    # We convert a python variable to aesara.shared
    input_sh = []
    i = geo_model.interpolator.get_python_input_block()
    for ii in i:
        input_sh.append(aesara.shared(ii))

    # We get the rescaling parameters:
    rf = geo_model.rescaling.df.loc['values', 'rescaling factor'].astype('float32')
    centers = geo_model.rescaling.df.loc['values', 'centers'].astype('float32')

    # We create pandas groups by id to be able to modify several points at the same time:
    g = geo_model.surface_points.df.groupby('id')
    l = aesara.shared(np.array([], dtype='float64'))

    # %%
    g_obs_p = 1e3 * np.array([-0.3548658 , -0.35558686, -0.3563156 , -0.35558686, -0.3548658 ,
           -0.3534237 , -0.35201198, -0.3534237 , -0.3548658 , -0.3563401 ,
           -0.3548658 , -0.35558686, -0.3548658 , -0.3541554 , -0.3534569 ,
           -0.3527707 , -0.35424498, -0.35575098, -0.3572901 , -0.35575098,
           -0.35424498, -0.35575098, -0.35424498, -0.35575098, -0.35424498,
           -0.35575098, -0.35643718, -0.35713565, -0.35643718], dtype='float32')

    y_obs_list = 1e3 * np.array([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                  2.19, 2.07, 2.16, 2.11, 2.13, 1.92])

    # %%
    # Python input variables
    i = geo_model.interpolator.get_python_input_block()
