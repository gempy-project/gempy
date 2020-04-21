import gempy as gp


def test_plot_data_default(one_fault_model_no_interp):
    gp._plot.plot_2d(one_fault_model_no_interp)


