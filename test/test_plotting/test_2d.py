import gempy as gp
import matplotlib.pyplot as plt


def test_plot_2d_data_default(one_fault_model_no_interp):
    gp._plot.plot_2d(one_fault_model_no_interp)
    plt.show()

