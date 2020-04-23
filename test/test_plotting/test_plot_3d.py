
import gempy as gp
import pyvista as pv
import matplotlib.pyplot as plt


def test_plot_3d_data_default(one_fault_model_no_interp):
    gpv = gp.plot.plot_3d(one_fault_model_no_interp,
                           plotter_type='basic', off_screen=True, notebook=False)
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()
