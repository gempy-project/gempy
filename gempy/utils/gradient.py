import numpy as np
import matplotlib.pyplot as plt
import gempy as gp


def plot_sig(n_surface_0, n_surface_1, a, b, drift,
             l=50, Z_x=None, sf_max=None, sf_min=None, sf_at_scalar=None, relu=None, **kwargs):
    """
    Plot the sigmoid function used by gempy to discretize space

    Args:
        n_surface_0 (list, int): Activation value
        n_surface_1 (list, int): Deactivation value
        a (list, float): value of the scalar field to trigger the activation
        b (list, float): value of the scalar field to trigger the deactivation
        drift (list, float): drift of the function
        l (float): sigmoid slope
        Z_x (ndarray): scalar field vector
        sf_max (Optional[float]): Maximum scalar field of the model. This represent the boundaries
        sf_min (Optional[float]): Minimum scalar field of the model. This represent the boundaries
        sf_at_scalar (Optional[ndarray]): Values where the layers are found

    Keyword Args:
        colors: List with the colors for the layers

    Returns:

    """
    if 'colors' not in kwargs:
        colors = ['#015482','#9f0052','#ffbe00','#728f02','#443988','#ff3f20','#325916','#5DA629']

    if Z_x is None:
        Z_x = np.linspace(-3, 3, 2000)
    f_x_s = np.zeros_like(Z_x)
    relu_up = np.copy(Z_x)
    relu_up -= b[0]
    relu_up[relu_up < 0] = 0
    relu_up = relu_up * -.01

    # print(relu_up)
    relu_down = np.copy(Z_x)
    relu_down -= a[-1]
    relu_down[relu_down > 0] = 0
    relu_down = relu_down * -.01

    relu = relu_up + relu_down

    if len(n_surface_0) == 1:

        f_x = -n_surface_0 / (1 + np.exp(-l * (Z_x - a))) - \
              (n_surface_1 / (1 + np.exp(l * (Z_x - b)))) + drift
        f_x += relu

        plt.plot(f_x, Z_x)

    else:
        len_ = len(n_surface_0)
        fig = plt.figure(figsize=(7, 12))
        for e in range(len_):
            f_x = - n_surface_0[e] / (1 + np.exp(-l * (Z_x - a[e]))) - \
                  (n_surface_1[e] / (1 + np.exp(l * (Z_x - b[e])))) + drift[e]
            f_x_s += f_x + relu
            # fig.add_subplot(len_, 1, e+1)
            plt.plot(f_x, Z_x, '--', label='Layer ' + str(drift[e]))

        if sf_max is not None:
            plt.hlines(sf_max, 0, f_x_s.max(), label='Model Extent')
        if sf_min is not None:
            plt.hlines(sf_min, 0, f_x_s.max())
        if sf_at_scalar is not None:
            plt.hlines(sf_at_scalar, 0, f_x_s.max(), linewidth=3,
                       color=colors[:len(sf_at_scalar)], label='Scalar value interfaces')

        plt.plot(f_x_s, Z_x, linewidth=5, alpha=.7, label='Actual exporty')
        plt.ylabel('Scalar field')
        plt.xlabel('Lith block')
        #  plt.gca().invert_yaxis()

        plt.legend(bbox_to_anchor=(1.8, 1))

    return plt.gcf()






