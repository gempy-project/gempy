from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt


def add_colorbar(im=None, axes=None, cs=None, label = None, aspect=30, location="right", pad_fraction=1, **kwargs):
    """
    Add a colorbar to a plot (im).
    Args:
        im:             plt imshow
        label:          label of the colorbar
        axes:
        cs:             Contourset
        aspect:         the higher, the smaller the colorbar is
        pad_fraction:
        **kwargs:

    Returns: A perfect colorbar, no matter the plot.

    """
    if axes is None:
        axes = im.axes
    divider = axes_grid1.make_axes_locatable(axes)
    width = axes_grid1.axes_size.AxesY(axes, aspect=2. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes(location, size=width, pad=pad)
    plt.sca(current_ax)
    if cs:
        cbar = axes.figure.colorbar(cs, cax=cax, **kwargs)
    else:
        if im is not None:
            cbar = axes.figure.colorbar(im, cax=cax, **kwargs)
    cbar.set_label(label)
    return cbar