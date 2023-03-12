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
  
    cax = axes.inset_axes([1.04, 0.2, 0.05, 0.6])
    if cs:
        cbar = axes.figure.colorbar(cs, ax=axes, cax=cax, location="right", **kwargs)
    else:
        if im is not None:
            cbar = axes.figure.colorbar(im, cax=cax, location="right",  **kwargs)
    cbar.set_label(label)
    return cbar