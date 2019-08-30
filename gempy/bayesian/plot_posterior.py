import numpy as np
import pandas as pn
import scipy.stats as stats
from .joyplot import joyplot

from typing import Union
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib.gridspec as gridspect

# Create cmap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
from arviz.plots.jointplot import *
from arviz.plots.jointplot import _var_names, _scale_fig_size
from arviz.stats import hpd
import arviz


# Seaborn style
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Discrete cmap
pal_disc = sns.cubehelix_palette(10, rot=-.25, light=.7)
pal_disc_l = sns.cubehelix_palette(10)
my_cmap = ListedColormap(pal_disc)
my_cmap_l = ListedColormap(pal_disc_l)


# Continuous cmap
pal_cont = sns.cubehelix_palette(250, rot=-.25, light=.7)
pal_cont_l = sns.cubehelix_palette(250)

my_cmap_full = ListedColormap(pal_cont)
my_cmap_full_l = ListedColormap(pal_cont_l)

default_red = '#DA8886'
default_blue = pal_cont.as_hex()[4]
default_l = pal_disc_l.as_hex()[4]


class PlotPosterior:
    def __init__(self, data: arviz.data.inference_data.InferenceData = None):
        self.data = data
        self.iteration = 1

    def _create_joint_axis(self, figure=None, subplot_spec=None, figsize=None, textsize=None):
        figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize)
        # Instantiate figure and grid

        if figure is None:
            fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
        else:
            fig = figure

        if subplot_spec is None:
            grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)
        else:
            grid = gridspect.GridSpecFromSubplotSpec(4, 4, subplot_spec=subplot_spec)

            # Set up main plot
        self.axjoin = fig.add_subplot(grid[1:, :-1])

        # Set up top KDE
        self.ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=self.axjoin)
        self.ax_hist_x.tick_params(labelleft=False, labelbottom=False)

        # Set up right KDE
        self.ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=self.axjoin)
        self.ax_hist_y.tick_params(labelleft=False, labelbottom=False)
        sns.despine(left=True, bottom=True)

        return self.axjoin, self.ax_hist_x, self.ax_hist_y

    def _create_likelihood_axis(self, figure=None, subplot_spec=None, textsize=None, **kwargs):
        # Making the axes:
        if figure is None:
            figsize = kwargs.get('figsize', None)
            fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
        else:
            fig = figure

        if subplot_spec is None:
            grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.1, figure=fig)
        else:
            grid = gridspect.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)

        ax_like = fig.add_subplot(grid[0, 0])
        ax_like.spines['bottom'].set_position(('data', 0.0))
        # ax_like.spines['left'].set_position(('data', -1))
        ax_like.yaxis.tick_right()

        ax_like.spines['right'].set_position(('data', 1.05))
        ax_like.spines['top'].set_color('none')
        ax_like.spines['left'].set_color('none')
        ax_like.set_xlabel('Thickness Obs.')
        # ax_like.set_ylabel('Likelihood')
        ax_like.set_title('Likelihood')
        return ax_like

    def _create_joy_axis(self, figure=None, subplot_spec=None, n_samples=21, overlap=.85):

        grid = gridspect.GridSpecFromSubplotSpec(n_samples, 1, hspace=-overlap, subplot_spec=subplot_spec)
        ax_joy = [figure.add_subplot(grid[i, 0]) for i in range(n_samples)]
        return ax_joy

    def create_figure(self, marginal=True, likelihood=True, joyploy=True,
                      figsize=None, textsize=None,
                      n_samples=21):

        figsize, self.ax_labelsize, _, self.xt_labelsize, self.linewidth, _ = _scale_fig_size(figsize, textsize)
        self.fig, axes = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
        gs_0 = gridspect.GridSpec(2, 2, figure=self.fig, hspace=.2)

        if marginal is True and likelihood is True and joyploy is True:
            # Testing
            self.marginal_axes = self._create_joint_axis(figure=self.fig, subplot_spec=gs_0[0, 0])
          #  self.likelihood_axes = self._create_likelihood_axis(figure=self.fig, subplot_spec=gs_0[0, 1])
           # self.joy = self._create_joy_axis(self.fig, gs_0[1, :])

    def plot_marginal_posterior(self, plotters, iteration=-1, **marginal_kwargs):
        marginal_kwargs.setdefault("plot_kwargs", {})
        marginal_kwargs["plot_kwargs"]["linewidth"] = self.linewidth
        marginal_kwargs.setdefault('fill_kwargs', {})

        marginal_kwargs["plot_kwargs"].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('alpha', .8)

        # Flatten data
        x = plotters[0][2].flatten()[:iteration]
        y = plotters[1][2].flatten()[:iteration]

        for val, ax, rotate in ((x, self.ax_hist_x, False), (y, self.ax_hist_y, True)):
            plot_dist(val, textsize=self.xt_labelsize, rotated=rotate, ax=ax, **marginal_kwargs)

    def plot_joint_posterior(self, plotters, iteration=-1, kind='kde', **joint_kwargs):

        # Set labels for axes
        x_var_name = make_label(plotters[0][0], plotters[0][1])
        y_var_name = make_label(plotters[1][0], plotters[1][1])

        self.axjoin.set_xlabel(x_var_name, fontsize=self.ax_labelsize)
        self.axjoin.set_ylabel(y_var_name, fontsize=self.ax_labelsize)
        self.axjoin.tick_params(labelsize=self.xt_labelsize)

        # Flatten data
        x = plotters[0][2].flatten()[:iteration]
        y = plotters[1][2].flatten()[:iteration]

        if kind == "scatter":
            self.axjoin.scatter(x, y, **joint_kwargs)
        elif kind == "kde":
            contour = joint_kwargs.get('contour', True)
            fill_last = joint_kwargs.get('fill_last', False)

            plot_kde(x, y, contour=contour, fill_last=fill_last, ax=self.axjoin, **joint_kwargs)
        else:
            gridsize = joint_kwargs.get('grid_size', 'auto')
            if gridsize == "auto":
                gridsize = int(len(x) ** 0.35)
            self.axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
            self.axjoin.grid(False)

    def plot_trace(self, plotters, iteration, n_iterations=20):
        if iteration < 3:
            iteration = 3
        i_0 = np.max([0, (iteration - n_iterations)])

        theta1_val_trace = plotters[0][2].flatten()[i_0:iteration]
        theta2_val_trace = plotters[1][2].flatten()[i_0:iteration]

        theta1_val = theta1_val_trace[-1]
        theta2_val = theta2_val_trace[-1]

        # Plot point of the given iteration
        self.axjoin.plot(theta1_val, theta2_val, 'bo', ms=6, color='k')

        # Plot a trace of n_iterations
        pair_x_array = np.vstack(
            (theta1_val_trace[:-1], theta1_val_trace[1:])).T  # np.tile(x, (2,1)).T # np.reshape(x, (-1, 2))
        pair_y_array = np.vstack((theta2_val_trace[:-1], theta2_val_trace[1:])).T
        for i, pair_x in enumerate(pair_x_array):
            alpha_val = i / pair_x_array.shape[0]
            pair_y = pair_y_array[i]
            self.axjoin.plot(pair_x, pair_y, linewidth=1, alpha=alpha_val, color='k')

    def plot_marginal(self, var_names=None, data=None, iteration=-1,
                      group='both',
                      plot_trace=True, n_iterations=20,
                      kind='kde',
                      coords=None, credible_interval=.98,
                      marginal_kwargs=None, marginal_kwargs_prior=None,
                      joint_kwargs=None, joint_kwargs_prior=None):
        self.axjoin.clear()
        self.ax_hist_x.clear()
        self.ax_hist_y.clear()

        if data is None:
            data = self.data

        valid_kinds = ["scatter", "kde", "hexbin"]
        if kind not in valid_kinds:
            raise ValueError(
                ("Plot type {} not recognized." "Plot type must be in {}").format(kind, valid_kinds)
            )

        if coords is None:
            coords = {}

        if joint_kwargs is None:
            joint_kwargs = {}

        if marginal_kwargs is None:
            marginal_kwargs = {}

        data_0 = convert_to_dataset(data, group="posterior")
        var_names = _var_names(var_names, data_0)

        plotters = list(xarray_var_iter(get_coords(data_0, coords), var_names=var_names, combined=True))

        if len(plotters) != 2:
            raise Exception(
                "Number of variables to be plotted must 2 (you supplied {})".format(len(plotters))
            )

        if kind == 'kde':
            joint_kwargs.setdefault('contourf_kwargs', {})
            joint_kwargs.setdefault('contour_kwargs', {})
            joint_kwargs['contourf_kwargs'].setdefault('cmap', my_cmap_l)
            joint_kwargs['contourf_kwargs'].setdefault('levels', 11)
            joint_kwargs['contourf_kwargs'].setdefault('alpha', .8)
            joint_kwargs['contour_kwargs'].setdefault('alpha', 0)

        marginal_kwargs.setdefault('fill_kwargs', {})
        marginal_kwargs.setdefault("plot_kwargs", {})
        marginal_kwargs["plot_kwargs"]["linewidth"] = self.linewidth

        marginal_kwargs["plot_kwargs"].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('alpha', .8)

        if group == 'both' or group == 'posterior':

            self.plot_joint_posterior(plotters, kind=kind, **joint_kwargs)
            self.plot_marginal_posterior(plotters, iteration=iteration, **marginal_kwargs)

        plot_prior = True if group == 'both' or group == 'prior' else False
        if plot_prior is True:
            if joint_kwargs_prior is None:
                joint_kwargs_prior = {}

            if marginal_kwargs_prior is None:
                marginal_kwargs_prior = {}

            joint_kwargs_prior.setdefault('contourf_kwargs', {})
            marginal_kwargs_prior.setdefault('fill_kwargs', {})
            marginal_kwargs_prior.setdefault("plot_kwargs", {})
            marginal_kwargs_prior["plot_kwargs"]["linewidth"] = self.linewidth

            if kind == 'kde':
                joint_kwargs_prior.setdefault('contourf_kwargs', {})
                joint_kwargs_prior.setdefault('contour_kwargs', {})
                joint_kwargs_prior['contourf_kwargs'].setdefault('cmap', my_cmap)
                joint_kwargs_prior['contourf_kwargs'].setdefault('levels', 11)
                alpha_p = .8 if group == 'prior' else .4
                joint_kwargs_prior['contourf_kwargs'].setdefault('alpha', alpha_p)
                joint_kwargs_prior['contour_kwargs'].setdefault('alpha', 0)

            marginal_kwargs_prior["plot_kwargs"].setdefault('color', default_blue)
            marginal_kwargs_prior['fill_kwargs'].setdefault('color', default_blue)
            marginal_kwargs_prior['fill_kwargs'].setdefault('alpha', .8)

            data_1 = convert_to_dataset(data, group="prior")
            plotters_prior = list(xarray_var_iter(get_coords(data_1, coords), var_names=var_names, combined=True))
            prior_x = plotters_prior[0][2].flatten()
            prior_y = plotters_prior[1][2].flatten()

            self.plot_joint_posterior(plotters_prior, kind=kind, **joint_kwargs_prior)
            self.plot_marginal_posterior(plotters_prior, **marginal_kwargs_prior)

            x_min, x_max = hpd(prior_x, credible_interval=credible_interval)
            y_min, y_max = hpd(prior_y, credible_interval=credible_interval)

        else:
            x = plotters[0][2].flatten()[:iteration]
            y = plotters[1][2].flatten()[:iteration]
            x_min, x_max = hpd(x, credible_interval=credible_interval)
            y_min, y_max = hpd(y, credible_interval=credible_interval)

        if plot_trace is True:
            self.plot_trace(plotters, iteration, n_iterations)

        self.axjoin.set_xlim(x_min, x_max)
        self.axjoin.set_ylim(y_min, y_max)
        self.ax_hist_x.set_xlim(self.axjoin.get_xlim())
        self.ax_hist_y.set_ylim(self.axjoin.get_ylim())

        return self.axjoin, self.ax_hist_x, self.ax_hist_y

def create_gempy_colors():
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    my_cmap = ListedColormap(pal)


def plot_joint_pro(
    data,
    var_names=None,
    group='posterior',
    iteration=None,
    plot_trace=False,
    credible_interval=.98,
    coords=None,
    figure=None,
    subplot_spec=None,
    figsize=None,
    textsize=None,
    kind="kde",
    gridsize="auto",
    contour=True,
    fill_last=True,
    joint_kwargs=None,
    marginal_kwargs=None,
    **kwargs
):
    """
    Plot a scatter or hexbin of two variables with their respective marginals distributions.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : Iter of 2 e.g. (var_1, var_2)
        Variables to be plotted, two variables are required.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    kind : str
        Type of plot to display (scatter, kde or hexbin)
    gridsize : int or (int, int), optional.
        The number of hexagons in the x-direction. Ignored when hexbin is False. See `plt.hexbin`
        for details
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    joint_kwargs : dicts, optional
        Additional keywords modifying the join distribution (central subplot)
    marginal_kwargs : dicts, optional
        Additional keywords modifying the marginals distributions (top and right subplot)

    Returns
    -------
    axjoin : matplotlib axes, join (central) distribution
    ax_hist_x : matplotlib axes, x (top) distribution
    ax_hist_y : matplotlib axes, y (right) distribution

    Examples
    --------
    Scatter Joint plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> az.plot_joint(data,
        >>>             var_names=['theta'],
        >>>             coords={'school': ['Choate', 'Phillips Andover']},
        >>>             kind='scatter',
        >>>             figsize=(6, 6))

    Hexbin Joint plot

    .. plot::
        :context: close-figs

        >>> az.plot_joint(data,
        >>>             var_names=['theta'],
        >>>             coords={'school': ['Choate', 'Phillips Andover']},
        >>>             kind='hexbin',
        >>>             figsize=(6, 6))

    KDE Joint plot

    .. plot::
        :context: close-figs

        >>> az.plot_joint(data,
        >>>                 var_names=['theta'],
        >>>                 coords={'school': ['Choate', 'Phillips Andover']},
        >>>                 kind='kde',
        >>>                 figsize=(6, 6))

    """
    # TODO check if data is posterior or prior


    valid_kinds = ["scatter", "kde", "hexbin"]
    if kind not in valid_kinds:
        raise ValueError(
            ("Plot type {} not recognized." "Plot type must be in {}").format(kind, valid_kinds)
        )

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize)

    if joint_kwargs is None:
        joint_kwargs = {}

    if marginal_kwargs is None:
        marginal_kwargs = {}

    marginal_kwargs.setdefault("plot_kwargs", {})
    marginal_kwargs["plot_kwargs"]["linewidth"] = linewidth
    if kind =='kde':
        joint_kwargs.setdefault('contourf_kwargs', {})
        joint_kwargs.setdefault('contour_kwargs', {})

    marginal_kwargs.setdefault('fill_kwargs', {})

    joint_kwargs2 = copy.deepcopy(joint_kwargs)
    marginal_kwargs2 = copy.deepcopy(marginal_kwargs)

    if iteration is None:
        iteration = -1
    elif iteration < 3:
        print('Minimum iteration must be 3.')
        iteration = 3

    if group == 'both':
        data_prior = convert_to_dataset(data, group="prior")
        data = convert_to_dataset(data, group="posterior")
        # For posterior
        if kind == 'kde':

            joint_kwargs['contourf_kwargs'].setdefault('cmap', my_cmap_l)
            joint_kwargs['contourf_kwargs'].setdefault('levels', 11)
            joint_kwargs['contourf_kwargs'].setdefault('alpha', .8)
            joint_kwargs['contour_kwargs'].setdefault('alpha', 0)

        marginal_kwargs["plot_kwargs"].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('alpha', .8)

        # For prior
        joint_kwargs2.setdefault('contourf_kwargs', {})
        marginal_kwargs2.setdefault('fill_kwargs', {})

        joint_kwargs2['contourf_kwargs'].setdefault('cmap', my_cmap)
        joint_kwargs2['contourf_kwargs'].setdefault('levels', 11)
        joint_kwargs2['contourf_kwargs'].setdefault('alpha', .2)
        joint_kwargs2['contour_kwargs'].setdefault('alpha', 0)

        marginal_kwargs2["plot_kwargs"].setdefault('color', default_blue)
        marginal_kwargs2['fill_kwargs'].setdefault('color', default_blue)
        marginal_kwargs2['fill_kwargs'].setdefault('alpha', .4)

    elif group == 'posterior':
        data = convert_to_dataset(data, group="posterior")
        if kind == 'kde':
            joint_kwargs['contourf_kwargs'].setdefault('cmap', my_cmap_l)
            joint_kwargs['contourf_kwargs'].setdefault('levels', 11)
            joint_kwargs['contourf_kwargs'].setdefault('alpha', .8)
            joint_kwargs['contour_kwargs'].setdefault('alpha', 0)

        marginal_kwargs["plot_kwargs"].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('color', default_l)
        marginal_kwargs['fill_kwargs'].setdefault('alpha', .8)

    elif group == 'prior':
        data = convert_to_dataset(data.prior, group='prior')
        joint_kwargs.setdefault('contourf_kwargs', {})
        marginal_kwargs.setdefault('fill_kwargs', {})

        joint_kwargs['contourf_kwargs'].setdefault('cmap', my_cmap)
        joint_kwargs['contourf_kwargs'].setdefault('levels', 11)
        joint_kwargs['contourf_kwargs'].setdefault('alpha', .8)
        joint_kwargs['contour_kwargs'].setdefault('alpha', 0)


        marginal_kwargs["plot_kwargs"].setdefault('color', default_blue)
        marginal_kwargs['fill_kwargs'].setdefault('color', default_blue)
        marginal_kwargs['fill_kwargs'].setdefault('alpha', .8)

    if coords is None:
        coords = {}

    var_names = _var_names(var_names, data)

    plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

    if len(plotters) != 2:
        raise Exception(
            "Number of variables to be plotted must 2 (you supplied {})".format(len(plotters))
        )

    # Instantiate figure and grid

    if figure is None:
        fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
    else:
        fig = figure

    if subplot_spec is None:
        grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)
    else:
        grid = gridspect.GridSpecFromSubplotSpec(4, 4, subplot_spec=subplot_spec)

    # Set up main plot
    axjoin = fig.add_subplot(grid[1:, :-1])

    # Set up top KDE
    ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=axjoin)
    ax_hist_x.tick_params(labelleft=False, labelbottom=False)

    # Set up right KDE
    ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=axjoin)
    ax_hist_y.tick_params(labelleft=False, labelbottom=False)

    # Set labels for axes
    x_var_name = make_label(plotters[0][0], plotters[0][1])
    y_var_name = make_label(plotters[1][0], plotters[1][1])

    axjoin.set_xlabel(x_var_name, fontsize=ax_labelsize)
    axjoin.set_ylabel(y_var_name, fontsize=ax_labelsize)
    axjoin.tick_params(labelsize=xt_labelsize)

    # Flatten data
    x = plotters[0][2].flatten()[:iteration]
    y = plotters[1][2].flatten()[:iteration]

    if kind == "scatter":
        axjoin.scatter(x, y, **joint_kwargs)
    elif kind == "kde":
        plot_kde(x, y, contour=contour, fill_last=fill_last, ax=axjoin, **joint_kwargs)
    else:
        if gridsize == "auto":
            gridsize = int(len(x) ** 0.35)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)

    for val, ax, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
        plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax, **marginal_kwargs)

    if plot_trace is True:
        n_iterations = kwargs.get('n_iterations', 20)
        i_0 = np.max([0, (iteration - n_iterations)])

        plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

        theta1_val_trace = plotters[0][2].flatten()[i_0:iteration]
        theta2_val_trace = plotters[1][2].flatten()[i_0:iteration]

        theta1_val = theta1_val_trace[-1]
        theta2_val = theta2_val_trace[-1]

        # Plot point of the given iteration
        axjoin.plot(theta1_val, theta2_val, 'bo', ms=6, color='k')

        # Plot a trace of n_iterations
        pair_x_array = np.vstack((theta1_val_trace[:-1], theta1_val_trace[1:])).T  # np.tile(x, (2,1)).T # np.reshape(x, (-1, 2))
        pair_y_array = np.vstack((theta2_val_trace[:-1], theta2_val_trace[1:])).T
        for i, pair_x in enumerate(pair_x_array):
            alpha_val = i / pair_x_array.shape[0]
            pair_y = pair_y_array[i]
            axjoin.plot(pair_x, pair_y, linewidth=1, alpha=alpha_val, color='k')

    if group == 'both':
        plotters = list(xarray_var_iter(get_coords(data_prior, coords), var_names=var_names, combined=True))
        prior_x = plotters[0][2].flatten()
        prior_y = plotters[1][2].flatten()

        x_min, x_max = hpd(prior_x, credible_interval=credible_interval)
        y_min, y_max = hpd(prior_y, credible_interval=credible_interval)

        if kind == "scatter":
            axjoin.scatter(prior_x, prior_y, **joint_kwargs)
        elif kind == "kde":
            plot_kde(prior_x, prior_y, contour=contour, fill_last=fill_last, ax=axjoin, **joint_kwargs2)
        else:
            if gridsize == "auto":
                gridsize = int(len(x) ** 0.35)
            axjoin.hexbin(prior_x, prior_y, mincnt=1, gridsize=gridsize, **joint_kwargs2)
            axjoin.grid(False)

        for val, ax, rotate in ((prior_x, ax_hist_x, False), (prior_y, ax_hist_y, True)):
            plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax, **marginal_kwargs2)
    else:
        x_min, x_max = hpd(x, credible_interval=credible_interval)
        y_min, y_max = hpd(y, credible_interval=credible_interval)
        print(x_min, x_max, y_min, y_max)
        
    axjoin.set_xlim(x_min, x_max)
    axjoin.set_ylim(y_min, y_max)
    ax_hist_x.set_xlim(axjoin.get_xlim())
    ax_hist_y.set_ylim(axjoin.get_ylim())
    sns.despine(left=True, bottom=True)

    return axjoin, ax_hist_x, ax_hist_y


def plot_normal_likelihood_pro(data, var_mean, var_std, x_range: tuple = None,
                               figure=None, subplot_spec=None, **kwargs):



    color_fill = kwargs.get('color_fill', pal_disc.as_hex()[4])
    # Plotting likelihood
    if x_range is not None:
        thick_min = x_range[0]
        thick_max = x_range[1]
    else:
        thick_max = model_mean + 3 * model_std
        thick_min = model_mean - 3 * model_std

    thick_vals = np.linspace(thick_min, thick_max, 100)
    observation = np.asarray(obs)

    thick_model = model_mean
    thick_std = model_std

    # Making the axes:
    if figure is None:
        figsize = kwargs.get('figsize', None)
        fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
    else:
        fig = figure

    if subplot_spec is None:
        grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.1, figure=fig)
    else:
        grid = gridspect.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)

    ax_like = fig.add_subplot(grid)
    ax_like.spines['bottom'].set_position(('data', 0.0))
    ax_like.spines['left'].set_position(('data', thick_min - thick_min * .1))
    ax_like.spines['right'].set_color('none')
    ax_like.spines['top'].set_color('none')

    nor_l = stats.norm.pdf(thick_vals, loc=thick_model, scale=thick_std)
    likelihood_at_observation = stats.norm.pdf(observation, loc=thick_model, scale=thick_std)

    y_min = (nor_l.min() - nor_l.max()) * .1
    y_max = nor_l.max() + nor_l.max() * .05

    ax_like.plot(thick_vals, nor_l, color= '#7eb1bc', linewidth=.5)
    ax_like.fill_between(thick_vals, nor_l, 0, color=color_fill, alpha=.8)
    ax_like.axvlines(observation, 0, likelihood_at_observation, linestyles='dashdot', color='#DA8886', alpha=1)
    ax_like.axhlines(likelihood_at_observation, observation, thick_min - thick_min * .1,
               linestyles='dashdot', color='#DA8886', alpha= 1)
    ax_like.scatter(observation, np.zeros_like(observation), s=50, c='#DA8886')

    ax_like.set_ylim(y_min, y_max)
    ax_like.set_xlim(thick_min, thick_max)
    ax_like.set_xlabel('Thickness Obs.')
    ax_like.set_ylabel('Likelihood')
    ax_like.title('Likelihood')
    ax_like.tight_layout()
    return ax_like

def plot_normal_likelihood(model_mean: float, model_std: float, obs: Union[list, float], x_range: tuple = None,
                           figure=None, subplot_spec=None, **kwargs):

    color_fill = kwargs.get('color_fill', pal_disc.as_hex()[4])
    # Plotting likelihood
    if x_range is not None:
        thick_min = x_range[0]
        thick_max = x_range[1]
    else:
        thick_max = model_mean + 3 * model_std
        thick_min = model_mean - 3 * model_std

    thick_vals = np.linspace(thick_min, thick_max, 100)
    observation = np.asarray(obs)

    thick_model = model_mean
    thick_std = model_std

    # Making the axes:
    if figure is None:
        figsize = kwargs.get('figsize', None)
        fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
    else:
        fig = figure

    if subplot_spec is None:
        grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.1, figure=fig)
    else:
        grid = gridspect.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)

    ax_like = fig.add_subplot(grid)
    ax_like.spines['bottom'].set_position(('data', 0.0))
    ax_like.spines['left'].set_position(('data', thick_min - thick_min * .1))
    ax_like.spines['right'].set_color('none')
    ax_like.spines['top'].set_color('none')

    nor_l = stats.norm.pdf(thick_vals, loc=thick_model, scale=thick_std)
    likelihood_at_observation = stats.norm.pdf(observation, loc=thick_model, scale=thick_std)

    y_min = (nor_l.min() - nor_l.max()) * .1
    y_max = nor_l.max() + nor_l.max() * .05

    ax_like.plot(thick_vals, nor_l, color= '#7eb1bc', linewidth=.5)
    ax_like.fill_between(thick_vals, nor_l, 0, color=color_fill, alpha=.8)
    ax_like.axvlines(observation, 0, likelihood_at_observation, linestyles='dashdot', color='#DA8886', alpha=1)
    ax_like.axhlines(likelihood_at_observation, observation, thick_min - thick_min * .1,
               linestyles='dashdot', color='#DA8886', alpha= 1)
    ax_like.scatter(observation, np.zeros_like(observation), s=50, c='#DA8886')

    ax_like.set_ylim(y_min, y_max)
    ax_like.set_xlim(thick_min, thick_max)
    ax_like.set_xlabel('Thickness Obs.')
    ax_like.set_ylabel('Likelihood')
    ax_like.title('Likelihood')
    ax_like.tight_layout()
    return ax_like


def plot_joyplot(trace_mean: pn.Series, trace_std: Union[float, pn.Series], trace_n, obs=None, ax=None,
                 cmap=my_cmap_full, **kwargs):

    hex_c = kwargs.get('hex_c', None)
    n_traces = kwargs.get('n_traces', 21)
    thinning = kwargs.get('thinning', 1)* -1
    samples_size = kwargs.get('samples_size', 100)

    l_1 = trace_n - np.round(n_traces / 2)
    l_0 = trace_n + np.round(n_traces / 2)

    df = pn.DataFrame()

    if isinstance(trace_std, pn.Series):
        iter_vals = pn.DataFrame([trace_mean, trace_std]).loc[l_0:l_1:thinning]
        for e, i in iter_vals.iteritems():
            num = np.random.normal(loc=i.iloc[0], scale=i.iloc[1], size=samples_size)
            name = e
            df[name] = num
    else:
        iter_vals = trace_mean.loc[l_0:l_1:thinning]
        for e, i in iter_vals.iteritems():
            num = np.random.normal(loc=i, scale=trace_std, size=samples_size)
            name = e
            df[name] = num


    pal = sns.cubehelix_palette(250, rot=-.25, light=.7)
    my_cmap_full = ListedColormap(pal)
    # Likilihood for color
    if obs is not None:
        if hex_c is None and cmap is not None:
            like = stats.norm.pdf(120, loc=trace_mean.loc[l_0:l_1:thinning], scale=20)
            cNorm = colors.Normalize(like.min(), like.max())
            # Continuous cmap
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            hex_c = [colors.to_hex(i) for i in scalarMap.to_rgba(like)]
    else:
        hex_c = pal.as_hex()[4]


    iteration_label = [int(y) if int(y) % 10 == 0 else None for y in df.columns]
    fig, axes = joyplot(df, bw_method=1, overlap=2, labels=iteration_label, ax=ax,
                              yrot=0,# ylabels=False,
                            #  xlabels='Thickness Obs',
                              title='Likelihood inference',
                              #range_style='own',
                              ylabels='foo',
                              color=hex_c,
                              grid='both',
                              fade=False,
                              linewidth=.1, alpha=1);
    n_axes = len(axes[:-1])
    print(int(n_axes / 2) > trace_n)
    if int(n_axes/2) > trace_n:

        axes[-trace_n-2].axhline(0, 0, 100, c='#DA8886', linewidth=3)

    else:
        print(int(n_axes/2) > trace_n)
        axes[int(n_axes/2)].axhline(0, 0, 100, c='#DA8886', linewidth=3)
      #  axes[int(n_axes / 2)].scatter(0, 0, 100, c='#DA8886', linewidth=3)

    if obs is not None:
        axes[-1].scatter(obs, np.zeros_like(obs), marker='v', s=100, c='#DA8886')
        axes[-2].scatter(120, np.zeros_like(obs)-.002, marker='^', s=300, c='#DA8886')
        axes[-1].axvline(obs, 0, 100, c='#DA8886', linestyle='-.')

 #   for a in axes:
 #       a.set_xlim([0, 250])
 #       axes[0].axvline(100, 0, 100, c='#DA8886')
#        plt.vlines(obs, -50000, 5, linewidth=5, linestyles='solid', color='#DA8886', alpha=.5)
    plt.xlabel('Thickness Obs')

    plt.show()
    return fig, axes


def plot_posterior(trace, theta1_loc, theta1_scale, theta2_loc, theta2_scale, iteration,
                  model_mean_name: str, model_std: float, obs: Union[list, float], x_range: tuple = None,
                   **kwargs):

    fig = plot_normal_marginal(theta1_loc, theta1_scale, theta2_loc, theta2_scale, cmap=my_cmap,
                               trace=trace, iteration=iteration,
                               fig=None,
                               subplot=121, **kwargs)

    model_mean = trace[model_mean_name].loc[iteration]

    if type(model_std) is float or type(model_std) is int:
        model_std = model_std
        trace_std = model_std
    elif type(model_std) is str:
        trace_std = trace[model_std]
        model_std = trace_std.loc[iteration]
    else:
        raise TypeError
    # ---------------------
    # This operations are for getting the same color in the likelihood plot as in the joy plot
    n_traces = kwargs.get('n_traces', 51)
    thinning = kwargs.get('thinning', 1) * -1
    trace_n = iteration

    l_1 = trace_n - np.round(n_traces / 2)
    l_0 = trace_n + np.round(n_traces / 2)

    like_range = stats.norm.pdf(obs, loc=trace[model_mean_name].loc[l_0:l_1:thinning], scale=20)
    like_point = stats.norm.pdf(obs, loc=trace[model_mean_name].loc[trace_n], scale=20)

    cNorm = colors.Normalize(like_range.min(), like_range.max())
    # Continuous cmap
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap_full)
    hex_c = [colors.to_hex(i) for i in scalarMap.to_rgba(like_range)]
    hex_point = [colors.to_hex(i) for i in scalarMap.to_rgba(np.atleast_1d(like_point))]

    #hex_val = np.min([int(len(hex_c)/2+1), iteration])
    #mid_hex = hex_c[-hex_val-1]
    # ---------------------

    plot_normal_likelihood(model_mean, model_std, obs, x_range, fig=fig, subplot=122, color_fill=hex_point)
    plot_joyplot(trace[model_mean_name], trace_std, iteration, obs=obs, cmap=my_cmap_full,
                 hex_c=hex_c, **kwargs)
    plt.tight_layout()


def plot_normal_marginal(theta1_loc, theta1_scale, theta2_loc, theta2_scale, trace=None, iteration=None,
                         cmap=my_cmap, figure=None, **kwargs):
    # Prior space
    rock2 = np.linspace(theta2_loc - theta2_scale * 3, theta2_loc + theta2_scale * 3, 500)
    rock1 = np.linspace(theta1_loc - theta1_scale * 3, theta1_loc + theta1_scale * 3, 500)

    # 2D prior space
    X, Y = np.meshgrid(rock2, rock1)

    # We set our normal distribution
    nor_x = stats.norm.pdf(rock2, loc=theta2_loc,
                           scale=theta2_scale)  # Related to estandar deviation: more unknown in beta
    nor_y = stats.norm.pdf(rock1, loc=theta1_loc, scale=theta1_scale)

    # Prior probability
    M = np.dot(nor_x[:, None], nor_y[None, :])

    if figure is None:

        figsize = kwargs.get('figsize', None)
        fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
    else:
        fig = figure

    # subplot = kwargs.get('subplot', 111)
    # plt.subplot(subplot)

    #    fig.add_subplot(subplot)
    M_ma = np.ma.masked_less_equal(M, 1e-6)

    im = plt.imshow(M_ma, interpolation='gaussian',
                    origin='lower', alpha=0.6,
                    cmap=cmap,
                    extent=(theta2_loc - theta2_scale * 3, theta2_loc + theta2_scale * 3,
                            theta1_loc - theta1_scale * 3, theta1_loc + theta1_scale * 3)
                    )

    theta1_name = kwargs.get('theta1_name', r"$\theta_1$")
    theta2_name = kwargs.get('theta2_name', r"$\theta_2$")
    print(kwargs)
    if trace is not None:
        if iteration is None:
            iteration = trace.index[-1]

        theta1_val = trace[theta1_name].loc[iteration]
        theta2_val = trace[theta2_name].loc[iteration]

        theta1_val_trace = trace[theta1_name].loc[iteration - 500:iteration]
        theta2_val_trace = trace[theta2_name].loc[iteration - 500: iteration]

        plt.plot(theta1_val, theta2_val, 'bo', ms=6, color='k')
        plt.plot(theta1_val_trace, theta2_val_trace, color='k', alpha=.5)
        plt.plot(trace[theta1_name].loc[iteration - 20:iteration], trace[theta2_name].loc[iteration - 20:iteration],
                 color='k', alpha=1)

    plt.xlabel(theta1_name)
    plt.ylabel(theta2_name)
    plt.title("Prior distribution landscape")
    sns.despine(left=True, bottom=True)
    return fig


