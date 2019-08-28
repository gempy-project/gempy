import numpy as np
import pandas as pn
import scipy.stats as stats
from .joyplot import joyplot

from typing import Union

import matplotlib.pyplot as plt
from matplotlib import cm

# Create cmap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
from arviz.plots.jointplot import *
from arviz.plots.jointplot import _var_names, _scale_fig_size
# Seaborn style
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Discrete cmap
pal_disc = sns.cubehelix_palette(10, rot=-.25, light=.7)
my_cmap = ListedColormap(pal_disc)

# Continuous cmap
pal_cont = sns.cubehelix_palette(250, rot=-.25, light=.7)
my_cmap_full = ListedColormap(pal_cont)

default_red = '#DA8886'
default_blue = pal_cont.as_hex()[4]


def create_gempy_colors():
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    my_cmap = ListedColormap(pal)


def plot_joint_pro(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    kind="scatter",
    gridsize="auto",
    contour=True,
    fill_last=True,
    joint_kwargs=None,
    marginal_kwargs=None,
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

    data = convert_to_dataset(data, group="posterior")

    if coords is None:
        coords = {}

    var_names = _var_names(var_names, data)

    plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

    if len(plotters) != 2:
        raise Exception(
            "Number of variables to be plotted must 2 (you supplied {})".format(len(plotters))
        )

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize)

    if joint_kwargs is None:
        joint_kwargs = {}

    if marginal_kwargs is None:
        marginal_kwargs = {}
    marginal_kwargs.setdefault("plot_kwargs", {})
    marginal_kwargs["plot_kwargs"]["linewidth"] = linewidth

    # Instantiate figure and grid
    fig, _ = plt.subplots(0, 0, figsize=figsize, constrained_layout=True)
    grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)

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
    x = plotters[0][2].flatten()
    y = plotters[1][2].flatten()

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

    ax_hist_x.set_xlim(axjoin.get_xlim())
    ax_hist_y.set_ylim(axjoin.get_ylim())

    if True:
        trace = True
        iteration = 1
        if trace is not None:
            if iteration is None:
                iteration = 80
                n_iterations = 20
                i_0 = np.max(0, (iteration - n_iterations))

            plotters = list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))

            theta1_val_trace = plotters[0][2].flatten()[i_0:iteration]
            theta2_val_trace = plotters[1][2].flatten()[i_0:iteration]

            theta1_val = theta1_val_trace[-1]
            theta2_val = theta2_val_trace[-1]

            # Plot point of the given iteration
            plt.plot(theta1_val, theta2_val, 'bo', ms=6, color='k')

            # Plot a trace of n_iterations
            pair_x_array = np.vstack((x[:-1], x[1:])).T  # np.tile(x, (2,1)).T # np.reshape(x, (-1, 2))
            pair_y_array = np.vstack((y[:-1], y[1:])).T
            for i, pair_x in enumerate(pair_x_array):
                alpha_val = i / pair_x_array.shape[0]
                pair_y = pair_y_array[i]
                plt.plot(pair_x, pair_y, '--', linewidth=3, alpha=alpha_val, color='k')

        group = 'both'
        if group == 'both':
            plotters = list(xarray_var_iter(get_coords(data.prior, coords), var_names=var_names, combined=True))
            prior_x = plotters[0][2].flatten()[:iteration]
            prior_y = plotters[1][2].flatten()[:iteration]
            if kind == "scatter":
                axjoin.scatter(prior_x, prior_y, **joint_kwargs)
            elif kind == "kde":
                plot_kde(prior_x, prior_y, contour=contour, fill_last=fill_last, ax=axjoin, **joint_kwargs)
            else:
                if gridsize == "auto":
                    gridsize = int(len(x) ** 0.35)
                axjoin.hexbin(prior_x, prior_y, mincnt=1, gridsize=gridsize, **joint_kwargs)
                axjoin.grid(False)

            for val, ax, rotate in ((prior_x, ax_hist_x, False), (prior_y, ax_hist_y, True)):
                plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax, **marginal_kwargs)

    return axjoin, ax_hist_x, ax_hist_y


def plot_normal_marginal(theta1_loc, theta1_scale, theta2_loc, theta2_scale, trace=None, iteration=None,
                         cmap=my_cmap, fig=None, **kwargs):

    # Prior space
    rock2 = np.linspace(theta2_loc - theta2_scale * 3, theta2_loc + theta2_scale * 3, 500)
    rock1 = np.linspace(theta1_loc - theta1_scale * 3, theta1_loc + theta1_scale * 3, 500)

    # 2D prior space
    X, Y = np.meshgrid(rock2, rock1)

    # We set our normal distribution
    nor_x = stats.norm.pdf(rock2, loc=theta2_loc, scale=theta2_scale)  # Related to estandar deviation: more unknown in beta
    nor_y = stats.norm.pdf(rock1, loc=theta1_loc, scale=theta1_scale)

    # Prior probability
    M = np.dot(nor_x[:, None], nor_y[None, :])

    if fig is None:
        fig = plt.figure()
    subplot = kwargs.get('subplot', 111)
    plt.subplot(subplot)

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

        theta1_val_trace = trace[theta1_name].loc[iteration-500:iteration]
        theta2_val_trace = trace[theta2_name].loc[iteration-500: iteration]

        plt.plot(theta1_val, theta2_val, 'bo', ms=6, color='k')
        plt.plot(theta1_val_trace, theta2_val_trace, color='k', alpha=.5)
        plt.plot(trace[theta1_name].loc[iteration-20:iteration], trace[theta2_name].loc[iteration-20:iteration],
                 color='k', alpha=1)

    plt.xlabel(theta1_name)
    plt.ylabel(theta2_name)
    plt.title("Prior distribution landscape")
    sns.despine(left=True, bottom=True)
    return fig


def plot_normal_likelihood(model_mean: float, model_std: float, obs: Union[list, float], x_range: tuple = None,
                           fig=None, **kwargs):

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
    if fig is None:
        fig = plt.figure()

    subplot = kwargs.get('subplot', 111)
    ax = fig.add_subplot(subplot)
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['left'].set_position(('data', thick_min - thick_min * .1))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    nor_l = stats.norm.pdf(thick_vals, loc=thick_model, scale=thick_std)
    likelihood_at_observation = stats.norm.pdf(observation, loc=thick_model, scale=thick_std)

    y_min = (nor_l.min() - nor_l.max()) * .1
    y_max = nor_l.max() + nor_l.max() * .05

    plt.plot(thick_vals, nor_l, color= '#7eb1bc', linewidth=.5)
    plt.fill_between(thick_vals, nor_l, 0, color=color_fill, alpha=.8)
    plt.vlines(observation, 0, likelihood_at_observation, linestyles='dashdot', color='#DA8886', alpha=1)
    plt.hlines(likelihood_at_observation, observation, thick_min - thick_min * .1,
               linestyles='dashdot', color='#DA8886', alpha= 1)
    plt.scatter(observation, np.zeros_like(observation), s=50, c='#DA8886')

    plt.ylim(y_min, y_max)
    plt.xlim(thick_min, thick_max)
    plt.xlabel('Thickness Obs.')
    plt.ylabel('Likelihood')
    plt.title('Likelihood')
    plt.tight_layout()


def plot_joyplot(trace_mean: pn.Series, trace_std: Union[float, pn.Series], trace_n, obs=None,
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
    fig, axes = joyplot(df, bw_method=1, overlap=2, labels=iteration_label,
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



