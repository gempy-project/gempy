import numpy as np
import pandas as pn
import scipy.stats as stats
import joypy

from typing import Union

import matplotlib.pyplot as plt
from matplotlib import cm

# Create cmap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns

# Seaborn style
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Discrete cmap
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
my_cmap = ListedColormap(pal)

# Continuous cmap
pal = sns.cubehelix_palette(250, rot=-.25, light=.7)
my_cmap_full = ListedColormap(pal)


def create_gempy_colors():
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    my_cmap = ListedColormap(pal)


def plot_marginal(theta1_loc, theta1_scale, theta2_loc, theta2_scale, trace=None, iteration=None,
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


    color_fill = kwargs.get('color_fill', pal.as_hex()[4])
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
    n_traces = kwargs.get('n_traces', 51)
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

    iteration = [int(y) if int(y) % 10 == 0 else None for y in df.columns]
    fig, axes = joypy.joyplot(df, bw_method=1, overlap=2, labels=iteration,  # ylabels=False,
                              title='Likelihood inference',
                              range_style='own',
                              color=hex_c,
                              grid='both',
                              fade=False,
                              linewidth=.1, alpha=1);
    n_axes = len(axes[:-1])
    axes[int(n_axes/2)].axhline(0, 0, 100)

    if obs is not None:
        axes[-2].scatter(obs, np.zeros_like(obs), s=500, c='#DA8886')
        plt.vlines(obs, -50000, 5, linewidth=5, linestyles='solid', color='#DA8886', alpha=.5)

    plt.show()


def plot_posterior(trace, theta1_loc, theta1_scale, theta2_loc, theta2_scale, iteration,
                  model_mean_name: str, model_std: float, obs: Union[list, float], x_range: tuple = None,
                   **kwargs):

    fig = plot_marginal(theta1_loc, theta1_scale, theta2_loc, theta2_scale, cmap=my_cmap,
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
    like = stats.norm.pdf(120, loc=trace[model_mean_name].loc[l_0:l_1:thinning], scale=20)
    cNorm = colors.Normalize(like.min(), like.max())
    # Continuous cmap
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap_full)
    hex_c = [colors.to_hex(i) for i in scalarMap.to_rgba(like)]
    mid_hex = hex_c[int(len(hex_c)/2)]
    # ---------------------

    plot_normal_likelihood(model_mean, model_std, obs, x_range, fig=fig, subplot=122, color_fill=mid_hex)
    plot_joyplot(trace[model_mean_name], trace_std, iteration, obs=obs, cmap=my_cmap_full, hex_c=hex_c, **kwargs)
    plt.tight_layout()



