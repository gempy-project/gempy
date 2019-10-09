"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


@author: Fabian A. Stamm
"""

import numpy as np
import scipy.optimize as sop
from matplotlib import pyplot as plt

import copy
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib.gridspec as gridspect

# Create cmap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import pandas as pn
import scipy.stats as stats
import seaborn as sns

def loss_abs(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Absolute-error loss function.

        Args:
            estimate_s (int or float): Value estimate.
            true_s (int, float or np.array): True value.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int or float, optional): Risk-affinity re-weighting factor.
        Returns:
            Loss incurred for an estimate given a true value.
            Based on absolute-error loss, i.e. the absolute distance of the estimate
            from the true value.

            true_s can be either a single value or an array of possible true values,
            the output will be a single determined absolute loss value or an array of determined
            loss values, accordingly.

            estimate_s has to be one single value.

    """
    true_s = np.array(true_s).astype(float)
    loss_s = np.zeros_like(true_s)
    underest = (estimate_s < true_s)
    overest = (estimate_s > true_s)
    loss_s[underest] = (true_s[underest] - estimate_s)  * (u * (r ** -0.5))
    loss_s[overest] = (estimate_s - true_s[overest]) * (o * r)
    if u_f != 1:
        underest_fatal = (estimate_s <= 0) & (true_s > 0)
        loss_s[underest_fatal] = (true_s[underest_fatal] - estimate_s) * (u_f * (r ** -0.5))
    if o_f != 1:
        overest_fatal = (estimate_s > 0) & (true_s <= 0)
        loss_s[overest_fatal] = np.abs((true_s[overest_fatal] - estimate_s)) * (o_f * r)
    return loss_s

def loss_abs_given_values(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Absolute-error loss function for the exclusive use with a single
    given true value.

            Args:
                estimate_s (int or float): Value estimate.
                true_s (int, float): True value.
                u (int or float, optional): Underestimation re-weighting factor.
                o (int or float, optional): Overestimation re-weighting factor.
                u_f (int or float, optional): Fatal underestimation re-weighting factor.
                o_f (int or float, optional): Fatal overestimation re-weighting factor.
                r (int or float, optional): Risk-affinity re-weighting factor.
            Returns:
                Loss incurred for an estimate given one determined true value.
                Based on absolute-error loss, i.e. the absolute distance of the estimate
                from the true value.

        """
    if estimate_s < true_s:
        if estimate_s <= 0 and true_s > 0:
            loss_s = (true_s - estimate_s) * (u_f * (r ** -0.5))  # bad case of underestimation
        else:
            loss_s = (true_s - estimate_s) * (u * (r ** -0.5)) # normal underestimation
    elif estimate_s > true_s:
        if estimate_s > 0 and true_s <= 0:
            loss_s = (estimate_s - true_s) * (o_f)  # bad case of overestimation
        else:
            loss_s = (estimate_s - true_s) * (o * r)  # normal overestimation
    else:
        loss_s = 0
    return loss_s

def loss_sqr(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Squared-error loss function.

        Args:
            estimate_s (int or float): Value estimate.
            true_s (int, float or np.array): True value.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int or float, optional): Risk-affinity re-weighting factor.
        Returns:
            Loss incurred for an estimate of the value given a true value.
            Based on squared-error loss, i.e. the absolute distance of the estimate
            from the true value squared.

            true_s can be either a single value or an array of possible true values,
            the output will be a single determined absolute loss value or an array of determined
            loss values, accordingly.

            estimate_s has to be one single value.

    """
    return np.square(loss_abs(estimate_s, true_s, u,o,u_f,o_f, r))

def loss_sqr_given_values(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Squared-error loss function for the exclusive use with a single
        given true value.

                Args:
                    estimate_s (int or float): Value estimate.
                    true_s (int, float): True value.
                    u (int or float, optional): Underestimation re-weighting factor.
                    o (int or float, optional): Overestimation re-weighting factor.
                    u_f (int or float, optional): Fatal underestimation re-weighting factor.
                    o_f (int or float, optional): Fatal overestimation re-weighting factor.
                    r (int or float, optional): Risk-affinity re-weighting factor.
                Returns:
                    Loss incurred for an estimate given one determined true value.
                    Based on squared-error loss, i.e. the absolute distance of the estimate
                    from the true value squared.
    """
    return loss_abs_given_values(estimate_s, true_s, u,o,u_f,o_f, r)**2

def expected_loss_for_estimate(estimate_s, true_s, function='absolute', u=1,o=1,u_f=1,o_f=1, r=1):
    """Function to attain expected loss for an estimate
    given a range of possible true values by taking the mean of all
    possible loss realizations.

        Args:
            estimate_s (int or float): Value estimate.
            true_s (int, float or np.array): True value.
            function ('absolute'(default) or 'squared'): Use of absolute-error or
                squared-error loss function.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int or float, optional): Risk-affinity re-weighting factor.
        Returns:
            Expected loss for a single estimate given a range of possible true values.

            Note: If only one possible true value is passed, the expected loss equals
            the actually incurred loss.

    """

    if function == 'absolute':
        return loss_abs(estimate_s, true_s, u,o,u_f,o_f, r).mean()
    elif function == 'squared':
        return loss_sqr(estimate_s, true_s, u, o, u_f, o_f, r).mean()
    else:
        print('Error: Type of loss function not recognized. '
              'Use "absolute" or "squared".')

def expected_loss_for_range(estimate_range, true_s, function='absolute', u=1,o=1,u_f=1,o_f=1, r=1):
    """Function to attain expected loss and Bayes action based on an absolute-error or
           squared error-loss function for a defined range of estimates.

                Args:
                    estimate_range (np.array): Range of value estimates.
                    true_s (np.array): Array of possible true value occurrences (from a probability distribution)
                    u (int or float, optional): Underestimation re-weighting factor.
                    o (int or float, optional): Overestimation re-weighting factor.
                    u_f (int or float, optional): Fatal underestimation re-weighting factor.
                    o_f (int or float, optional): Fatal overestimation re-weighting factor.
                    r (int, float or np.array, optional): Risk-affinity re-weighting factor.
                Returns:
                    [0]: Expected loss for the range of estimates.
                    [1]: Bayes action (estimate with minimal expected loss)
                    [2]: Expected loss of Bayes action.
        """
    expected_loss_s = lambda estimate_s, r: expected_loss_for_estimate(estimate_s, true_s, function, u, o, u_f, o_f, r)
    loss_e = [expected_loss_s(e, r) for e in estimate_range]
    bayes_action = sop.fmin(expected_loss_s, -40, args=(r,), disp=False)
    bayes_action_loss_e = expected_loss_s(bayes_action, r)
    return loss_e, bayes_action, bayes_action_loss_e

def expected_loss_plot(estimate_range, true_s, risk_range=1, function='absolute', u=1,o=1,u_f=1,o_f=1,
                        verbose=False):
    """Function to plot expected losses and the Bayes action for a range of estimates
    relative to a distribution of possible true values.
    It is possible to plot this for several risk factors at once.

            Args:
                estimate_range (np.array): Range of value estimates.
                true_s (np.array): Array of possible true value occurrences (from a probability distribution)
                u (int or float, optional): Underestimation re-weighting factor.
                o (int or float, optional): Overestimation re-weighting factor.
                u_f (int or float, optional): Fatal underestimation re-weighting factor.
                o_f (int or float, optional): Fatal overestimation re-weighting factor.
                r (int, float or np.array, optional): Risk-affinity re-weighting factor.
            Returns:
                Plot of expected losses for risk neutrality
                or several risk factors.

    """
    ax = plt.subplot(111)
    if isinstance(risk_range, (int,float)):
        r_range=[risk_range]
    else:
        r_range=risk_range
    for r in r_range:
        loss_e, bayes_a, bayes_a_loss_e = expected_loss_for_range(estimate_range, true_s, function, u,o,u_f,o_f, r)
        _color = next(ax._get_lines.prop_cycler)
        plt.plot(estimate_range, loss_e, label="r =" + str(r), color=_color['color'])
        plt.scatter(bayes_a, bayes_a_loss_e, s=70,
                        color=_color['color'])  # , label = "Bayes action r "+str(r))
        plt.vlines(bayes_a, 0, 10 * np.max(loss_e), color=_color['color'], linestyles="--")
        if verbose == True:
            print("Bayes action (minimum) at risk r %.2f: %.2f --- expected loss: %.2f"\
                  % (r, bayes_a, bayes_a_loss_e))
    plt.legend(loc="upper left", scatterpoints=1, title="Legend")
    plt.xlabel("Estimate")
    plt.ylabel("Expected loss")
    plt.xlim(estimate_range[0], estimate_range[-1])
    plt.ylim(0, 1.1 * np.max(loss_e))
    plt.grid()
    plt.show()

def loss_for_estimate(estimate_s, true_s, function='absolute', u=1,o=1,u_f=1,o_f=1, r=1):
    """Function to attain actually incurred loss for an estimate
    given a single true value.

        Args:
            estimate_s (int or float): Value estimate.
            true_s (int or float): True value.
            function ('absolute'(default) or 'squared'): Use of absolute-error or
                squared-error loss function.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int or float, optional): Risk-affinity re-weighting factor.
        Returns:
            Loss for a single estimate given a single determined true value.
    """
    if function == 'absolute':
        return loss_abs_given_values(estimate_s, true_s, u,o,u_f,o_f, r)
    elif function == 'squared':
        return loss_sqr_given_values(estimate_s, true_s, u, o, u_f, o_f, r)
    else:
        print('Error: Type of loss function not recognized. '
              'Use "absolute" or "squared".')

def loss_for_range(estimate_range, true_s, function='absolute', u=1,o=1,u_f=1,o_f=1, r=1):
    """Function to attain loss and Bayes action based on an absolute-error or
               squared error-loss function for a defined range of estimates
               given one single true value.

                    Args:
                        estimate_range (np.array): Range of value estimates.
                        true_value (int or float): Array of possible true value occurrences (from a probability distribution)
                        u (int or float, optional): Underestimation re-weighting factor.
                        o (int or float, optional): Overestimation re-weighting factor.
                        u_f (int or float, optional): Fatal underestimation re-weighting factor.
                        o_f (int or float, optional): Fatal overestimation re-weighting factor.
                        r (int, float or np.array, optional): Risk-affinity re-weighting factor.
                    Returns:
                        [0]: Loss for the range of estimates.
                        [1]: Bayes action (estimate with minimal expected loss)
                        [2]: Expected loss of Bayes action.

                        Note: Since the is only one possible true value,
                        the Bayes action will always equal this value.
            """
    incurred_loss = lambda estimate_s, r: loss_for_estimate(estimate_s, true_s, function, u, o, u_f, o_f, r)
    loss_i = [incurred_loss(e, r) for e in estimate_range]
    bayes_action = sop.fmin(incurred_loss, -40, args=(r,), disp=False)
    bayes_action_loss = incurred_loss(bayes_action, r)
    return loss_i, bayes_action, bayes_action_loss

def loss_plot(estimate_range, true_s, risk_range=1, function='absolute', u=1,o=1,u_f=1,o_f=1,
                        verbose=False):
    """Function to plot losses for a range of estimates
    relative to a single given true value.
    It is possible to plot this for several risk factors at once.

            Args:
                estimate_range (np.array): Range of value estimates.
                true_s (int or float): Array of possible true value occurrences (from a probability distribution)
                u (int or float, optional): Underestimation re-weighting factor.
                o (int or float, optional): Overestimation re-weighting factor.
                u_f (int or float, optional): Fatal underestimation re-weighting factor.
                o_f (int or float, optional): Fatal overestimation re-weighting factor.
                r (int, float or np.array, optional): Risk-affinity re-weighting factor.
            Returns:
                Plot of losses for risk neutrality
                or several risk factors given a single determined true value.

    """
    ax = plt.subplot(111)
    if isinstance(risk_range, (int,float)):
        r_range=[risk_range]
    else:
        r_range=risk_range
    for r in r_range:
        loss_i, bayes_a, bayes_a_loss = loss_for_range(estimate_range, true_s, function, u,o,u_f,o_f, r)
        _color = next(ax._get_lines.prop_cycler)
        plt.plot(estimate_range, loss_i, label="r =" + str(r), color=_color['color'])
        plt.scatter(bayes_a, bayes_a_loss, s=70,
                        color=_color['color'])  # , label = "Bayes action r "+str(r))
        plt.vlines(bayes_a, 0, 10 * np.max(loss_i), color=_color['color'], linestyles="--")
        if verbose == True:
            print("Bayes action (minimum) at risk r %.2f: %.2f --- expected loss: %.2f"\
                  % (r, bayes_a, bayes_a_loss))
    plt.legend(loc="upper left", scatterpoints=1, title="Legend")
    plt.xlabel("Estimate")
    plt.ylabel("Expected loss")
    plt.xlim(estimate_range[0], estimate_range[-1])
    plt.ylim(0, 1.1 * np.max(loss_i))
    plt.grid()
    plt.show()


def plot_multiple_loss():

    def plot_axis(xvals, nor_l, loss, subplot_spec, e, depth=False):
        grid = gridspect.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)
        ax = fig.add_subplot(grid[:, :])
        # ax0.yaxis.set_visible(False)
        axr = ax.twinx()

        labels = 'Thickness Score' if depth is True else None
        c = default_red if depth is True else default_blue

        ax.plot(xvals, nor_l, color=c, linewidth=.5, label=labels)
        ax.fill_between(xvals, nor_l, 0, color=c, alpha=.8)
        ax.set_ylabel('Likelihood')
        ax.set_ylim(0, .16)
        ax.set_xlim(-25+6, 6+25)

        axr.set_ylim(0, 40)
        axr.set_ylabel('Expected Loss')

        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        if e == 0:
            axr.plot(xvals, loss, linewidth=3, color='white')
            axr.plot(xvals, loss, linewidth=2, color='#496155', label='Loss')
            axr.legend(frameon=True, facecolor='white', framealpha=1)

        else:
            axr.plot(xvals, loss, linewidth=3, color='white')
            axr.plot(xvals, loss, linewidth=2, color='#496155', label='Risk Neutral')
        if e%2 == 0:

            axr.yaxis.set_visible(False)
           # axr.yaxis.label.set_visible(False)

        else:

            for tick in axr.get_yticklines():
                tick.set_visible(False)
           # axr.yaxis.label.set_visible(False)
            ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        axr.xaxis.set_visible(False)
        if e>3:
            ax.xaxis.set_visible(True)
            axr.xaxis.set_visible(True)
            ax.set_xlabel('Score')
        return ax, axr

    def res_score_loss(true_s, estimate_s, ov=1.25, uv_b=1.5, ov_b=2, risk_s=1):

        underest = (estimate_s < true_s)
        # loss_s = np.zeros((estimate_s.shape[0], true_s.shape[0]))
        underest_bad = (estimate_s <= 0) & (true_s > 0)
        overest = (estimate_s > true_s)
        overest_bad = (estimate_s > 0) & (true_s <= 0)
        a = underest * abs(true_s - estimate_s) * (risk_s ** -0.5)
        b = underest_bad * abs(true_s - estimate_s) * (uv_b * (risk_s ** -0.5))
        c = overest * abs(estimate_s - true_s) * (ov * risk_s)
        d = overest_bad * abs(estimate_s - true_s) * (ov_b * risk_s)
        loss_s = (a + b + c + d).mean(axis=1)
        return loss_s, (a,b,c,d)

    def abs_loss(true_s, estimate_s, ov=1, uv=1, risk_s=1, uv_b=1):
        print('foooooo')
        underest = (estimate_s < true_s)
        # loss_s = np.zeros((estimate_s.shape[0], true_s.shape[0]))
        underest_bad = 1# (estimate_s <= 0) & (true_s > 0)
        overest = (estimate_s > true_s)
        overest_bad = (estimate_s > 0) & (true_s <= 0)
        b = 0
        d = 0
        a = underest * abs(true_s - estimate_s) * (uv * risk_s)
      #  b = underest_bad * (true_s - estimate_s) * (uv_b * (risk_s))
        c = overest * abs(estimate_s - true_s) * (ov * risk_s)
        #d = overest_bad * (estimate_s - true_s) * (ov_b * risk_s)
        loss_s = (a + b + c + d).mean(axis=1)
        return loss_s, (a,b,c,d)

    def compute_values(mu, sigma, loss_type, x_range=(-24+6, 6+24), risk=1, depth=False):
        samples_size = 100000

        if x_range is None:
            x_max = mu + 5 * sigma
            x_min = mu - 5 * sigma
        else:
            x_min, x_max = x_range
       # xvals = np.linspace(-23+6, 6+23, 100)

        xvals = np.linspace(x_min, x_max, 100)
        nor_l = stats.norm.pdf(xvals, loc=mu, scale=sigma)
        samples = np.random.normal(loc=mu, scale=sigma, size=samples_size)

        if depth is True:
            #xvals = np.linspace(0, 60, 100)
            # nor_d = stats.norm.pdf(xvals, loc=30, scale=5)
            samples_depth = np.random.normal(loc=30, scale=5, size=samples_size)
            depth_score = samples_depth

            # Transformation to cost
            d_cost = - ((depth_score-depth_score.min()) / 8) ** 2.6
            samples = samples + d_cost
        elif depth is False:
            pass

        else:
            samples = depth

        if loss_type == 'abs':
            loss, partial = abs_loss(samples, xvals.reshape(-1, 1), 1, 1, risk_s=risk)
            #return xvals, nor_l, loss, partial
        elif loss_type == 'custom':
            loss, partial = res_score_loss(samples, xvals.reshape(-1, 1), risk_s=risk)

        if depth is True:
            return xvals, nor_l, loss, partial, samples, d_cost
        else:
            return xvals, nor_l, loss, partial

    sns.set(style="white")  # , rc={"axes.facecolor": (0, 0, 0, 0)})
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

    # %matplotlib notebook
    figsize = (12, 12)
    fig, axes = plt.subplots(0, 0, figsize=figsize, constrained_layout=False)
    sns.despine(left=True, bottom=True)

    gs_0 = gridspect.GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.009)

    axis = []

    mu = [12]*6
    sigma = [2.3, 2.3 ,4,4, 4, 4]
    type = ['abs', 'custom', 'abs', 'custom', 'abs', 'custom']
    for e, ax in enumerate(gs_0):

        if e==0:
            x, n, l, partial = compute_values(mu[e], sigma[e], type[e])
            ax_, axr_ = plot_axis(x, n, l, ax, e)
            ax_.vlines(mu[e], 0, 40, linestyles='--', alpha=.5)
            ax_.vlines(0, 0, 40, linestyles='--', alpha=.5)
            axr_.hlines(l.min(), -25, x[l.argmin()],
                       linestyles='--', alpha=.5)

        if e==1:
            x, n, l, partial = compute_values(mu[e], sigma[e], type[e])
            ax_, axr_ = plot_axis(x, n, l, ax, e)
            ax_.vlines(mu[e], 0, 40, linestyles='--', alpha=.5)
            ax_.vlines(0, 0, 40, linestyles='--', alpha=.5)
            labels = ['Underes.', 'Crit. Underest.', 'Overest.', 'Crit. Overest.']
            for i in range(4):
                axr_.plot(x, partial[i].mean(axis=1), '--', linewidth=1, label=labels[i])

            axr_.legend(loc=2, frameon=True, facecolor='white', framealpha=1)

        if e==2 or e==3:
            x, n, l, partial = compute_values(mu[e], sigma[e], type[e])
            ax_, axr_ = plot_axis(x, n, l, ax, e)
            axr_.plot(x[l.argmin()], 0, 'o',  color='#496155', markersize=12)
            if e == 2:
                axr_.hlines(l.min(), -25, x[l.argmin()],
                            linestyles='--', alpha=.5)

            x, n, l, partial = compute_values(mu[e], sigma[e], type[e], risk=.2)
            axr_.plot(x, l, linewidth=3, color='white')
            axr_.plot(x, l, linewidth=2, label='Risk Friendly')
            axr_.plot(x[l.argmin()], 0, 'o',  color='b', markersize=12)

            x, n, l, partial = compute_values(mu[e], sigma[e], type[e], risk=5)
            axr_.plot(x, l, linewidth=3, color='white')
            axr_.plot(x, l, linewidth=2, label='Risk Averse')
            axr_.plot(x[l.argmin()], 0, 'o',  color='orange', markersize=12)

            ax_.vlines(mu[e], 0, 40, linestyles='--', alpha=.5)
            ax_.vlines(0, 0, 40, linestyles='--', alpha=.5)
            if e==2:
                axr_.legend(frameon=True, facecolor='white', framealpha=1)

        if e > 3:
            x, n, l, partial, ss, d_cost = compute_values(mu[e], sigma[e], type[e],
                                                          x_range=(-24+6, 6+24),
                                                          risk=1, depth=True)
            ax_, axr_ = plot_axis(x, n, l, ax, e, depth=True)
            axr_.plot(x[l.argmin()], 0, 'o',  color='#496155', markersize=12)

            x_max = ss.mean() + 6 * ss.std()
            x_min = ss.mean() - 6 * ss.std()
            print(x_max, x_min, 'foo')
            print(ss.mean(), d_cost.mean())
            x, n, l, partial = compute_values(mu[e], sigma[e], type[e],
                                                      x_range=(-24 + 6, 6 + 24),
                                                      risk=.2, depth=ss)
            #print(_1.mean(),  d_cost.mean())
            axr_.plot(x, l, linewidth=3, color='white')
            axr_.plot(x, l, linewidth=2, label='Risk Friendly')
            axr_.plot(x[l.argmin()], 0, 'o',  color='b', markersize=12)

            x, n, l, partial = compute_values(mu[e], sigma[e], type[e],
                                                      x_range=(-24 + 6, 6 + 24),
                                                      risk=5, depth=ss)
          #  print(_1.mean(),  d_cost.mean())
            axr_.plot(x, l, linewidth=3, color='white')
            axr_.plot(x, l, linewidth=2, label='Risk Adverse')
            axr_.plot(x[l.argmin()], 0, 'o',  color='orange', markersize=12)

            k = sns.kdeplot(d_cost, ax=ax_, shade=True, label='Depth Score', color=default_red,
                        alpha=.8,linewidth=.5)
            #k.legend_.set_frame_on(True)
            sns.kdeplot(ss, ax=ax_, label='Final Score', shade=True, color=default_blue,
                        alpha=1, linewidth=.5)
            k.legend_.set_frame_on(True)
            k.legend_.set_alpha(1)

            if e == 4:
                k.legend_.set_visible(False)
            # ax_.vlines(mu[e], 0, 40, linestyles='--', alpha=.5)
            ax_.vlines(np.median(ss), 0, 40, linestyles='--', alpha=.5)
            ax_.vlines(0, 0, 40, linestyles='--', alpha=.5)
          #  ax_.legend(loc=2)

        axis.append((ax_, axr_))
    return k
