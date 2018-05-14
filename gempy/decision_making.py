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