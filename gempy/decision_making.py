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

def loss_abs(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Absolute-error loss function.

        Args:
            estimate_s (int or float): Score estimate.
            true_s (int, float or np.array): True score value.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int, float or np.array, optional): Risk-affinity re-weighting factor.
        Returns:
            Loss incurred for an estimate of the score given a true score.
            Based on absolute-error, i.e. the absolute distance of the estimate
            from the true value.

            True_s can be either a single value or an array of possible true values,
            the output will be a single determined value or an array of determined
            values, accordingly.

            Estimate_s has to be one single value.

    """
    true_s = np.array(true_s).astype(float)
    loss_s = np.zeros_like(true_s)
    underest = (estimate_s < true_s)
    underest_fatal = (estimate_s == 0) & (true_s > 0)
    overest = (estimate_s > true_s)
    overest_fatal = (estimate_s > 0) & (true_s == 0)
    loss_s[underest] = (true_s[underest] - estimate_s)  * (u * r)
    loss_s[underest_fatal] = (true_s[underest_fatal] - estimate_s) * (u_f * (r ** -0.5))
    loss_s[overest] = (estimate_s - true_s[overest]) * (o * r)
    loss_s[overest_fatal] = np.abs((true_s[overest_fatal] - estimate_s)) * (o_f * r)
    return loss_s

def loss_sqr(estimate_s, true_s, u=1,o=1,u_f=1,o_f=1, r=1):
    """Squared-error loss function.

        Args:
            estimate_s (int or float): Score estimate.
            true_s (int, float or np.array): True score value.
            u (int or float, optional): Underestimation re-weighting factor.
            o (int or float, optional): Overestimation re-weighting factor.
            u_f (int or float, optional): Fatal underestimation re-weighting factor.
            o_f (int or float, optional): Fatal overestimation re-weighting factor.
            r (int, float or np.array, optional): Risk-affinity re-weighting factor.
        Returns:
            Loss incurred for an estimate of the score given a true score.
            Based on absolute-error, i.e. the absolute distance of the estimate
            from the true value.

            True_s can be either a single value or an array of possible true values,
            the output will be a single determined value or an array of determined
            values, accordingly.

            Estimate_s has to be one single value.

    """
    true_s = np.array(true_s).astype(float)
    loss_s = np.zeros_like(true_s)
    underest = (estimate_s < true_s)
    underest_fatal = (estimate_s == 0) & (true_s > 0)
    overest = (estimate_s > true_s)
    overest_fatal = (estimate_s > 0) & (true_s == 0)
    loss_s[underest] = (true_s[underest] - estimate_s)  * (u * r)
    loss_s[underest_fatal] = (true_s[underest_fatal] - estimate_s) * (u_f * (r ** -0.5))
    loss_s[overest] = (estimate_s - true_s[overest]) * (o * r)
    loss_s[overest_fatal] = np.abs((true_s[overest_fatal] - estimate_s)) * (o_f * r)
    return np.square(loss_s)

def expected_loss(estimate_s, true_s, function='absolute', u=1,o=1,u_f=1,o_f=1, r=1):
    """Function to attain expected loss, based on an absolute-error or
    squared error-loss function.

    :param estimate_s:
    :param true_s:
    :param function:
    :param u:
    :param o:
    :param u_f:
    :param o_f:
    :param r:
    :return:
    """
    if function == 'absolute':
        return loss_abs(estimate_s, true_s, u,o,u_f,o_f, r).mean()
    elif function == 'squared':
        return loss_sqr(estimate_s, true_s, u, o, u_f, o_f, r).mean()
    else:
        print('Error: Type of loss function not recognized. '
              'Use "absolute" or "squared".)

def expected_loss_range(estimate_range, true_s, risk_range=np.array(1), function='absolute', u=1,o=1,u_f=1,o_f=1):
    expected_loss_s = lambda estimate_s, r: expected_loss(estimate_s, true_s, function, u,o,u_f,o_f,r)

    for r in risk_range:

