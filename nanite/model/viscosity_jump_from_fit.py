"""Compute viscosity from jump at maximum indentation for FEM model

Changelog
---------
2020-01-11
 - paul: commented out plotting command
 - paul: added missing parameter mu
2020-01-10
 - paul: move seaborn stuff into __main__
2020-01-08
 - email from Shada
"""

import lmfit
import numpy as np


def segment_for_fit(tip_position, force, fraction_force):
    """
    The function specifies the segment of the retract curve that will be
    used for fitting the viscosity model.

    Parameters
    ----------
    tip_position: 1D numpy array
        the tip position values of the cantilever's retract motion [m]
    force: 1D numpy array
        the force values of the cantilever's retract motion [N]
    fraction_force: float
        fraction of the data points that should be taken into account for
        fitting, the decision is made according to the maximal value of
        the force

    Returns
    -------
    position_seg: 1D numpy array
        the segment of tip position array that will be used for the fitting [m]
    force_seg: 1D numpy array
        the segment of force array that will be used for the fitting [N]
    max_position: float
        the maximal absolute value of the tip position [m]
    max_force: float
        the maximal value of the force [N]

    Notes
    -----
    max_position is assumed to be the first point of the retract
    curve even though it is not always the largest indentation value. This
    is because the movement of the Piezo changes the direction at the
    beginning of the retract signal and larger indentation values after the
    first point are assumed to be as a result of the measurement noise.
    """

    fraction = int(0.8*len(force))
    max_force = np.max(force[:fraction])
    max_position = -1 * tip_position[0]
    fit_stop = int(np.argwhere(force < (max_force * fraction_force))[0])
    position_seg = tip_position[:fit_stop].copy()
    force_seg = force[:fit_stop].copy()

    return position_seg, force_seg, max_position, max_force


def model_fit(position_seg, force_seg):
    """
    piecewise fitting of two functions, the first part of the signal with a
    vertical line followed by a third order polynomial.
    ..math:: y = \\y0
    when x<x0 \\y0 + a*(x-x0) + b*(x-x0)^2 + c*(x-x0)^3 when x>=x0

    Parameters
    ----------
    position_seg: 1D numpy array [m]
        the tip position values used for the fitting
    force_seg: 1D numpy array [N]
        the force values used for the fitting

    Returns
    -------
    fit: 1D numpy array
        fit results
    res.params: lmfit.parameter
        fitting parameters
    """
    params = lmfit.Parameters()
    params.add("x0",
               value=force_seg[0]*0.98,
               max=force_seg[0],
               min=force_seg[0] * 0.6,
               vary=True)

    # beginning of signal - top part
    params.add("y0",
               value=position_seg[0],
               max=position_seg[0] * 0.98,
               min=position_seg[0] * 1.02,
               vary=False)

    #
    params.add('a',
               value=-1.0e1,
               # max=0,
               # brute_step=0.1,
               vary=True)
    params.add("b",
               value=-1.0e2,
               vary=True)
    params.add("c",
               value=1.0e2,
               vary=True)
    params.add('d',
               expr="y0")

    def fcn(params, x, y):
        parvals = params.valuesdict()
        split = x < parvals["x0"]
        mod1 = np.zeros_like(y)
        mod1 += parvals["d"] + parvals["a"] * (x - parvals["x0"]) + parvals[
            "b"] * (x - parvals["x0"]) ** 2 + parvals[
            "c"] * (x - parvals["x0"]) ** 3
        mod1[~split] = 0

        mod2 = np.zeros_like(y)
        mod2 += parvals["y0"]
        mod2[split] = 0

        return y - mod1 - mod2

    res = lmfit.minimize(fcn=fcn,
                         params=params,
                         method='nelder',
                         args=(force_seg, position_seg))
    fit = position_seg - res.residual

    return fit, res.params


def force_jump(params, max_force):
    """
    calculating the value of the force jump from the difference between
     the maximal force and the force value at the end of the fitted
     vertical line in the piecewise function fitting.

    Parameters
    ----------
    params: lmfit.parameter
        piecewise fitting parameters
    max_force: float [N]
        the maximal value of the force

    Returns
    -------
    force_jump_max_indent: float
        force jump value [N]
    jump_end: float
        the force value at the end (smallest force) of the fitted
        vertical line [N]
    fitted_max_indent: float
        the fitted maximal indentation [m]
    """
    fitting_parvals = params.valuesdict()
    jump_end = fitting_parvals['x0']
    fitted_max_indent = fitting_parvals['y0']
    force_jump_max_indent = max_force - jump_end
    return force_jump_max_indent, jump_end, fitted_max_indent


def fit_viscosity_jump_model(tip_position: object, force: object, R: object,
                             velocity: object, fraction_force: object =
                             0.85) -> object:
    """ finding the viscosity value from the force jump in the retract curve
    .. math::
        mu =
            \frac{F_jump}{2*7.25*delta_max^{1/2}*R^{1/2}*velocity}

    Parameters
    ----------
    tip_position: 1D numpy array
        the tip position values of the cantilever's retract motion [m]
    force: 1D numpy array
        the force values of the cantilever's retract motion [N]
    R: float [m]
        the radius of the spherical tip used for indentation
    velocity: float [m/s]
        the absolute value of the Piezo velocity set by the user
    fraction_force: float
        fraction of the data points that should be taken into account for
        fitting, the decision is made according to the maximal value of the
        force

    Returns
    -------
    force_jump_max_indent: float
        value of the the force jump at the maximal indentation [N]
    fitted_max_indent: float
        value of the fitted maximal indentation [m]
    """
    position_seg, force_seg, max_position, max_force = segment_for_fit(
        tip_position, force, fraction_force)
    position_fit, params = model_fit(position_seg, force_seg)
    force_jump_max_indent, jump_end, fitted_max_indent = force_jump(params,
                                                                    max_force)

    return force_jump_max_indent, fitted_max_indent
