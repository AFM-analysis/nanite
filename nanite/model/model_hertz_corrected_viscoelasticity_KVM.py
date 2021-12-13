import lmfit
import numpy as np
import copy


def get_parameter_defaults():
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=1e3, min=0, max=20e3, vary=True)
    params.add("E1", value=1e3, min=0, max=20e3, vary=True)
    params.add("time_ind", value=1, min=0.0, max=10, vary=False)
    params.add("R", value=2.5e-6, vary=False)
    params.add("eta", value=.5, min=0, vary=False)
    params.add("nu", value=.5, vary=False)
    params.add('_constraint', value=0.5, min=1 / 5, max=10, vary=True)
    params.add("lmd", expr='_constraint*time_ind')
    params.add("velocity", value=5e-6, vary=False)
    params.add("contact_point", value=0, vary=True)
    params.add("baseline", value=0, vary=False)
    return params


def hertz_corrected_viscelasticity_KVM(delta, E, E1, time_ind, R, eta, nu,
                                       _constraint, lmd, velocity,
                                       contact_point,
                                       baseline):
    r"""Hertz model for a spherical indenter icluding correction for
    viscoelasticity"""
    """Hertz model for Spherical indenter - approximation

    .. math::

        F = \\(frac{4}{3}
            \\frac{E}{1-\\nu^2}
            \\sqrt{R}
            \\delta^{3/2})
            \\(1-0.15frac{delta}{R})
            \\+(frac{4}{3}
            \\frac{E1}{1-\\nu^2}
            \\sqrt{R}
            \\delta^{3/2})
            \\(1-0.15frac{delta}{R})(\\exp^(frac{-0.365delta}{\\lambdaV}))
            \\+7.25delta^{1/2}R^{1/2}\\muV

    Parameters
    ----------
    E: float
        Young's modulus of Hertz theory [N/m²]
    E1: float
        Young's modulus of Maxwell element [N/m²]
    delta: 1d ndarray
        Indentation [m]
    time_ind: float
        Indentation time [s]
    R: float
        Tip radius [m]
    eta: float
        Dashpot value of Kelvin Voigt model [Pa.s]
    nu: float
        Poisson's ratio
    _constraint: float
        constraint for Maxwell relaxation (lmd)
    lmd: float
        relaxation time of Maxwell elements [s]
    velocity: float
        Velocity of indentation [m/s]
    contact_point: float
        Indentation offset [m]
    baseline: float
        Force offset [N]
    negindent: bool
        If `True`, will assume that the indentation value(s) given by
        `delta` are negative and must be multiplied by -1.

    Returns
    -------
    F: float
        Force [N]

    Notes
    -----
    These approximations are made by the Hertz model:

    - The sample is isotropic.
    - The sample is behave as Kelvin-Voigt-Maxwell material.
    - The sample is extended infinitely in one half space.
    - The indenter is not deformable.
    - There are no additional interactions between sample and indenter.

    Additional assumptions:

    - no surface forces

    References
    ----------
    Sneddon (1965) :cite:`Sneddon1965`,
    Dobler (personal communication, 2018) :cite:`Dobler`
    Ding et. al (2017) :cite:'Ding'
    Abuhattum et al. (2021) :cite:'Abuhattum'
    """
    root = (contact_point - delta)
    pos = root > 0
    D = (1 - nu ** 2)

    aa0 = 4 / 3 * np.sqrt(R) * E / D
    aa1 = 4 / 3 * np.sqrt(R) * E1 / D

    bb = np.zeros_like(delta)
    bb[pos] = (root[pos]) ** (3 / 2)
    cc = np.zeros_like(delta)

    cc[pos] = 1 - 0.15 * root[pos] / R

    dd = np.zeros_like(delta)
    dd[pos] = (np.exp(-0.365 * root[pos] / (velocity * (lmd))))

    ee = np.zeros_like(delta)
    ee[pos] = 7.25 * (root[pos]) ** (1 / 2) * R ** (1 / 2) * eta * velocity
    return bb * cc * (aa0 + aa1 * dd) + ee + baseline


def model(params, x):
    if x[0] < x[-1]:
        revert = True
    else:
        revert = False
    if revert:
        x = x[::-1]

    mf = hertz_corrected_viscelasticity_KVM(delta=x,
                                            E=params["E"],
                                            E1=params["E1"].value,
                                            time_ind=params["time_ind"].value,
                                            R=params["R"].value,
                                            eta=params["eta"].value,
                                            nu=params["nu"].value,
                                            _constraint=params["_constraint"]
                                            .value,
                                            lmd=params["lmd"].value,
                                            velocity=params["velocity"].value,
                                            contact_point=params
                                            ["contact_point"].value,
                                            baseline=params["baseline"].value)

    if revert:
        return mf[::-1]
    return mf


def compute_ancillaries(idnt):
    # This function takes a nanite.indent.Indentation instance
    # (https://nanite.readthedocs.io/en/stable/sec_code_reference.html
    # nanite.indent.Indentation)
    # as an argument and returns additional parameters as a
    # dictionary.
    """first, find the contact point that is computed from simple Hertz
    model"""
    parms = idnt.get_initial_fit_parameters(model_key=model_key,
                                            model_ancillaries=False)
    R = parms["R"].value
    velocity = parms["velocity"].value
    indentation = copy.deepcopy(idnt)
    indentation.fit_properties["model_key"] = "hertz_para"
    params_ind = indentation.get_initial_fit_parameters()
    params_ind["R"].value = R

    indentation.fit_model(model_key="hertz_para",
                          params_initial=params_ind,
                          range_x=(-2e-6, 1e-6), range_type='absolute')

    if "params_fitted" in indentation.fit_properties:
        params_ind_updated = indentation.fit_properties["params_fitted"]
        contact_point_simple = params_ind_updated["contact_point"].value
    else:
        contact_point_simple = 0

    force = indentation.data["force"]
    segment = indentation.data["segment"]
    tip_position = indentation.data["tip position"]
    force = force[(segment == 1)]
    tip_position = tip_position[(segment == 1)]

    force_jump_max_indent, fitted_max_indent = fit_viscosity_jump_model(
            tip_position=tip_position,
            force=force,
            R=R,
            velocity=velocity)

    fitted_max_indent = fitted_max_indent - contact_point_simple
    force_jump_max_indent = force_jump_max_indent

    eta_constants = 2 * 7.25 * velocity * R ** (1 / 2) * (
        abs(fitted_max_indent)) ** (1 / 2)
    time_ind = -(fitted_max_indent) / velocity
    eta = force_jump_max_indent / eta_constants

    anc_dict = {"force_jump_max_indent": force_jump_max_indent,
                "eta": eta,
                "time_ind": time_ind}
    return anc_dict


def helper_retract_fraction(tip_position, force, fraction_force):
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


def helper_line_polynomial_fit(position_seg, force_seg):
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

    params.add("y0",
               value=position_seg[0],
               max=position_seg[0] * 0.98,
               min=position_seg[0] * 1.02,
               vary=False)

    params.add('a',
               value=-1.0e1,
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
    position_seg, force_seg, max_position, max_force = helper_retract_fraction(
        tip_position, force, fraction_force)
    position_fit, params = helper_line_polynomial_fit(position_seg, force_seg)
    force_jump_max_indent, jump_end, fitted_max_indent = force_jump(params,
                                                                    max_force)

    return force_jump_max_indent, fitted_max_indent


model_doc = hertz_corrected_viscelasticity_KVM.__doc__
model_func = hertz_corrected_viscelasticity_KVM
model_key = "hertz_corr_visco_KVM"
model_name = "Hertz model corrected for viscoelasticity using KVM model"
parameter_keys = ["E", "E1", "time_ind", "R", "eta", "nu",
                  "_constraint", "lmd", "velocity", "contact_point",
                  "baseline"]
parameter_names = ["Young's Modulus of Kelvin",
                   "Young's Modulus of Maxwell",
                   "Time to indent",
                   "Tip Radius", "viscosity", "Poisson's Ratio",
                   "Constraint",
                   "Maxwell element relaxation",
                   "Velocity of indenter",
                   "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "Pa", "s", "m", "Pa·s", " ", " ", "s", "m/s", "m",
                   "N"]
parameter_anc_keys = ["force_jump_max_indent", "eta",
                      "time_ind"]
parameter_anc_names = ["Force jump at max indent", "viscosity",
                       "Time to indent"]
parameter_anc_units = ["N", "Pa·s", "s"]

valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
