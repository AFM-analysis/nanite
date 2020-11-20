import lmfit
import numpy as np


def get_parameter_defaults():
    """Return the default model parameters"""
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=3e3, min=0)
    params.add("R", value=10e-6, min=0, vary=False)
    params.add("nu", value=.5, min=0, max=0.5, vary=False)
    params.add("contact_point", value=0)
    params.add("baseline", value=0)
    return params


def hertz_sneddon_spherical_approx(delta, E, R, nu, contact_point=0,
                                   baseline=0):
    r"""Hertz model for Spherical indenter - approximation

    .. math::

        F = \frac{4}{3} \frac{E}{1-\nu^2} \sqrt{R} \delta^{3/2}
            \left(1
             - \frac{1}{10} \frac{\delta}{R}
             - \frac{1}{840} \left(\frac{\delta}{R}\right)^2
             + \frac{11}{15120} \left(\frac{\delta}{R}\right)^3
             + \frac{1357}{6652800} \left(\frac{\delta}{R}\right)^4
             \right)

    Parameters
    ----------
    delta: 1d ndarray
        Indentation [m]
    E: float
        Young's modulus [N/m²]
    R: float
        Tip radius [m]
    nu: float
        Poisson's ratio
    contact_point: float
        Indentation offset [m]
    baseline: float
        Force offset [N]

    Returns
    -------
    F: float
        Force [N]

    Notes
    -----
    These approximations are made by the Hertz model:

    - The sample is isotropic.
    - The sample is a linear elastic solid.
    - The sample is extended infinitely in one half space.
    - The indenter is not deformable.
    - There are no additional interactions between sample and indenter.

    Additional assumptions:

    - no surface forces

    References
    ----------
    Sneddon (1965) :cite:`Sneddon1965`,
    Dobler (personal communication, 2018) :cite:`Dobler`
    """
    aa = 4/3 * E/(1-nu**2)*np.sqrt(R)
    root = contact_point-delta
    pos = root > 0
    bb = np.zeros_like(delta)
    bb[pos] = (root[pos])**(3/2)*(
        + 1
        - 1/10*(root[pos]/R)
        - 1/840*(root[pos]/R)**2
        + 11/15120*(root[pos]/R)**3
        + 1357/6652800*(root[pos]/R)**4)
    return aa*bb + baseline


model_doc = hertz_sneddon_spherical_approx.__doc__
model_func = hertz_sneddon_spherical_approx
model_key = "sneddon_spher_approx"
model_name = "spherical indenter (Sneddon, approximative)"
parameter_keys = ["E", "R", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Tip Radius",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "m", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
