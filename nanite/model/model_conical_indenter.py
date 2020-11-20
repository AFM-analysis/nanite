import lmfit
import numpy as np
from numpy import pi


def get_parameter_defaults():
    """Return the default model parameters"""
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=3e3, min=0)
    params.add("alpha", value=25, min=0, max=90, vary=False)
    params.add("nu", value=.5, min=0, max=0.5, vary=False)
    params.add("contact_point", value=0)
    params.add("baseline", value=0)
    return params


def hertz_conical(delta, E, alpha, nu, contact_point=0, baseline=0):
    r"""Hertz model for a conical indenter

    .. math::

        F = \frac{2\tan\alpha}{\pi}
            \frac{E}{1-\nu^2}
            \delta^2

    Parameters
    ----------
    delta: 1d ndarray
        Indentation [m]
    E: float
        Young's modulus [N/m²]
    alpha: float
        Half cone angle [degrees]
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

    - infinitely sharp probe

    References
    ==========
    Love (1939) :cite:`Love1939`
    """
    aa = 2*np.tan(alpha*pi/180)/pi * E/(1-nu**2)
    root = contact_point-delta
    pos = root > 0
    bb = np.zeros_like(delta)
    bb[pos] = root[pos]**2
    return aa*bb + baseline


model_doc = hertz_conical.__doc__
model_func = hertz_conical
model_key = "hertz_cone"
model_name = "conical indenter (Hertz)"
parameter_keys = ["E", "alpha", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Half Cone Angle",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "°", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
