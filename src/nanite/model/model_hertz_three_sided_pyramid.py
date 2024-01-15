import lmfit
import numpy as np
from numpy import pi


def get_parameter_defaults():
    """Return the default model parameters"""
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=3e3, min=0)
    params.add("alpha", value=5, min=0, max=30, vary=False)
    params.add("nu", value=.5, min=0, max=0.5, vary=False)
    params.add("contact_point", value=0)
    params.add("baseline", value=0)
    return params


def hertz_three_sided_pyramid(delta, E, alpha, nu, contact_point=0,
                              baseline=0):
    r"""Hertz model for three sided pyramidal indenter

    .. math::

        F = 0.887 \tan\alpha
            \cdot \frac{E}{1-\nu^2}
            \delta^2

    Parameters
    ----------
    delta: 1d ndarray
        Indentation [m]
    E: float
        Young's modulus [N/m²]
    alpha: float
        Inclination angle of the pyramidal face [degrees]
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
    - The inclination angle of the pyramidal face (in radians)
      must be close to zero.

    References
    ----------
    Bilodeau et al. 1992 :cite:`Bilodeau:1992`
    """
    aa = 0.8887*np.tan(alpha*pi/180) * E/(1-nu**2)
    root = contact_point-delta
    pos = root > 0
    bb = np.zeros_like(delta)
    bb[pos] = (root[pos])**(2)
    return aa*bb + baseline


model_doc = hertz_three_sided_pyramid.__doc__
model_func = hertz_three_sided_pyramid
model_key = "hertz_pyr3s"
model_name = "pyramidal indenter, three-sided (Hertz)"
parameter_keys = ["E", "alpha", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Face Angle",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "°", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
