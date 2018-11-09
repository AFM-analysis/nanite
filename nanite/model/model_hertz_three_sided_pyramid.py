import lmfit
import numpy as np
from numpy import pi

from . import weight


def get_parameter_defaults():
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=3e3, min=0)
    params.add("alpha", value=25, vary=False)
    params.add("nu", value=.5, vary=False)
    params.add("contact_point", value=0)
    params.add("baseline", value=0, vary=False)
    return params


def hertz_three_sided_pyramid(E, delta, alpha, nu, contact_point=0,
                              baseline=0):
    """Hertz model for three sided pyramidal indenter

    .. math::

        F = 0.887 \\tan\\alpha
            \\cdot \\frac{E}{1-\\nu^2}
            \\delta^2

    Parameters
    ----------
    E: float
        Young's modulus [N/m²]
    delta: 1d ndarray
        Indentation [m]
    alpha: float
        Face angle of the pyramid [degrees]
    nu: float
        Poisson's ratio
    contact_point: float
        Indentation offset [m]
    baseline: float
        Force offset [N]
    negindent: bool
        If `True`, will assume that the indentation value(s) given by
        `delta` are negative and must be mutlitplied by -1.

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


def model(params, x):
    if x[0] < x[-1]:
        revert = True
    else:
        revert = False
    if revert:
        x = x[::-1]
    mf = hertz_three_sided_pyramid(E=params["E"].value,
                                   delta=x,
                                   alpha=params["alpha"].value,
                                   nu=params["nu"].value,
                                   contact_point=params["contact_point"].value,
                                   baseline=params["baseline"].value)
    if revert:
        return mf[::-1]
    return mf


def residual(params, delta, force, weight_cp=5e-7):
    """ Compute residuals for fitting

    Parameters
    ----------
    params: lmfit.Parameters
        The fitting parameters for `model`
    delta: 1D ndarray of lenght M
        The indentation distances
    force: 1D ndarray of length M
        The corresponding force data
    weight_cp: positive float or zero/False
        The distance from the contact point until which
        linear weights will be applied. Set to zero to
        disable weighting.
    """
    md = model(params, delta)
    resid = force-md

    if weight_cp:
        # weight the curve so that the data around the contact_point do
        # not affect the fit so much.
        weights = weight.weight_cp(cp=params["contact_point"].value,
                                   delta=delta,
                                   weight_dist=weight_cp)
        resid *= weights
    return resid


model_doc = hertz_three_sided_pyramid.__doc__
model_key = "hertz_pyr3s"
model_name = "pyramidal indenter, three-sided (Hertz)"
parameter_keys = ["E", "alpha", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Face Angle",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "°", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
