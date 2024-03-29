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


def hertz_paraboloidal(delta, E, R, nu, contact_point=0, baseline=0):
    r"""Hertz model for a paraboloidal indenter

    .. math::

        F = \frac{4}{3}
            \frac{E}{1-\nu^2}
            \sqrt{R}
            \delta^{3/2}

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
    The derivation in :cite:`Sneddon1965` reads

    .. math::

        F = \frac{4}{3}
            \frac{E}{1-\nu^2}
            \sqrt{2k}
            \delta^{3/2},

    where :math:`k` is defined by the paraboloid equation

    .. math::

        \rho^2 = 4kz.

    As follows from the derivations in :cite:`LandauLifshitz`, this
    model is valid for either of the two cases:

    - Indentation of a plane (infinite radius) with Young's modulus
      :math:`E` by a sphere with infinite Young's modulus and radius
      :math:`R`, or
    - Indentation of a sphere with radius :math:`R` and Young's modulus
      :math:`E` by a plane (infinite radius) with infinite Young's modulus.


    These approximations are made by the Hertz model:

    - The sample is isotropic.
    - The sample is a linear elastic solid.
    - The sample is extended infinitely in one half space.
    - The indenter is not deformable.
    - There are no additional interactions between sample and indenter.

    Additional assumptions:

    - no surface forces
    - If the indenter is spherical, then its radius :math:`R` is much
      larger than the indentation depth :math:`\delta`.

    References
    ----------
    Sneddon (1965) :cite:`Sneddon1965` (equation 6.9),
    Theory of Elasticity by Landau and Lifshitz (1959)
    :cite:`LandauLifshitz` (§9 Solid bodies in contact, equation 9.14)
    """
    aa = 4/3 * E/(1-nu**2)*np.sqrt(R)
    root = contact_point-delta
    pos = root > 0
    bb = np.zeros_like(delta)
    bb[pos] = (root[pos])**(3/2)
    return aa*bb + baseline


model_doc = hertz_paraboloidal.__doc__
model_func = hertz_paraboloidal
model_key = "hertz_para"
model_name = "parabolic indenter (Hertz)"
parameter_keys = ["E", "R", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Tip Radius",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "m", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
