import lmfit
import numpy as np


def get_parameter_defaults():
    """Return the default model parameters"""
    # The order of the parameters must match the order
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E_S", value=3e3, min=0)
    params.add("E_L", value=20, min=0, max=1000)
    params.add("R", value=10e-6, min=0, vary=False)
    params.add("nu_S", value=.3, min=0, max=0.5, vary=False)
    params.add("nu_L", value=.3, min=0, max=0.5, vary=False)
    params.add("t", value=0.1e-6, min=1e-12)
    params.add("contact_point", value=0)
    params.add("baseline", value=0)
    return params


def power_layer_clifford_2009(delta, E_S, E_L, R, nu_S, nu_L, t,
                              contact_point=0, baseline=0):
    r"""Power law model including one elastic layer covering the sample

    .. math::

        F &= \frac{4}{3}
            E^*
            \sqrt{R}
            \delta^{3/2}

        E^* &= E_\mathrm{L}
              + (E_\mathrm{S} - E_\mathrm{L})
              \frac{P \xi^n}{1 + P \xi^n}

        \xi &= \frac{\sqrt{R\delta}}{t}
               \left(
               \frac{E_\mathrm{L}}{E_\mathrm{S}}
               \right)^m
               \frac{1 - B_\mathrm{S}\nu_\mathrm{S}^2}
               {1 - B_\mathrm{L}\nu_\mathrm{L}^2}

    with the fixed constants determined from the finite element fits
    :math:`P=2.25`, :math:`n=1.5`, :math:`m=2/3`,
    :math:`B_\mathrm{S}=0.22`, and :math:`B_\mathrm{L}=1.92`.


    Parameters
    ----------
    delta: 1d ndarray
        Indentation [m]
    E_S: float
        Young's modulus of the sample [N/m²]
    E_L: float
        Young's modulus of the layer [N/m²]
    R: float
        Tip radius [m]
    nu_S: float
        Poisson's ratio of the sample
    nu_L: float
        Poisson's ratio of the layer
    t: float
        Thickness of the layer covering the sample [m]
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
    The power law model was derived by fitting finite element
    model simulations for materials in the ~100 GPa range
    and a layer thickness in the range of ~1 µm.

    Neither for :math:`t \rightarrow 0`
    nor for :math:`E_\mathrm{L} \rightarrow E_\mathrm{S}`
    will the power law model converge to the paraboloidal hertz model.

    References
    ----------
    Clifford (2009) :cite:`Clifford2009` (equations 9 and 10)
    """
    # Note that this code is optimized for readability, not for speed.
    # roots of delta
    root = contact_point - delta
    pos = root > 0
    dr12 = np.zeros_like(delta)
    dr12[pos] = np.sqrt(root[pos])
    dr32 = np.zeros_like(delta)
    dr32[pos] = root[pos]**(3/2)
    # constants
    P = 2.25
    n = 1.5
    m = 2/3
    B_S = 0.22
    B_L = 1.92
    # inner term (equation 9)
    a = np.sqrt(R) * dr12
    xi = (a / t
          * (E_L/E_S)**m
          * (1 - B_S * nu_S**2) / (1 - B_L * nu_L**2)
          )
    # outer term for emodulus (equation 10)
    E = E_L + (E_S - E_L) * (P * xi**n) / (1 + P * xi**n)
    # original "hertz" first term
    hertz = 4/3 * E * np.sqrt(R) * dr32
    return hertz + baseline


model_doc = power_layer_clifford_2009.__doc__
model_func = power_layer_clifford_2009
model_key = "power_layer_clifford_2009"
model_name = "elastic layer power law (Clifford 2009)"
parameter_keys = ["E_S", "E_L", "R", "nu_S", "nu_L", "t",
                  "contact_point", "baseline"]
parameter_names = ["Sample Young's Modulus", "Layer Young's Modulus",
                   "Tip Radius", "Sample Poisson's Ratio",
                   "Layer Poisson's Ratio", "Layer Thickness",
                   "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "Pa", "m", "", "", "m", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
