#cython: language_level=3
import lmfit
import numpy as np
from . import weight
cimport numpy as np


def get_parameter_defaults():
    # The order of the parameters must match the order 
    # of ´parameter_names´ and ´parameter_keys´.
    params = lmfit.Parameters()
    params.add("E", value=3e3, min=0)
    params.add("R", value=10e-6, min=0, vary=False)
    params.add("nu", value=.5, min=0, max=0.5, vary=False)
    params.add("contact_point", value=0)
    params.add("baseline", value=0)
    return params


def hertz_spherical(double E, delta, double R, double nu, double contact_point=0, double baseline=0):
    r"""Hertz model for Spherical indenter - modified by Sneddon


    .. math::

        F &= \frac{E}{1-\nu^2} \left( \frac{R^2+a^2}{2} \ln \! \left(
             \frac{R+a}{R-a}\right) -aR  \right)\\
        \delta &= \frac{a}{2} \ln
             \! \left(\frac{R+a}{R-a}\right)

    (:math:`a` is the radius of the circular contact area between bead
    and sample.)

    Parameters
    ----------
    E: float
        Young's modulus [N/m²]
    delta: 1d ndarray
        Indentation [m]
    R: float
        Tip radius [m]
    nu: float
        Poisson's ratio
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
    - The sample is a linear elastic solid.
    - The sample is extended infinitely in one half space.
    - The indenter is not deformable.
    - There are no additional interactions between sample and indenter.

    Additional assumptions:

    - no surface forces

    References
    ----------
    Sneddon (1965) :cite:`Sneddon1965`
    """
    cdef double a
    cdef int ii
    FF = np.zeros_like(delta)
    root = delta-contact_point
    root[root >0]=0
    root=-1*root
    for ii in range(root.shape[0]):
        if root[ii]==0:
            FF[ii]=0
        else:
            a=_get_a(R, root[ii])
            FF[ii]=E/(1-nu**2)*((R**2+a**2)/2*np.log((R+a)/(R-a))-a*R) 
    return FF + baseline


cdef double _get_a(double R, double delta, double accuracy=1e-12):
    cdef double a_left, a_center, a_right, d_delta, delta_predict
    a_left=0
    a_center=0.5*R
    a_right=R
    d_delta=1e200 #np.inf
    while d_delta>accuracy:
        delta_predict=_delta_of_a(a_center, R)
        if delta_predict>delta:
            a_left,a_right=a_left,a_center
            a_center=a_left+(a_right-a_left)/2
        elif delta_predict<delta:
            a_left,a_right=a_center,a_right
            a_center=a_left+(a_right-a_left)/2
        else:
            break
        d_delta=abs((delta_predict-delta)/delta)
    return a_center


def get_a(R, delta, accuracy=1e-12):
    """Compute the contact area radius (wrapper)"""
    return _get_a(R, delta, accuracy)


cdef double _delta_of_a(double a, double R):
    cdef double delta
    delta=a/2*np.log((R+a)/(R-a))
    return delta


def delta_of_a(a, R):
    """Compute indentation from contact area radius (wrapper)"""
    return _delta_of_a(a, R)


def model(params, x):
    if x[0]<x[-1]:
        revert = True
    else:
        revert = False
    if revert:
        x = x[::-1]
    mf = hertz_spherical(E=params["E"].value,
                         delta=x,
                         R=params["R"].value,
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

model_doc = hertz_spherical.__doc__
model_key = "sneddon_spher"
model_name = "spherical indenter (Sneddon)"
parameter_keys = ["E", "R", "nu", "contact_point", "baseline"]
parameter_names = ["Young's Modulus", "Tip Radius",
                   "Poisson's Ratio", "Contact Point", "Force Baseline"]
parameter_units = ["Pa", "m", "", "m", "N"]
valid_axes_x = ["tip position"]
valid_axes_y = ["force"]
