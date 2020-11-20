import numpy as np


def get_default_residuals_wrapper(model_function):
    """Return a wrapper around the default nanite residual function"""
    default_modeling_wrapper = get_default_modeling_wrapper(model_function)

    def default_residuals_wrapper(params, delta, force, weight_cp=5e-7):
        return residual(params=params,
                        delta=delta,
                        force=force,
                        model=default_modeling_wrapper,
                        weight_cp=weight_cp)

    return default_residuals_wrapper


def get_default_modeling_wrapper(model_function):
    """Return a wrapper around the default nanite modeling function"""
    def default_modeling_wrapper(params, delta):
        return model_direction_agnostic(model_function=model_function,
                                        params=params,
                                        delta=delta)
    return default_modeling_wrapper


def model_direction_agnostic(model_function, params, delta):
    if delta[0] < delta[-1]:
        revert = True
    else:
        revert = False
    if revert:
        delta = delta[::-1]

    mf = model_function(delta=delta, **params.valuesdict())

    if revert:
        return mf[::-1]
    else:
        return mf


def residual(params, delta, force, model, weight_cp=5e-7):
    """Compute residuals for fitting

    Parameters
    ----------
    params: lmfit.Parameters
        The fitting parameters for `model`
    delta: 1D ndarray of lenght M
        The indentation distances
    force: 1D ndarray of length M
        The corresponding force data
    model: callable
        A model function accepting the arguments ``params``
        and ``delta``
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
        weights = compute_contact_point_weights(
            cp=params["contact_point"].value,
            delta=delta,
            weight_dist=weight_cp)
        resid *= weights
    return resid


def compute_contact_point_weights(cp, delta, weight_dist=5e-7):
    """Compute contact point weights

    Parameters
    ----------
    cp: float
        Fitted contact point value
    delta: 1d ndarray of length N
        The indentation array along which weights will be computed.
    weight_width: float
        The distance from `cp` until which weights will be applied.

    Returns
    -------
    weights: 1d ndarray of length N
        The weights.

    Notes
    -----
    All variables should be given in the same units. The weights increase
    linearly from increasing distances of `delta-cp` from 0 to 1 and are
    1 outside of the weight width `abs(delta-cp)>weight_width`.
    """
    # weights are proportional to distance from contact point
    # normalized by weight_width.
    x = np.abs(delta-cp)
    x /= weight_dist
    x[x > 1] = 1
    return x
