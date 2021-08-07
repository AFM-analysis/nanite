"""Methods for estimating the point of contact (POC)"""
import lmfit
import numpy as np
import scipy.signal as spsig


#: List of all methods available for contact point estimation
POC_METHODS = []


def compute_preproc_clip_approach(force):
    # get data
    fg0 = np.array(force, copy=True)
    # Only use the (initial) approach part of the curve.
    idmax = np.argmax(fg0)
    fg = fg0[:idmax]
    return fg


def compute_preproc_gradient(force):
    """Compute the gradient of the force curve

    This method also removes the tilt in the approach part.

    1. Compute the rolling average of the force
       (Otherwise the gradient would be too wild)
    2. Compute the gradient
       (Converting to gradient space gets rid of linear
       contributions in the approach part)
    3. Compute the rolling average of the gradient
       (Makes the curve to analyze more smooth so that the
      methods below don't hit the alarm too early)
    """
    # apply rolling average filter to force
    p1_fs = min(47, force.size // 2 // 2 * 2 + 1)
    assert p1_fs % 2 == 1, "must be odd"
    p1_cumsum_vec = np.cumsum(np.insert(np.copy(force), 0, 0))
    p1 = (p1_cumsum_vec[p1_fs:] - p1_cumsum_vec[:-p1_fs]) / p1_fs
    # take the gradient
    if p1.size > 1:
        p1g = np.gradient(p1)
        # apply rolling average filter to the gradient
        p1g_cumsum_vec = np.cumsum(np.insert(np.copy(p1g), 0, 0))
        p1gm = (p1g_cumsum_vec[p1_fs:] - p1g_cumsum_vec[:-p1_fs]) / p1_fs
    else:
        # fallback for bad data (array with very few elements)
        p1gm = p1
    return p1gm


def compute_poc(force, method="scheme_2020"):
    """Compute the contact point from force data

    If the POC method returns np.nan, then the center of the
    force data is used.
    """
    # compute POC according to method chosen
    for mfunc in POC_METHODS:
        if mfunc.identifier == method:
            if "clip_approach" in mfunc.preprocessing:
                force = compute_preproc_clip_approach(force)
            if "gradient" in mfunc.preprocessing:
                force = compute_preproc_gradient(force)
            cp = mfunc(force)
            break
    else:
        raise ValueError(f"Undefined POC method '{method}'!")
    if np.isnan(cp):
        cp = force.size // 2
    return cp


def poc(identifier, name, preprocessing):
    """Decorator for point of contact (POC) methods

    The name and identifier are stored as a property of the wrapped
    function.

    Parameters
    ----------
    identifier: str
        identifier of the POC method (e.g. "baseline_deviation")
    name: str
        human-readble name of the POC method
        (e.g. "Deviation from baseline")
    preprocessing: list of str
        list of preprocessing methods that should be applied;
        may contain ["gradient", "clip_approach"].
    """
    def attribute_setter(func):
        """Decorator that sets the necessary attributes

        The outer decorator is used to obtain the attributes.
        This inner decorator returns the actual function that
        wraps the preprocessor.
        """
        POC_METHODS.append(func)
        func.identifier = identifier
        func.name = name
        func.preprocessing = preprocessing
        return func

    return attribute_setter


@poc(identifier="deviation_from_baseline",
     name="Deviation from baseline",
     preprocessing=["gradient", "clip_approach"])
def poc_deviation_from_baseline(force):
    """Deviation from baseline

    1. Obtain the baseline (initial 10% of the gradient curve)
    2. Compute average and maximum deviation of the baseline
    3. The CP is the index of the curve where it exceeds
       twice of the maximum deviation
    """
    cp = np.nan
    # Crop the slow approach trace (10% of the curve)
    baseline = force[:int(force.size * .1)]
    if baseline.size:
        bl_avg = np.average(baseline)
        bl_rng = np.max(np.abs(baseline - bl_avg)) * 2
        bl_dev = (force - bl_avg) > bl_rng
        if np.sum(bl_dev):
            cp = np.where(bl_dev)[0][0]
    return cp


@poc(identifier="fit_constant_line",
     name="Piecewise fit with constant and line",
     preprocessing=["gradient", "clip_approach"])
def poc_fit_constant_line(force):
    """Piecewise fit with constant and line

    Fit a piecewise function (constant+linear) to the baseline
    and indentation part.

    The point of contact is the intersection of a horizontal line
    (constant) for the baseline and a linear function (constant slope)
    for the indentation part.
    """
    def residual(params, x, data):
        off = params["off"]
        x0 = params["x0"]
        m = params["m"]
        one = off
        two = m * (x - x0) + off
        return data - np.maximum(one, two)

    cp = np.nan
    if force.size > 4:  # 3 fit parameters
        x = np.arange(force.size)

        params = lmfit.Parameters()
        params.add('off', value=np.mean(force[:10]))
        params.add('x0', value=force.size // 2)
        params.add('m', value=(force.max() - force.min()) / force.size)

        out = lmfit.minimize(residual, params, args=(x, force))
        if out.success:
            cp = int(out.params["x0"])

    return cp


@poc(identifier="gradient_zero_crossing",
     name="Gradient zero-crossing of indentation part",
     preprocessing=["gradient", "clip_approach"])
def poc_gradient_zero_crossing(force):
    """Gradient zero-crossing of indentation part

    1. Apply a median filter to the curve
    2. Compute the gradient
    3. Cut off trailing 10 points from the gradient (noise)
    4. The CP is the index of the gradient curve when the
       sign changes, measured from the point of maximal
       indentation.
    """
    cp = np.nan
    # Perform a median filter to smooth the array
    filtsize = 15
    y = spsig.medfilt(force, filtsize)
    # Cut off the trailing 10 points (noise)
    cutoff = 10
    if y.size > cutoff + 1:
        grad = np.gradient(y)[:-cutoff]
        # Use the point where the gradient becomes positive for the
        # first time.
        gradpos = grad > 0
        if np.sum(gradpos):
            # The contains positive values.
            # Flip `gradpos`, because we want the first value from the
            # end of the array.
            cp = y.size - np.where(gradpos[::-1])[0][0] - cutoff - 1
    return cp


@poc(identifier="scheme_2020",
     name="Heuristic analysis pipeline 2020",
     preprocessing=["gradient", "clip_approach"])
def poc_scheme_2020(force):
    """Heuristic analysis pipeline 2020

    This pipeline was first implemented in nanite 1.6.1 in the
    year of 2020.

    Preprocessing:

    1. Compute the rolling average of the force
       (Otherwise the gradient would be too wild)
    2. Compute the gradient
       (Converting to gradient space gets rid of linear
       contributions in the approach part)
    3. Compute the rolling average of the gradient
       (Makes the curve to analyze more smooth so that the
       methods below don't hit the alarm too early)

    Method 1: baseline deviation

    1. Obtain the baseline (initial 10% of the gradient curve)
    2. Compute average and maximum deviation of the baseline
    3. The CP is the index of the curve where it exceeds
       twice of the maximum deviation

    Method 2: sign of gradient

    1. Apply a median filter to the approach curve
    2. Compute the gradient
    3. Cut off trailing 10 points from the gradient (noise)
    4. The CP is the index of the gradient curve when the
       sign changes, measured from the point of maximal
       indentation.

    If one of the methods fail, then a combined constant+linear
    function (max(constant, linear) is fitted to the gradient to
    determine the contact point.
    """
    cp1 = poc_deviation_from_baseline(np.array(force, copy=True))
    cp2 = poc_gradient_zero_crossing(np.array(force, copy=True))

    if np.isnan(cp1) or np.isnan(cp2):
        cp = poc_fit_constant_line(force)
    else:
        cp = min(cp1, cp2)
    return cp
