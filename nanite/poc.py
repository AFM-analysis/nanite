"""Methods for estimating the point of contact (POC)"""
import lmfit
import numpy as np
from scipy.ndimage.filters import uniform_filter1d


#: List of all methods available for contact point estimation
POC_METHODS = []


def compute_preproc_clip_approach(force):
    """Clip the approach part (discard the retract part)

    This POC preprocessing method may be applied before
    applying the POC estimation method.
    """
    # get data
    fg0 = np.array(force, copy=True)
    # Only use the (initial) approach part of the curve.
    idmax = np.argmax(fg0)
    fg = fg0[:idmax]
    return fg


def compute_poc(force, method="deviation_from_baseline"):
    """Compute the contact point from force data

    If the POC method returns np.nan, then the center of the
    force data is used.
    """
    # compute POC according to method chosen
    for mfunc in POC_METHODS:
        if mfunc.identifier == method:
            if "clip_approach" in mfunc.preprocessing:
                force = compute_preproc_clip_approach(force)
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
        may contain ["clip_approach"].
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
     preprocessing=["clip_approach"])
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
        # argmax gets the first largest value
        maxid = np.argmax(bl_dev)
        if bl_dev[maxid]:
            cp = maxid
    return cp


@poc(identifier="fit_constant_line",
     name="Piecewise fit with constant and line",
     preprocessing=["clip_approach"])
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
        curve = np.maximum(one, two)
        return data - curve

    y = np.array(force, copy=True)
    cp = np.nan
    if y.size > 4:  # 3 fit parameters
        ymin, ymax = np.min(y), np.max(y)
        y = (y - ymin) / (ymax - ymin)
        x = np.arange(y.size)
        x0 = poc_deviation_from_baseline(force)
        if np.isnan(x0):
            x0 = y.size // 2
        params = lmfit.Parameters()
        params.add('off', value=np.mean(y[:10]))
        params.add('x0', value=x0)
        params.add('m', value=1)

        out = lmfit.minimize(residual, params, args=(x, y))
        if out.success:
            cp = int(out.params["x0"])
    return cp


@poc(identifier="gradient_zero_crossing",
     name="Gradient zero-crossing of indentation part",
     preprocessing=["clip_approach"])
def poc_gradient_zero_crossing(force):
    """Gradient zero-crossing of indentation part

    1. Apply a moving average filter to the curve
    2. Compute the gradient
    3. Cut off gradient at maxiumum with a 10 point reserve
    4. Apply a moving average filter to the gradient
    5. The POC is the index of the averaged gradient curve where
       the values are below 1% of the gradient maximum, measured
       from the indentation maxium (not from baseline).
    """
    cp = np.nan
    # Perform a median filter to smooth the array
    filtsize = max(5, int(force.size*.01))
    y = uniform_filter1d(force, size=filtsize)
    if y.size > 1:
        # Cutoff at maximum plus some reserve
        cutoff = y.size - np.argmax(y) + 10
        grad = np.gradient(y)[:-cutoff]
        if grad.size > 50:
            # Use the point where the gradient becomes small enough.
            gradn = uniform_filter1d(grad, size=filtsize)
            gradpos = gradn <= 0.01 * np.max(gradn)
            if np.sum(gradpos):
                # The gradient contains positive values.
                # Flip `gradpos`, because we want the first value from the
                # end of the array.
                # Weight with two times "filtsize//2", because we actually
                # want the rolling median filter from the edge and not at the
                # center of the array (and two times, because we did two
                # filter operations).
                cp = y.size - np.where(gradpos[::-1])[0][0] - cutoff + filtsize
    return cp
