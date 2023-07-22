"""Methods for estimating the point of contact (POC)"""
import lmfit
import numpy as np
from scipy.ndimage import uniform_filter1d


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


def compute_poc(force, method="deviation_from_baseline", ret_details=False):
    """Compute the contact point from force data

    Parameters
    ----------
    force: 1d ndarray
        Force data
    method: str
        Name of the method for computing the POC (see :const:`POC_METHODS`)
    ret_details: bool
        Whether or not to return a dictionary with details alongside the
        POC estimate.

    Notes
    -----
    If the POC method returns np.nan, then the center of the
    force data is returned (to allow fitting algorithms to proceed).
    """
    # compute POC according to method chosen
    for mfunc in POC_METHODS:
        if mfunc.identifier == method:
            if "clip_approach" in mfunc.preprocessing:
                force = compute_preproc_clip_approach(force)
            data = mfunc(force, ret_details=ret_details)
            if ret_details:
                cp, details = data
                details["method"] = method
            else:
                cp, details = data, None
            break
    else:
        raise ValueError(f"Undefined POC method '{method}'!")
    if np.isnan(cp):
        cp = force.size // 2
    if ret_details:
        return cp, details
    else:
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
def poc_deviation_from_baseline(force, ret_details=False):
    """Deviation from baseline

    1. Obtain the baseline (initial 10% of the gradient curve)
    2. Compute average and maximum deviation of the baseline
    3. The CP is the index of the curve where it exceeds
       twice of the maximum deviation
    """
    cp = np.nan
    details = {}
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
        if ret_details:
            x = [0, force.size-1]
            details["plot force"] = [np.arange(force.size), force]
            details["plot baseline mean"] = [x, [bl_avg, bl_avg]]
            details["plot baseline threshold"] = [x, [bl_avg + bl_rng,
                                                      bl_avg + bl_rng]]
            details["plot poc"] = [[cp, cp],
                                   [force.min(), force.max()]]
            details["norm"] = "force"

    if ret_details:
        return cp, details
    else:
        return cp


@poc(identifier="fit_constant_line",
     name="Piecewise fit with constant and line",
     preprocessing=["clip_approach"])
def poc_fit_constant_line(force, ret_details=False):
    r"""Piecewise fit with constant and line

    Fit a piecewise function (constant+linear) to the baseline
    and indentation part:

    .. math::

       F = \text{max}(d, m\delta + d)

    The point of contact is the intersection of a horizontal line
    at :math:`d` (baseline) and a linear function with slope :math:`m`
    for the indentation part.

    The point of contact is defined as :math:`\delta=0` (It's another
    fitting parameter).
    """
    def model(params, x):
        d = params["d"]
        x0 = params["x0"]
        m = params["m"]
        one = d
        two = m * (x - x0) + d
        return np.maximum(one, two)

    def residual(params, x, data):
        return data - model(params, x)

    cp = np.nan
    details = {}
    if force.size > 4:  # 3 fit parameters
        # normalize force
        fmin = np.min(force)
        fptp = np.max(force) - fmin
        y = (force - fmin) / fptp
        x = np.arange(y.size)
        # get estimate for cp
        x0 = poc_frechet_direct_path(force)
        if np.isnan(x0):
            x0 = y.size // 2
        params = lmfit.Parameters()
        params.add('d', value=np.mean(y[:10]))
        params.add('x0', value=x0)
        params.add('m', value=(1 - y[x0])/(x.size - x0))

        out = lmfit.minimize(residual, params, args=(x, y), method="nelder")
        if out.success:
            cp = int(out.params["x0"])
            if ret_details:
                details["plot force"] = [x, force]
                details["plot fit"] = [np.arange(force.size),
                                       model(out.params, x) * fptp + fmin]
                details["plot poc"] = [[cp, cp],
                                       [fmin, fmin + fptp]]
                details["norm"] = "force"

    if ret_details:
        return cp, details
    else:
        return cp


@poc(identifier="fit_constant_polynomial",
     name="Piecewise fit with constant and polynomial",
     preprocessing=["clip_approach"])
def poc_fit_constant_polynomial(force, ret_details=False):
    r"""Piecewise fit with constant and polynomial

    Fit a piecewise function (constant + polynomial) to the baseline
    and indentation part.

    .. math::

       F = \frac{\delta^3}{a\delta^2 + b\delta + c} + d

    This function is defined for all :math:`\delta>0`. For all
    :math:`\delta<0` the model evaluates to :math:`d` (baseline).

    I'm not sure where this has been described initially, but it is
    used e.g. in :cite:`Rusaczonek19`.

    For small indentations, this function exhibits a cubic behavior:

    .. math::

       y \approx \delta^3/c + d

    And for large indentations, this function is linear:

    .. math::

       y \approx \delta/a

    The point of contact is defined as :math:`\delta=0` (It's another
    fitting parameter).
    """
    def model(params, x):
        d = params["d"].value
        x0 = params["x0"].value
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        x1 = x - x0
        curve = x1**3 / (a*x1**2 + b*x1 + c) + d
        curve[x1 <= 0] = d
        return curve

    def residual(params, x, data):
        curve = model(params, x)
        return data - curve

    cp = np.nan
    details = {}
    if force.size > 6:  # 5 fit parameters
        fmin = np.min(force)
        fptp = np.max(force) - fmin
        y = (force - fmin) / fptp
        x = np.arange(y.size)
        x0 = poc_frechet_direct_path(force)
        if np.isnan(x0):
            x0 = y.size // 2
        params = lmfit.Parameters()
        params.add('d', value=np.mean(y[:10]))
        params.add('x0', value=x0)
        # The polynomial fitting parameters are supposed to be
        # greater than zero (source?). We set the minimum to 1e-3 so
        # the fitting algorithm becomes more stable. Also, the initial
        # values for b and c are more or less arbitrary (this is a heuristic
        # approach).
        # for larger x, a is something like an inverse slope. Since we
        # normalized the y-values to 1, we just take the x-difference.
        params.add('a', value=(y.size-x0), min=1e-3, max=100*(y.size-x0))
        params.add('b', value=y.size, min=1e-3)
        params.add('c', value=.5, min=1e-3)

        out = lmfit.minimize(residual, params, args=(x, y), method="nelder")

        if out.success:
            cp = int(out.params["x0"])
            if ret_details:
                details["plot force"] = [x, force]
                details["plot fit"] = [np.arange(force.size),
                                       model(out.params, x) * fptp + fmin]
                details["plot poc"] = [[cp, cp],
                                       [fmin, fmin + fptp]]
                details["norm"] = "force"

    if ret_details:
        return cp, details
    else:
        return cp


@poc(identifier="fit_line_polynomial",
     name="Piecewise fit with line and polynomial",
     preprocessing=["clip_approach"])
def poc_fit_line_polynomial(force, ret_details=False):
    r"""Piecewise fit with line and polynomial

    Fit a piecewise function (line + polynomial) to the baseline
    and indentation part.

    The linear basline (:math:`\delta<0`) is modeled with:

    .. math::

       F = m \delta + d

    The indentation part (:math:`\delta>0`) is modeled with:

    .. math::

       F = \frac{\delta^3}{a\delta^2 + b\delta + c} + m \delta  + d

    For small indentations, this function exhibits a linear and
    only slightly cubic behavior:

    .. math::

       y \approx \delta^3/c + m \delta + d

    And for large indentations, this function is linear:

    .. math::

       y \approx \left( \frac{1}{a} + m \right) \delta

    The point of contact is defined as :math:`\delta=0` (It's another
    fitting parameter).


    See Also
    --------
    poc_fit_constant_polynomial: polynomial-only version
    """
    def model(params, x):
        d = params["d"].value
        x0 = params["x0"].value
        m = params["m"].value
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        x1 = x - x0
        curve = m * x1 + d
        curve[x1 > 0] += x1[x1 > 0]**3 / (a*x1[x1 > 0]**2 + b*x1[x1 > 0] + c)
        return curve

    def residual(params, x, data):
        curve = model(params, x)
        return data - curve

    cp = np.nan
    details = {}
    if force.size > 7:  # 6 fit parameters
        fmin = np.min(force)
        fptp = np.max(force) - fmin
        y = (force - fmin) / fptp
        x = np.arange(y.size)
        x0 = poc_frechet_direct_path(force)
        if np.isnan(x0):
            x0 = y.size // 2
        params = lmfit.Parameters()
        params.add('d', value=np.mean(y[:10]))
        params.add('x0', value=x0)
        # slope
        params.add('m', value=y[x0]/x0)
        # The polynomial fitting parameters are supposed to be
        # greater than zero (source?). We set the minimum to 1e-3 so
        # the fitting algorithm becomes more stable. Also, the initial
        # values for b and c are more or less arbitrary (this is a heuristic
        # approach).
        # for larger x, a is something like an inverse slope. Since we
        # normalized the y-values to 1, we just take the x-difference.
        params.add('a', value=(y.size-x0), min=1e-3, max=100*(y.size-x0))
        params.add('b', value=y.size, min=1e-3)
        params.add('c', value=.5, min=1e-3)

        out = lmfit.minimize(residual, params, args=(x, y), method="nelder")

        if out.success:
            cp = int(out.params["x0"])
            if ret_details:
                details["plot force"] = [x, force]
                details["plot fit"] = [np.arange(force.size),
                                       model(out.params, x) * fptp + fmin]
                details["plot poc"] = [[cp, cp],
                                       [fmin, fmin + fptp]]
                details["norm"] = "force"

    if ret_details:
        return cp, details
    else:
        return cp


@poc(identifier="frechet_direct_path",
     name="Fréchet distance to direct path",
     preprocessing=["clip_approach"])
def poc_frechet_direct_path(force, ret_details=False):
    """Fréchet distance to direct path

    The indentation part is transformed to normalized coordinates
    (force and corresponding x in range [0, 1]). The point with the
    largest distance to the line from (0, 0) to (1, 1) is the contact
    point.

    This method is robust with regard to tilted baselines and is a
    good initial guess for fitting-based POC estimation approaches.

    Note that the length of the baseline influences the returned
    contact point. For shorter baselines, the contact point will
    be closer to the point of maximum indentation.
    """
    x = np.linspace(0, 1, len(force), endpoint=True)
    y = (force - force.min()) / (force.max() - force.min())

    # rotate the curve towards x
    # (computation of Frechet distance with curve is now just distance
    # from x-axis, i.e. minimum)
    alpha = - np.pi / 4
    yr = x * np.sin(alpha) + y * np.cos(alpha)
    cp = np.argmin(yr)

    if ret_details:
        details = {"plot normalized rotated force": [np.arange(len(force)),
                                                     yr],
                   "plot poc": [[cp, cp],
                                [yr.min(), yr.max()]],
                   "norm": "force-rotated",
                   }
        return cp, details
    else:
        return cp


@poc(identifier="gradient_zero_crossing",
     name="Gradient zero-crossing of indentation part",
     preprocessing=["clip_approach"])
def poc_gradient_zero_crossing(force, ret_details=False):
    """Gradient zero-crossing of indentation part

    1. Apply a moving average filter to the curve
    2. Compute the gradient
    3. Cut off gradient at maximum with a 10 point reserve
    4. Apply a moving average filter to the gradient
    5. The POC is the index of the averaged gradient curve where
       the values are below 1% of the gradient maximum, measured
       from the indentation maximum (not from baseline).
    """
    cp = np.nan
    details = {}
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
            thresh = 0.01 * np.max(gradn)
            gradpos = gradn <= thresh
            if np.sum(gradpos):
                # The gradient contains positive values.
                # Flip `gradpos`, because we want the first value from the
                # end of the array.
                # Weight with two times "filtsize//2", because we actually
                # want the rolling median filter from the edge and not at the
                # center of the array (and two times, because we did two
                # filter operations).
                cp = y.size - np.where(gradpos[::-1])[0][0] - cutoff + filtsize

                if ret_details:
                    # scale the gradient so that it aligns with the force
                    x = np.arange(gradn.size)
                    details["plot force gradient"] = [x, gradn]
                    details["plot threshold"] = [[x[0], x[-1]],
                                                 [thresh, thresh]]
                    details["plot poc"] = [[cp, cp],
                                           [gradn.min(), gradn.max()]]
                    details["norm"] = "force-gradient"

    if ret_details:
        return cp, details
    else:
        return cp
