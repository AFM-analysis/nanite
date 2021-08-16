import copy
import inspect
import functools
import warnings

import numpy as np

from . import poc
from .smooth import smooth_axis_monotone


#: Available preprocessors
PREPROCESSORS = []


class CannotSplitWarning(UserWarning):
    pass


def apply(apret, identifiers=None, options=None, ret_details=False,
          preproc_names=None):
    """Perform force-distance preprocessing steps

    Parameters
    ----------
    apret: nanite.Indentation
        The afm data to preprocess
    identifiers: list
        A list of preprocessing identifiers that will be
        applied (in the order given).
    options: dict of dict
        Preprocessing options for each identifier
    ret_details:
        Return preprocessing details dictionary
    preproc_names: list
        Deprecated - use `identifiers` instead
    """
    if preproc_names is not None:
        identifiers = preproc_names
        warnings.warn(
            "Please use 'identifiers' instead of 'preproc_names'!",
            DeprecationWarning)
    details = {}
    # Reset all user-defined data of the dataset, because we
    # probably edited "tip position", "force", etc. If this
    # is too slow in the future, then we would have to remember
    # the previous preprocessing pipeline (such code is currently
    # implemented on the other side in `indent.py` (2021-08-16).
    apret.reset_data()
    for ii, pid in enumerate(identifiers):
        if pid in available():
            meth = get_func(pid)
            req = meth.steps_required
            act = identifiers[:ii]
            if req is not None and ((set(req) & set(act)) != set(req)):
                raise ValueError(f"The preprocessing step '{pid}' requires"
                                 f" the steps {meth.steps_required}!")
            # create a copy of the dictionary (if it exists) so that
            # `ret_details` is not written to it
            kwargs = copy.deepcopy(options.get(pid, {}))
            if "ret_details" in inspect.signature(meth).parameters:
                # only set `ret_details` if method accepts it
                kwargs["ret_details"] = ret_details
            details[pid] = meth(apret, **kwargs)
        else:
            msg = "The preprocessing method '{}' does not exist!"
            raise KeyError(msg.format(pid))
    # only return details if required
    return details if ret_details else None


def autosort(identifiers):
    """Automatically sort preprocessing identifiers

    This takes into account `steps_required` and `steps_optional`.
    """
    sorted_identifiers = copy.copy(identifiers)
    for pid in identifiers:
        meth = get_func(pid)
        steps_precursor = []
        if meth.steps_required is not None:
            steps_precursor += meth.steps_required
        if meth.steps_optional is not None:
            for ostep in meth.steps_optional:
                if ostep in identifiers:
                    steps_precursor.append(ostep)
        for step in steps_precursor:
            # We have a requirement, check whether it is fulfilled
            cix = sorted_identifiers.index(pid)
            rix = sorted_identifiers.index(step)
            if rix > cix:
                # We pop the wrong requirement and insert it before
                # the current pid.
                sorted_identifiers.remove(step)
                sorted_identifiers.insert(cix, step)

    # Perform a sanity check
    check_order(sorted_identifiers)

    return sorted_identifiers


@functools.lru_cache()
def available():
    """Return list of available preprocessor identifiers"""
    av = [pp.identifier for pp in PREPROCESSORS]
    return autosort(av)


def check_order(identifiers):
    """Check preprocessing steps for correct order"""
    for cix, pid in enumerate(identifiers):
        meth = get_func(pid)
        if meth.steps_required:
            rix = [identifiers.index(r) for r in meth.steps_required]
            if np.any(np.array(rix) > cix):
                raise ValueError(
                    f"Wrong required step order for {pid}: {identifiers}!")
        if meth.steps_optional:
            rio = []
            for rr in meth.steps_optional:
                if rr in identifiers:
                    rio.append(identifiers.index(rr))
            if np.any(np.array(rio) > cix):
                raise ValueError(
                    f"Wrong optional step order for {pid}: {identifiers}!")


def get_func(identifier):
    """Return preprocessor function for identifier"""
    for func in PREPROCESSORS:
        if func.identifier == identifier:
            return func
    else:
        raise KeyError(f"Preprocessor '{identifier}' unknown!")


def get_name(identifier):
    """Return preprocessor name for identifier"""
    func = get_func(identifier)
    return func.name


def get_steps_required(identifier):
    """Return requirement identifiers for identifier"""
    func = get_func(identifier)
    return func.steps_required


def preprocessing_step(identifier, name, steps_required=None,
                       steps_optional=None, options=None):
    """Decorator for Indentation preprocessors

    The name and identifier are stored as a property of the wrapped
    function.

    Parameters
    ----------
    identifier: str
        identifier of the preprocessor (e.g. "correct_tip_offset")
    name: str
        human-readble name of the preprocessor
        (e.g. "Estimate contact point")
    steps_required: list of str
        list of preprocessing steps that must be added before this
        step
    steps_optional: list of str
        unlike `steps_required`, these steps do not have to be set,
        but if they are set, they should come before this step
    options: list of dict
        if the preprocessor accepts optional keyword arguments,
        this list yields valid values or dtypes
    """

    def attribute_setter(func):
        """Decorator that sets the necessary attributes

        The outer decorator is used to obtain the attributes.
        This inner decorator returns the actual function that
        wraps the preprocessor.
        """
        func.identifier = identifier
        assert isinstance(name, str)
        func.name = name
        func.options = options
        func.steps_required = steps_required
        func.steps_optional = steps_optional
        PREPROCESSORS.append(func)
        return func

    return attribute_setter


@preprocessing_step(identifier="compute_tip_position",
                    name="tip-sample separation")
def preproc_compute_tip_position(apret):
    """Perform tip-sample separation

    Populate the "tip position" column by adding the force
    normalized by the spring constant to the cantilever
    height ("height (measured)").

    This computation correctly reproduces the column
    "Vertical Tip Position" as it is exported by the
    JPK analysis software with the checked option
    "Use Unsmoothed Height".
    """
    has_hm = "height (measured)" in apret
    has_fo = "force" in apret
    has_sc = "spring constant" in apret.metadata
    if "tip position" in apret.columns_innate:
        # nothing to do
        pass
    elif has_hm and has_fo and has_sc:
        k = apret.metadata["spring constant"]
        force = apret["force"]
        zcant = apret["height (measured)"]
        apret["tip position"] = zcant + force / k
    else:
        missing = []
        if not has_hm:
            missing.append("missing data column 'height (measured)'")
        if not has_fo:
            missing.append("missing data column 'force'")
        if not has_sc:
            missing.append("missing metadata 'spring constant'")
        mt = ", ".join(missing)
        raise ValueError("Cannot compute tip position: {}".format(mt))


@preprocessing_step(identifier="correct_force_offset",
                    name="baseline correction")
def preproc_correct_force_offset(apret):
    """Correct the force offset with an average baseline value
    """
    idp = poc.compute_poc(force=apret["force"],
                          method="deviation_from_baseline")
    if idp:
        apret["force"] = apret["force"] - np.average(apret["force"][:idp])
    else:
        apret["force"] = apret["force"] - apret["force"][0]


@preprocessing_step(
    identifier="correct_tip_offset",
    name="contact point estimation",
    steps_required=["compute_tip_position"],
    options=[
        {"name": "method",
         "type": str,
         "choices": [p.identifier for p in poc.POC_METHODS],
         "choices_human_readable": [p.name for p in poc.POC_METHODS]}
    ]
)
def preproc_correct_tip_offset(apret, method="deviation_from_baseline",
                               ret_details=False):
    """Estimate the point of contact

    An estimate of the contact point is subtracted from the
    tip position.
    """
    data = poc.compute_poc(force=apret["force"],
                           method=method,
                           ret_details=ret_details)
    if ret_details:
        cpid, details = data
    else:
        cpid, details = data, None
    apret["tip position"] = (apret["tip position"]
                             - apret["tip position"][cpid])
    return details


@preprocessing_step(identifier="correct_split_approach_retract",
                    name="segment discovery",
                    steps_required=["compute_tip_position"])
def preproc_correct_split_approach_retract(apret):
    """Split the approach and retract curves (farthest point method)

    Approach and retract curves are defined by the microscope. When the
    direction of piezo movement is flipped, the force at the sample tip
    is still increasing. This can be either due to a time lag in the AFM
    system or due to a residual force acting on the sample due to the
    bent cantilever.

    To repair this time lag, we append parts of the retract curve to the
    approach curve, such that the curves are split at the minimum height.
    """
    x = np.array(apret["tip position"], copy=True)
    y = np.array(apret["force"], copy=True)

    idp = poc.poc_deviation_from_baseline(y)
    if idp and not np.isnan(idp):
        # Flip and normalize tip position so that maximum is at minimum
        # z-position (set to 1) which coincides with maximum indentation.
        x -= x[idp]
        x /= x.min()
        x[x < 0] = 0

        # Flip and normalize force so that maximum force is set to 1.
        y -= np.average(y[:idp])
        y /= y.max()
        y[y < np.std(y[:idp])] = 0

        idmin = np.argmax(x ** 2 + y ** 2)

        segment = np.zeros(len(apret), dtype=np.uint8)
        segment[idmin:] = 1
        apret["segment"] = segment
    else:
        msg = "Cannot correct splitting of approach and retract curve " + \
              "because the contact point position could not be estimated."
        warnings.warn(msg, CannotSplitWarning)


@preprocessing_step(identifier="smooth_height",
                    name="monotonic height data",
                    steps_optional=[
                        # Otherwise we lose the location of the point
                        # of deepest indentation:
                        "correct_split_approach_retract",
                        # Otherwise it might not be applied to
                        # "tip position":
                        "compute_tip_position"])
def preproc_smooth_height(apret):
    """Make height data monotonic

    For the columns "height (measured)", "height (piezo), and
    "tip position", this method ensures that the approach and
    retract segments are monotonic.
    """
    orig = ["height (measured)",
            "height (piezo)",
            "tip position"]
    for col in orig:
        if col not in apret:
            continue
        # Apply smoothing
        sm_app = smooth_axis_monotone(apret.appr[col])
        sm_ret = smooth_axis_monotone(apret.retr[col])

        # Replace the column data
        apret.appr[col] = sm_app
        apret.retr[col] = sm_ret


def _deprecate_call(method):
    def wrapper(*args, **kwargs):
        warnings.warn(
            "Using `IndentationPreprocessor` class is deprecated, please use "
            f"the method '{method.__module__}.{method.__name__}' instead!",
            DeprecationWarning)
        return method(*args, **kwargs)
    return wrapper


class IndentationPreprocessor:
    apply = _deprecate_call(apply)
    autosort = _deprecate_call(autosort)
    get_func = _deprecate_call(get_func)
    get_name = _deprecate_call(get_name)
    get_steps_required = _deprecate_call(get_steps_required)
    check_order = _deprecate_call(check_order)
    available = _deprecate_call(available)
