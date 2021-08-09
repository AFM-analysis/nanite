import copy
import functools
import warnings

import numpy as np

from . import poc
from .smooth import smooth_axis_monotone


class CannotSplitWarning(UserWarning):
    pass


def preprocessing_step(identifier, name, require_steps=None, options=None):
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
    require_steps: list of str
        list of preprocessing steps that must be added before this
        step
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
        func.require_steps = require_steps
        return func

    return attribute_setter


class IndentationPreprocessor(object):
    @staticmethod
    def apply(apret, identifiers=None, options=None, preproc_names=None):
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
        preproc_names: list
            Deprecated - use identifiers instead.

        Notes
        -----
        This method is usually called from within the `Indentation`
        class instance. If you are using this class directly and
        apply it more than once, you might need to call
        `apret.reset()` before preprocessing a second time.
        """
        if preproc_names is not None:
            identifiers = preproc_names
            warnings.warn(
                "Please use 'identifiers' instead of 'preproc_names'!",
                DeprecationWarning)
        for ii, pid in enumerate(identifiers):
            if pid in IndentationPreprocessor.available():
                meth = IndentationPreprocessor.get_func(pid)
                req = meth.require_steps
                act = identifiers[:ii]
                if req is not None and ((set(req) & set(act)) != set(req)):
                    raise ValueError(f"The preprocessing step '{pid}' requires"
                                     f" the steps {meth.require_steps}!")
                kwargs = options.get(pid)
                if kwargs:  # also apply preprocessing options
                    meth(apret, **kwargs)
                else:
                    meth(apret)
            else:
                msg = "The preprocessing method '{}' does not exist!"
                raise KeyError(msg.format(pid))

    @staticmethod
    def autosort(identifiers):
        """Automatically sort preprocessing identifiers via require_steps"""
        sorted_identifiers = copy.copy(identifiers)
        for pid in identifiers:
            meth = IndentationPreprocessor.get_func(pid)
            if meth.require_steps is not None:
                # We have a requirement, check whether it is fulfilled
                cix = sorted_identifiers.index(pid)
                rix = [sorted_identifiers.index(r) for r in meth.require_steps]
                if np.any(np.array(rix) > cix):
                    # We change the order by popping the original cix and
                    # then inserting the step after the largest rix.
                    sorted_identifiers.remove(pid)
                    new_cix = np.max(rix) + 1
                    sorted_identifiers.insert(new_cix, pid)
        return sorted_identifiers

    @staticmethod
    @functools.lru_cache()
    def available():
        """Return list of available preprocessor identifiers"""
        av = []
        for key in dir(IndentationPreprocessor):
            func = getattr(IndentationPreprocessor, key)
            if hasattr(func, "identifier"):
                av.append(func.identifier)
        return sorted(av)

    @staticmethod
    def get_func(identifier):
        """Return preprocessor function for identifier"""
        for key in dir(IndentationPreprocessor):
            func = getattr(IndentationPreprocessor, key)
            if hasattr(func, "identifier") and func.identifier == identifier:
                return func
        else:
            raise KeyError(f"Preprocessor '{identifier}' unknown!")

    @staticmethod
    def get_name(identifier):
        """Return preprocessor name for identifier"""
        func = IndentationPreprocessor.get_func(identifier)
        return func.name

    @staticmethod
    def get_require_steps(identifier):
        """Return requirement identifiers for identifier"""
        func = IndentationPreprocessor.get_func(identifier)
        return func.require_steps

    @staticmethod
    @preprocessing_step(identifier="compute_tip_position",
                        name="tip-sample separation")
    def compute_tip_position(apret):
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
        if "tip position" in apret:
            # nothing to do
            pass
        elif has_hm and has_fo and has_sc:
            k = apret.metadata["spring constant"]
            force = apret["force"]
            zcant = apret["height (measured)"]
            apret["tip position"] = zcant + force/k
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

    @staticmethod
    @preprocessing_step(identifier="correct_force_offset",
                        name="baseline correction")
    def correct_force_offset(apret):
        """Correct the force offset with an average baseline value
        """
        idp = poc.compute_poc(force=apret["force"],
                              method="deviation_from_baseline")
        if idp:
            apret["force"] -= np.average(apret["force"][:idp])
        else:
            apret["force"] -= apret["force"][0]

    @staticmethod
    @preprocessing_step(
        identifier="correct_tip_offset",
        name="contact point estimation",
        require_steps=["compute_tip_position"],
        options=[
            {"name": "method",
             "type": str,
             "choices": [p.identifier for p in poc.POC_METHODS],
             "choices_human_readable": [p.name for p in poc.POC_METHODS]}
        ]
        )
    def correct_tip_offset(apret, method="deviation_from_baseline"):
        """Estimate the point of contact

        An estimate of the contact point is subtracted from the
        tip position.
        """
        cpid = poc.compute_poc(force=apret["force"], method=method)
        apret["tip position"] -= apret["tip position"][cpid]

    @staticmethod
    @preprocessing_step(identifier="correct_split_approach_retract",
                        name="segment discovery",
                        require_steps=["compute_tip_position"])
    def correct_split_approach_retract(apret):
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

            idmin = np.argmax(x**2+y**2)

            segment = np.zeros(len(apret), dtype=np.uint8)
            segment[idmin:] = 1
            apret["segment"] = segment
        else:
            msg = "Cannot correct splitting of approach and retract curve " +\
                  "because the contact point position could not be estimated."
            warnings.warn(msg, CannotSplitWarning)

    @staticmethod
    @preprocessing_step(identifier="smooth_height",
                        name="spatial smoothing")
    def smooth_height(apret):
        """Smoothen height data

        For the columns "height (measured)" and "tip position",
        and for the approach and retract data separately, this
        method adds the columns "height (measured, smoothed)" and
        "tip position (smoothed)" to `apret`.
        """
        orig = ["height (measured)",
                "tip position"]
        dest = ["height (measured, smoothed)",
                "tip position (smoothed)"]
        for o, d in zip(orig, dest):
            if o not in apret.columns:
                continue
            # Get approach and retract data
            app_idx = apret["segment"] == 0
            app = np.array(apret[o][app_idx])
            ret_idx = apret["segment"] == np.max(apret["segment"])
            ret = np.array(apret[o][ret_idx])
            # Apply smoothing
            sm_app = smooth_axis_monotone(app)
            sm_ret = smooth_axis_monotone(ret)

            # Make sure that approach always comes before retract
            begin = np.where(app_idx)[0]
            end = np.where(ret_idx)[0]
            assert(np.all(end-begin > 0)), "Found retract before approach!"

            # If everything is ok, we can add the new columns
            apret[d] = np.concatenate((sm_app, sm_ret))


#: Available preprocessors
available_preprocessors = IndentationPreprocessor.available()
