import functools
import warnings

import numpy as np

from .smooth import smooth_axis_monotone


class CannotSplitWarning(UserWarning):
    pass


def preprocessing_step(identifier, name):
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
    """
    def attribute_setter(func):
        """Decorator that sets the necessary attributes

        The outer decorator is used to obtain the attributes.
        This inner decorator returns the actual function that
        wraps the preprocessor.
        """
        func.identifier = identifier
        func.name = name
        return func

    return attribute_setter


class IndentationPreprocessor(object):
    @staticmethod
    def apply(apret, preproc_names):
        """Perform force-distance preprocessing steps

        Parameters
        ----------
        apret: nanite.Indentation
            The afm data to preprocess
        preproc_names: list
            A list of names for static methods in
            `IndentationPreprocessor` that will be
            applied (in the order given).

        Notes
        -----
        This method is usually called from within the `Indentation`
        class instance. If you are using this class directly and
        apply it more than once, you might need to call
        `apret.reset()` before preprocessing a second time.
        """
        for mm in preproc_names:
            if mm in IndentationPreprocessor.available():
                meth = getattr(IndentationPreprocessor, mm)
                meth(apret)
            else:
                msg = "The preprocessing method '{}' does not exist!"
                raise KeyError(msg.format(mm))

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
        for key in dir(IndentationPreprocessor):
            func = getattr(IndentationPreprocessor, key)
            if hasattr(func, "identifier") and func.identifier == identifier:
                return func.name
        else:
            raise KeyError(f"Preprocessor '{identifier}' unknown!")

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
        idp = apret.estimate_contact_point_index()
        if idp:
            apret["force"] -= np.average(apret["force"][:idp])
        else:
            apret["force"] -= apret["force"][0]

    @staticmethod
    @preprocessing_step(identifier="correct_tip_offset",
                        name="contact point estimation")
    def correct_tip_offset(apret):
        """Correct the offset of the tip position

        An estimate of the tip position is used to compute the
        contact point.
        """
        cpid = apret.estimate_contact_point_index()
        apret["tip position"] -= apret["tip position"][cpid]

    @staticmethod
    @preprocessing_step(identifier="correct_split_approach_retract",
                        name="segment discovery")
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

        idp = apret.estimate_contact_point_index()
        if idp:
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

            segment = np.zeros(len(apret), dtype=bool)
            segment[idmin:] = True
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
            app_idx = ~apret["segment"]
            app = np.array(apret[o][app_idx])
            ret_idx = apret["segment"]
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
