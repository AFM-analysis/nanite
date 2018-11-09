import warnings

import numpy as np

from .smooth import smooth_axis_monotone


class CannotSplitWarning(UserWarning):
    pass


class IndentationPreprocessor(object):
    @staticmethod
    def apply(apret, preproc_names):
        """Perform force-indentation preprocessing steps

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
            if hasattr(IndentationPreprocessor, mm):
                meth = getattr(IndentationPreprocessor, mm)
                meth(apret)
            else:
                msg = "The preprocessing method '{}' does not exist!"
                raise KeyError(msg.format(mm))

    @staticmethod
    def available():
        """List available preprocessor names"""
        ignore = ["available", "apply"]
        funcs = IndentationPreprocessor.__dict__
        av = []
        for ff in funcs:
            if (not ff.startswith("_")
                and ff not in ignore
                    and isinstance(funcs[ff], staticmethod)):
                av.append(ff)
        return sorted(av)

    @staticmethod
    def compute_tip_position(apret):
        """Compute the tip-sample separation

        This computation correctly reproduces the column
        "Vertical Tip Position" as it is exported by the
        JPK analysis software with the checked option
        "Use Unsmoothed Height".
        """
        k = apret.metadata["spring constant [N/m]"]
        force = apret.data["force"]
        zcant = apret.data["height (measured)"]
        apret.data["tip position"] = zcant + force/k

    @staticmethod
    def correct_force_offset(apret):
        """Correct the force offset with an average baseline value
        """
        idp = apret.estimate_contact_point_index()
        if idp:
            apret.data["force"] -= np.average(apret.data["force"][:idp])
        else:
            apret.data["force"] -= apret.data["force"][0]

    @staticmethod
    def correct_tip_offset(apret):
        """Correct the offset of the tip position

        An estimate of the tip position is used to compute the
        contact point.
        """
        cpid = apret.estimate_contact_point_index()
        apret.data["tip position"] -= apret.data["tip position"][cpid]

    @staticmethod
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
        x = np.array(apret.data["tip position"], copy=True)
        y = np.array(apret.data["force"], copy=True)

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

            # approach
            apret.data.loc[apret.data.index < idmin, "segment"] = False
            # retract
            apret.data.loc[apret.data.index >= idmin, "segment"] = True
        else:
            msg = "Cannot correct splitting of approach and retract curve " +\
                  "because the contact point position could not be estimated."
            warnings.warn(msg, CannotSplitWarning)

    @staticmethod
    def smooth_height(apret):
        """Smoothen height data

        For the columns "height (measured)" and "tip position",
        and for the approach and retract data separately, this
        method adds the columns "height (measured, smoothed)" and
        "tip position (smoothed)" to `self.data`.
        """
        orig = ["height (measured)",
                "tip position"]
        dest = ["height (measured, smoothed)",
                "tip position (smoothed)"]
        for o, d in zip(orig, dest):
            if o not in apret.data.columns:
                continue
            # Get approach and retract data
            app_idx = ~apret.data["segment"].values
            app = np.array(apret.data[o].loc[app_idx])
            ret_idx = apret.data["segment"].values
            ret = np.array(apret.data[o].loc[ret_idx])
            # Apply smoothing
            sm_app = smooth_axis_monotone(app)
            sm_ret = smooth_axis_monotone(ret)

            # Make sure that approach always comes before retract
            begin = np.where(app_idx)[0]
            end = np.where(ret_idx)[0]
            assert(np.all(end-begin > 0)), "Found retract before approach!"

            # If everything is ok, we can add the new columns
            apret.data[d] = np.concatenate((sm_app, sm_ret))


#: Available preprocessors
available_preprocessors = IndentationPreprocessor.available()
