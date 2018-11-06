import copy
import hashlib
import warnings

import lmfit
import numpy as np
import scipy.signal as spsig

from . import model


FP_DEFAULT = dict(model_key="hertz_para",
                  optimal_fit_edelta=False,
                  optimal_fit_num_samples=100,
                  params_initial=None,
                  preprocessing=[],
                  range_type="absolute",
                  range_x=[0, 0],
                  segment="approach",
                  weight_cp=1e-6,
                  x_axis="tip position",
                  y_axis="force",
                  )

FP_RESULTS = ["chi_sqr",
              "hash",
              "optimal_fit_delta_array",
              "optimal_fit_delta",
              "optimal_fit_E_array",
              "params_fitted",
              "success",
              "xmax",
              "xmin",
              ]


class FitKeyError(BaseException):
    pass


class FitDataError(BaseException):
    pass


class FitWarning(UserWarning):
    pass


class FitProperties(dict):
    """Fit property manager class

    Provide convenient access to fit properties as a dictionary
    and dynamically manage resets due to new initial parameters.

    Dynamic properties include:

    - set "params_initial" to `None` if the "model_key" changes
    - remove all keys except those in `FP_DEFAULT` if a key that is
      in `FP_DEFAULT` changes (All other keys are considered to be
      obsolete fitting results).

    Additional attributes:

    - "segment_bool": bool
        `False` for "approach" and `True` for "retract"
    """

    def __getitem__(self, key):
        if key == "segment_bool":
            if self["segment"] == "approach":
                return False
            if self["segment"] == "retract":
                return True
            else:
                msg = "Unknown segment: {}".format(self["segment"])
                raise FitKeyError(msg)
        else:
            return super(FitProperties, self).__getitem__(key)

    def __setitem__(self, key, value):
        if key in FP_DEFAULT:
            if (key in self and
                    self[key] == value):
                pass
            else:
                if key == "model_key":
                    # Other model has other parameters.
                    self["params_initial"] = None
                elif key == "range_x":
                    if ("optimal_fit_edelta" in self and
                        self["optimal_fit_edelta"] and
                        "range_x" in self and
                            self["range_x"][1] == value[1]):
                        # Ignore changes in range[0]
                        return
                # Trigger `self.reset`
                self.reset()
        elif key not in FP_RESULTS:
            msg = "Key '{}' not in FP_DEFAULT".format(key)
            raise FitKeyError(msg)
        super(FitProperties, self).__setitem__(key, value)

    def reset(self):
        for key in list(self.keys()):
            if key not in FP_DEFAULT:
                self.pop(key)

    def restore(self, props):
        """update the dictionary without removing any keys"""
        for key in props:
            super(FitProperties, self).__setitem__(key, props[key])


class IndentationFitter(object):
    def __init__(self, data_set, **kwargs):
        """Fit force-indentation curves

        Parameters
        ----------
        model_key: str
            A key referring to a model in
            `nanite.model.models_available`
        params_initial: instance of lmfit.Parameters
            Parameters for fitting. If not given,
            default parameters are used.
        range_x: tuple of 2
            The range for fitting, see `range_type` below.
        range_type: str
            One of:

            - absolute:
                Set the absolute fitting range in values
                given by the `x_axis`.
            - relative cp:
                In some cases it is desired to be able to
                fit a model only up until a certain indentation
                depth (tip position) measured from the contact
                point. Since the contact point is a fit parameter
                as well, this requires a two-pass fitting.
        preprocessing: list of str
            Preprocessing
        segment: str
            One of "approach" or "retract".
        weight_cp: float
            Weight the contact point region which shows artifacts
            that are difficult to model with e.g. Hertz.
        optimal_fit_edelta: bool
            Search for the optimal fit by varying the maximal
            indentation depth and determining a plateau in the
            resulting Young's modulus (fitting parameter "E").
        optimal_fit_num_samples: int
            Number of samples to use for searching the optimal fit
        """
        # Get initial fit parameters
        # IMPORTANT:
        # If there are new additions in the default values,
        # make sure to take these into account in `FP_DEFAULT`.
        self.fp = FitProperties(**FP_DEFAULT)

        # Get parameters from data set
        for key in data_set.fit_properties:
            if key in FP_DEFAULT:
                self.fp[key] = data_set.fit_properties[key]
        # Get parameters from kwargs
        for key in self.fp:
            if key in kwargs:
                if key not in FP_DEFAULT:
                    msg = "Key '{}' not in FP_DEFAULT".format(key)
                    raise FitKeyError(msg)
                self.fp[key] = kwargs[key]
        # Set initial fitting parameters
        if self.fp["params_initial"] is None:
            self.fp["params_initial"] = self.get_initial_parameters(
                data_set=data_set,
                model_key=self.fp["model_key"]
            )

        # Set arrays
        self.segment = (data_set["segment"] ==
                        self.fp["segment_bool"]).values
        self.segment.setflags(write=False)

        self.x_axis = data_set[self.fp["x_axis"]].values
        self.x_axis.setflags(write=False)

        self.y_axis = data_set[self.fp["y_axis"]].values
        self.y_axis.setflags(write=False)

        self.fit_range = np.zeros_like(self.segment)
        self.fit_curve = np.zeros_like(self.y_axis)
        self.fit_residuals = np.zeros_like(self.y_axis)

        # Fitting parameters might be changed due to iterative fitting.
        # Store them in separate variables to prevent resetting the
        # `FitPropreties` instance `self.fp`.
        self.optimal_fit_edelta = self.fp["optimal_fit_edelta"]
        self.range_type = self.fp["range_type"]
        self.range_x = list(self.fp["range_x"])

        self.hash = self._hash()
        self.fp["hash"] = self.hash

        # Perform sanity checks
        if self.fp["range_type"] not in ["absolute", "relative cp"]:
            msg = "`range_type` must be  'absolute' or 'relative cp'!"
            raise FitKeyError(msg)
        if len(self.fp["range_x"]) != 2:
            raise FitKeyError("`range_x` must have length 2!")
        if (np.isnan(self.fp["range_x"][0]) or
                np.isnan(self.fp["range_x"][1])):
            raise FitKeyError("`range_x` must not contain NaN!")
        if self.fp["segment"] not in ["approach", "retract"]:
            msg = "`segment` must be  'approach' or 'retract'!"
            raise FitKeyError(msg)
        if self.fp["model_key"] not in model.models_available:
            msg = "unknown model '{}'".format(self.fp["model_key"])
            raise FitKeyError(msg)

        if self.fp["optimal_fit_edelta"]:
            # Make sure we have emodulus
            if "E" not in self.fp["params_initial"]:
                msg = "Search for optimal fit requires the parameter 'E'!"
                raise FitKeyError(msg)
            # This only works with absolute range
            if self.fp["range_type"] != "absolute":
                msg = "Only absolute range relative to contact point allowed!"
                raise FitKeyError(msg)

        md_key = self.fp["model_key"]
        md = model.models_available[md_key]
        params = md.get_parameter_defaults()
        for p in params:
            msg = "Unknown fitting parameter '{}' for model '{}'!"
            if p not in self.fp["params_initial"]:
                raise FitKeyError(msg.format(p, md_key))

        if self.fp["range_x"][0] > self.fp["range_x"][1]:
            msg = "Fitting range is inverted: {}".format(self.fp["range_x"])
            warnings.warn(msg, FitWarning)

    def compute_emodulus_vs_mindelta(self, callback=None):
        """Compute elastic modulus vs. minimal indentation curve"""
        segid = self.segment
        xseg = self.x_axis[segid]
        yseg = self.y_axis[segid]

        xmax = np.max(self.fp["range_x"])
        if np.isinf(xmax):
            xmax = np.max(xseg)
        # Disable and remember `optimal_fit_edelta`
        optimal_fit_edelta = self.optimal_fit_edelta
        self.optimal_fit_edelta = False

        # We are agnostic concerning the direction of indentation.
        # `xseg` should start at the baseline.
        seems_approach = np.average(yseg[:10]) < np.average(yseg[-10:])
        if seems_approach and self.fp["segment"] == "approach":
            if xseg[0] < xseg[-1]:
                msg = "Unexpected trend in approach x data!"
                raise FitDataError(msg)
        elif seems_approach:
            msg = "Data appears to be 'approach', but is 'retract'!"
            raise FitDataError(msg)
        elif not seems_approach and self.fp["segment"] == "retract":
            if xseg[0] > xseg[-1]:
                msg = "Unexpected trend in retract x data!"
                raise FitDataError(msg)
            msg = "Unexpected trend in retract curve!"
            raise FitDataError(msg)
        elif not seems_approach:
            msg = "Data appears to be 'retract', but is 'approach'!"
            raise FitDataError(msg)

        # Fit the range of parameters
        xmin = xseg.min()
        if xmin >= 0:
            msg = "No negative values (indentation) found! " \
                  + "Did you correct for tip offset?"
            raise FitKeyError(msg)
        num_samp = self.fp["optimal_fit_num_samples"]
        indentations = np.linspace(xmin, xmin*.05, num_samp)
        emoduli = np.zeros_like(indentations)
        for ii, x0 in enumerate(indentations):
            # Perform a fit and record the elastic modulus
            self.range_x = [x0, xmax]
            self.fit()
            emoduli[ii] = self.fp["params_fitted"]["E"].value
            if callback and ii % 5 == 0:
                callback(emoduli, indentations)

        self.optimal_fit_edelta = optimal_fit_edelta

        return emoduli, indentations

    @staticmethod
    def compute_opt_mindelta(emoduli, indentations):
        """Determine the plateau of an emodulus-indentation curve

        The following procedure is performed:

        1. Smooth the emodulus data with a Butterworth filter
        2. Label sequences that have similar values by binning
           into ten regions between the min and max.
        3. Ignore sequences with emodulus that is smaller than
           the binning size.
        4. Determine the longest sequence.
        """
        # Perform smoothing of the curve
        # First, perform filtering with butterworth filter
        nb = 1      # Filter order
        wb = 0.05    # Cutoff frequency
        b, a = spsig.butter(nb, wb, output='ba')
        smooth_e = spsig.filtfilt(b, a, emoduli)
        # Second, determine the longest sequence of values
        # that have the same smoothed value.
        ni = 10
        ivals, istep = np.linspace(smooth_e.min(), smooth_e.max(), ni,
                                   endpoint=False, retstep=True)
        ivals += istep/2
        labelarray = np.zeros_like(smooth_e, dtype=int)
        valarray = np.zeros_like(smooth_e, dtype=int)
        # label each sequence with an individual `idx`
        for ii in range(smooth_e.shape[0]):
            valid = np.argmin(np.abs(ivals - smooth_e[ii]))
            if ii == 0:
                idx = 0
            elif valid == valarray[ii-1]:
                pass
            else:
                idx += 1
            labelarray[ii] = idx
            valarray[ii] = valid
        # Determine the longest sequence
        counts = list(np.bincount(labelarray))
        # Ignore values that are below cutoff

        for ii in range(len(counts)):
            labmax = np.argmax(counts)
            labid = np.where(labelarray == labmax)[0][0]
            valmax = ivals[valarray[labid]]
            if valmax > istep:
                break
            counts.pop(labmax)
        else:
            # Nothing found: select the middle value
            warnings.warn("Could not find correct plateau.", FitWarning)
            labmax = 5
        # Determine the interval in the original array
        indices = np.where(labelarray == labmax)[0]
        if len(indices) == 1:
            dopt = indentations[indices[0]]
        else:
            # compute optimal indentation as center
            dopt = np.average(indentations[indices[0]:indices[-1]])
        return dopt

    def _fit(self):
        """Fit a model to the data

        Notes
        -----
        This method is private because it requires the array
        `self.fit_range` before the actual fitting.
        """
        model_key = self.fp["model_key"]
        params_initial = self.fp["params_initial"]
        weight_cp = self.fp["weight_cp"]

        # boolean array indexing the segment
        segid = self.segment
        # x: the entire segment
        xseg = self.x_axis[segid]
        # y: the entire segment
        yseg = self.y_axis[segid]
        # x: the values being fitted
        x = self.x_axis[self.fit_range]
        # y: the values being fitted
        y = self.y_axis[self.fit_range]

        md = model.models_available[model_key]

        # short reference for better readability
        fit_cur = self.fit_curve
        fit_res = self.fit_residuals

        # reset values to nan
        fit_cur[:] = np.nan
        fit_res[:] = np.nan

        # Make sure that we can actually fit by comparing variable fitting
        # parameters and size of x.
        npvaried = np.sum([p[1].vary for p in list(params_initial.items())])
        if npvaried < x.shape[0] - 1:
            # perform fit
            fit = lmfit.minimize(
                md.residual, params_initial, args=(x, y, weight_cp))
            # fitted method
            fit_cur[segid] = md.model(fit.params, xseg)
            # residuals
            fit_res[segid] = md.residual(fit.params, xseg, yseg, weight_cp)
            # add fit results to fp dictionary
            self.fp.update({"params_fitted": fit.params,
                            "chi_sqr": fit.chisqr,
                            "xmin": x.min(),
                            "xmax": x.max(),
                            "success": True,
                            })
        else:
            self.fp["success"] = False

    def fit(self):
        """Fit the approach-retract data to a model function
        """
        range_type = self.range_type
        range_x = copy.copy(self.range_x)
        # Set to True in `self._fit`
        self.fp["success"] = False

        if self.optimal_fit_edelta:
            # emodulus-indentation curve:
            emoduli, indentations = self.compute_emodulus_vs_mindelta()
            # determine optimal x0
            dopt = self.compute_opt_mindelta(emoduli, indentations)
            # add indentations/emoduli to self.fp
            self.fp["optimal_fit_E_array"] = emoduli
            self.fp["optimal_fit_delta_array"] = indentations
            self.fp["optimal_fit_delta"] = dopt
            # update fitting parameters for final fit
            self.range_x = [dopt, np.max(self.fp["range_x"])]
            # perform final fit
            self.optimal_fit_edelta = False
            self.fit()
            self.optimal_fit_edelta = True

        elif self.range_type == "absolute":
            # This is easy. Simply set the boolean array of fitting values
            # Exclude data points from other segment
            if range_x[0] != range_x[1]:
                x_data = self.x_axis.copy()
                range_bool = self.segment.copy()
                rmin, rmax = np.min(range_x), np.max(range_x)
                range_bool[x_data < rmin] = False
                range_bool[x_data > rmax] = False
            else:
                range_bool = self.segment
            self.fit_range[:] = range_bool
            self._fit()

        elif range_type == "relative cp":
            # Let's try to solve this with four passes
            # First, get the approximate contact point
            self.range_type = "absolute"
            # Set full range to get estimate of cp
            self.range_x = [0, 0]
            self.fit()
            # Do the following three-times.
            for _i in range(3):
                # get the fitted contact point
                cp = self.fp["params_fitted"]["contact_point"].value
                self.range_x = list(np.array(range_x)+cp)
                # Second, fit with the new contact point as range parameters
                self.fit()

        # Reset range data to original values
        self.range_type = range_type
        self.range_x = range_x

    def get_initial_parameters(self, data_set=None, model_key="hertz_para"):
        """Get initial fit parameters for a specific model

        Parameters
        ----------

        """
        if (model_key == self.fp["model_key"] and
                self.fp["params_initial"] is not None):
            params = self.fp["params_initial"]
        else:
            md = model.models_available[model_key]
            params = md.get_parameter_defaults()
            # Guess initial contact point from actual tip position
            # (see `self.compute_tip_position`)
            # Depending on how the current data set was pre-processed,
            # the column "tip position" might already be corrected by
            # an estimated contact point offset.
            # (see `self.compute_tip_offset`)
            if data_set is None:
                msg = "Need `data_set` to get initial parameters!"
                raise ValueError(msg)
            if "tip position" in data_set:
                cpid = data_set.estimate_contact_point_index()
                cp = data_set["tip position"][cpid]
                params["contact_point"].set(cp)
            else:
                msg = "Cannot estimate contact point, because of missing "\
                      + "column 'tip position'"
                warnings.warn(msg, FitWarning)
        return params

    def _hash(self):
        """Compute hash identifier for current fit

        The hash is computed without applying the fit, making
        it suitable for caching fits.
        """
        hashlist = []
        # preprocessing
        hashlist.append(self.fp["preprocessing"])
        # axes data
        hashlist.append(self.x_axis)
        hashlist.append(self.y_axis)
        # fit parameters
        for key in FP_DEFAULT:
            if (key == "range_x" and
                    self.fp["optimal_fit_edelta"]):
                # range only partly if "optimal_fit_edelta" is True
                hashlist.append(self.fp["range_x"][1])
            elif (key == "optimal_fit_num_samples" and
                  not self.fp["optimal_fit_edelta"]):
                # ignore number of samples if optimal fit is not used
                pass
            else:
                hashlist.append(self.fp[key])
        # join and hash
        myhash = hashlib.md5(obj2str(hashlist)).hexdigest()
        return myhash


def obj2str(obj):
    """String representation of an object for hashing"""
    if isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, (bool, int, float)):
        return str(float(obj)).encode("utf-8")
    elif obj is None:
        return b"none"
    elif isinstance(obj, np.ndarray):
        return obj.tostring()
    elif isinstance(obj, tuple):
        return obj2str(list(obj))
    elif isinstance(obj, list):
        return b"".join(obj2str(o) for o in obj)
    elif isinstance(obj, dict):
        return obj2str(list(obj.items()))
    elif isinstance(obj, lmfit.parameter.Parameter):
        return obj2str([obj.value, obj.max, obj.min, obj.vary,
                        obj.expr, obj.name])
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))
