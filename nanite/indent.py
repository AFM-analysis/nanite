from collections import OrderedDict
import copy
import inspect

import lmfit
import numpy as np
import scipy.signal as spsig

from .fit import IndentationFitter, FitProperties, guess_initial_parameters, \
    FP_DEFAULT
from . import model
from .preproc import IndentationPreprocessor
from .rate import get_rater


class Indentation(object):
    def __init__(self, idnt_data):
        """Force-indentation

        Parameters
        ----------
        idnt_data: nanite.read.IndentationData
            Object holding the experimental data
        """
        self.metadata = idnt_data.metadata
        self.path = idnt_data.path
        self.enum = idnt_data.enum

        #: All data as afmformats.AFMForceDistance
        self.data = idnt_data
        #: Default preprocessing steps steps,
        #: see :func:`Indentation.apply_preprocessing`.
        self.preprocessing = []
        # protected fit properties
        self._fit_properties = FitProperties()

        # Curve rating (see `self.rate_quality`)
        self._rating = None

        # Store initial parameters for reset (see `self.reset`)
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self._init_kwargs = {}
        args.remove("self")
        for arg in args:
            self._init_kwargs[arg] = copy.deepcopy(values[arg])

    def __contains__(self, key):
        return self.data.__contains__(key)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __repr__(self):
        return "Indentation {: 6d} in '{}'".format(
            self.enum,
            self.path
        )

    @property
    def fit_properties(self):
        """Fitting results, see :func:`Indentation.fit_model`)"""
        return self._fit_properties

    @fit_properties.setter
    def fit_properties(self, fp):
        self._fit_properties.update(fp)

    def apply_preprocessing(self, preprocessing=None):
        """Perform curve preprocessing steps

        Parameters
        ----------
        preprocessing: list
            A list of preprocessing method names that are
            stored in the `IndentationPreprocessor` class.
            If set to `None`, `self.preprocessing` will be
            used.
        """
        if preprocessing is None:
            preprocessing = self.preprocessing

        if "preprocessing" in self.fit_properties:
            preproc_past = self.fit_properties["preprocessing"]
        else:
            preproc_past = []

        if preproc_past != preprocessing:
            # Remember initial fit parameters for user convenience
            fp = self.fit_properties
            fp["preprocessing"] = preprocessing
            # Reset all data
            self.reset()
            # Apply preprocessing
            IndentationPreprocessor.apply(self, preprocessing)
            # Check availability of axes
            for ax in ["x_axis", "y_axis"]:
                # make sure the fitting axes are defined
                if ax in fp and not fp[ax] in self.data:
                    fp.pop(ax)
            # Set new fit properties
            self.fit_properties = fp
        # remember preprocessing
        self.preprocessing = preprocessing

    def compute_emodulus_mindelta(self, callback=None):
        """Elastic modulus in dependency of maximum indentation

        The fitting interval is varied such that the maximum
        indentation depth ranges from the lowest tip position
        to the estimated contact point. For each interval, the
        current model is fitted and the elastic modulus is
        extracted.

        Parameters
        ----------
        callback: callable
            A method that is called with the `emoduli` and
            `indentations` as the computation proceeds every
            five steps.

        Returns
        -------
        emoduli, indentations: 1d ndarrays
            The fitted elastic moduli at the corresponding
            maximal indentation depths.

        Notes
        -----
        The information about emodulus and mindelta is also stored in
        `self.fit_properties` with the keys "optimal_fit_E_array" and
        "optimal_fit_delta_array", if `self.fit_model` is called with
        the argument `search_optimal_fit` set to `True`.
        """

        if "optimal_fit_E_array" in self.fit_properties:
            emoduli = self.fit_properties["optimal_fit_E_array"]
            indentations = self.fit_properties["optimal_fit_delta_array"]
        else:
            fitter = IndentationFitter(self)
            emoduli, indentations = fitter.compute_emodulus_vs_mindelta(
                callback=callback
            )
            self.fit_properties["optimal_fit_E_array"] = emoduli
            self.fit_properties["optimal_fit_delta_array"] = indentations
        return emoduli, indentations

    def estimate_optimal_mindelta(self):
        """Estimate the optimal indentation depth

        This is a convenience function that wraps around
        `compute_emodulus_mindelta` and
        `IndentationFitter.compute_opt_mindelta`.
        """
        emoduli, indentations = self.compute_emodulus_mindelta()
        dopt = IndentationFitter.compute_opt_mindelta(
            emoduli=emoduli,
            indentations=indentations
        )
        return dopt

    @classmethod
    def _estimate_contact_point_index_from_baseline(cls, fg):
        idp1 = np.nan
        # Method 1: base line deviation
        # Crop the slow approach trace (10% of the curve)
        baseline = fg[:int(fg.size*.1)]
        if baseline.size:
            bl_avg = np.average(baseline)
            bl_rng = np.max(np.abs(baseline-bl_avg))*2
            bl_dev = (fg-bl_avg) > bl_rng
            if np.sum(bl_dev):
                idp1 = np.where(bl_dev)[0][0]
        return idp1

    @classmethod
    def _estimate_contact_point_index_from_cl_fit(cls, fg):
        """This is probably the most robust version"""
        # TODO:
        # - test whether this is really slower than the other methods
        def residual(params, x, data):
            off = params["off"]
            x0 = params["x0"]
            m = params["m"]
            one = off
            two = m*(x-x0) + off
            return data - np.maximum(one, two)

        if fg.size > 4:
            x = np.arange(fg.size)

            params = lmfit.Parameters()
            params.add('off', value=np.mean(fg[:10]))
            params.add('x0', value=fg.size//2)
            params.add('m', value=(fg.max() - fg.min()) / fg.size)

            out = lmfit.minimize(residual, params, args=(x, fg))
            if out.success:
                idp = int(out.params["x0"])
            else:
                idp = fg.size // 2
        else:
            # approach part too short to be reasonable
            idp = 0
        return idp

    @classmethod
    def _estimate_contact_point_index_from_sign_gradient(cls, fg):
        idp2 = np.nan
        # Method 2: gradient change
        # Perform a median filter to smooth the array
        filtsize = 15
        y = spsig.medfilt(fg, filtsize)
        # Cut off the trailing 10 points (noise)
        cutoff = 10
        if y.size > cutoff+1:
            grad = np.gradient(y)[:-cutoff]
            # Use the point where the gradient becomes positive for the
            # first time.
            gradpos = grad > 0
            if np.sum(gradpos):
                # The contains positive values.
                # Flip `gradpos`, because we want the first value from the
                # end of the array.
                idp2 = y.size - np.where(gradpos[::-1])[0][0] - cutoff - 1
        return idp2

    @classmethod
    def _estimate_contact_point_index_preprocess_gradient(cls, force):
        # Preprocessing (remove tilt from curve)
        # apply rolling average filter to force
        p1_fs = min(47, force.size//2//2*2 + 1)
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

    def estimate_contact_point_index(self):
        """Estimate the contact point

        Contact point (CP) estimation involves a preprocessing step
        where the force data are transformed into gradient space
        (to account for a slope in the approach curve) and a
        subsequent analysis with two different methods to determine
        when the gradient changes significantly enough to qualify for
        a CP. Of those two methods, the one which yields the smallest
        index (measured from the beginning of the approach curve)
        is returned. If one of the methods fail, then a fit function
        with a constant and linear part is used to determine the CP.

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
        determine the contact point. If that fails as well, then
        the CP defaults to the center of the entire approach curve.

        .. versionchanged:: 1.6.0
            Add the gradient preprocessing step to circumvent issues
            with tilted baselines. This feature does not significantly
            affect fitting results.

        .. versionchanged:: 1.6.1
            Added max(constant, linear) fit when the other methods
            fail.
        """
        # get data
        y0 = np.array(self.data["force"], copy=True)
        # Only use the (initial) approach part of the curve.
        idmax = np.argmax(y0)
        y = y0[:idmax]

        fg = self._estimate_contact_point_index_preprocess_gradient(y)
        idp1 = self._estimate_contact_point_index_from_baseline(fg)
        idp2 = self._estimate_contact_point_index_from_sign_gradient(fg)

        if np.isnan(idp1) or np.isnan(idp2):
            idp = self._estimate_contact_point_index_from_cl_fit(fg)
        else:
            idp = min(idp1, idp2)

        return idp

    def export(self, path, fmt="tab"):
        """Saves the current data as tab separated values"""
        self.data.export(path, fmt=fmt)

    def fit_model(self, **kwargs):
        """Fit the approach-retract data to a model function

        Parameters
        ----------
        model_key: str
            A key referring to a model in
            `nanite.model.models_available`
        params_initial: instance of lmfit.Parameters or dict
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
        """
        if "preprocessing" in kwargs:
            self.apply_preprocessing(kwargs["preprocessing"])
        # self.fit_properties is an instance of FitProperties that
        # stores previous fit kwargs. If the given kwargs are
        # different than in the previous fit, the following two
        # lines will reset the "hash" in the fit properties, triggering
        # a new fit.
        # (sorted, such that `model_key` is set before `params_initial`)
        for arg in sorted(kwargs.keys()):
            self.fit_properties[arg] = kwargs[arg]

        # set a default model (needed for self.get_initial_fit_parameters)
        if "model_key" not in self.fit_properties:
            self.fit_properties["model_key"] = FP_DEFAULT["model_key"]

        # set default initial parameters
        if ("params_initial" not in self.fit_properties
                or self.fit_properties["params_initial"] is None):
            # We need the initial parameters (to modify them).
            # Guesses common parameters like the contact point that
            # would have otherwise been done in `IndentationFitter`:
            fp_guess = self.get_initial_fit_parameters(
                common_ancillaries=True,
                model_ancillaries=True)
            self.fit_properties["params_initial"] = fp_guess

        if "hash" in self.fit_properties:
            # There is nothing to do, because the initial fit
            # properties are the same.
            pass
        else:
            fitter = IndentationFitter(self)
            # Perform fitting
            # Note: if `fitter.fp["success"]` is `False`, then
            # the `fit_residuals` and `fit_curve` are `nan`.
            fitter.fit()
            self["fit"] = fitter.fit_curve
            self["fit residuals"] = fitter.fit_residuals
            self["fit range"] = fitter.fit_range
            self.fit_properties = fitter.fp

    def get_ancillary_parameters(self, model_key=None):
        """Compute ancillary parameters for the current model"""
        if model_key is None:
            if "model_key" in self.fit_properties:
                model_key = self.fit_properties["model_key"]
            else:
                model_key = FP_DEFAULT["model_key"]
        return model.get_anc_parms(idnt=self,
                                   model_key=model_key)

    def get_initial_fit_parameters(self, model_key=None,
                                   common_ancillaries=True,
                                   model_ancillaries=True):
        """Return the initial fit parameters

        If there are not initial fit parameters set in
        `self.fit_properties`, then they are computed.

        Parameters
        ----------
        model_key: str
            Optionally set a model key. This will override the
            "model_key" key in `self.fit_properties`.
        common_ancillaries: bool
            Guess global ancillaries such as the contact point.
        model_ancillaries: bool
            Guess model-related ancillaries

        Notes
        -----
        `global_ancillaries` and `model_ancillaries` only have an
        effect if self.fit_properties["params_initial"] is set.
        """
        if model_key is not None:
            self.fit_properties["model_key"] = model_key
        if self.fit_properties.get("params_initial", False):
            parms = self.fit_properties["params_initial"]
        elif "model_key" in self.fit_properties:
            parms = guess_initial_parameters(
                self,
                model_key=self.fit_properties["model_key"],
                model_ancillaries=model_ancillaries,
                common_ancillaries=common_ancillaries)
        else:
            # for user convenience (with default model)
            parms = guess_initial_parameters(
                self,
                model_key=FP_DEFAULT["model_key"],
                model_ancillaries=model_ancillaries,
                common_ancillaries=common_ancillaries)
        return parms

    def get_rating_parameters(self):
        """Return current rating parameters"""
        rdict = OrderedDict()
        if self._rating is None:
            rt = [np.nan] * 6
        else:
            rt = self._rating
        rdict["Hash"] = rt[0]
        rdict["Regressor"] = rt[1]
        rdict["Training set"] = rt[2]
        rdict["Feature names"] = rt[3]
        rdict["Linear discriminant analysis"] = rt[4]
        rdict["Rating"] = rt[5]
        return rdict

    def rate_quality(self, regressor="Extra Trees", training_set="zef18",
                     names=None, lda=None):
        """Compute the quality of the obtained curve

        Uses heuristic approaches to rate a curve.

        Parameters
        ----------
        regressor: str
            The regressor name used for rating.
        training_set: str
            A label for a training set shipped with nanite or a
            path to a training set.

        Returns
        -------
        rating: float
            A value between 0 and 10 where 0 is the lowest rating.
            If no fit has been performed, a rating of -1 is returned.

        Notes
        -----
        The rating is cached based on the fitting hash
        (see `IndentationFitter._hash`).
        """
        if (self.fit_properties and "hash" in self.fit_properties):
            curhash = self.fit_properties["hash"]
        else:
            curhash = "none"
        if regressor.lower() == "none":
            rt = -1
        elif (self._rating is None or
              self._rating[0] != curhash or
              self._rating[1] != regressor or
              self._rating[2] != training_set or
              self._rating[3] != names or
              self._rating[4] != lda):
            # Perform rating
            rater = get_rater(regressor=regressor,
                              training_set=training_set,
                              names=names,
                              lda=lda)
            rt = rater.rate(datasets=self)[0]
            self._rating = (curhash, regressor, training_set, names, lda, rt)
        else:
            # Use cached rating
            rt = self._rating[-1]
        return rt

    def reset(self):
        """Resets all data operations"""
        self.__init__(**self._init_kwargs)
