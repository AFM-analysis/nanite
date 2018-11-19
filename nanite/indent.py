import copy
import inspect
import pathlib

import numpy as np
import scipy.signal as spsig

from .fit import IndentationFitter, FitProperties
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

        #: All data in a Pandas DataFrame
        self.data = idnt_data.data
        #: Default preprocessing steps steps,
        #: see :func:`Indentation.apply_preprocessing`.
        self.preprocessing = []
        #: Fitting results, see :func:`Indentation.fit_model`)
        self.fit_properties = FitProperties()

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
            fp.reset()  # TODO: this is redundant (check FitProperties)
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

    def estimate_contact_point_index(self):
        """Estimate the contact point

        Contact point (CP) estimation is performed with two methods and
        that one which returns the smallest index is returned.

        Method 1: baseline deviation

        1. Obtain the baseline (initial 10% of the approach curve)
        2. Compute average and maximum deviation of the baseline
        3. The CP is the index of the approach curve where it exceeds
           twice of the maximum deviation

        Method 2: sign of gradient

        1. Perform a median filter on the approach curve
        2. Compute the gradient
        3. Cut off trailing 10 points from the gradient (noise)
        4. The CP is the index of the gradient curve when the
           sign changes, measured from the point of maximal
           indentation.

        If one of the methods fail, the index 0 is returned.

        """
        # initial assumptions
        idp1 = 0
        idp2 = 0
        # get data
        y0 = np.array(self.data["force"], copy=True)
        # Only use the (initial) approach part of the curve.
        idmax = np.argmax(y0)
        y = y0[:idmax]

        # Method 1: base line deviation
        # Crop the slow approach trace (10% of the curve)
        baseline = y[:int(y.size*.1)]
        if baseline.size:
            bl_avg = np.average(baseline)
            bl_rng = np.max(np.abs(baseline-bl_avg))*2
            bl_dev = (y-bl_avg) > bl_rng
            if np.sum(bl_dev):
                idp1 = np.where(bl_dev)[0][0]

        # Method 2: gradient change
        # Perform a median filter to smooth the array
        filtsize = 15
        y = spsig.medfilt(y, filtsize)
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

        return min(idp1, idp2)

    def export(self, path):
        """Saves the current data as tab separated values"""
        path = pathlib.Path(path)
        with path.open("w") as fd:
            self.data.to_csv(path_or_buf=fd,
                             sep="\t",
                             encoding="utf-8",
                             na_rep="nan",
                             index=False)

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
        for arg in kwargs:
            self.fit_properties[arg] = kwargs[arg]

        if "hash" in self.fit_properties:
            # There is nothing to do, because the initial fit
            # properties are the same.
            pass
        else:
            fitter = IndentationFitter(self, **kwargs)
            # Perform fitting
            # Note: if `fitter.fp["success"]` is `False`, then
            # the `fit_residuals` and `fit_curve` are `nan`.
            fitter.fit()
            self["fit"] = fitter.fit_curve
            self["fit residuals"] = fitter.fit_residuals
            self["fit range"] = fitter.fit_range
            self.fit_properties = fitter.fp

    def get_initial_fit_parameters(self):
        if ("params_initial" in self.fit_properties and
                self.fit_properties["params_initial"] is not None):
            parms = self.fit_properties["params_initial"]
        else:
            parms = IndentationFitter(self).fp["params_initial"]
        return parms

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
