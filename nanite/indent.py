from collections import OrderedDict
import copy
import warnings

import afmformats
import numpy as np

from .fit import IndentationFitter, FitProperties, guess_initial_parameters, \
    FP_DEFAULT
from . import model
from . import poc
from . import preproc
from .rate import get_rater


class Indentation(afmformats.AFMForceDistance):
    def __init__(self, data, metadata, diskcache=None):
        """Additional functionalities for afmformats.AFMForceDistance"""
        super(Indentation, self).__init__(data=data,
                                          metadata=metadata,
                                          diskcache=diskcache)
        #: Default preprocessing steps,
        #: see :func:`Indentation.apply_preprocessing`.
        self.preprocessing = []
        #: Preprocessing options
        self.preprocessing_options = {}
        # preprocessing details (for user convenience)
        self._preprocessing_details = {}
        # protected fit properties
        self._fit_properties = FitProperties()

        # Curve rating (see `self.rate_quality`)
        self._rating = None

    @property
    def data(self):
        warnings.warn("Please use __getitem__ instead!", DeprecationWarning)
        return self

    @property
    def fit_properties(self):
        """Fitting results, see :func:`Indentation.fit_model`)"""
        return self._fit_properties

    @fit_properties.setter
    def fit_properties(self, fp):
        self._fit_properties.update(fp)

    def apply_preprocessing(self, preprocessing=None, options=None,
                            ret_details=False):
        """Perform curve preprocessing steps

        Parameters
        ----------
        preprocessing: list
            A list of preprocessing method identifiers that are
            stored in the `nanite.preproc.PREPROCESSORS` list.
            If set to `None`, `self.preprocessing` will be
            used.
        options: dict of dict
            Dictionary of keyword arguments for each preprocessing
            step (if applicable)
        ret_details:
            Return preprocessing details dictionary
        """
        if preprocessing is None:
            preprocessing = self.preprocessing

        if options is None:
            options = self.preprocessing_options

        if "preprocessing" in self.fit_properties:
            preproc_past = [self.fit_properties["preprocessing"],
                            self.fit_properties["preprocessing_options"]]
        else:
            preproc_past = []

        if ((preproc_past != [preprocessing, options])
                or (not self._preprocessing_details and ret_details)):
            # Remember initial fit parameters for user convenience
            fp = self.fit_properties
            # Reset fit properties
            fp.reset()
            # Set preprocessing options
            fp["preprocessing"] = preprocessing
            fp["preprocessing_options"] = options
            # Reset rating
            self._rating = None
            # Apply preprocessing
            # (This will call `AFMData.reset_data` on self)
            details = preproc.apply(apret=self,
                                    identifiers=preprocessing,
                                    options=options,
                                    ret_details=ret_details)
            self._preprocessing_details = details
            # Check availability of axes
            for ax in ["x_axis", "y_axis"]:
                # make sure the fitting axes are defined
                if ax in fp and not fp[ax] in self:
                    fp.pop(ax)

        # remember preprocessing
        self.preprocessing = preprocessing
        self.preprocessing_options = copy.deepcopy(options)

        return self._preprocessing_details

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

    def estimate_contact_point_index(self, method="deviation_from_baseline"):
        """Estimate the contact point index

        See the `poc` submodule for more information.
        """
        idp = poc.compute_poc(force=np.array(self["force"], copy=True),
                              method=method)
        return idp

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
        preprocessing_options: list of str
            Preprocessing
        segment: str
            Segment index (e.g. 0 for approach)
        weight_cp: float
            Weight the contact point region which shows artifacts
            that are difficult to model with e.g. Hertz.
        gcf_k: float
            Geometrical correction factor :math:`k` for non-single-contact
            data. The measured indentation is multiplied by this factor to
            correct for experimental geometries during fitting,
            e.g. ``gcf_k=0.5`` for parallel-place compression.
        optimal_fit_edelta: bool
            Search for the optimal fit by varying the maximal
            indentation depth and determining a plateau in the
            resulting Young's modulus (fitting parameter "E").
        """
        if "preprocessing" in kwargs:
            options = kwargs.get("preprocessing_options",
                                 self.preprocessing_options)
            self.apply_preprocessing(preprocessing=kwargs["preprocessing"],
                                     options=options)
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
        return model.compute_anc_parms(idnt=self,
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
        names: list of str
            Only use these features for rating
        lda: bool
            Perform linear discriminant analysis

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
        if self.fit_properties and "hash" in self.fit_properties:
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
