import pathlib
from pkg_resources import resource_filename
from typing import List, Literal

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import FunctionTransformer


from .features import IndentationFeatures
from .regressors import reg_dict, reg_names, reg_trees


class IndentationRater(IndentationFeatures):
    def __init__(self, regressor=None, scale=None, lda=None,
                 training_set=None, names=None,
                 weight=True, sample_weight=None,
                 *args, **kwargs):
        """Rate quality

        Parameters
        ----------
        regressor: sciki-learn RegressorMixin
            The regressor used for rating
        scale: bool
            If True, apply a Standard Scaler. If a regressor based on
            decision trees is used, the Standard Scaler is not used
            by default, otherwise it is.
        lda: bool
            If True, apply a Linear Discriminant Analysis (LDA). If a
            regressor based on a decision tree is used, LDA is not
            used by default, otherwise it is.
        training_set: tuple of (X, y)
            The training set (samples, response)
        names: list of str
            Feature names to use
        weight: bool
            Weight the input samples by the number of occurrences
            or with `sample_weight`. For tree-based classifiers, set this
            to True to avoid bias.
        sample_weight: list-like
            The sample weights. If set to `None` sample weights
            are computed from the training set.
        *args: list
            Positional arguments for :class:`IndentationFeatures`
        **kwargs:
            Keyword arguments for :class:`IndentationFeatures`

        See Also
        --------
        sklearn.preprocessing.StandardScaler:
            Standard scaler
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
            Linear discriminant analysis
        nanite.rate.regressors.reg_trees:
            List of regressors that are identified as tree-based
        """
        if regressor is not None:
            _name = regressor.__class__.__name__
            if lda is None:
                lda = False if _name in reg_trees else True
            if scale is None:
                scale = False if _name in reg_trees else True

        # training set
        if training_set is None:
            # default
            training_set = self.load_training_set(names=names)
        # sample weights
        if sample_weight is None:
            sample_weight = self.compute_sample_weight(*training_set)
        steps = []

        # scaling (does not affect decision trees / random forests)
        if scale:
            steps.append(StandardScaler())
        # linear discriminant analysis
        if lda:
            steps.append(LinearDiscriminantAnalysis())

        if regressor is not None:
            steps.append(regressor)

        if len(steps) == 0:
            dummy = FunctionTransformer(lambda x: x)
            steps.append(dummy)

        #: sklearn pipeline with transforms (and regressor if given)
        self.pipeline = make_pipeline(*steps)

        fit_params = {}
        if regressor is not None and weight:
            # set weighting for regressor
            key = "{}__sample_weight".format(self.pipeline.steps[-1][0])
            fit_params[key] = sample_weight

        if regressor is not None:
            self.pipeline.fit(*training_set, **fit_params)

        names = self.get_feature_names(names=names, which_type="all")

        #: feature names used by the regressor pipeline
        self.names = sorted(names)
        super(IndentationRater, self).__init__(*args, **kwargs)

    def _pre_rate(self, bsample):
        """exclude based on boolean training set"""
        if np.sum(bsample == 0):
            # bad curve
            return False
        else:
            # good curve
            return True

    def _rate(self, sample):
        gd = self.pipeline.predict(np.atleast_2d(sample))
        return gd[0]

    @staticmethod
    def compute_sample_weight(X, y):
        """Weight samples according to occurrence in y"""
        if not np.all(np.array(y, dtype=int) == y):
            msg = "Only integer ratings allowed."
            raise NotImplementedError(msg)
        weight = np.zeros(y.shape[0], dtype=float)
        for ii in range(11):
            idxii = y == ii
            occur = np.sum(idxii)
            if occur:
                # Sometimes the training set is not large enough.
                # If no occurences were found, the weights remain
                # zero.
                weight[idxii] = 1 / occur
        # normalize
        weight /= np.sum(weight)
        return weight

    @staticmethod
    def get_training_set_path(label="zef18"):
        """Return the path to a training set shipped with nanite

        Training sets are stored in the `nanite.rate`
        module path with ``ts_`` prepended to `label`.
        """
        data_loc = "nanite.rate"
        resp_path = resource_filename(data_loc, "ts_{}".format(label))
        return resp_path

    @classmethod
    def load_training_set(
            cls,
            path: pathlib.Path | str = None,
            names: List[str] = None,
            which_type: Literal["all", "binary", "continuous"] | List = None,
            replace_inf: bool = True,
            impute_zero_rated_nan: bool = True,
            remove_nan: bool = True,
            ret_names: bool = False):
        """Load a training set from a directory

        Parameters
        ----------
        path: pathlib.Path or str
            Optional path to the training set directory. If none
            is specified, the default "zef18" is loaded.
        names: list of str
            List of features to use, defaults to all features.
        which_type: str
            Which type of feature to return see :const:`.VALID_FEATURE_TYPES`
            for valid options. By default, only the "continuous" features
            are imported. The "binary" features are not needed for training;
            they are used to sort out new force-distance data.
        replace_inf: bool
            Replace infinity-valued feature values with
            `2 * sign * max(abs(values))`.
        impute_zero_rated_nan: bool
            If there are nan-valued features that have a zero response
            (rated worst), replace those feature values with the mean
            of the zero-response features that are not nan-valued.
        remove_nan: bool
            Remove any nan-valued features (after `impute_zero_rated_nan`
            was applied). This is necessary, since skimage cannot handle
            nan-valued sample values.
        ret_names: bool
            Return the names of the features in addition to the samples
            and response.

        Returns
        -------
        samples: 2d ndarray
            Sample values with axes `(data_size, num_features)`
        response: 1d ndarray
            Response array of length `data_size`
        names: list, optional
            List of feature names corresponsing to axis `1` in `samples`
        """
        if which_type is None:
            which_type = ["continuous"]
        fnames = cls.get_feature_names(which_type=which_type, names=names)
        sample_paths = []
        if path is None:
            path = cls.get_training_set_path()
        path = pathlib.Path(path).resolve()

        resp_path = str(path / "train_response.txt")
        for fn in fnames:
            resf = str(path / "train_{}.txt".format(fn))
            sample_paths.append(resf)

        samples = [np.loadtxt(sp, dtype=float, ndmin=2) for sp in sample_paths]
        samples = np.concatenate(samples, axis=1)
        response = np.loadtxt(resp_path, dtype=float)

        # Deal with NaN-valued feature data with a response of 0.
        if impute_zero_rated_nan:
            resp0 = response == 0
            # For each feature, find values that are NaN where the
            # response is zero. Those values are then be set to values
            # where the response is zero and the values are not NaN.
            for ii, fn in enumerate(fnames):
                # locations where the feature is nan
                fdat = samples[:, ii]
                fnans = np.isnan(fdat)
                # locations where feature is nan AND response is 0
                # (those are the locations we would like to change)
                coloc = np.logical_and(resp0, fnans)
                # location where the feature is not nan AND response is 0
                # (those are the reference locations)
                ref = np.logical_and(resp0, ~fnans)
                if np.any(coloc) and np.any(ref):
                    # We have values
                    refval = np.mean(fdat[ref])
                    samples[coloc, ii] = refval

        # Deal with remaining NaN-valued feature data.
        if remove_nan:
            # Remove nan-values from training set
            valid = ~np.array(np.sum(np.isnan(samples), axis=1), dtype=bool)
            samples = samples[valid, :]
            # remove corresponding responses
            response = response[valid]

        # Deal with infinite feature data.
        if replace_inf:
            for ii in range(len(fnames)):
                si = samples[:, ii]
                isinf = np.isinf(si)
                if np.any(isinf):
                    extreme = np.nanmax(np.abs(si[~isinf]))
                    posinf = np.isposinf(si)
                    if np.any(posinf):
                        samples[posinf, ii] = 2 * extreme
                    neginf = np.isneginf(si)
                    if np.any(neginf):
                        samples[neginf, ii] = -2 * extreme

        res = [samples, response]

        if ret_names:
            res.append(fnames)

        return res

    def rate(self, samples=None, datasets=None):
        """Perform rating step

        Parameters
        ----------
        samples: 1d or 2d ndarray (cast to 2d ndarray) or None
            Measured samples, if set to None, `dataset` must be given.
        dataset: list of nanite.Indentation
            Full, fitted measurement

        Returns
        -------
        ratings: list
            Resulting ratings
        """
        if samples is None and datasets is None:
            # use dataset from IndentationFeature
            datasets = [self.dataset]
        elif datasets is None:
            # distinguish between binary and other samples
            fsamples = []
            bsamples = []
            fnames = self.get_feature_names(
                names=self.names,
                which_type=["continuous"])
            for samp in samples:
                fsamp = []  # continuous samples
                bsamp = []  # binary samples
                for ii, name in enumerate(self.names):
                    if name in fnames:
                        fsamp.append(samp[ii])
                    else:
                        assert name.startswith("feat_bin_")
                        bsamp.append(samp[ii])
                fsamples.append(fsamp)
                bsamples.append(bsamp)
        else:
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            fsamples = []
            bsamples = []
            # continuous features
            for idnt in datasets:
                samp = self.compute_features(
                    idnt=idnt,
                    names=self.names,
                    which_type=["continuous"])
                fsamples.append(samp)
            # binary features
            for idnt in datasets:
                bsamp = self.compute_features(idnt=idnt,
                                              names=self.names,
                                              which_type="binary")
                bsamples.append(bsamp)

        fsamples = np.atleast_2d(fsamples)
        bsamples = np.atleast_2d(bsamples)

        ratings = []
        for bsamp, fsamp in zip(bsamples, fsamples):
            if not self._pre_rate(bsamp):
                # certainly a bad curve
                gd = 0
            elif np.isnan(np.sum(fsamp)):
                # ignore nan-valued samples
                gd = -1
            else:
                gd = self._rate(fsamp)
            ratings.append(gd)

        return np.array(ratings).flatten()


def get_available_training_sets():
    """List of internal training sets"""
    data_loc = "nanite"
    resp_path = resource_filename(data_loc, "rate")
    avail = []
    for pp in pathlib.Path(resp_path).glob("ts_*"):
        avail.append(pp.name[3:])
    return sorted(avail)


def get_rater(regressor, training_set="zef18", names=None,
              lda=None, **reg_kwargs):
    """Convenience method to get a rater

    Parameters
    ----------
    regressor: str or RegressorMixin
        If a string, must be in `reg_names`.
    training_set: str or pathlib.Path or tuple (X, y)
        A string label representing a training set shipped with
        nanite, the path to a training set, or a tuple
        representing the training set (samples, response)
        for use with sklearn.
    names: list of str
        Only use these features for rating
    lda: bool
        Perform linear discriminant analysis

    Returns
    -------
    irater: nanite.IndentationRater
        The rating instance.
    """
    avr = get_available_training_sets()
    if isinstance(training_set, tuple):
        pass
    else:
        if training_set in avr:
            ts_path = IndentationRater.get_training_set_path(
                label=training_set)
        else:
            ts_path = training_set
        training_set = IndentationRater.load_training_set(
            path=ts_path,
            names=names)

    if len(training_set) != 2:
        raise ValueError("Expected training_set of the form (X, y)!")

    if isinstance(regressor, str):
        if regressor not in reg_names:
            msg = "Unknown regressor name: '{}'!".format(regressor) \
                  + " Please pass your own sklearn RegressorMixin."
            raise ValueError(msg)
        reg_cl, default_kw = reg_dict[regressor]
        kw = default_kw.copy()
        kw.update(reg_kwargs)
        regr = reg_cl(**kw)
    else:
        regr = regressor

    rater = IndentationRater(regressor=regr,
                             training_set=training_set,
                             names=names,
                             lda=lda)
    return rater
