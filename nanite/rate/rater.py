import pathlib
from pkg_resources import resource_filename

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
            If True, apply StandardScaler (important for non-tree
            based regressors).
        lda: bool
            If True, apply LinearDiscriminantAnalysis. See notes for defaults.
        training_set: tuple of (X, y)
            The training set (samples, response)
        names: list of str
            Feature names to use
        weight: bool
            Weight the input samples by the number of occurences
            or with `sample_weight`. For tree-based classifiers, set this
            to True to avoid bias.
        sample_weight: list-like
            The sample weights. If set to `None` sample weights
            are computed from the training set.
        *args, **kwargs:
            Arguments for the :class:`IndentationFeatures` base class.

        Notes
        -----
        The default value for lda is different
        depending on the regressor (see source code below).
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
        """Weight samples according to occurence in y"""
        occur = np.zeros(y.shape[0], dtype=float)
        for ii in range(11):
            idxii = y == ii
            occur[idxii] = np.sum(idxii)
        weight = 1/occur
        # normalize
        weight /= np.sum(weight)
        return weight

    @staticmethod
    def get_training_set_path(label="zef18"):
        """Return the path to a training set shipped with afmfit

        Training sets are stored in the `afmfit.rate`
        module path with "ts_" prepended to `label`.
        """
        data_loc = "afmfit.rate"
        resp_path = resource_filename(data_loc, "ts_{}".format(label))
        return resp_path

    @classmethod
    def load_training_set(cls, path=None, names=None,
                          which_type=["continuous", "discrete"],
                          remove_nan=True, ret_names=False,
                          ret_sample_weights=False):
        """
        currently, only "float" features are used for sklearn stuff
        """
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
        if remove_nan:
            # Remove nan-values from training set
            valid = ~np.isnan(np.sum(samples, axis=1))
            samples = samples[valid, :]
            # remove corresponding responses
            response = response[valid]

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
        dataset: list of afmfit.Indentation
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
                which_type=["continuous", "discrete"])
            for samp in samples:
                fsamp = []  # continuous or discrete samples
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
            # float features
            for dataset in datasets:
                samp = self.compute_features(
                    dataset=dataset,
                    names=self.names,
                    which_type=["continuous", "discrete"])
                fsamples.append(samp)
            for dataset in datasets:
                bsamp = self.compute_features(dataset=dataset,
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


def get_rater(regressor, training_set="zef18", names=None,
              lda=None, **reg_kwargs):
    """Convenience method to get a rater

    Parameters
    ----------
    regressor: str or RegressorMixin
        If a string, must be in `reg_names`.
    training_set: str or pathlib.Path or tuple (X, y)
        A string label representing a training set shipped with
        afmfit, the path to a training set, or a tuple
        representing the training set (samples, response)
        for use with sklearn.

    Returns
    -------
    irater: IndentationRater
        The rating instance.
    """

    if isinstance(training_set, pathlib.Path):
        training_set = IndentationRater.load_training_set(
            path=training_set,
            names=names)
    elif isinstance(training_set, str):
        ts_path = IndentationRater.get_training_set_path(label=training_set)
        training_set = IndentationRater.load_training_set(
            path=ts_path,
            names=names)
    else:
        pass
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
