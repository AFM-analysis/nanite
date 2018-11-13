"""Save and load user-rated datasets"""
from functools import lru_cache
import hashlib
import pathlib
import shutil
import tempfile
import time

import h5py
import lmfit
import numpy as np
from sklearn import model_selection

from ..group import IndentationGroup
from . import rater


class RateManager():
    def __init__(self, path, verbose=0):
        """Manage user-defined rates"""
        #: Path to the manual ratings (directory or .h5 file)
        self.path = pathlib.Path(path)
        #: verbosity level
        self.verbose = verbose
        self._ratings = None

    @staticmethod
    def _get_samples(path, verbose=0):
        rm = RateManager(path, verbose=verbose)
        samples = []
        idr = rater.IndentationRater
        for ds in rm.datasets:
            assert "success" in ds.fit_properties
            features = idr.compute_features(ds)
            samples.append(features)
        return np.array(samples, dtype=float)

    @property
    def datasets(self):
        return [r["data_set"] for r in self.ratings]

    @property
    def ratings(self):
        if self._ratings is None:
            self._ratings = load(self.path, verbose=self.verbose)
        return self._ratings

    @property
    def samples(self):
        """The individual sample ratings computed by afmlib"""
        return RateManager._get_samples(self.path, verbose=self.verbose)

    def export_training_set(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        raters = rater.IndentationRater.get_feature_funcs()
        samples = self.samples
        for ii, rti in enumerate(raters):
            rpath = path / "train_{}.txt".format(rti[0])
            np.savetxt(rpath, samples[:, ii].flatten(), fmt="%.2e")
        user = self.get_rates(which="user")
        upath = str(path / "train_response.txt")
        np.savetxt(upath, user.flatten(), fmt="%.2e")

    def get_cross_validation_score(self, regressor, training_set=None,
                                   n_splits=20, random_state=42):
        """Regressor cross-validation scoring

        Cross-validation is used to identify regressors that
        over-fit the train set by splitting the train set into
        multiple learn/test sets and quantifying the regressor
        performance for each split.

        Parameters
        ----------
        regressor: str or RegressorMixin
            If a string, must be in `reg_names`.
        training_set: X, y
            If given, do not use self.samples

        Notes
        -----
        A :class:`skimage.model_selection.KFold` cross validator is used
        in combination with the mean squared error score.

        Cross-validation score is computed from samples that are filtered
        with the binary features and only from samples that do not contain
        any nan values.
        """
        ir = rater.get_rater(regressor)

        if training_set:
            X, y = training_set
        else:
            # remove binary features and nans otherwise cross-validation
            # will not work
            X, y = self.get_training_set(prefilter_binary=True,
                                         remove_nans=True)

        # The score is maximized, therefore it is "neg_"
        scoring = 'neg_mean_squared_error'
        loo = model_selection.KFold(n_splits=n_splits, shuffle=True,
                                    random_state=random_state)
        scores = model_selection.cross_val_score(ir.pipeline, X, y,
                                                 scoring=scoring, cv=loo)
        return -scores

    def get_rates(self, which="user", training_set="zef18"):
        """
        which: str
            Which rating to return: "user" or a regressor name
        """
        if which == "user":
            rtngs = np.array([ri["rating"]
                              for ri in load(self.path, meta_only=True)])
        else:
            rt = rater.get_rater(regressor=which, training_set=training_set)
            rtngs = rt.rate(self.samples)
        return rtngs

    def get_training_set(self, which_type="all", prefilter_binary=False,
                         remove_nans=False, transform=False):
        """Return (X, y) training set"""
        X = self.samples
        y = self.get_rates(which="user")

        if prefilter_binary:
            # remove samples with `False` in a binary feature
            ir = rater.IndentationRater(regressor=None)
            bnames, bind = ir.get_feature_names(which_type="binary",
                                                ret_indices=True)
            if bnames:
                X_bool = X[:, bind]

                X_f = []
                y_f = []
                for ii in range(len(y)):
                    if ir._pre_rate(X_bool[ii]):
                        X_f.append(X[ii])
                        y_f.append(y[ii])
                X, y = np.array(X_f), np.array(y_f)

        # must come after prefilter_binary
        if which_type != "all":
            _unames, indices = rater.IndentationRater.get_feature_names(
                which_type=which_type,
                ret_indices=True)
            # remove unwanted features
            if X.size:
                X = X[:, list(indices)]

        if remove_nans:
            X_n = []
            y_n = []
            for ii in range(len(y)):
                if not np.sum(np.isnan(X[ii])):
                    X_n.append(X[ii])
                    y_n.append(y[ii])
            X, y = np.array(X_n), np.array(y_n)

        if transform:
            # Transform data
            ir = rater.IndentationRater(regressor=None,
                                        training_set=(X, y)
                                        )
            X = ir.pipeline.transform(X)
        return X, y


@lru_cache(maxsize=100)
def hash_file(path, blocksize=65536):
    """Compute sha256 hex-hash of a file

    Parameters
    ----------
    path: str or pathlib.Path
        path to the file
    blocksize: int
        block size read from the file

    Returns
    -------
    hex: str
        The first six characters of the hash
    """
    fname = pathlib.Path(path)
    hasher = hashlib.sha256()
    with fname.open('rb') as fd:
        buf = fd.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = fd.read(blocksize)
    return hasher.hexdigest()[:6]


def load(path, meta_only=False, verbose=0):
    """

    Notes
    -----
    The .fit_properties attribute of each Indentation instance
    is overridden by a simple dictionary,
    so its functionalities are not available anymore.
    """
    path = pathlib.Path(path)
    ratings = []
    if path.is_dir():
        if verbose:
            print("Performing iterative folder search.")
        for fpath in sorted(path.rglob("*.h5")):
            ratings += load(fpath, meta_only=meta_only)
    elif path.suffix == ".h5":
        if verbose:
            print("Loading file '{}'... ".format(path), end="", flush=True)
        ratings += load_hdf5(path, meta_only=meta_only)
        if verbose:
            print("Done.")
    return ratings


def load_hdf5(path, meta_only=False):
    ratings = []
    path = pathlib.Path(path)
    # temporary directory for original data
    tdir = tempfile.mkdtemp(prefix="nanite_rate_data_")
    with h5py.File(path, mode="r") as h5:
        if not meta_only:
            # extract experimental data
            dataset_dict = {}
            for dkey in h5["data"]:
                dset = h5["data"][dkey]
                dbin = dset.value
                name = dkey + "_" + pathlib.Path(dset.attrs["path"]).name
                dpath = pathlib.Path(tdir) / name
                dbin.tofile(str(dpath))
                dataset_dict[dkey] = IndentationGroup(dpath)
        # load individual curves
        for akey in h5["analysis"]:
            h5gr = h5["analysis"][akey]
            attrs = h5gr.attrs
            if not meta_only:
                indent = dataset_dict[attrs["data hash"]][attrs["data enum"]]
                indent["fit"] = h5gr["fit"].value
                indent["fit range"] = h5gr["fit range"].value
                indent["force"] = h5gr["force"].value
                indent["fit residuals"] = h5gr["fit residuals"].value
                indent["tip position"] = h5gr["tip position"].value
                indent["segment"] = h5gr["segment"].value
            fit_properties = {}
            fkeys = [key for key in attrs if key.startswith("fit ")]
            for fkey in fkeys:
                key = fkey[4:]
                val = attrs[fkey]
                if key.startswith("params"):
                    parms = lmfit.Parameters()
                    parms.loads(val)
                    val = parms
                elif key == "preprocessing":
                    val = val.split(",")
                elif key == "range_x":
                    val = val.strip("[]() ").split(",")
                    val = (float(val[0]), float(val[1]))
                fit_properties[key] = val
            rating = {
                "name": attrs["user name"],
                "rating": attrs["user rate"],
                "comment": attrs["user comment"],
                "enum": attrs["data enum"],
                "fit properties": fit_properties,
            }
            if not meta_only:
                # This overrides the FitProperties class!
                indent.fit_properties = fit_properties
                rating["data_set"] = indent
            ratings.append(rating)

    shutil.rmtree(tdir, ignore_errors=True)
    return ratings


def save_hdf5(h5path, indent, user_rate, user_name, user_comment, h5mode="a"):
    """Store all relevant data of a user rating into an hdf5 file

    Parameters
    ----------
    h5path: str
        Path to HDF5 file where data will be stored
    indent: nanite.Indentation
        The experimental data processed and fitted with nanite
    user_rate: float
        Rating given by the user
    user_name: str
        Name of the rating user
    """
    dkw = {"fletcher32": True,
           "compression": "gzip",
           "compression_opts": 9}
    with h5py.File(h5path, mode=h5mode) as h5:
        # store raw experimental data as binary array
        data = h5.require_group("data")
        dhash = hash_file(indent.path)
        if dhash not in data:
            meas = data.create_dataset(
                dhash,
                data=np.fromfile(str(indent.path), dtype=bool),
                **dkw
            )
            meas.attrs["path"] = str(indent.path)
        # store indentation data along with the user rate
        ana = h5.require_group("analysis")
        idd = "{}_{}".format(dhash, indent.enum)
        if idd in ana:
            # Only allow overriding of user data if fit matches.
            # Otherwise, the rating might be wrong.
            if not np.allclose(indent["fit"], ana[idd]["fit"], equal_nan=True):
                raise ValueError(
                    "Cannot store rating for different fit in hdf5 file!")
            out = ana[idd]
        else:
            out = ana.create_group(idd)
            out.attrs["data enum"] = indent.enum
            out.attrs["data hash"] = dhash
            for key in indent.fit_properties:
                val = indent.fit_properties[key]
                if key.startswith("params_"):
                    val = val.dumps()
                elif key == "preprocessing":
                    val = ",".join(val)
                elif key == "range_x":
                    val = str(val)
                out.attrs["fit {}".format(key)] = val

            out.create_dataset("fit",
                               data=indent["fit"].values,
                               **dkw)
            out.create_dataset("fit range",
                               data=indent["fit range"].values,
                               **dkw)
            out.create_dataset("force",
                               data=indent["force"].values,
                               **dkw)
            out.create_dataset("fit residuals",
                               data=indent["fit residuals"].values,
                               **dkw)
            out.create_dataset("tip position",
                               data=indent["tip position"].values,
                               **dkw)
            out.create_dataset("segment",
                               data=indent["segment"].values,
                               **dkw)
        # update user data in any case
        out.attrs["user comment"] = user_comment
        out.attrs["user name"] = user_name
        out.attrs["user rate"] = user_rate
        out.attrs["user time"] = time.time()
        out.attrs["user time str"] = time.ctime()


def hdf5_rated(h5path, indent):
    """Test whether an indentation has already been rated

    Returns
    -------
    is_rated, rating, comment
    """
    is_rated = False
    rating = -1
    comment = ""
    h5path = pathlib.Path(h5path)
    if h5path.exists():
        with h5py.File(h5path, mode="r") as h5:
            if "analysis" in h5:
                ana = h5["analysis"]
                dhash = hash_file(indent.path)
                idd = "{}_{}".format(dhash, indent.enum)
                if idd in ana:
                    is_rated = True
                    rating = ana[idd].attrs["user rate"]
                    comment = ana[idd].attrs["user comment"]
    return is_rated, rating, comment
