"""Test usage of user-defined training set"""
import pathlib
import tempfile

import numpy as np

from nanite import IndentationGroup
from nanite.rate import IndentationRater


data_path = pathlib.Path(__file__).parent / "data"
jpkfile = data_path / "fmt-jpk-fd_map-data-reference-points.jpk-force-map"


def setup_training_set(n=300):
    tdir = tempfile.mkdtemp(prefix="test_nanite_rate_ts_")
    tdir = pathlib.Path(tdir)
    np.random.set_state(np.random.RandomState(47).get_state())
    for bb in IndentationRater.get_feature_names(which_type="binary"):
        bvals = np.random.choice([0, 1], size=n, p=[.05, .95])
        np.savetxt(tdir / f"train_{bb}.txt", bvals)
    for cc in IndentationRater.get_feature_names(which_type="continuous"):
        cvals = np.random.random_sample(size=n)
        np.savetxt(tdir / f"train_{cc}.txt", cvals)
    rating = np.random.choice(range(11), size=n)
    np.savetxt(tdir / "train_response.txt", rating)
    return tdir


def test_user_training_set():
    tdir = setup_training_set()
    # load a curve
    idnt = IndentationGroup(jpkfile)[0]
    # fit it
    idnt.fit_model(model_key="sneddon_spher_approx",
                   preprocessing=["compute_tip_position",
                                  "correct_force_offset"])
    r1 = idnt.rate_quality(regressor="Extra Trees", training_set="zef18")
    assert r1 > 9, "sanity check"
    r2 = idnt.rate_quality(regressor="Extra Trees", training_set=tdir)
    assert 4 < r2 < 5, "with the given random state we end up at 4.55"


def test_training_set_impute_nans():
    tdir = setup_training_set()
    # edit one of the training feature data to contain an inf value
    fpath = tdir / "train_feat_con_apr_flatness.txt"
    rpath = tdir / "train_response.txt"

    fdat = np.loadtxt(fpath)
    fdat[10] = 1.1
    fdat[11] = 1.2
    fdat[12] = np.nan
    fdat[13] = np.nan
    np.savetxt(fpath, fdat)

    rdat = np.loadtxt(rpath)
    rdat[rdat == 0] = 1
    rdat[10] = 0
    rdat[11] = 0
    rdat[12] = 0
    rdat[13] = 0
    np.savetxt(rpath, rdat)

    samples, response, names = IndentationRater.load_training_set(
        path=tdir,
        which_type="continuous",
        impute_zero_rated_nan=True,  # This should be the default
        replace_inf=True,  # This should be the default
        ret_names=True,
    )
    idf = names.index("feat_con_apr_flatness")
    data = samples[:, idf]

    assert np.allclose(data[10], 1.1)
    assert np.allclose(data[11], 1.2)
    assert np.allclose(data[12], 1.15)
    assert np.allclose(data[13], 1.15)


def test_training_set_inf_values():
    tdir = setup_training_set()
    # edit one of the training feature data to contain an inf value
    fpath = tdir / "train_feat_con_apr_flatness.txt"
    fdat = np.loadtxt(fpath)
    fdat[10] = np.inf
    fdat[11] = -np.inf
    fdat[12] = 1.5
    fdat[13] = -1.4
    np.savetxt(fpath, fdat)

    samples, response, names = IndentationRater.load_training_set(
        path=tdir,
        which_type="continuous",
        replace_inf=True,  # This should be the default
        ret_names=True,
    )
    idf = names.index("feat_con_apr_flatness")
    data = samples[:, idf]

    assert np.allclose(data[10], 3)
    assert np.allclose(data[11], -3)
    assert np.allclose(data[12], 1.5)
    assert np.allclose(data[13], -1.4)
