"""Test input/output of user rating data"""
import pathlib
import shutil
import tempfile

import h5py
import numpy as np
import pytest

from nanite import model, IndentationGroup
from nanite.rate.io import RateManager, hdf5_rated, load_hdf5, save_hdf5
from nanite.rate.rater import IndentationRater

datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def setuph5(ret_idnt=False, path=jpkfile):
    tdir = tempfile.mkdtemp(prefix="test_nanite_rate_io_")
    tdir = pathlib.Path(tdir)
    h5path = tdir / "simple.h5"
    grp = IndentationGroup(path)
    for idnt in grp:
        idnt.apply_preprocessing(["compute_tip_position"])

        inparams = model.model_hertz_paraboloidal.get_parameter_defaults()
        inparams["baseline"].vary = True
        inparams["contact_point"].set(1.8e-5)

        # Fit with absolute full range
        idnt.fit_model(model_key="hertz_para",
                       params_initial=inparams,
                       range_x=(0, 0),
                       range_type="absolute",
                       x_axis="tip position",
                       y_axis="force",
                       segment="approach",
                       weight_cp=False)

        save_hdf5(h5path=h5path,
                  indent=idnt,
                  user_rate=5,
                  user_name="hans",
                  user_comment="this is a comment",
                  h5mode="a")
    else:
        idnt = grp[0]

    if ret_idnt:
        return tdir, h5path, idnt
    else:
        return tdir, h5path


def test_hdf5rated():
    tdir, h5path, idnt = setuph5(ret_idnt=True)
    is_rated, rating, comment = hdf5_rated(h5path, idnt)
    assert is_rated
    assert rating == 5
    assert comment == "this is a comment"
    shutil.rmtree(tdir, ignore_errors=True)


def test_rate_manager_basic():
    tdir, h5path, idnt = setuph5(ret_idnt=True)
    rmg = RateManager(h5path)
    # sanity checks
    rr = rmg.ratings[0]
    assert rr["name"] == "hans"
    assert rr["rating"] == 5
    # file name preserved
    ds = rmg.datasets[0]
    assert jpkfile.name in ds.path.name
    # features are the same
    idr = IndentationRater
    ss = rmg.samples[0]
    assert np.allclose(ss, idr.compute_features(idnt))
    # rates
    assert np.ndarray.item(rmg.get_rates(which="user")) == 5
    # This will fail when the hyper-parameters for "Extra Trees" change
    # or when new features are added.
    assert np.allclose(
        np.ndarray.item(rmg.get_rates(which="Extra Trees",
                                      training_set="zef18")),
        3.5492840783289035)
    shutil.rmtree(tdir, ignore_errors=True)


def test_rate_manager_crossval():
    path = datadir / "map-data-reference-points.jpk-force-map"
    tdir, h5path = setuph5(path=path)
    rmg = RateManager(h5path)
    cv = rmg.get_cross_validation_score(regressor="Extra Trees",
                                        training_set=None,
                                        n_splits=2,
                                        random_state=42)
    assert np.all(cv == 0)
    shutil.rmtree(tdir, ignore_errors=True)


def test_rate_manager_export():
    tdir, h5path = setuph5()
    rmg = RateManager(h5path)
    rmg.export_training_set(tdir)

    ss = rmg.samples[0]
    feats = IndentationRater.get_feature_names()

    for ff, si in zip(feats, ss):
        fi = np.loadtxt(tdir / "train_{}.txt".format(ff))
        # :.2e, because features are not stored with high accuracy
        assert np.ndarray.item(fi) == float("{:.2e}".format(si))

    shutil.rmtree(tdir, ignore_errors=True)


def test_rate_manager_get_ts():
    path = datadir / "map-data-reference-points.jpk-force-map"
    tdir, h5path = setuph5(path=path)
    rmg = RateManager(h5path)
    x2, _ = rmg.get_training_set(which_type="binary")
    assert np.all(x2 == 1)
    x3, _ = rmg.get_training_set(which_type="continuous",
                                 prefilter_binary=True)
    x4, _ = rmg.get_training_set(remove_nans=True)
    assert np.all(np.hstack((x2, x3)) == x4)
    shutil.rmtree(tdir, ignore_errors=True)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_rate_manager_get_ts_bad():
    path = datadir / "bad_map-data-2013.05.27-13.50.21.jpk-force-map"
    tdir, h5path = setuph5(path=path)
    rmg = RateManager(h5path)
    x2, _ = rmg.get_training_set(which_type="binary")
    assert np.allclose(x2.flatten(), [1, 0, 1])
    x3, _ = rmg.get_training_set(which_type="continuous",
                                 prefilter_binary=True)
    assert x3.size == 0
    x4, _ = rmg.get_training_set(remove_nans=True)
    assert x4.size == 0
    shutil.rmtree(tdir, ignore_errors=True)


def test_rate_manager_get_ts_single():
    tdir, h5path = setuph5()
    rmg = RateManager(h5path)
    x2, _ = rmg.get_training_set(which_type="binary")
    assert np.all(x2 == 1)
    x3, _ = rmg.get_training_set(which_type="continuous",
                                 prefilter_binary=True)
    x4, _ = rmg.get_training_set(remove_nans=True)
    assert np.all(np.hstack((x2, x3)) == x4)
    shutil.rmtree(tdir, ignore_errors=True)


def test_write():
    tdir, h5path, idnt = setuph5(ret_idnt=True)

    with h5py.File(str(h5path), mode="r") as hi:
        # experimental data
        assert "4443b7" in hi["data"]
        # a few attributes
        attrs = hi["analysis/4443b7_0"].attrs
        assert attrs["fit model_key"] == "hertz_para"
        assert not attrs["fit optimal_fit_edelta"]
        assert attrs["fit preprocessing"] == "compute_tip_position"
        assert np.allclose(hi["analysis/4443b7_0"]["fit"], idnt.data["fit"],
                           equal_nan=True)

    shutil.rmtree(tdir, ignore_errors=True)


def test_write_read():
    tdir, h5path = setuph5()

    datalist = load_hdf5(h5path)
    assert datalist[0]["rating"] == 5
    shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
