import pathlib
import tempfile

import numpy as np

from nanite.rate import IndentationRater

import pytest

pytest.importorskip("appdirs", reason="appdirs not installed")

from nanite.cli import profile, rating  # noqa: E402


data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"
jpkfile2 = data_path / "fmt-jpk-fd_map-data-reference-points.jpk-force-map"


def setup_training_set(n=300):
    tdir = tempfile.mkdtemp(prefix="test_nanite_rate_ts_")
    tdir = pathlib.Path(tdir)
    np.random.set_state(np.random.RandomState(47).get_state())
    for bb in IndentationRater.get_feature_names(which_type="binary"):
        bvals = np.random.choice([0, 1], size=n, p=[.05, .95])
        np.savetxt(tdir / "train_{}.txt".format(bb), bvals)
    for cc in IndentationRater.get_feature_names(which_type="continuous"):
        cvals = np.random.random_sample(size=n)
        np.savetxt(tdir / "train_{}.txt".format(cc), cvals)
    thisrating = np.random.choice(range(11), size=n)
    np.savetxt(tdir / "train_response.txt", thisrating)
    return tdir


def test_fit_data():
    # use temporary file
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_cli_rate_")
    name = pathlib.Path(name)
    profile.Profile(path=name)

    # this will fit with the profile default parameters
    idnt = rating.fit_data(path=jpkfile, profile_path=name)
    assert idnt.path == jpkfile
    assert idnt.fit_properties["success"]


def test_fit_data_with_user_training_set():
    tdir = setup_training_set()
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_cli_rate_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    pf["rating training set"] = tdir
    pf["fit param R value"] = 37.28e-6 / 2
    pout = tempfile.mkdtemp(prefix="test_nanite_cli_rate_ts")
    pout = pathlib.Path(pout)
    rating.fit_perform(path=jpkfile2, path_results=pout, profile_path=name)
    stats = np.loadtxt(pout / "statistics.tsv", skiprows=1, usecols=(1, 2, 3))
    assert np.all(stats[:, 0] == [109, 129, 416])
    assert np.all((3.5 < stats[:, 2]) * (stats[:, 2] < 5))


def test_fit_data_with_zef18():
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_cli_rate_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    pf["rating training set"] = "zef18"
    pf["weight_cp"] = 2e-6
    pf["fit param R value"] = 137.28e-6 / 2
    pout = tempfile.mkdtemp(prefix="test_nanite_cli_rate_ts")
    pout = pathlib.Path(pout)
    rating.fit_perform(path=jpkfile2, path_results=pout, profile_path=name)
    stats = np.loadtxt(pout / "statistics.tsv", skiprows=1, usecols=(1, 2, 3))
    assert np.all(stats[:, 0] == [109, 129, 416])
    assert np.allclose(stats[0, 2], 9.6, atol=0.11, rtol=0)
    assert np.allclose(stats[1, 2], 2.5, atol=0.11, rtol=0)
    assert np.allclose(stats[2, 2], 5.0, atol=0.11, rtol=0)
