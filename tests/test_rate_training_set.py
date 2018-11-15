"""Test usage of user-defined training set"""
import pathlib
import shutil
import tempfile

import numpy as np

from nanite import IndentationGroup
from nanite.rate import IndentationRater


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "map-data-reference-points.jpk-force-map"


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
    shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
