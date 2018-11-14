import pathlib
import tempfile

from nanite.cli import profile


def test_profile_getter():
    # use temporary file
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_profile_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    # sanity checks (run twice to trigger loading and saving)
    assert pf["segment"] == "approach"
    assert pf["segment"] == "approach"
    assert pf["preprocessing"] == ["compute_tip_position",
                                   "correct_force_offset",
                                   "correct_tip_offset"]
    assert pf["preprocessing"] == ["compute_tip_position",
                                   "correct_force_offset",
                                   "correct_tip_offset"]
    pf["preprocessing"] = ["compute_tip_position"]
    assert pf["preprocessing"] == ["compute_tip_position"]
    assert pf["range_x"] == [0, 0]
    assert pf["range_x"] == [0, 0]
    assert pf["weight_cp"] == 5e-7
    assert pf["weight_cp"] == 5e-7

    try:
        name.unlink()
    except OSError:
        pass


def test_profile_fitparams():
    # use temporary file
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_profile_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    # sanity checks (run twice to trigger loading and saving)
    pf["model_key"] = "hertz_cone"
    params = pf.get_fit_params()
    params2 = pf.get_fit_params()
    assert params == params2
    assert len(params) == 5
    assert "E" in params.keys()

    try:
        name.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
