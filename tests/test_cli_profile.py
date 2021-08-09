import pathlib
import shutil
import tempfile

from nanite.cli import profile


data_path = pathlib.Path(__file__).parent / "data"


def test_profile_getter():
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_profile_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    # sanity checks (run twice to trigger loading and saving)
    assert pf["segment"] == 0
    assert pf["segment"] == 0
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


def test_profile_getter_1_7_8():
    """Load a profile from version 1.7.8"""
    tdir = pathlib.Path(tempfile.mkdtemp(prefix="cli_profile_"))
    name = "cli-profile-1.7.8.cfg"
    cfgpath = tdir / name
    shutil.copy2(data_path / name, cfgpath)
    pf = profile.Profile(path=cfgpath)
    assert pf["segment"] == 0
    assert pf["segment"] == 0


def test_profile_getter_2_1_0():
    """Load a profile from version 2.1.0"""
    tdir = pathlib.Path(tempfile.mkdtemp(prefix="cli_profile_"))
    name = "cli-profile-2.1.0.cfg"
    cfgpath = tdir / name
    shutil.copy2(data_path / name, cfgpath)
    pf = profile.Profile(path=cfgpath)
    assert pf["segment"] == 0
    assert pf["segment"] == 0
    assert pf["rating training set"] == "zef18"


def test_profile_fitparams():
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


def test_single_fitparam():
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_profile_")
    name = pathlib.Path(name)
    pf = profile.Profile(path=name)
    # sanity checks (run twice to trigger loading and saving)
    pf["fit param E value"] = 50
    pf["fit param R value"] = 16e-6
    params = pf.get_fit_params()
    assert params["E"].value == 50
    assert params["R"].value == 16e-6


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
