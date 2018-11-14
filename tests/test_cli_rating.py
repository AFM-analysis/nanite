import pathlib
import tempfile

from nanite.cli import profile, rating

datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_plotting():
    # use temporary file
    _, name = tempfile.mkstemp(suffix=".cfg", prefix="test_nanite_profile_")
    name = pathlib.Path(name)
    profile.Profile(path=name)

    # this will fit with the profile default parameters
    idnt = rating.fit_data(path=jpkfile, profile_path=name)
    assert idnt.path == jpkfile
    assert idnt.fit_properties["success"]

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
