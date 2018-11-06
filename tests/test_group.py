"""Test group functionalities"""
import pathlib

from nanite import Indentation, IndentationGroup


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_base():
    ds1 = IndentationGroup(jpkfile)
    ds2 = IndentationGroup(jpkfile)

    ds3 = ds1 + ds2
    assert len(ds3) == 2
    assert len(ds2) == 1
    assert len(ds1) == 1

    ds2 += ds3
    assert len(ds3) == 2
    assert len(ds2) == 3

    for apret in ds3:
        assert isinstance(apret, Indentation)
    # test repr
    print(ds3)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
