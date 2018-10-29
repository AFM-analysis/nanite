"""Test of data set functionalities"""
import pathlib

from nanite import Indentation, IndentationDataSet


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_afm_data_set():
    ds1 = IndentationDataSet(jpkfile)
    ds2 = IndentationDataSet(jpkfile)

    ds3 = ds1 + ds2
    assert len(ds3) == 2
    assert len(ds2) == 1
    assert len(ds1) == 1

    ds2 += ds3
    assert len(ds3) == 2
    assert len(ds2) == 3

    for apret in ds3:
        assert isinstance(apret, Indentation)

    print(ds3)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
