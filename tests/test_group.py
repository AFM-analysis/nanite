"""Test group functionalities"""
import pathlib
import tempfile
import shutil

from nanite import Indentation, IndentationGroup, load_group


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


def test_subgroup():
    tmp = tempfile.mkdtemp(prefix="test_nanite_group")
    shutil.copy(datadir / "map-data-reference-points.jpk-force-map", tmp)
    shutil.copy(datadir / "map2x2_extracted.jpk-force-map", tmp)
    shutil.copy(datadir / "flipsign_2015.05.22-15.31.49.352.jpk-force", tmp)

    grp = load_group(tmp)
    exp = pathlib.Path(tmp) / "map2x2_extracted.jpk-force-map"
    subgrp = grp.subgroup_with_path(path=exp)
    assert len(grp) == 8
    assert len(subgrp) == 4
    assert subgrp[0].path == exp

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
