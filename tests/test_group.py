"""Test group functionalities"""
import pathlib
import tempfile
import shutil

from afmformats.errors import MissingMetaDataError
import pytest

from nanite import Indentation, IndentationGroup, load_group


data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


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
    shutil.copy(
        data_path / "fmt-jpk-fd_map-data-reference-points.jpk-force-map", tmp)
    shutil.copy(data_path / "fmt-jpk-fd_map2x2_extracted.jpk-force-map", tmp)
    shutil.copy(
        data_path / "fmt-jpk-fd_flipsign_2015.05.22-15.31.49.352.jpk-force",
        tmp)

    grp = load_group(tmp)
    exp = pathlib.Path(tmp) / "fmt-jpk-fd_map2x2_extracted.jpk-force-map"
    subgrp = grp.subgroup_with_path(path=exp)
    assert len(grp) == 8
    assert len(subgrp) == 4
    assert subgrp[0].path == exp


def test_open_dat_without_spring_constant():
    # There are data files without spring constant but with force in nN.
    # nanite requires the spring constant for computation of the tip
    # position which we check for here.
    tmp = tempfile.mkdtemp(prefix="test_nanite_group")
    shutil.copy(data_path / "fmt-afm-workshop-fd_single_2021-10-22_14.16.csv",
                tmp)

    # try without explicit metadata
    with pytest.raises(MissingMetaDataError, match="spring constant"):
        load_group(tmp)

    # try with metadata
    grp = load_group(tmp, meta_override={"spring constant": 20})
    assert grp[0].metadata["spring constant"] == 20
