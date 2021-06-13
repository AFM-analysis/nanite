"""Test of data set functionalities"""
import pathlib

import numpy as np
import pytest

from nanite import IndentationGroup


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


@pytest.mark.filterwarnings('ignore::nanite.smooth.'
                            + 'DoubledSmoothingWindowWarning')
def test_app_ret():
    grp = IndentationGroup(jpkfile)
    idnt = grp[0]
    idnt.apply_preprocessing(["smooth_height"])
    hms = np.array(idnt.data["height (measured, smoothed)"])
    idnt.apply_preprocessing(["compute_tip_position",
                              "smooth_height"])
    hms2 = np.array(idnt.data["height (measured, smoothed)"])
    assert np.all(hms == hms2)
    np.array(idnt.data["tip position (smoothed)"])


def test_tip_sample_separation():
    grp = IndentationGroup(jpkfile)
    idnt = grp[0]
    # This computation correctly reproduces the column
    # "Vertical Tip Position" as it is exported by the
    # JPK analysis software with the checked option
    # "Use Unsmoothed Height".
    idnt.apply_preprocessing(["compute_tip_position"])
    tip = np.array(idnt.data["tip position"])
    assert tip[0] == 2.2803841798545836e-05


def test_correct_app_ret():
    grp = IndentationGroup(jpkfile)
    idnt = grp[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_split_approach_retract"])
    a = idnt.data["segment"][~(idnt.data["segment"])]
    assert len(a) == 2006


def test_correct_force_offset():
    grp = IndentationGroup(jpkfile)
    idnt = grp[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset"])
    idp = idnt.estimate_contact_point_index()
    assert np.allclose(np.average(idnt.data["force"][:idp]), 0)


@pytest.mark.parametrize(
    "metadata,software",
    [
        (None, "JPK"),
        ({"software": "custom1a"}, "custom1a"),
    ])
def test_metadata_override(metadata, software):
    grp = IndentationGroup(jpkfile, meta_override=metadata)
    idnt = grp[0]
    assert idnt.metadata["software"] == software
