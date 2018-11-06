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
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["smooth_height"])
    hms = np.array(ar.data["height (measured, smoothed)"])
    ar.apply_preprocessing(["compute_tip_position",
                            "smooth_height"])
    hms2 = np.array(ar.data["height (measured, smoothed)"])
    assert np.all(hms == hms2)
    np.array(ar.data["tip position (smoothed)"])


def test_tip_sample_separation():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    # This computation correctly reproduces the column
    # "Vertical Tip Position" as it is exported by the
    # JPK analysis software with the checked option
    # "Use Unsmoothed Height".
    ar.apply_preprocessing(["compute_tip_position"])
    tip = np.array(ar.data["tip position"])
    assert tip[0] == 2.2803841798545836e-05


def test_correct_app_ret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_split_approach_retract"])
    a = ar.data.loc[~(ar.data["segment"].values)]
    assert len(a) == 2006


def test_correct_force_offset():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    idp = ar.estimate_contact_point_index()
    assert np.allclose(np.average(ar.data["force"][:idp]), 0)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
