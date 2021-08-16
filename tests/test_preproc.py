"""Test preprocessing"""
import pathlib

import numpy as np
import pytest

from nanite import IndentationGroup
from nanite import preproc


data_path = pathlib.Path(__file__).resolve().parent / "data"
bad_files = list(data_path.glob("bad*"))


def test_autosort():
    unsorted = ["correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position"]
    expected = ["correct_force_offset",
                "compute_tip_position",
                "correct_tip_offset"]
    actual = preproc.autosort(unsorted)
    assert expected == actual


def test_autosort2():
    unsorted = ["correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position",
                ]
    expected = ["compute_tip_position",
                "correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                ]
    actual = preproc.autosort(unsorted)
    assert (actual.index("correct_split_approach_retract")
            > actual.index("compute_tip_position"))
    assert (actual.index("correct_tip_offset")
            > actual.index("compute_tip_position"))
    assert expected == actual


def test_autosort3():
    unsorted = ["smooth_height",
                "correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position",
                ]
    expected = ["compute_tip_position",
                "correct_split_approach_retract",
                "smooth_height",
                "correct_force_offset",
                "correct_tip_offset",
                ]
    actual = preproc.autosort(unsorted)
    assert expected == actual
    assert (actual.index("correct_split_approach_retract")
            > actual.index("compute_tip_position"))
    assert (actual.index("correct_tip_offset")
            > actual.index("compute_tip_position"))
    assert (actual.index("smooth_height")
            > actual.index("correct_split_approach_retract"))


def test_check_order():
    with pytest.raises(ValueError, match="Wrong optional step order"):
        preproc.check_order([
            "smooth_height",
            "correct_split_approach_retract"])


def test_correct_split_approach_retract():
    fd = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]

    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"])
    assert fd.appr["segment"].size == 2000
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset",
                            "correct_split_approach_retract"])
    assert fd.appr["segment"].size == 2006


def test_get_steps_required():
    req_act = preproc.get_steps_required("correct_tip_offset")
    req_exp = ["compute_tip_position"]
    assert req_act == req_exp


@pytest.mark.filterwarnings('ignore::nanite.preproc.CannotSplitWarning',
                            'ignore::UserWarning')
def test_process_bad():
    for bf in bad_files:
        ds = IndentationGroup(bf)
        for idnt in ds:
            # Walk through the standard analysis pipeline without
            # throwing any exceptions.
            idnt.apply_preprocessing(["compute_tip_position",
                                      "correct_force_offset",
                                      "correct_tip_offset",
                                      "correct_split_approach_retract"])
            idnt.fit_model()


@pytest.mark.filterwarnings(
    'ignore::nanite.smooth.DoubledSmoothingWindowWarning:')
def test_smooth():
    idnt = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset"])
    orig = np.array(idnt["tip position"], copy=True)
    # now apply smoothing filter
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset",
                              "smooth_height"])
    new = np.array(idnt["tip position"], copy=True)
    assert not np.all(orig == new)


def test_unknown_method():
    idnt = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    with pytest.raises(KeyError, match="unknown_method"):
        idnt.apply_preprocessing(["compute_tip_position",
                                  "unknown_method"])


def test_wrong_order():
    idnt = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    with pytest.raises(ValueError, match="requires the steps"):
        # order matters
        idnt.apply_preprocessing(["correct_tip_offset",
                                  "compute_tip_position"])


def test_wrong_order_2():
    idnt = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    with pytest.raises(ValueError, match="requires the steps"):
        # order matters
        idnt.apply_preprocessing(["correct_split_approach_retract",
                                  "compute_tip_position"])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
