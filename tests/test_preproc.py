"""Test preprocessing"""
import pathlib

import pytest

from nanite import IndentationGroup


data_path = pathlib.Path(__file__).resolve().parent / "data"
bad_files = list(data_path.glob("bad*"))


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
