"""Test preprocessing"""
import pathlib

import pytest

from nanite import IndentationGroup


datadir = pathlib.Path(__file__).resolve().parent / "data"
bad_files = list(datadir.glob("bad*"))


@pytest.mark.filterwarnings('ignore::nanite.preproc.CannotSplitWarning')
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


def test_unknown_method():
    idnt = IndentationGroup(datadir / "spot3-0192.jpk-force")[0]
    try:
        idnt.apply_preprocessing(["compute_tip_position",
                                  "unknown_method"])
    except KeyError:
        pass
    else:
        assert False, "Preprocessing with an unknown method must not work."


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
