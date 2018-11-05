"""Test of basic opening functionalities"""
import pathlib

import pytest

from nanite import IndentationDataSet


datadir = pathlib.Path(__file__).resolve().parent / "data"
bad_files = list(datadir.glob("bad*"))


@pytest.mark.filterwarnings('ignore::nanite.preproc.CannotSplitWarning')
def test_process_bad():
    for bf in bad_files:
        ds = IndentationDataSet(bf)
        for apret in ds:
            # Walk through the standard analysis pipeline without
            # throwing any exceptions.
            apret.apply_preprocessing(["compute_tip_position",
                                       "correct_force_offset",
                                       "correct_tip_offset",
                                       "correct_split_approach_retract"])
            apret.fit_model()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
