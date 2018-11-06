"""Test opening of file format variations"""
import pathlib
import numpy as np

from nanite import IndentationGroup

datadir = pathlib.Path(__file__).resolve().parent / "data"


def test_process_flipsign():
    # This is a curve extracted from a map file. When loading it
    # with nanite, the sign of the force curve was flipped.
    flipped = datadir / "flipsign_2015.05.22-15.31.49.352.jpk-force"
    apret = IndentationGroup(flipped)[0]
    apret.apply_preprocessing(["compute_tip_position",
                               "correct_force_offset",
                               "correct_tip_offset",
                               "correct_split_approach_retract"])

    apret.fit_model(model_key="hertz_para", weight_cp=False)
    assert np.allclose(apret.fit_properties["params_fitted"]["E"].value,
                       5230.0087777251456)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
