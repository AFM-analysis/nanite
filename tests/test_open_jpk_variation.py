"""Test opening of file format variations"""
import pathlib
import numpy as np

from nanite import IndentationGroup

data_path = pathlib.Path(__file__).resolve().parent / "data"


def test_process_flipsign():
    # This is a curve extracted from a map file. When loading it
    # with nanite, the sign of the force curve was flipped.
    flipped = data_path / "flipsign_2015.05.22-15.31.49.352.jpk-force"
    idnt = IndentationGroup(flipped)[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset",
                              "correct_split_approach_retract"])
    # We set the baseline fixed, because this test was written so)
    params_initial = idnt.get_initial_fit_parameters(model_key="hertz_para")
    params_initial["baseline"].set(vary=False)
    idnt.fit_model(model_key="hertz_para", weight_cp=False,
                   params_initial=params_initial)
    assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                       5230.008779761989)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
