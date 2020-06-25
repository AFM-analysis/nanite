"""Test of data set functionalities"""
import copy
import pathlib

from nanite.fit import FitProperties, FP_DEFAULT, FitKeyError
from nanite import IndentationGroup
from nanite import model


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


def test_changed_fit_properties():
    ar = IndentationGroup(jpkfile)[0]
    # Initially, fit properties are not set
    assert not ar.fit_properties
    assert isinstance(ar.fit_properties, dict)
    # Prepprocessing
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_tip_offset",
                            "correct_force_offset"])
    ar.fit_model()
    hash1 = ar.fit_properties["hash"]
    pinit = copy.deepcopy(ar.fit_properties["params_initial"])
    pinit["E"].vary = False
    assert "hash" in ar.fit_properties, "make sure we didn't change anything"
    ar.fit_properties["params_initial"] = pinit
    assert "hash" not in ar.fit_properties
    ar.fit_model()
    assert hash1 != ar.fit_properties["hash"]


def test_fp_reset():
    fp = FitProperties(**FP_DEFAULT)
    fp["weight_cp"] = 1
    assert "weight_cp" in fp
    # Adding other than default keys is possible.
    fp["hash"] = True
    assert "hash" in fp
    # Updating a default key with same value does not
    # reset thedict.
    fp["weight_cp"] = 1
    assert "hash" in fp
    # Changing the default value removes all other keys.
    fp["hash"] = "laskdlai"
    fp["weight_cp"] = 2
    assert "test" not in fp
    assert "rhababer" not in fp
    # test manual resetting
    fp["hash"] = "2"
    fp.reset()
    assert "hash" not in fp


def test_with_dataset():
    ar = IndentationGroup(jpkfile)[0]
    # Initially, fit properties are not set
    assert not ar.fit_properties
    assert isinstance(ar.fit_properties, dict)
    # Prepprocessing
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_tip_offset",
                            "correct_force_offset"])

    # Fitting the data set will populate the dict
    ar.fit_model()
    assert isinstance(ar.fit_properties, dict)
    assert isinstance(ar.fit_properties, FitProperties)
    assert "hash" in ar.fit_properties
    assert ar.fit_properties["success"]
    hash1 = ar.fit_properties["hash"]
    # Change something
    ar.fit_properties["weight_cp"] = 0
    assert "hash" not in ar.fit_properties
    ar.fit_model()
    hash2 = ar.fit_properties["hash"]
    assert hash1 != hash2
    # Change it back
    ar.fit_properties["weight_cp"] = FP_DEFAULT["weight_cp"]
    ar.fit_model()
    hash3 = ar.fit_properties["hash"]
    assert hash1 == hash3


def test_change_model_key():
    ar = IndentationGroup(jpkfile)[0]
    # Prepprocessing
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_tip_offset",
                            "correct_force_offset"])

    # Fitting the data set will populate the dict
    ar.fit_model(model_key="hertz_para")
    # Change the model
    assert ar.fit_properties["params_initial"] is not None
    ar.fit_properties["model_key"] = "hertz_cone"
    # Changing the model key should reset the initial parameters
    assert ar.fit_properties["params_initial"] is None


def test_wrong_key():
    fp = FitProperties(**FP_DEFAULT)
    # unknown properties raise KeyError
    try:
        fp["dolce"] = False
    except FitKeyError:
        pass
    else:
        raise ValueError("Should not be able to set unknown key!")


def test_wrong_params_initial():
    ar = IndentationGroup(jpkfile)[0]
    # Prepprocessing
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_tip_offset",
                            "correct_force_offset"])
    md = model.models_available["hertz_para"]
    params = md.get_parameter_defaults()
    ar.fit_properties["model_key"] = "hertz_cone"
    try:
        ar.fit_model(params_initial=params)
    except FitKeyError:
        # We forced the wrong fitting parameters.
        pass
    else:
        raise ValueError("Should not be able to use wrong fit parameters!")


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
