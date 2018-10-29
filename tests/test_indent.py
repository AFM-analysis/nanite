"""Test of data set functionalities"""
import pathlib

import numpy as np

import nanite


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


def test_afm_data_set_basic():
    ds1 = nanite.IndentationDataSet(jpkfile)
    apret = ds1[0]
    # tip-sample separation
    apret.apply_preprocessing(["compute_tip_position"])
    assert apret["tip position"].values[0] == 2.2803841798545836e-05
    # correct for an offset in the tip
    apret.apply_preprocessing(["compute_tip_position",
                               "correct_tip_offset"])
    # This value is subject to change if a better way to estimate the
    # contact point is found:
    assert apret["tip position"].values[0] == 4.765854684370548e-06


def test_afm_data_set_fitting():
    ds1 = nanite.IndentationDataSet(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_parabolic.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    apret.fit_model(model_key="hertz_para",
                    params_initial=inparams,
                    range_x=(0, 0),
                    range_type="absolute",
                    x_axis="tip position",
                    y_axis="force",
                    segment="approach",
                    weight_cp=False)
    params = apret.fit_properties["params_fitted"]
    assert np.allclose(params["contact_point"].value, 1.8029310201193193e-05)
    assert np.allclose(params["E"].value, 14741.958242422093)

    # Fit with absolute short
    apret.fit_model(model_key="hertz_para",
                    params_initial=inparams,
                    range_x=(17e-06, 19e-6),
                    range_type="absolute",
                    x_axis="tip position",
                    y_axis="force",
                    segment="approach",
                    weight_cp=False)
    params2 = apret.fit_properties["params_fitted"]
    assert np.allclose(params2["contact_point"].value, 1.8028461828272924e-05)
    assert np.allclose(params2["E"].value, 14840.840404880484)

    # Fit with relative to initial fit
    apret.fit_model(model_key="hertz_para",
                    params_initial=params2,
                    range_x=(-2e-6, 1e-6),
                    range_type="relative cp",
                    x_axis="tip position",
                    y_axis="force",
                    segment="approach",
                    weight_cp=False)
    params3 = apret.fit_properties["params_fitted"]
    # These results are subject to change if the "relative cp" method is
    # changed.
    assert np.allclose(params3["contact_point"].value, 1.8028478083499856e-05)
    assert np.allclose(params3["E"].value, 14839.821714634612)


def test_get_model():
    md = nanite.model.model_hertz_parabolic
    model_name = "parabolic indenter (Hertz)"
    md2 = nanite.model.get_model_by_name(model_name)
    assert md2 is md


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
