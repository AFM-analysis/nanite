"""Test of data set functionalities"""
import pathlib
import tempfile

import numpy as np
import pytest

import afmformats
import nanite
import nanite.model


data_path = pathlib.Path(__file__).parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_apply_preprocessing():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # apply preprocessing by manually setting the list
    idnt.preprocessing = ["compute_tip_position"]
    idnt.apply_preprocessing()


def test_apply_preprocessing_remember_fit_properties():
    """
    Normally, the fit properties would be overridden
    if the preprocessing changes. For user convenience,
    nanite remembers it. This is the test
    """
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    idnt.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    cp1 = 1.8029310065572342e-05
    inparams["contact_point"].set(cp1)
    inparams["contact_point"].vary = False

    # Fit with absolute full range
    idnt.fit_model(model_key="hertz_para",
                   params_initial=inparams,
                   range_x=(0, 0),
                   range_type="absolute",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)
    assert cp1 == idnt.fit_properties["params_initial"]["contact_point"].value
    assert cp1 == idnt.fit_properties["params_fitted"]["contact_point"].value

    # Change preprocessing
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset"])
    assert cp1 == idnt.fit_properties["params_initial"]["contact_point"].value


def test_basic():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # tip-sample separation
    idnt.apply_preprocessing(["compute_tip_position"])
    assert idnt.preprocessing == ["compute_tip_position"]
    assert idnt["tip position"][0] == 2.2803841798545836e-05
    # correct for an offset in the tip
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_tip_offset"])
    # This value is subject to change if a better way to estimate the
    # contact point is found:
    assert idnt["tip position"][0] == 4.765854684370548e-06


def test_export():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # tip-sample separation
    idnt.apply_preprocessing(["compute_tip_position"])
    # create temporary file
    _, path = tempfile.mkstemp(suffix=".tab", prefix="nanite_idnt_export")
    idnt.export_data(path)
    data = afmformats.load_data(path)[0]
    assert len(data) == 4000
    assert np.allclose(data["force"][100], -4.853736717639109e-10)
    assert np.allclose(data["height (measured)"][100], 2.256791903750211e-05,
                       atol=1e-10, rtol=0)
    assert data["segment"][100] == 0
    assert np.allclose(data["time"][100], 0.04999999999999999)
    assert np.allclose(data["tip position"][100], 2.255675939721752e-05,
                       atol=1e-10, rtol=0)
    assert data["segment"][3000] == 1


def test_fitting():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    idnt.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    idnt.fit_model(model_key="hertz_para",
                   params_initial=inparams,
                   range_x=(0, 0),
                   range_type="absolute",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)
    params = idnt.fit_properties["params_fitted"]
    assert np.allclose(params["contact_point"].value, 1.8029310201193193e-05)
    assert np.allclose(params["E"].value, 14741.958242422093,
                       atol=1,
                       rtol=0)

    # Fit with absolute short
    idnt.fit_model(model_key="hertz_para",
                   params_initial=inparams,
                   range_x=(17e-06, 19e-6),
                   range_type="absolute",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)
    params2 = idnt.fit_properties["params_fitted"]
    assert np.allclose(params2["contact_point"].value, 1.8028461828272924e-05)
    assert np.allclose(params2["E"].value, 14838.89245576058,
                       atol=3,
                       rtol=0)

    # Fit with relative to initial fit
    idnt.fit_model(model_key="hertz_para",
                   params_initial=params2,
                   range_x=(-2e-6, 1e-6),
                   range_type="relative cp",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)
    params3 = idnt.fit_properties["params_fitted"]
    # These results are subject to change if the "relative cp" method is
    # changed.
    assert np.allclose(params3["contact_point"].value, 1.8028478083499856e-05)
    assert np.allclose(params3["E"].value, 14838.010069354472,
                       atol=2,
                       rtol=0)


@pytest.mark.filterwarnings('ignore::nanite.fit.FitWarning')
def test_get_initial_fit_parameters():
    """This is a convenience function"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # A: sanity check
    fp = idnt.get_initial_fit_parameters()
    assert fp["contact_point"].value == 0, "need to get tip position first"
    # B: get default fit parameters (hertz_para)
    idnt.apply_preprocessing(["compute_tip_position"])
    fp = idnt.get_initial_fit_parameters()
    for kk in ['E', 'R', 'nu', 'contact_point', 'baseline']:
        assert kk in fp.keys()
    # C: set other model and get fit parameters
    idnt.fit_properties["model_key"] = "hertz_pyr3s"
    fp2 = idnt.get_initial_fit_parameters()
    for kk in ['E', 'alpha', 'nu', 'contact_point', 'baseline']:
        assert kk in fp2.keys()
    # D: fit and get from fit_properties
    idnt.fit_model(params_initial=fp2)
    fp3 = idnt.get_initial_fit_parameters()
    assert fp2 == fp3


def test_get_model():
    md = nanite.model.model_hertz_paraboloidal
    model_name = "parabolic indenter (Hertz)"
    md2 = nanite.model.get_model_by_name(model_name).module
    assert md2 is md


def test_preprocessing_reset():
    fd = nanite.IndentationGroup(jpkfile)[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"],
                           options={"correct_tip_offset": {
                               "method": "fit_constant_line"}})

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Perform fit and make sure that all properties are set
    fd.fit_model(model_key="hertz_para",
                 params_initial=inparams,
                 range_x=(0, 0),
                 range_type="absolute",
                 x_axis="tip position",
                 y_axis="force",
                 segment="approach",
                 weight_cp=False)
    fd.rate_quality(training_set="zef18",
                    regressor="Extra Trees")
    assert fd._rating is not None
    assert fd.preprocessing == ["compute_tip_position",
                                "correct_force_offset",
                                "correct_tip_offset"]
    assert fd.preprocessing_options == {"correct_tip_offset": {
                                        "method": "fit_constant_line"}}
    assert not fd._preprocessing_details
    assert np.allclose(
        fd._fit_properties["params_fitted"]["contact_point"].value,
        -3.514699519428463e-06,
        atol=0,
        rtol=1e-4)
    assert "tip position" in fd

    # Change preprocessing and make sure properties are reset
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"],
                           options={"correct_tip_offset": {
                               "method": "fit_constant_polynomial"}})
    assert fd._rating is None
    assert fd.preprocessing == ["compute_tip_position",
                                "correct_force_offset",
                                "correct_tip_offset"]
    assert fd.preprocessing_options == {"correct_tip_offset": {
                                        "method": "fit_constant_polynomial"}}
    assert not fd._preprocessing_details
    assert "params_fitted" not in fd._fit_properties
    assert "tip position" in fd

    # Change preprocessing, removing tip position
    fd.apply_preprocessing(["correct_force_offset"])
    assert "tip position" not in fd


def test_rate_quality_cache():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    idnt.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    idnt.fit_model(model_key="hertz_para",
                   params_initial=inparams,
                   range_x=(0, 0),
                   range_type="absolute",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)
    r1 = idnt.rate_quality(training_set="zef18",
                           regressor="Extra Trees")
    assert idnt._rating[-1] == r1
    r2 = idnt.rate_quality(training_set="zef18",
                           regressor="Extra Trees")
    assert r1 == r2


def test_rate_quality_disabled():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    idnt.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    idnt.fit_model(model_key="hertz_para",
                   params_initial=inparams,
                   range_x=(0, 0),
                   range_type="absolute",
                   x_axis="tip position",
                   y_axis="force",
                   segment="approach",
                   weight_cp=False)

    r1 = idnt.rate_quality(training_set="zef18",
                           regressor="none")
    assert r1 == -1


def test_rate_quality_nofit():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    r1 = idnt.rate_quality()
    assert r1 == -1


def test_repr_str():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    assert "AFMForceDistance" not in str(idnt)
    assert "Indentation" in str(idnt)
    assert "fmt-jpk-fd_spot3-0192.jpk-force" in str(idnt)
    assert "Indentation" in repr(idnt)
    assert "fmt-jpk-fd_spot3-0192.jpk-force" in repr(idnt)
