"""Test of data set functionalities"""
import pathlib
import tempfile

import numpy as np
import pytest

import nanite
import nanite.model


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


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
    assert idnt["tip position"].values[0] == 2.2803841798545836e-05
    # correct for an offset in the tip
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_tip_offset"])
    # This value is subject to change if a better way to estimate the
    # contact point is found:
    assert idnt["tip position"].values[0] == 4.765854684370548e-06


def test_export():
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # tip-sample separation
    idnt.apply_preprocessing(["compute_tip_position"])
    # create temporary file
    _, path = tempfile.mkstemp(suffix=".tsv", prefix="nanite_idnt_export")
    idnt.export(path)
    converters = {3: lambda x: x.decode() == "True"}  # segment
    data = np.loadtxt(path, skiprows=1, converters=converters)
    assert data.shape == (4000, 5)
    assert np.allclose(data[100, 0], 0.04999999999999999)
    assert np.allclose(data[100, 1], -4.853736717639109e-10)
    assert np.allclose(data[100, 2], 2.256791903750211e-05, atol=1e-10, rtol=0)
    assert data[100, 3] == 0
    assert np.allclose(data[100, 4], 2.255675939721752e-05, atol=1e-10, rtol=0)
    assert data[3000, 3] == 1

    try:
        pathlib.Path(path).unlink()
    except OSError:
        pass


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
    assert np.allclose(params["E"].value, 14741.958242422093)

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
    assert np.allclose(params2["E"].value, 14840.840404880484)

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
    assert np.allclose(params3["E"].value, 14839.821714634612)


@pytest.mark.filterwarnings('ignore::nanite.fit.FitWarning')
def test_get_initial_fit_parameters():
    """This is a convenience function"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]
    # A: sanity check
    try:
        idnt.get_initial_fit_parameters()
    except KeyError:
        pass
    else:
        assert False, "need to get tip position first"
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
    md2 = nanite.model.get_model_by_name(model_name)
    assert md2 is md


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


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
