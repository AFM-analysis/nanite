"""Test rating features"""
import pathlib

import numpy as np

from nanite import IndentationGroup, model
from nanite.rate import features


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def setup_indent():
    idnt = IndentationGroup(jpkfile)[0]
    idnt.apply_preprocessing(["compute_tip_position"])
    inparams = model.model_hertz_paraboloidal.get_parameter_defaults()
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
    return idnt


def test_basic_nofit():
    idnt = IndentationGroup(jpkfile)[0]
    feat = features.IndentationFeatures(idnt)
    assert not feat.is_fitted
    assert not feat.is_valid
    assert not feat.has_contact_point
    try:
        feat.contact_point
    except ValueError:
        pass
    else:
        assert False, "without fit - should not have a CP"


def test_basic_withfit():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    assert feat.is_fitted
    assert feat.is_valid
    assert feat.has_contact_point

    cp = feat.contact_point
    # This will change if the the default method of CP detection
    # changes.
    assert np.allclose(cp, 1.8029310201193193e-05)
    # Simple check for meta
    assert feat.meta["points"] == 2000


def test_get_features():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    samples, names = feat.compute_features(idnt, ret_names=True)

    # These are the samples from the original release
    ref = {
        'feat_bin_apr_spikes_count': 1.0,
        'feat_bin_cp_position': 1.0,
        'feat_bin_size': 1.0,
        'feat_con_apr_flatness': 0.4157949790794979,
        'feat_con_apr_size': 0.04400000000000004,
        'feat_con_apr_sum': 0.3293102976667975,
        'feat_con_bln_slope': 0.6287923407474181,
        'feat_con_bln_variation': 0.46602504647335763,
        'feat_con_cp_curvature': -0.15481927352155514,
        'feat_con_cp_magnitude': 0.005076769161668967,
        'feat_con_idt_maxima_75perc': 0.3184029609825442,
        'feat_con_idt_monotony': 0.0,
        'feat_con_idt_spike_area': 0.4262935122879405,
        'feat_con_idt_sum': 0.19263421098385752,
        'feat_con_idt_sum_75perc': 0.015026423315718608
    }
    for key in ref:
        idx = names.index(key)
        assert np.allclose(samples[idx], ref[key]), "Mismatch '{}'".format(key)


def test_get_feature_funcs_order():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    funcs = feat.get_feature_funcs()
    samples = feat.compute_features(idnt)
    for (_, func), samp in zip(funcs, samples):
        assert np.allclose(func(feat), samp)


def test_get_feature_names_bad_names():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    try:
        feat.get_feature_names(names=["feat_con_apr_flatness",
                                      "feat_con_unknown"])
    except ValueError:
        pass
    else:
        assert False, "Unknown names should not work"


def test_get_feature_names_bad_type():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    try:
        feat.get_feature_names(which_type="unknown")
    except ValueError:
        pass
    else:
        assert False, "Unknown types should not work"


def test_get_feature_names_type():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    # binary
    names = feat.get_feature_names(which_type="binary")
    for nn in names:
        assert nn.startswith("feat_bin_")
    # continuous
    names = feat.get_feature_names(which_type="continuous")
    for nn in names:
        assert nn.startswith("feat_con_")


def test_get_feature_names_mix():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    # binary
    names = feat.get_feature_names(which_type="binary",
                                   names=["feat_bin_size",
                                          "feat_con_idt_sum"])
    assert len(names) == 1
    assert names[0] == "feat_bin_size"


def test_get_feature_names_indices():
    idnt = setup_indent()
    feat = features.IndentationFeatures(idnt)
    # binary
    names, idx = feat.get_feature_names(which_type="binary",
                                        names=["feat_bin_size",
                                               "feat_con_idt_sum"],
                                        ret_indices=True)
    nall = feat.get_feature_names()
    assert idx[0] == nall.index(names[0])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
