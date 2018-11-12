"""Test qmap feature"""
import pathlib
import warnings

import numpy as np

from nanite import model, qmap, IndentationGroup


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "map2x2_extracted.jpk-force-map"
jpkfile2 = datadir / "map-data-reference-points.jpk-force-map"


def test_feat_scan_order():
    qm = qmap.QMap(jpkfile)
    order = qm.get_qmap("meta scan order", qmap_only=True)
    assert order[0, 0] == 0
    assert order[0, -1] == 1
    assert order[-1, -1] == 2
    assert order[-1, 0] == 3
    assert np.isnan(order[0, 1])


def test_feat_min_height():
    qm = qmap.QMap(jpkfile)
    qd = qm.get_qmap("data min height", qmap_only=True)
    assert np.allclose(qd[0, 0], 40.55030392499141)
    assert np.allclose(qd[0, -1], 47.354988549298945)
    assert np.allclose(qd[-1, -1], 96.1627883099352)
    assert np.allclose(qd[-1, 0], 89.95170867840217)


def test_feat_emod_nofit():
    qm = qmap.QMap(jpkfile)
    with warnings.catch_warnings(record=True) as w:
        # No data availabale, because there is no fit
        qd = qm.get_qmap("fit young's modulus", qmap_only=True)
        assert len(w) == 4
        assert w[0].category is qmap.DataMissingWarning
    assert np.alltrue(np.isnan(qd))


def test_feat_emod_withfit():
    qm = qmap.QMap(jpkfile2)
    # fit data
    for idnt in qm.group:
        idnt.apply_preprocessing(["compute_tip_position",
                                  "correct_force_offset",
                                  "correct_tip_offset",
                                  ])
        inparams = model.model_sneddon_spherical_approximation \
            .get_parameter_defaults()
        inparams["E"].value = 50
        inparams["R"].value = 37.28e-6 / 2

        # Fit with absolute full range
        idnt.fit_model(model_key="sneddon_spher_approx",
                       params_initial=inparams,
                       range_x=(0, 0),
                       range_type="absolute",
                       x_axis="tip position",
                       y_axis="force",
                       segment="approach",
                       weight_cp=2e-6)

    qd = qm.get_qmap("fit young's modulus", qmap_only=True)
    vals = qd.flat[~np.isnan(qd.flat)]
    assert np.allclose(vals[0], 57.629464729399096), "gray matter"
    assert np.allclose(vals[2], 46.614068655067435), "white matter"
    assert np.allclose(vals[1], 17605.034108797558), "background"


def test_feat_rating():
    """Reproduces rating in figures 5K-M"""
    qm = qmap.QMap(jpkfile2)
    # fit data
    for idnt in qm.group:
        idnt.apply_preprocessing(["compute_tip_position",
                                  "correct_force_offset",
                                  "correct_tip_offset"])
        inparams = model.model_sneddon_spherical_approximation \
            .get_parameter_defaults()
        inparams["E"].value = 50
        inparams["R"].value = 37.28e-6 / 2

        # Fit with absolute full range
        idnt.fit_model(model_key="sneddon_spher_approx",
                       params_initial=inparams,
                       range_x=(0, 0),
                       range_type="absolute",
                       x_axis="tip position",
                       y_axis="force",
                       segment="approach",
                       weight_cp=2e-6)
        idnt.rate_quality(training_set="zef18",
                          regressor="Extra Trees")

    qd = qm.get_qmap("meta rating", qmap_only=True)
    vals = qd.flat[~np.isnan(qd.flat)]
    assert np.allclose(vals[0], 9.471932624275558), "gray matter"
    assert np.allclose(vals[2], 4.75182041147194), "white matter"
    assert np.allclose(vals[1], 2.568823857492953), "background"


def test_feat_rating_nofit():
    qm = qmap.QMap(jpkfile)
    with warnings.catch_warnings(record=True) as w:
        # No data availabale, because there is no fit
        qd = qm.get_qmap("meta rating", qmap_only=True)
        assert len(w) == 4
        assert w[0].category is qmap.DataMissingWarning
    assert np.alltrue(np.isnan(qd))


def test_get_coords():
    qm = qmap.QMap(jpkfile)

    px = qm.get_coords(which="px")
    refpx = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])
    assert np.all(px == refpx)

    um = qm.get_coords(which="um")
    refum = np.array([[31.972656250000004, -753.5351562500001],
                      [571.8359375000001, -753.90625],
                      [571.8359375000001, -213.73046875000003],
                      [31.855468750000004, -213.73046875000003]])
    assert np.all(um == refum)


def test_get_coords_bad():
    qm = qmap.QMap(jpkfile)
    try:
        qm.get_coords(which="mm")
    except ValueError:
        pass
    else:
        assert False, "Units [mm] should not be supported."


def test_get_qmap():
    qm = qmap.QMap(jpkfile)
    x, y, _ = qm.get_qmap(feature="data min height", qmap_only=False)
    assert x.size == 10
    assert y.size == 10


def test_init_with_dataset():
    ds = IndentationGroup(jpkfile)
    qm = qmap.QMap(ds)
    assert qm.shape == (10, 10)


def test_metadata():
    qm = qmap.QMap(jpkfile)
    assert np.allclose(qm.extent,
                       [1.97265625, 601.97265625,
                        -783.53515625, -183.53515625000006])
    assert qm.shape == (10, 10)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
