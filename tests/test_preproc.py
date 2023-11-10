"""Test preprocessing"""
import pathlib

import numpy as np
import pytest

from nanite import IndentationGroup
from nanite import preproc


data_path = pathlib.Path(__file__).resolve().parent / "data"
bad_files = list(data_path.glob("bad*"))


def test_autosort():
    unsorted = ["correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position"]
    expected = ["correct_force_offset",
                "compute_tip_position",
                "correct_tip_offset"]
    actual = preproc.autosort(unsorted)
    assert expected == actual


def test_autosort2():
    unsorted = ["correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position",
                ]
    expected = ["compute_tip_position",
                "correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                ]
    actual = preproc.autosort(unsorted)
    assert (actual.index("correct_split_approach_retract")
            > actual.index("compute_tip_position"))
    assert (actual.index("correct_tip_offset")
            > actual.index("compute_tip_position"))
    assert expected == actual


def test_autosort3():
    unsorted = ["smooth_height",
                "correct_split_approach_retract",
                "correct_force_offset",
                "correct_tip_offset",
                "compute_tip_position",
                ]
    expected = ["compute_tip_position",
                "correct_split_approach_retract",
                "smooth_height",
                "correct_force_offset",
                "correct_tip_offset",
                ]
    actual = preproc.autosort(unsorted)
    assert expected == actual
    assert (actual.index("correct_split_approach_retract")
            > actual.index("compute_tip_position"))
    assert (actual.index("correct_tip_offset")
            > actual.index("compute_tip_position"))
    assert (actual.index("smooth_height")
            > actual.index("correct_split_approach_retract"))


def test_check_order():
    with pytest.raises(ValueError, match="Wrong optional step order"):
        preproc.check_order([
            "smooth_height",
            "correct_split_approach_retract"])


def test_get_steps_required():
    req_act = preproc.get_steps_required("correct_tip_offset")
    req_exp = ["compute_tip_position"]
    assert req_act == req_exp


def test_preproc_correct_force_slope_bad_monotonic_data_issue_25():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    # put a custom, bad tip position (this requires some hacking)
    s = len(fd["force"])
    raw_data = {}
    for col in fd.columns_innate:
        raw_data[col] = fd[col]
    tippos = np.linspace(0, 5e-9, s)
    tippos = np.roll(tippos, s//2)
    tippos[:s//2] = 0
    raw_data["tip position"] = tippos
    fd._raw_data = raw_data
    # sanity check
    assert np.all(fd["tip position"] == tippos)
    assert "tip position" in fd.columns_innate
    # This caused "TypeError: expected non-empty vector for x" in
    # np.polynomial in nanite 3.7.3.
    fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope",
         "correct_force_offset"],
        options={
            "correct_tip_offset": {"method": "deviation_from_baseline"},
            "correct_force_slope": {"region": "approach",
                                    "strategy": "drift"},
        },
        ret_details=True)


def test_preproc_correct_force_slope_drift_approach():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope",
         "correct_force_offset"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
            "correct_force_slope": {"region": "approach",
                                    "strategy": "drift"},
        },
        ret_details=True)
    slopedet = details["correct_force_slope"]
    for key in ["plot slope data",
                "plot slope fit",
                "norm"]:
        assert key in slopedet
    # Sanity check for size of baseline (determined by POC via line+poly)
    assert len(slopedet["plot slope data"][0]) == 15602
    # Check for flatness of baseline.
    assert np.ptp(fd["force"][:15000]) < .094e-9, "ptp less than 0.1 nN"
    bl0 = np.mean(fd["force"][:100])
    bl1 = np.mean(fd["force"][10000:10100])
    assert np.allclose(bl0, bl1,
                       atol=1e-11,
                       rtol=0)
    assert np.abs(bl1 - bl0) < 0.0085e-9
    # Now check whether the maximum force is lower than normal.
    assert np.allclose(np.max(fd["force"]), 1.6688553846662955e-09,
                       atol=0)
    assert np.allclose(np.min(fd["force"]), -5.935613094149927e-10,
                       atol=0)
    # Compare it to the correction obtained with just "baseline".
    # For this dataset, the drift is resulting in a continuous increase
    # in the force. Thus, if we compare the "approach" correction with
    # just the "baseline" correction, we should arrive at a lower number
    # for the "approach" correction.
    assert np.max(fd["force"]) < 1.6903757572616587e-09


def test_preproc_correct_force_slope_drift_full():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
            "correct_force_slope": {"region": "all",
                                    "strategy": "drift"},
        },
        ret_details=True)
    slopedet = details["correct_force_slope"]
    for key in ["plot slope data",
                "plot slope fit",
                "norm"]:
        assert key in slopedet
    # Sanity check for size of baseline (determined by POC via line+poly)
    assert len(slopedet["plot slope data"][0]) == 15602
    # Check for flatness of baseline.
    assert np.ptp(fd["force"][:15000]) < .094e-9, "ptp less than 0.1 nN"
    bl0 = np.mean(fd["force"][:100])
    bl1 = np.mean(fd["force"][10000:10100])
    assert np.allclose(bl0, bl1,
                       atol=0,
                       rtol=.01)
    assert np.abs(bl1 - bl0) < 0.0085e-9
    # Now check for flatness of retract tail
    rt0 = np.mean(fd["force"][-100:])
    rt1 = np.mean(fd["force"][-10100:-10000])
    assert rt1 < rt0, "There is still a little drift left in the retract tail"
    # The following test would fail if we set "strategy" to "shift" or if
    # we set region to "baseline" or "approach".
    assert np.allclose(rt0, rt1,
                       atol=5e-11, rtol=0)


def test_preproc_correct_force_slope_shift():
    fd = IndentationGroup(
        data_path
        # Note: this dataset actually resembles a drift over time, not
        # a shift. But we are only looking at the baseline, so it's OK.
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
            "correct_force_slope": {"region": "baseline",
                                    "strategy": "shift"},
        },
        ret_details=True)
    slopedet = details["correct_force_slope"]
    for key in ["plot slope data",
                "plot slope fit",
                "norm"]:
        assert key in slopedet
    # Sanity check for size of baseline (determined by POC via line+poly)
    assert len(slopedet["plot slope data"][0]) == 15602
    # Check for flatness of baseline.
    assert np.ptp(fd["force"][:15000]) < .095e-9, "ptp less than 0.1 nN"
    bl0 = np.mean(fd["force"][:100])
    bl1 = np.mean(fd["force"][10000:10100])
    assert np.allclose(bl0, bl1,
                       atol=0,
                       rtol=.01)
    assert np.abs(bl1 - bl0) < 0.009e-9


def test_preproc_correct_force_slope_shift_control():
    """Same test as above, but without slope correction"""
    fd = IndentationGroup(
        data_path
        # Note: this dataset actually resembles a drift over time, not
        # a shift. But we are only looking at the baseline, so it's OK.
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
        })
    # Make sure the baseline is not as flat as in the test above.
    assert np.ptp(fd["force"][:15000]) > .35e-9, "larger ptp with tilt"
    bl0 = np.mean(fd["force"][:100])
    bl1 = np.mean(fd["force"][10000:10100])
    assert not np.allclose(bl0, bl1,
                           atol=0,
                           rtol=.01)
    assert bl1 > bl0
    assert bl1 - bl0 > 0.19e-9


def test_preproc_correct_force_slope_shift_full():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-shift-"
          "adyp_2023-06-26.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
            "correct_force_slope": {"region": "all",
                                    "strategy": "shift"},
        },
        ret_details=True)
    slopedet = details["correct_force_slope"]
    for key in ["plot slope data",
                "plot slope fit",
                "norm"]:
        assert key in slopedet
    # Sanity check for size of baseline (determined by POC via line+poly)
    assert len(slopedet["plot slope data"][0]) == 3201
    # Check for flatness of baseline.
    assert np.ptp(fd["force"][:3000]) < .04e-9, "ptp less than 0.04 nN"
    bl0 = np.mean(fd["force"][:50])
    bl1 = np.mean(fd["force"][3000:3050])
    assert np.allclose(bl0, bl1,
                       atol=0.004e-9,
                       rtol=0)
    assert np.abs(bl1 - bl0) < 0.0085e-9
    # Now check for flatness of retract tail
    rt0 = np.mean(fd["force"][-50:])
    rt1 = np.mean(fd["force"][-1050:-1000])
    # The following test would fail if we set "strategy" to "drift" or if
    # we set region to "baseline" or "approach".
    assert np.allclose(rt0, rt1,
                       atol=2e-11, rtol=0)


def test_preproc_correct_split_approach_retract():
    fd = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]

    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"])
    assert fd.appr["segment"].size == 2000
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset",
                            "correct_split_approach_retract"])
    assert fd.appr["segment"].size == 2006


@pytest.mark.filterwarnings('ignore::nanite.preproc.CannotSplitWarning',
                            'ignore::UserWarning')
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


@pytest.mark.filterwarnings(
    'ignore::nanite.smooth.DoubledSmoothingWindowWarning:')
def test_smooth():
    idnt = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset"])
    orig = np.array(idnt["tip position"], copy=True)
    # now apply smoothing filter
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset",
                              "smooth_height"])
    new = np.array(idnt["tip position"], copy=True)
    assert not np.all(orig == new)


def test_unknown_method():
    idnt = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    with pytest.raises(KeyError, match="unknown_method"):
        idnt.apply_preprocessing(["compute_tip_position",
                                  "unknown_method"])


def test_wrong_order():
    idnt = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    with pytest.raises(ValueError, match="requires the steps"):
        # order matters
        idnt.apply_preprocessing(["correct_tip_offset",
                                  "compute_tip_position"])


def test_wrong_order_2():
    idnt = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    with pytest.raises(ValueError, match="requires the steps"):
        # order matters
        idnt.apply_preprocessing(["correct_split_approach_retract",
                                  "compute_tip_position"])
