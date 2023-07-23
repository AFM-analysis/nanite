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


def test_preproc_correct_force_slope():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-mitotic_2021-01-29.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset", "correct_force_slope"],
        options={
            "correct_tip_offset": {"method": "fit_line_polynomial"},
            "correct_force_slope": {"region": "baseline"},
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


def test_preproc_correct_force_slope_control():
    """Same test as above, but without slope correction"""
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-mitotic_2021-01-29.jpk-force")[0]
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
