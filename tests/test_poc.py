"""Test point of contact (POC) estimation"""
import pathlib

import numpy as np
import pytest

from nanite import poc, IndentationGroup


data_path = pathlib.Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize("method,contact_point", [
    ["gradient_zero_crossing", 1895],
    ["fit_constant_line", 1919],
    ["fit_constant_polynomial", 1898],
    ["fit_line_polynomial", 1899],
    ["frechet_direct_path", 1903],
    ["deviation_from_baseline", 1908],
])
def test_poc_estimation(method, contact_point):
    fd = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    assert poc.compute_poc(fd["force"], method) == contact_point


@pytest.mark.parametrize("method,contact_point", [
    ["gradient_zero_crossing", 1895],
    ["fit_constant_line", 1919],
    ["fit_constant_polynomial", 1898],
    ["fit_line_polynomial", 1899],
    ["frechet_direct_path", 1903],
    ["deviation_from_baseline", 1908],
])
def test_poc_estimation_details(method, contact_point):
    fd = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    _, details = poc.compute_poc(fd["force"], method, ret_details=True)
    assert np.all(np.array(details["plot poc"][0]) == contact_point)


@pytest.mark.parametrize("method,contact_point", [
    ["gradient_zero_crossing", 1895],
    ["fit_constant_line", 1919],
    ["fit_constant_polynomial", 1898],
    ["fit_line_polynomial", 1899],
    ["frechet_direct_path", 1903],
    ["deviation_from_baseline", 1908],
])
def test_poc_estimation_via_indent(method, contact_point):
    fd = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"],
                           options={"correct_tip_offset": {"method": method}})
    assert np.argmin(np.abs(fd["tip position"])) == contact_point


def test_poc_details_deviation_from_baseline():
    fd = IndentationGroup(data_path / "fmt-jpk-fd_spot3-0192.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_force_offset", "correct_tip_offset"],
        options={"correct_tip_offset": {"method": "deviation_from_baseline"}},
        ret_details=True)
    pocd = details["correct_tip_offset"]
    for key in ["plot force",
                "plot baseline mean",
                "plot baseline threshold",
                "plot poc"]:
        assert key in pocd
    assert np.allclose(
        np.mean(pocd["plot baseline threshold"][1]),
        8.431890003513514e-11,
        atol=0)
    assert np.allclose(
        np.mean(pocd["plot baseline mean"][1]),
        -7.573822405258677e-12,
        atol=0)


def test_poc_details_fit_line_polynomial():
    fd = IndentationGroup(
        data_path
        / "fmt-jpk-fd_single_tilted-baseline-drift-"
          "mitotic_2021-01-29.jpk-force")[0]
    details = fd.apply_preprocessing(
        ["compute_tip_position", "correct_tip_offset"],
        options={"correct_tip_offset": {"method": "fit_line_polynomial"}},
        ret_details=True)
    pocd = details["correct_tip_offset"]
    for key in ["plot force",
                "plot fit",
                "plot poc"]:
        assert key in pocd
    assert np.allclose(
        pocd["plot poc"][0][0],
        15602,
        atol=0)
    assert np.allclose(
        fd["force"][15602],
        1.7313584441928315e-09,
        atol=0,
    )


def test_poc_frechet_direct_path():
    force = np.zeros(100)
    force[50:] = np.arange(50)**2
    cp = poc.poc_frechet_direct_path(force)
    assert cp == 62  # makes sense
