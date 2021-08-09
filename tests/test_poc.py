"""Test point of contact (POC) estimation"""
import pathlib

import numpy as np
import pytest

from nanite import poc, IndentationGroup


data_path = pathlib.Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize("method,contact_point", [
    ["gradient_zero_crossing", 1895],
    ["fit_constant_line", 1919],
    ["deviation_from_baseline", 1908],
    ])
def test_poc_estimation(method, contact_point):
    fd = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    assert poc.compute_poc(fd["force"], method) == contact_point


@pytest.mark.parametrize("method,contact_point", [
    ["gradient_zero_crossing", 1895],
    ["fit_constant_line", 1919],
    ["deviation_from_baseline", 1908],
    ])
def test_poc_estimation_via_indent(method, contact_point):
    fd = IndentationGroup(data_path / "spot3-0192.jpk-force")[0]
    fd.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"],
                           options={"correct_tip_offset": {"method": method}})
    assert np.argmin(np.abs(fd["tip position"])) == contact_point
