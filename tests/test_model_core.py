"""Test NaniteFitModel class"""
import pathlib

import numpy as np
import pytest

import nanite.model.core

from common import MockModelModule

data_path = pathlib.Path(__file__).parent / "data"


def test_compute_anc_max_indent():
    ds1 = nanite.IndentationGroup(
        data_path / "fmt-jpk-fd_spot3-0192.jpk-force")
    fd = ds1[0]
    # correct for an offset in the tip
    fd.apply_preprocessing(preprocessing=["compute_tip_position",
                                          "correct_tip_offset"],
                           options={
                               "correct_tip_offset": {
                                   "method": "deviation_from_baseline"
                               }
                           })
    # fit a model
    fd.fit_model(model_key="hertz_para")
    # get the ancillary parameter
    max_indent = nanite.model.core.compute_anc_max_indent(fd)
    assert np.allclose(max_indent, 1.295070502610294e-07,
                       atol=0,
                       rtol=1e-5)


def test_model_bad_incomplete():
    class BadModel:
        model_key = "peterpan"
    bad_mod = BadModel()
    with pytest.raises(nanite.model.core.ModelIncompleteError,
                       match="parameter_names"):
        nanite.model.core.NaniteFitModel(bad_mod)


def test_model_bad_incomplete_anc():
    bad_mod = MockModelModule(model_key="peterpan",
                              compute_ancillaries=lambda x: {"peter": 1.2})
    with pytest.raises(nanite.model.core.ModelIncompleteError,
                       match="parameter_anc_keys"):
        nanite.model.core.NaniteFitModel(bad_mod)


def test_model_bad_labels_not_unique():
    parameter_names_wrong = [
        "SAME", "SAME", "Two", "Three", "Four"]
    bad_mod = MockModelModule(model_key="peterpan",
                              parameter_names=parameter_names_wrong)
    with pytest.raises(nanite.model.core.ModelImplementationError,
                       match="'parameter_names' should be unique for"):
        nanite.model.core.NaniteFitModel(bad_mod)


def test_model_bad_spaces_in_units():
    parameter_units_wrong = ["Pa", "m", " ", "m", "N"]
    bad_mod = MockModelModule(model_key="peterpan",
                              parameter_units=parameter_units_wrong)
    with pytest.warns(nanite.model.core.ModelImplementationWarning,
                      match="should not contain leading or trailing spaces"):
        nanite.model.core.NaniteFitModel(bad_mod)


def test_model_bad_spaces_in_units_anc():
    bad_mod = MockModelModule(
                compute_ancillaries=lambda x: {"E": np.nan},
                parameter_anc_keys=["E"],
                parameter_anc_names=["ancillary E guess"],
                parameter_anc_units=["Pa "],
                model_key="test2")
    with pytest.warns(nanite.model.core.ModelImplementationWarning,
                      match="should not contain leading or trailing spaces"):
        nanite.model.core.NaniteFitModel(bad_mod)


def test_model_get_anc_parm_keys():
    with MockModelModule(model_key="peterpan") as md:
        akeys = md.get_anc_parm_keys()
        assert "max_indent" in akeys


def test_model_get_anc_parm_keys_2():
    with MockModelModule(model_key="peterpan",
                         compute_ancillaries=lambda x: {"hans": 1.2},
                         parameter_anc_keys=["hans"],
                         parameter_anc_names=["Hans"],
                         parameter_anc_units=["N"],
                         ) as md:
        akeys = md.get_anc_parm_keys()
        assert "max_indent" in akeys
        assert "hans" in akeys


def test_model_get_parm_name():
    with MockModelModule(model_key="peterpan",
                         compute_ancillaries=lambda x: {"hans": 1.2},
                         parameter_anc_keys=["hans"],
                         parameter_anc_names=["Hans"],
                         parameter_anc_units=["N"],
                         ) as md:
        assert md.get_parm_name("E") == "Young's Modulus"
        assert md.get_parm_name("hans") == "Hans"
        assert md.get_parm_name("max_indent") == "Maximum indentation"
        with pytest.raises(KeyError, match="Could not find parameter name"):
            md.get_parm_name("peter")


def test_model_get_parm_unit():
    with MockModelModule(model_key="peterpan",
                         compute_ancillaries=lambda x: {"hans": 1.2},
                         parameter_anc_keys=["hans"],
                         parameter_anc_names=["Hans"],
                         parameter_anc_units=["N"],
                         ) as md:
        assert md.get_parm_unit("E") == "Pa"
        assert md.get_parm_unit("hans") == "N"
        assert md.get_parm_unit("max_indent") == "m"
        with pytest.raises(KeyError, match="Could not find parameter unit"):
            md.get_parm_unit("peter")


def test_model_repr_str():
    with MockModelModule(model_key="peterpan") as md:
        assert "peterpan" in repr(md)
        assert "NaniteFitModel" in repr(md)
        assert "peterpan" in str(md)
        assert "NaniteFitModel" in str(md)
