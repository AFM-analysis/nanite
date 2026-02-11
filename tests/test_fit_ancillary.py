"""Test of ancillary parameters"""
import pathlib

import numpy as np

import nanite
import nanite.model

from common import MockModelModule

data_path = pathlib.Path(__file__).parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_error_ancillary_yields_nan():
    """nan values are not used and should be ignored"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    def compute_ancillaries(*args, **kwargs):
        raise ValueError("Not computed")

    with MockModelModule(
            compute_ancillaries=compute_ancillaries,
            parameter_anc_keys=["J"],
            parameter_anc_names=["ancillary J guess"],
            parameter_anc_units=["Pa"],
            model_key="test2"):
        # We need to perform preprocessing first, if we want to get the
        # correct initial contact point.
        idnt.apply_preprocessing(["compute_tip_position"])
        # We set the baseline fixed, because this test was written so)
        params_initial = idnt.get_initial_fit_parameters(model_key="test2")
        params_initial["baseline"].set(vary=False)
        idnt.fit_model(model_key="test2",
                       params_initial=params_initial)
        anc_params = idnt.get_ancillary_parameters()
        assert np.isnan(anc_params["J"])


def test_simple_ancillary_override():
    """basic test for ancillary parameters"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    with MockModelModule(
            compute_ancillaries=lambda x: {"E": 1580},
            parameter_anc_keys=["E"],
            parameter_anc_names=["ancillary E guess"],
            parameter_anc_units=["Pa"],
            model_key="test1"):
        # We need to perform preprocessing first, if we want to get the
        # correct initial contact point.
        idnt.apply_preprocessing(["compute_tip_position"])
        # We set the baseline fixed, because this test was written so)
        params_initial = idnt.get_initial_fit_parameters(model_key="test1")
        params_initial["baseline"].set(vary=False)
        idnt.fit_model(model_key="test1",
                       params_initial=params_initial)
        assert idnt.fit_properties["params_initial"]["E"].value == 1580
        assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                           1584.8941261696934,
                           atol=1,
                           rtol=0)


def test_simple_ancillary_override_nan():
    """nan values are not used and should be ignored"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    with MockModelModule(
            compute_ancillaries=lambda x: {"E": np.nan},
            parameter_anc_keys=["E"],
            parameter_anc_names=["ancillary E guess"],
            parameter_anc_units=["Pa"],
            model_key="test2"):
        # We need to perform preprocessing first, if we want to get the
        # correct initial contact point.
        idnt.apply_preprocessing(["compute_tip_position"])
        # We set the baseline fixed, because this test was written so)
        params_initial = idnt.get_initial_fit_parameters(model_key="test2")
        params_initial["baseline"].set(vary=False)
        idnt.fit_model(model_key="test2",
                       params_initial=params_initial)
        assert idnt.fit_properties["params_initial"]["E"].value == 3000
        assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                           1584.8876592662375,
                           atol=1,
                           rtol=0)


def test_request_ancillary_parameters():
    """request the ancillary parameters after fitting"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    model_key = "test4"
    with MockModelModule(model_key=model_key):
        # We need to perform preprocessing first, if we want to get the
        # correct initial contact point.
        idnt.apply_preprocessing(["compute_tip_position"])
        # We set the baseline fixed, because this test was written so)
        params_initial = idnt.get_initial_fit_parameters(model_key=model_key)
        params_initial["baseline"].set(vary=False)
        idnt.fit_model(model_key=model_key,
                       params_initial=params_initial)

        # ancillary parameters are not yet requested
        assert idnt._anc_cache is None
        assert not idnt._anc_valid

        # actually request the ancillary parameters, as done in pyjibe
        anc = idnt.get_ancillary_parameters(
            model_key=model_key)

        # check ancillaries
        assert idnt._anc_cache is not None
        assert idnt._anc_valid
        assert np.allclose(anc["max_indent"], 3.669487775650337e-07,
                           atol=1e-10, rtol=0)

        # check params_initial and params_fitted
        assert idnt.fit_properties["params_initial"]["E"].value == 3000
        assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                           1584.8876592662375, atol=1, rtol=0)
