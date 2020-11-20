"""Test of ancillary parameters"""
import pathlib

import numpy as np

import nanite
import nanite.model


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


class MockModel:
    def __init__(self, model_key, **kwargs):
        # rebase on hertz model
        md = nanite.model.models_available["hertz_para"]
        for akey in dir(md):
            setattr(self, akey, getattr(md, akey))
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        self.model_key = model_key

    def __enter__(self):
        nanite.model.register_model(self, self.__repr__())

    def __exit__(self, a, b, c):
        nanite.model.models_available.pop(self.model_key)


def test_simple_ancillary_override():
    """basic test for ancillary parameters"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    with MockModel(
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
                           1572.7685940809245,
                           atol=0,
                           rtol=1e-5)


def test_simple_ancillary_override_nan():
    """nan values are not used and should be ignored"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    with MockModel(
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
                           1584.9805568368859,
                           atol=0,
                           rtol=1e-5)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
