"""Test of ancillary parameters"""
import pathlib

import numpy as np

import nanite
import nanite.model


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


class MockModel():
    def __init__(self, model_key, **kwargs):
        # rebase on hertz model
        md = nanite.model.models_available["hertz_para"]
        for key in dir(md):
            setattr(self, key, getattr(md, key))
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
            model_key="test1"):
        idnt.fit_model(preprocessing=["compute_tip_position"],
                       model_key="test1")
        assert idnt.fit_properties["params_initial"]["E"].value == 1580
        assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                           1584.8941257802458,
                           atol=0,
                           rtol=1e-12)


def test_simple_ancillary_override_nan():
    """nan values are not used and should be ignored"""
    ds1 = nanite.IndentationGroup(jpkfile)
    idnt = ds1[0]

    with MockModel(
        compute_ancillaries=lambda x: {"E": np.nan},
        parameter_anc_keys=["E"],
        parameter_anc_names=["ancillary E guess"],
            model_key="test2"):
        idnt.fit_model(preprocessing=["compute_tip_position"],
                       model_key="test2")
        assert idnt.fit_properties["params_initial"]["E"].value == 3000
        assert np.allclose(idnt.fit_properties["params_fitted"]["E"].value,
                           1584.8876592662375,
                           atol=0,
                           rtol=1e-12)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
