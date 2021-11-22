"""Test basic fitting"""
import pathlib

import numpy as np

import nanite
from nanite import IndentationGroup


data_path = pathlib.Path(__file__).parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_lmfit_method():
    ds1 = IndentationGroup(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8321e-5)

    # Fit with absolute full range
    kwargs = dict(model_key="hertz_para",
                  params_initial=inparams,
                  range_x=(0, 0),
                  range_type="absolute",
                  x_axis="tip position",
                  y_axis="force",
                  segment="approach",
                  weight_cp=False)

    apret.fit_model(**kwargs)
    params1 = apret.fit_properties["params_fitted"]
    assert np.allclose(params1["contact_point"].value,
                       1.802931023582261e-05,
                       rtol=0,
                       atol=0.000000000005,
                       )

    # make sure leastsq is the default
    apret.fit_model(method="leastsq", **kwargs)
    params2 = apret.fit_properties["params_fitted"]
    assert params2["contact_point"].value == params1["contact_point"]

    # use a different method
    apret.fit_model(method="nelder", method_kws={"max_nfev": 5}, **kwargs)
    params3 = apret.fit_properties["params_fitted"]
    assert params3["contact_point"].value != params1["contact_point"]
    assert np.allclose(params3["contact_point"].value,
                       1.8779025e-05,
                       rtol=0,
                       atol=0.000000000005,
                       )


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
