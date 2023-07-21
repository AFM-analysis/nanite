"""Test basic fitting"""
import pathlib

import numpy as np

import nanite
from nanite import IndentationGroup
import pytest


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
                       atol=2e-10,
                       rtol=0,
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
                       atol=2e-10,
                       rtol=0,
                       )


@pytest.mark.parametrize("gcf_k", [0.1, 0.23, 0.3, 1/np.pi, 0.5, 0.6, 1.0])
def test_gcf_k_no_change_in_contact_point(gcf_k):
    """Fit result for contact point does not change with gcf_k"""
    ds1 = IndentationGroup(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8321e-5)
    # sane initial fit parameters with gcf_k in mind
    inparams["E"].set(inparams["E"].value * (1/gcf_k)**(3/2))

    # Fit with absolute full range
    kwargs = dict(model_key="hertz_para",
                  params_initial=inparams,
                  range_x=(0, 0),
                  range_type="absolute",
                  x_axis="tip position",
                  y_axis="force",
                  segment="approach",
                  weight_cp=False,
                  gcf_k=gcf_k)

    apret.fit_model(**kwargs)
    params1 = apret.fit_properties["params_fitted"]
    assert np.allclose(params1["contact_point"].value,
                       1.802931023582261e-05,
                       atol=2e-11,
                       rtol=0,
                       )


@pytest.mark.parametrize("gcf_k", [0.1, 0.23, 0.3, 1/np.pi, 0.5, 0.6, 1.0])
def test_gcf_k_no_change_in_fitted_curve(gcf_k):
    """Fit result for contact point does not change with gcf_k"""
    ds1 = IndentationGroup(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8321e-5)
    # sane initial fit parameters with gcf_k in mind
    inparams["E"].set(inparams["E"].value * (1/gcf_k)**(3/2))

    # Fit with absolute full range
    kwargs = dict(model_key="hertz_para",
                  params_initial=inparams,
                  range_x=(0, 0),
                  range_type="absolute",
                  x_axis="tip position",
                  y_axis="force",
                  segment="approach",
                  weight_cp=False,
                  gcf_k=gcf_k)

    apret.fit_model(**kwargs)
    assert np.allclose(np.nanmax(apret["fit"]),
                       3.496152328502394e-09,
                       atol=0,
                       rtol=1e-4)
    assert np.allclose(np.nanmax(apret["fit residuals"]),
                       2.2340798430821952e-10,
                       atol=0,
                       rtol=1e-3)


@pytest.mark.parametrize("gcf_k", [0.1, 0.23, 0.3, 1/np.pi, 0.5, 0.6, 1.0])
def test_gcf_k_scaling_of_youngs_modulus(gcf_k):
    """Fit result for Young's modulus should scale with gcf_k"""
    ds1 = IndentationGroup(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position", "correct_tip_offset"])

    inparams = nanite.model.model_hertz_paraboloidal.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(0)

    # Fit with absolute full range
    kwargs = dict(model_key="hertz_para",
                  params_initial=inparams,
                  range_x=(0, 0),
                  range_type="absolute",
                  x_axis="tip position",
                  y_axis="force",
                  segment="approach",
                  weight_cp=False,
                  gcf_k=gcf_k)

    apret.fit_model(**kwargs)
    params1 = apret.fit_properties["params_fitted"]
    # proportionality between E and gcf_k
    corrval = 14741.950622347102 * (1/gcf_k)**(3/2)
    assert np.allclose(params1["E"].value,
                       corrval,
                       rtol=0.0001,
                       atol=0,
                       )


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
