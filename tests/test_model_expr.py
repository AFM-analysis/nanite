"""Test of models using expressions"""
import pathlib

import numpy as np

from nanite import IndentationGroup

from common import MockModelModuleExpr


data_dir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_dir / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_expr_model():
    """Make sure that a model with an expression is fitted correctly"""
    # Reference fit
    rgrp = IndentationGroup(jpkfile)
    ridnt = rgrp[0]
    ridnt.apply_preprocessing(["compute_tip_position",
                               "correct_force_offset"])
    ridnt.fit_model(model_key="hertz_para",
                    params_initial=None,
                    x_axis="tip position",
                    y_axis="force",
                    weight_cp=False,
                    segment="retract")
    rparms = ridnt.fit_properties["params_fitted"]
    remod = rparms["E"].value

    with MockModelModuleExpr() as mod:
        grp = IndentationGroup(jpkfile)
        idnt = grp[0]
        idnt.apply_preprocessing(["compute_tip_position",
                                  "correct_force_offset"])
        idnt.fit_model(model_key=mod.model_key,
                       params_initial=None,
                       x_axis="tip position",
                       y_axis="force",
                       weight_cp=False,
                       segment="retract")
        parms = idnt.fit_properties["params_fitted"]
        # make sure the expression survives
        assert parms["E1"].expr == "virtual_parameter+E"
        emod = parms["E1"].value
        # There are some difference due to heuristics
        assert np.allclose(emod, remod, atol=0, rtol=3e-3)


def test_expr_model_limit():
    """Fit with a limit towards the correct solution"""
    # Reference fit
    rgrp = IndentationGroup(jpkfile)
    ridnt = rgrp[0]
    ridnt.apply_preprocessing(["compute_tip_position",
                               "correct_force_offset"])
    ridnt.fit_model(model_key="hertz_para",
                    params_initial=None,
                    x_axis="tip position",
                    y_axis="force",
                    weight_cp=False,
                    segment="retract")
    rparmsi = ridnt.fit_properties["params_initial"]

    with MockModelModuleExpr() as mod:
        grp = IndentationGroup(jpkfile)
        idnt = grp[0]
        idnt.apply_preprocessing(["compute_tip_position",
                                  "correct_force_offset"])
        params_initial = mod.get_parameter_defaults()
        params_initial["E"].set(value=19000)
        params_initial["virtual_parameter"].set(value=10, min=0, max=200)
        params_initial["contact_point"].set(
            value=rparmsi["contact_point"].value)
        idnt.fit_model(model_key=mod.model_key,
                       params_initial=params_initial,
                       x_axis="tip position",
                       y_axis="force",
                       weight_cp=False,
                       segment="retract")
        parms = idnt.fit_properties["params_fitted"]
        # make sure the expression survives
        assert parms["E1"].expr == "virtual_parameter+E"
        emod = parms["E1"].value
        # It should have gone to the boundary
        assert np.allclose(emod, 19200)


def test_expr_model_sign():
    """Fit with negative limit"""
    # Reference fit
    rgrp = IndentationGroup(jpkfile)
    ridnt = rgrp[0]
    ridnt.apply_preprocessing(["compute_tip_position",
                               "correct_force_offset"])
    ridnt.fit_model(model_key="hertz_para",
                    params_initial=None,
                    x_axis="tip position",
                    y_axis="force",
                    weight_cp=False,
                    segment="retract")
    rparms = ridnt.fit_properties["params_fitted"]
    rparmsi = ridnt.fit_properties["params_initial"]
    remod = rparms["E"].value

    with MockModelModuleExpr() as mod:
        grp = IndentationGroup(jpkfile)
        idnt = grp[0]
        idnt.apply_preprocessing(["compute_tip_position",
                                  "correct_force_offset"])
        params_initial = mod.get_parameter_defaults()
        params_initial["E"].set(value=20000)
        params_initial["virtual_parameter"].set(value=-1, min=-np.inf, max=0)
        params_initial["contact_point"].set(
            value=rparmsi["contact_point"].value)
        idnt.fit_model(model_key=mod.model_key,
                       params_initial=params_initial,
                       x_axis="tip position",
                       y_axis="force",
                       weight_cp=False,
                       segment="retract")
        parms = idnt.fit_properties["params_fitted"]
        # make sure the expression survives
        assert parms["E1"].expr == "virtual_parameter+E"
        emod = parms["E1"].value
        # There are some difference due to heuristics
        assert np.allclose(emod, remod, atol=0, rtol=3e-3)
