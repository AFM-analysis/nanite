"""Test of models using expressions"""
import pathlib

import lmfit
import numpy as np

import nanite
from nanite import IndentationGroup
import nanite.model
from nanite.model import residuals


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


class MockModelExpr:
    def __init__(self, **kwargs):
        """E and E1 add up to the actual emodulus. E1 is varied indirectly"""
        self.model_doc = """Mock model with constraint"""
        self.model_func = MockModelExpr.hertz_constraint
        self.model_key = "hertz_constraint"
        self.model_name = "Hertz with constraint "
        self.parameter_keys = ["E", "R", "nu", "virtual_parameter",
                               "E1", "contact_point", "baseline"]
        self.parameter_names = ["Young's Modulus", "Tip Radius",
                                "Poisson's Ratio", "Virtual Parameter",
                                "Another Modulus", "Contact Point",
                                "Force Baseline"]
        self.parameter_units = ["Pa", "m", "", "Pa", "Pa", "m", "N"]
        self.valid_axes_x = ["tip position"]
        self.valid_axes_y = ["force"]

    @staticmethod
    def get_parameter_defaults():
        # The order of the parameters must match the order
        # of ´parameter_names´ and ´parameter_keys´.
        params = lmfit.Parameters()
        params.add("E", value=1e3, min=0, vary=False)
        params.add("R", value=10e-6, vary=False)
        params.add("nu", value=.5, vary=False)
        params.add("virtual_parameter", value=10, min=0, vary=True)
        params.add("E1", expr="virtual_parameter+E")
        params.add("contact_point", value=0)
        params.add("baseline", value=0)
        return params

    @staticmethod
    def hertz_constraint(delta, E, R, nu, virtual_parameter, E1,
                         contact_point=0, baseline=0):
        aa1 = 4 / 3 * E1 / (1 - nu ** 2) * np.sqrt(R)

        root = contact_point - delta
        pos = root > 0
        bb = np.zeros_like(delta)
        bb[pos] = (root[pos]) ** (3 / 2)
        cc = np.zeros_like(delta)
        cc[pos] = 1 - 0.15 * root[pos] / R
        return aa1 * bb * cc + baseline

    @staticmethod
    def model(params, x):
        if x[0] < x[-1]:
            revert = True
        else:
            revert = False
        if revert:
            x = x[::-1]
        mf = MockModelExpr.hertz_constraint(
            E=params["E"].value,
            delta=x,
            R=params["R"].value,
            nu=params["nu"].value,
            virtual_parameter=params["virtual_parameter"].value,
            E1=params["E1"].value,
            contact_point=params["contact_point"].value,
            baseline=params["baseline"].value)
        if revert:
            return mf[::-1]
        return mf

    @staticmethod
    def residual(params, delta, force, weight_cp=5e-7):
        """ Compute residuals for fitting

        Parameters
        ----------
        params: lmfit.Parameters
            The fitting parameters for `model`
        delta: 1D ndarray of lenght M
            The indentation distances
        force: 1D ndarray of length M
            The corresponding force data
        weight_cp: positive float or zero/False
            The distance from the contact point until which
            linear weights will be applied. Set to zero to
            disable weighting.
        """
        md = MockModelExpr.model(params, delta)
        resid = force - md

        if weight_cp:
            # weight the curve so that the data around the contact_point do
            # not affect the fit so much.
            weights = residuals.compute_contact_point_weights(
                cp=params["contact_point"].value,
                delta=delta,
                weight_dist=weight_cp)
            resid *= weights
        return resid

    def __enter__(self):
        nanite.model.register_model(self, self.__repr__())
        return self

    def __exit__(self, a, b, c):
        nanite.model.models_available.pop(self.model_key)


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

    with MockModelExpr() as mod:
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

    with MockModelExpr() as mod:
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

    with MockModelExpr() as mod:
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


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
