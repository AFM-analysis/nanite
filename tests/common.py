import lmfit
import nanite
import nanite.model.logic
from nanite.model import residuals
import numpy as np


class MockModelModule:
    def __init__(self, model_key, **kwargs):
        super(MockModelModule, self).__init__()
        # rebase on hertz model
        md = nanite.model.models_available["hertz_para"].module
        for akey in dir(md):
            setattr(self, akey, getattr(md, akey))
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        self.model_key = model_key

    def __enter__(self):
        return nanite.model.logic.register_model(self)

    def __exit__(self, a, b, c):
        nanite.model.models_available.pop(self.model_key)


class MockModelModuleExpr:
    def __init__(self, **kwargs):
        """E and E1 add up to the actual emodulus. E1 is varied indirectly"""
        self.model_doc = """Mock model with constraint"""
        self.model_func = MockModelModuleExpr.hertz_constraint
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

    def __enter__(self):
        nanite.model.logic.register_model(self)
        return self

    def __exit__(self, a, b, c):
        nanite.model.logic.deregister_model(self)

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
        mf = MockModelModuleExpr.hertz_constraint(
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
        md = MockModelModuleExpr.model(params, delta)
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
