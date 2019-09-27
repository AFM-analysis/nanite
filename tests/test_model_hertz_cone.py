"""Test of data set functionalities"""
import pathlib

import lmfit
import numpy as np

from nanite import IndentationGroup
from nanite.model import model_conical_indenter as hertz_conical


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_app_ret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    idp = ar.estimate_contact_point_index()

    aprid = ~ar["segment"]
    x = ar["tip position"][aprid]
    y = ar["force"][aprid]
    contact_point = x[idp]

    # crop x and y around contact_point
    distcp = x.shape[0]-idp
    x = x[-3*distcp:]
    y = y[-3*distcp:]

    params = lmfit.Parameters()
    params.add("contact_point", value=contact_point)
    params.add("baseline", value=0, vary=False)
    params.add("E", value=300e3, min=0)
    params.add("nu", value=.5, vary=False)
    params.add("alpha", value=30, vary=False)

    fit_w = lmfit.minimize(hertz_conical.residual, params, args=(x, y, True))
    fit_n = lmfit.minimize(hertz_conical.residual, params, args=(x, y, False))

    # Correctly reproduces fit results in the JPK analysis software
    # with "Vertical Tip Position", "Switchable Baseline Operation",
    # and a conical indenter with the "Hertz/Sneddon" "Model type".
    # The contact point was calculated through Hertz model and not adjusted
    # before fitting by JPK Software.
    # Set half cone angle to 30 Deg.
    E_jpk = 348.6e3
    cp_jpk = 18.06e-6
    assert np.allclose(fit_n.params["E"].value, E_jpk, rtol=8e-2, atol=0)
    assert np.allclose(
        fit_n.params["contact_point"].value, cp_jpk, rtol=4e-3, atol=0)

    if __name__ == "__main__" and False:
        import matplotlib.pylab as plt
        _fig, axes = plt.subplots(2, 1)
        xf = np.linspace(x[0], x[-1], 100)
        axes[0].plot(xf, hertz_conical.model(
            fit_w.params, xf), label="fit with weights")
        axes[0].plot(xf, hertz_conical.model(
            fit_n.params, xf), label="fit no weights")
        axes[0].plot(x, y, label="data")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(x, hertz_conical.residual(fit_w.params, x, y,
                                               weight_cp=True),
                     label="residuals with weight")
        axes[1].plot(x, hertz_conical.residual(fit_n.params, x, y,
                                               weight_cp=False),
                     label="residuals no weight")
        axes[1].legend()
        axes[1].grid()

        plt.show()


def test_fit_apret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    ar.fit_model(model_key="hertz_cone",
                 params_initial=None,
                 x_axis="tip position",
                 y_axis="force",
                 weight_cp=False,
                 segment="retract")

    if __name__ == "__main__":
        import matplotlib.pylab as plt
        _fig, axes = plt.subplots(2, 1)
        axes[0].plot(ar["tip position"], ar["force"], label="data")
        axes[0].plot(ar["tip position"], ar["fit"], label="fit")
        axes[0].legend()
        axes[0].grid()
        plt.show()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
