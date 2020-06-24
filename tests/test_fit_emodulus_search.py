"""Test of data set functionalities"""
import pathlib
import time

import numpy as np

from nanite import IndentationGroup
from nanite.fit import FitDataError


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"
badjpk = datapath / "bad_GWATspot1-data-2017.10.17-16.39.42.396-5.jpk-force"


def test_emodulus_search():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"])
    ar.fit_model(model_key="hertz_cone",
                 params_initial=None,
                 x_axis="tip position",
                 y_axis="force",
                 weight_cp=False,
                 segment="approach",
                 optimal_fit_edelta=True,
                 )
    assert "optimal_fit_delta_array" in ar.fit_properties
    assert "optimal_fit_E_array" in ar.fit_properties
    assert "optimal_fit_delta" in ar.fit_properties

    # This assertion might fail when the preprocessing changes
    # or when the search algorithm for the optimal fit changes.
    dopt = -2.07633035802137e-07
    assert np.allclose(ar.fit_properties["optimal_fit_delta"], dopt)

    if __name__ == "__main__":
        import matplotlib.pylab as plt
        _fig, axes = plt.subplots(2, 1)
        axes[0].plot(ar["tip position"]*1e6, ar["force"]*1e9, label="data")
        axes[0].plot(ar["tip position"]*1e6, ar["fit"]*1e9, label="fit")
        axes[0].legend()
        axes[0].grid()
        axes[0].set_xlabel("indentation [µm]")
        axes[0].set_ylabel("force [nN]")

        deltas = ar.fit_properties["optimal_fit_delta_array"]*1e6
        emod = ar.fit_properties["optimal_fit_E_array"]
        axes[1].plot(deltas, emod, label="emodulus/minimal-indentation-curve")
        axes[1].set_xlabel("minimal indentation [µm]")
        axes[1].set_ylabel("emodulus [Pa]")
        axes[1].vlines(ar.fit_properties["optimal_fit_delta"]*1e6,
                       emod.min(), emod.max(),
                       label="optimal minimal indentation at plateau",
                       color="r")
        axes[1].grid()
        axes[1].legend()
        plt.tight_layout()
        plt.show()


def test_cache_emodulus():
    # Check that the fitting procedure does not perform unneccessary
    # double fits and uses the cached variables for emoduli and
    # minimal indentations.
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"])
    t01 = time.perf_counter()
    ar.fit_model(model_key="hertz_cone",
                 params_initial=None,
                 x_axis="tip position",
                 y_axis="force",
                 weight_cp=False,
                 segment="approach",
                 optimal_fit_edelta=True,
                 )
    t02 = time.perf_counter()

    t11 = time.perf_counter()
    ar.fit_model()
    t12 = time.perf_counter()
    assert (t02-t01)*1e-4 > t12 - \
        t11, "Second computation should yield cached value (faster)!"

    # Make sure that changing the fit model is longer again
    t21 = time.perf_counter()
    ar.fit_model(model_key="hertz_para")
    t22 = time.perf_counter()

    assert ((t22-t21)*1e-4
            > (t12 - t11)), "Changing model_key should slow down computation!"


def test_fit_data_error():
    # a FitDataError is raised when it is not possible to compute an
    # E(delta) curve:
    ds = IndentationGroup(badjpk)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset",
                            "correct_tip_offset"])
    try:
        ar.compute_emodulus_mindelta()
    except FitDataError:
        pass
    else:
        assert False, "Invalid input data should not allow E(delt) computation"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
