"""Test of data set functionalities"""
import pathlib
import time

import nanite
from nanite import IndentationGroup


datapath = pathlib.Path(__file__).parent / "data"
jpkfile = datapath / "spot3-0192.jpk-force"


def test_hash_time():
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
    t0 = time.perf_counter()
    apret.fit_model(**kwargs)
    t1 = time.perf_counter()
    apret.fit_model()
    t2 = time.perf_counter()
    kwargs["weight_cp"] = 1e-5
    apret.fit_model(**kwargs)
    t3 = time.perf_counter()
    apret.fit_model()
    t4 = time.perf_counter()

    assert t1-t0 >= 100 * \
        (t2-t1), "Consecutive fits with same parameters should be instant"
    assert t3-t2 >= 100 * \
        (t2-t1), "Changing parameters again should cause a new fit"
    assert t3-t2 >= 100*(t4-t3), "And computing the same should be faster"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
