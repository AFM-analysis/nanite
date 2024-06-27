"""Test of data set functionalities"""
import pathlib

import lmfit
import numpy as np

from nanite import IndentationGroup
from nanite.model import model_power_layer_clifford_2009 as clifford


data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_app_ret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    idp = ar.estimate_contact_point_index()

    aprid = ar["segment"] == 0
    x = ar["tip position"][aprid]
    y = ar["force"][aprid]
    contact_point = x[idp]

    # crop x and y around contact_point
    distcp = x.shape[0]-idp
    x = x[-3*distcp:]
    y = y[-3*distcp:]

    params = clifford.get_parameter_defaults()
    params["t"].vary = False
    params["contact_point"].value = contact_point

    fit_res = lmfit.minimize(clifford.residual, params, args=(x, y, 0))

    # This is only a sanity check to make sure the
    # model is reproducible.
    assert np.allclose(fit_res.params["E_S"].value,
                       42218,
                       rtol=1e-2, atol=0)
    assert np.allclose(fit_res.params["E_L"].value,
                       191,
                       rtol=1e-2, atol=0)
    assert np.allclose(fit_res.params["contact_point"].value,
                       1.805e-5,
                       rtol=1e-2, atol=0)
