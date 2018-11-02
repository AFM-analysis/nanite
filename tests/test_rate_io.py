"""Test input/output of user rating data"""
import pathlib
import shutil
import tempfile

import h5py
import numpy as np

from nanite import model, IndentationDataSet
from nanite.rate.io import save_hdf5, load_hdf5


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_wite():
    ds1 = IndentationDataSet(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = model.model_hertz_parabolic.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    apret.fit_model(model_key="hertz_para",
                    params_initial=inparams,
                    range_x=(0, 0),
                    range_type="absolute",
                    x_axis="tip position",
                    y_axis="force",
                    segment="approach",
                    weight_cp=False)

    tdir = tempfile.mkdtemp(prefix="test_nanite_rate_io_")
    h5path = pathlib.Path(tdir) / "simple.h5"
    save_hdf5(h5path=h5path,
              indent=apret,
              user_rate=5,
              user_name="hans",
              user_comment="this is a comment",
              h5mode="a")

    with h5py.File(str(h5path), mode="r") as hi:
        # experimental data
        assert "4443b7" in hi["data"]
        # a few attributes
        attrs = hi["analysis/4443b7_0"].attrs
        assert attrs["fit model_key"] == "hertz_para"
        assert not attrs["fit optimal_fit_edelta"]
        assert attrs["fit preprocessing"] == "compute_tip_position"
        assert np.allclose(hi["analysis/4443b7_0"]["fit"], apret.data["fit"],
                           equal_nan=True)

    shutil.rmtree(tdir, ignore_errors=True)


def test_write_read():
    ds1 = IndentationDataSet(jpkfile)
    apret = ds1[0]
    apret.apply_preprocessing(["compute_tip_position"])

    inparams = model.model_hertz_parabolic.get_parameter_defaults()
    inparams["baseline"].vary = True
    inparams["contact_point"].set(1.8e-5)

    # Fit with absolute full range
    apret.fit_model(model_key="hertz_para",
                    params_initial=inparams,
                    range_x=(0, 0),
                    range_type="absolute",
                    x_axis="tip position",
                    y_axis="force",
                    segment="approach",
                    weight_cp=False)

    tdir = tempfile.mkdtemp(prefix="test_nanite_rate_io_")
    h5path = pathlib.Path(tdir) / "simple.h5"
    save_hdf5(h5path=h5path,
              indent=apret,
              user_rate=5,
              user_name="hans",
              user_comment="this is a comment",
              h5mode="a")

    datalist = load_hdf5(h5path)
    assert datalist[0]["rating"] == 5
    shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
