"""Test of data set functionalities"""
import pathlib

import lmfit
import numpy as np
import matplotlib.pylab as plt

from nanite import IndentationGroup
from nanite.model import model_hertz_corrected_viscoelasticity_KVM as \
    hertz_corr_visco_KVM


data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_KVM_5um_sec_B-2018-08-02.jpk-force"

def test_fit_apret():
    # arrange
    grp = IndentationGroup(jpkfile)
    E_exp = 1557

    idnt = grp[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_tip_offset",
                              "correct_force_offset"],
                             {"correct_tip_offset":{
                                 'method':"fit_line_polynomial"}})
    # act
    idnt.fit_model(model_key="hertz_corr_visco_KVM",
                   params_initial=None,
                   x_axis="tip position",
                   y_axis="force",
                   range_x=(-5e-6, 3e-7),
                   range_type='absolute',
                   weight_cp=False,
                   method="nelder")
    # assert
    assert np.allclose(idnt.fit_properties["params_fitted"]["E"], E_exp,
                       rtol=1e-02, atol=0)

    if __name__ == "__main__":
        _fig, axes = plt.subplots(2, 1)
        axes[0].plot(idnt["tip position"], idnt["force"], label="data")
        axes[0].plot(idnt["tip position"], idnt["fit"], label="fit")
        axes[0].legend()
        axes[0].grid()
        plt.show()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()