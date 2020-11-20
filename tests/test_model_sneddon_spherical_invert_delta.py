"""Test of inverse function"""
import numpy as np

from nanite.model import model_sneddon_spherical as hertz_spherical


def test_get_a_sneddon():
    r = 40e-6
    delta_real = [0.1e-6, 2e-6, 5e-6]
    delta_predict = np.zeros_like(delta_real)
    for i in range(len(delta_real)):
        a = hertz_spherical.get_a(r, delta_real[i], accuracy=1e-09)
        delta_predict[i] = hertz_spherical.delta_of_a(a, r)

    assert np.allclose(delta_real, delta_predict, rtol=1e-06, atol=0)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
