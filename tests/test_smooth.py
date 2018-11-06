"""Test smoothing functionalities"""
import numpy as np
import pytest

from nanite import smooth


def test_smooth_monotone():
    x = np.array([0]*30
                 + [1]*30
                 + [2]*30,
                 dtype=float)
    sm = smooth.smooth_axis_monotone(data=x, window=5)
    assert np.unique(sm).size == sm.size


@pytest.mark.filterwarnings('ignore::nanite.smooth.'
                            + 'DoubledSmoothingWindowWarning')
def test_smooth_monotone2():
    x = np.array([0]*30
                 + [1]*30
                 + [0]*10  # the algorithm will smooth over this part
                 + [1]*30,
                 dtype=float)
    sm = smooth.smooth_axis_monotone(data=x, window=5)
    assert np.unique(sm).size == sm.size


@pytest.mark.filterwarnings('ignore::nanite.smooth.'
                            + 'DoubledSmoothingWindowWarning')
def test_smooth_monotone_maxiter():
    # this is not solvable
    x = np.array([0]*30
                 + [1]*30
                 + [0]*30,
                 dtype=float)
    try:
        smooth.smooth_axis_monotone(data=x, window=5, max_iter=100)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.filterwarnings('ignore::nanite.smooth.'
                            + 'DoubledSmoothingWindowWarning')
def test_smooth_monotone_maxiter2():
    # this is not solvable
    x = np.array([0]*30
                 + [1]*30
                 + [0]*30,
                 dtype=float)
    try:
        smooth.smooth_axis_monotone(data=x, window=5,
                                    max_iter=0  # stop in first loop
                                    )
    except ValueError:
        pass
    else:
        assert False


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
