import numpy as np


def weight_cp(cp, delta, weight_dist=5e-7):
    """Compute contact point weights

    Parameters
    ----------
    cp: float
        Fitted contact point value
    delta: 1d ndarray of length N
        The indentation array along which weights will be computed.
    weight_width: float
        The distance from `cp` until which weights will be applied.

    Returns
    -------
    weights: 1d ndarray of length N
        The weights.

    Notes
    -----
    All variables should be given in the same units. The weights increase
    linearly from increasing distances of `delta-cp` from 0 to 1 and are
    1 outside of the weight width `abs(delta-cp)>weight_width`.
    """
    # weights are proportional to distance from contact point
    # normalized by weight_width.
    x = np.abs(delta-cp)
    x /= weight_dist
    x[x > 1] = 1
    return x
