"""smooth data"""
import warnings

import numpy as np
import scipy.ndimage as im


class DoubledSmoothingWindowWarning(UserWarning):
    pass


def smooth_axis(data, window=15):
    """
    Smooth a 1D  data array with a median filter of `width`.

    See Also
    --------
    smooth_axis_monotone
    """
    smooth = im.median_filter(data, size=(window,), mode="nearest")
    return smooth


def smooth_axis_monotone(data, window=15, max_iter=1000):
    """
    Smooth a 1D  data array with a median filter of `width`.
    This method makes sure that the data is monotonously
    increasing or decreasing, by increasing the window
    size automatically. If this happens, a
    :class:`DoubledSmoothingWindowWarning` is raised.

    Window should be an odd number.

    See Also
    --------
    smooth_axis
    """
    smooth = smooth_axis(data, window=window)
    gradient = np.gradient(smooth)

    for _ in range(max_iter):
        if np.abs(np.sum(gradient)) == np.sum(np.abs(gradient)):
            break
        window = window * 2 + 1
        smooth = smooth_axis(data, window=window)
        gradient = np.gradient(smooth)
        warnings.warn("Automatically doubled smoothing `window` size to "
                      + "{}. You might consider using a ".format(window)
                      + "larger value by default.",
                      DoubledSmoothingWindowWarning)
    else:
        raise ValueError("Reached `max_iter`={}".format(max_iter))

    for _ in range(max_iter):
        if np.unique(smooth).size == smooth.size:
            break
        # Keep axis monotonous.
        # get the first element with equal values
        equal = []
        myset = False
        for ii in range(smooth.size):
            if smooth[ii+1] == smooth[ii]:
                equal.append(ii+1)
                myset = True
            elif myset is True:
                # continue with these values in the next for-loop
                break
            if smooth.size == ii+2:
                # abort
                break

        for count, idx in enumerate(equal):
            try:
                smooth[idx] += (smooth[equal[-1]+1] -
                                smooth[equal[0]])/(len(equal)+5)*(count+1)
            except BaseException:
                # we have last element
                dx = (smooth[1]-smooth[0])/10
                smooth[-1] += dx
    else:
        raise ValueError("Reached `max_iter`={}".format(max_iter))

    return smooth
