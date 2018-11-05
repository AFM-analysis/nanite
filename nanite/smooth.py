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


def smooth_axis_monotone(data, window=15):
    """
    Smooth a 1D  data array with a median filter of `width`.
    This method makes sure that the data is monotonously
    increasing or decreasing, by increasing the window
    size automatically. If this happens, a warning will
    be issued.

    Window should be an uneven number.

    See Also
    --------
    smooth_axis
    """
    smooth = smooth_axis(data, window=window)
    gradient = np.gradient(smooth)

    while np.abs(np.sum(gradient)) != np.sum(np.abs(gradient)):
        window = window * 2 + 1
        smooth = smooth_axis(data, window=window)
        gradient = np.gradient(smooth)
        warnings.warn("Automatically doubled smoothing `window` size to "
                      + "{}. You might consider using a ".format(window)
                      + "larger value by default.",
                      DoubledSmoothingWindowWarning)

    while (np.unique(smooth).shape != smooth.shape):
        # keep axis unique
        # get first non-unique batch:
        equal = []
        myset = False
        for ii in range(smooth.shape[0]):
            if smooth[ii+1] == smooth[ii]:
                equal.append(ii+1)
                myset = True
            elif myset is True:
                break
            if smooth.shape[0] == ii+2:
                break

        for count, idx in enumerate(equal):
            try:
                smooth[idx] += (smooth[equal[-1]+1] -
                                smooth[equal[0]])/(len(equal)+5)*(count+1)
            except BaseException:
                # we have last element
                dx = (smooth[1]-smooth[0])/10
                smooth[-1] += dx
        if len(equal) == 0:
            break

    return smooth
