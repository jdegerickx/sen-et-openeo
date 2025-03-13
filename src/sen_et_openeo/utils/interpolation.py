import numpy as np
from numba import guvectorize


def _interp1d(xnew, xvals, yvals, ynew):
    i = 0
    N = len(xvals)
    if xnew[0] < xvals[0]:
        # x_a = 0.0
        # y_a = 0.0
        # x_b = xvals[0]
        # y_b = yvals[0]
        ynew[0] = yvals[0]
        return
    elif xnew[-1] > xvals[-1]:
        ynew[-1] = yvals[-1]
        return
    else:
        while xnew[0] >= xvals[i] and i < N:
            i += 1
        if xnew[0] == xvals[i]:
            ynew[0] = yvals[i]
            return
        if i == N:
            i = N-1
        x_a = xvals[i-1]
        y_a = yvals[i-1]
        x_b = xvals[i]
        y_b = yvals[i]
    slope = (xnew[0] - x_a)/(x_b - x_a)
    ynew[0] = slope * (y_b-y_a) + y_a
    return


interp1d = guvectorize(
    ['int64[:], int64[:], float32[:], float32[:]'],
    "(),(n),(n) -> ()", nopython=True)(_interp1d)

interp1d_uint16 = guvectorize(
    ['int64[:], int64[:], uint16[:], uint16[:]'],
    "(),(n),(n) -> ()", nopython=True)(_interp1d)


def interpolate_fast(arrs):

    for band in range(arrs.shape[0]):
        for px in range(arrs.shape[2]):
            y = arrs[band, :, px]

            nans_ids = np.isnan(y)
            xvals = np.where(~nans_ids)[0]
            yvals = y[xvals]
            xnew = np.where(nans_ids)[0]
            y[xnew] = interp1d(xnew, xvals, yvals)

    return arrs


def _interpolate_4d_float32(arrs):

    for band in range(arrs.shape[0]):
        for px in range(arrs.shape[2]):
            for py in range(arrs.shape[3]):
                y = arrs[band, :, px, py]

                nans_ids = np.isnan(y)
                xvals = np.where(~nans_ids)[0]
                yvals = y[xvals]
                xnew = np.where(nans_ids)[0]
                y[xnew] = interp1d(xnew, xvals, yvals)

    return arrs


def _interpolate_4d_uint16(arrs):

    for band in range(arrs.shape[0]):
        for px in range(arrs.shape[2]):
            for py in range(arrs.shape[3]):
                y = arrs[band, :, px, py]

                nans_ids = (y == 0)
                xvals = np.where(~nans_ids)[0]
                yvals = y[xvals]
                xnew = np.where(nans_ids)[0]
                y[xnew] = interp1d(xnew, xvals, yvals)

    return arrs


def interpolate_4d(arrs):

    if arrs.dtype == np.float32:
        return _interpolate_4d_float32(arrs)
    elif arrs.dtype == np.uint16:
        return _interpolate_4d_uint16(arrs)
    else:
        raise ValueError("Interpolate function is only available for "
                         "arrays of type float32 or uint16")
