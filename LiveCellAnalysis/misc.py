import numpy as np
import scipy
from scipy import optimize
from skimage import transform
from numbers import Number
from tllab_common.misc import *


def translateFrame(frame, shift):
    if np.all([i==0 for i in shift]):
        return frame
    else:
        transobject = transform.AffineTransform(translation=shift)
        return transform.warp(frame, transobject, mode='edge')


def CDF(d):
    d = np.array(d)
    d = np.sort(d).flatten()
    d = d[np.isfinite(d)]
    y = np.arange(len(d))
    y = np.vstack((y, y+1)).T.reshape(2*y.size).astype('float')
    x = np.vstack((d, d)).T.reshape(2*d.size).astype('float')
    return x, y


def distfit(d):
    if np.size(d) == 0:
        return np.nan, np.nan
    x, y = CDF(d)
    g = lambda p: np.nansum(((scipy.special.erf((x-p[0])/np.sqrt(2)/p[1])+1)/2-y/np.nanmax(y))**2)
    return scipy.optimize.minimize(g, (np.nanmean(d), np.nanstd(d)), options={'disp': False, 'maxiter': 1e5}).x


def line(x, p):
    """ 2 linear lines, p = (a, b, xa, xb) """
    a, b, xa, xb = np.array(p, dtype=float)
    return b * (1 - xa / xb) + (xa - x) * (b / xb + a / xa * np.heaviside(xa - x, 0.5))


def fit_line(x, y, dy):
    """ code to calculate fit of 2 linear lines, p = (a, b, xa, xb) """

    def fit_sub(x, y, xa):
        """ lsq fit for a given xa """
        t = xa - x
        H = np.heaviside(t, 0.5)
        n = len(t)
        St = np.sum(t)
        Sy = np.sum(y)
        Stt = np.sum(t ** 2)
        Sty = np.sum(t * y)
        SHt = np.sum(H * t)
        SHty = np.sum(H * t * y)
        SHtt = np.sum(H * t ** 2)
        A = np.array(((n, St, SHt), (St, Stt, SHtt), (SHt, SHtt, SHtt)))
        z = np.array((Sy, Sty, SHty))
        d, q, p = np.matmul(np.linalg.pinv(A), z)
        R2 = np.sum((y - d - q * t - p * t * H) ** 2)
        a = xa * p
        b = d + xa * q
        xb = b / q
        return R2, (a, b, xa, xb)

    if isinstance(dy, Number):
        dy = np.array([dy]*len(y))
    idx = np.isfinite(x) * np.isfinite(y) * np.isfinite(dy)
    if not np.sum(idx):
        return [np.nan] * 4
    x = x[idx]
    y = y[idx]
    dy = dy[idx]
    z = np.linspace(np.nanmin(x), np.nanmax(x), 25)
    R2 = [fit_sub(x, y, xa)[0] for xa in z]
    za = z[np.nanargmin(R2)]
    xa = optimize.minimize(lambda xa: fit_sub(x, y, xa)[0], za).x[0]
    p = fit_sub(x, y, xa)[1]
    dp = fminerr(lambda q: line(x, q), p, y, dy)[1]
    return p, dp


def fminerr(fun, a, y, dy=None, diffstep=1e-6):
    """ Error estimation of a fit

        Inputs:
        fun: function which was fitted to data
        a:   function parameters
        y:   ydata
        dy:  errors on ydata

        Outputs:
        chisq: Chi^2
        da:    error estimates of the function parameters
        R2:    R^2

        Example:
        x = np.array((-3,-1,2,4,5))
        a = np.array((2,-3))
        y = (15,0,5,30,50)
        fun = lambda a: a[0]*x**2+a[1]
        chisq,dp,R2 = fminerr(fun,p,y)

        adjusted from Matlab version by Thomas Schmidt, Leiden University
        wp@tl2020
    """
    eps = np.spacing(1)
    a = np.array(a).flatten()
    y = np.array(y).flatten()
    if dy is None:
        dy = np.ones(np.shape(y))
    else:
        dy = np.array(dy).flatten()
    nData = np.size(y)
    nPar = np.size(a)
    dy = 1 / (dy + eps)
    f0 = np.array(fun(a)).flatten()
    chisq = np.sum(((f0 - y) * dy) ** 2) / (nData - nPar)

    # calculate R^2
    sstot = np.sum((y - np.nanmean(y)) ** 2)
    ssres = np.sum((y - f0) ** 2)
    R2 = 1 - ssres / sstot

    # calculate derivatives
    deriv = np.zeros((nData, nPar))
    for i in range(nPar):
        ah = a.copy()
        ah[i] = a[i] * (1 + diffstep) + eps
        f = np.array(fun(ah)).flatten()
        deriv[:, i] = (f - f0) / (ah[i] - a[i]) * dy

    hesse = np.matmul(deriv.T, deriv)

    if np.linalg.matrix_rank(hesse) == np.shape(hesse)[0]:
        da = np.sqrt(chisq * np.diag(np.linalg.inv(hesse)))
    else:
        try:
            da = np.sqrt(chisq * np.diag(np.linalg.pinv(hesse)))
        except:
            da = np.zeros(a.shape)
        # da = np.full(np.shape(a),np.nan)
        # print('Hessian not invertible, size: {}, rank: {}'.format(np.shape(hesse)[0],np.linalg.matrix_rank(hesse)))
    return chisq, da, R2
