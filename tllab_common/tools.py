import numpy as np
import scipy
from numba import jit


def errwrap(fun, default, *args):
    """ Run a function fun, and when an error is caught return the default value
        wp@tl20190321
    """
    try:
        return fun(*args)
    except:
        return default


def fixpar(N, fix):
    """ Returns a function which will add fixed parameters in fix into an array
        N: total length of array which will be input in the function
        fix: dictionary, {2: 5.6}: fix parameter[2] = 5.6

        see its use in functions.fitgauss

        wp@tl20190816
    """
    # indices with variable parameters
    idx = sorted(list(set(range(N)) - set(fix)))

    # put the fixed paramters in place
    f = np.zeros(N)
    for i, v in fix.items():
        f[i] = v

    # make array used to construct variable part
    P = np.zeros((N, len(idx)))
    for i, j in enumerate(idx):
        P[j, i] = 1

    return lambda par: np.dot(P, par) + f


def unfixpar(p, fix):
    """ reverse of fixpar, but just returns the array immediately instead of returning
        a function which will do it

        wp@tl20190816
    """
    p = list(p)
    [p.pop(i) for i in sorted(list(fix), reverse=True)]
    return np.array(p)


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


def crop(im, x, y=None, z=None, m=np.nan):
    """ crops image im, limits defined by min(x)..max(y), when these limits are
        outside im the resulting pixels will be filled with mean(im)
        wp@tl20181129
    """
    if isinstance(x, np.ndarray) and x.shape == (3,2):
        z = x[2,:].copy().astype('int')
        y = x[1,:].copy().astype('int')
        x = x[0,:].copy().astype('int')
    elif isinstance(x, np.ndarray) and x.shape == (2,2):
        y = x[1,:].copy().astype('int')
        x = x[0,:].copy().astype('int')
    else:
        x = np.array(x).astype('int')
        y = np.array(y).astype('int')
    if not z is None: #3D
        z = np.array(z).astype('int')
        S = np.array(np.shape(im))
        R = np.array([[min(y),max(y)],[min(x),max(x)],[min(z),max(z)]]).astype('int')
        r = R.copy()
        r[R[:,0]<0,0] = 1
        r[R[:,1]>S,1] = S[R[:,1]>S]
        jm = im[r[0,0]:r[0,1], r[1,0]:r[1,1], r[2,0]:r[2,1]]
        jm =   np.concatenate((np.full((r[0,0]-R[0,0],jm.shape[1],jm.shape[2]),m),jm,np.full((R[0,1]-r[0,1],jm.shape[1],jm.shape[2]),m)),0)
        jm =   np.concatenate((np.full((jm.shape[0],r[1,0]-R[1,0],jm.shape[2]),m),jm,np.full((jm.shape[0],R[1,1]-r[1,1],jm.shape[2]),m)),1)
        return np.concatenate((np.full((jm.shape[0],jm.shape[1],r[2,0]-R[2,0]),m),jm,np.full((jm.shape[0],jm.shape[1],R[2,1]-r[2,1]),m)),2)
    else: #2D
        S = np.array(np.shape(im))
        R = np.array([[min(y),max(y)],[min(x),max(x)]]).astype(int)
        r = R.copy()
        r[R[:,0]<1,0] = 1
        r[R[:,1]>S,1] = S[R[:,1]>S]
        jm = im[r[0,0]:r[0,1], r[1,0]:r[1,1]]
        jm =   np.concatenate((np.full((r[0,0]-R[0,0],np.shape(jm)[1]),m),jm,np.full((R[0,1]-r[0,1],np.shape(jm)[1]),m)),0)
        return np.concatenate((np.full((np.shape(jm)[0],r[1,0]-R[1,0]),m),jm,np.full((np.shape(jm)[0],R[1,1]-r[1,1]),m)),1)


def fit_tilted_plane(im):
    """ Linear regression to determine z0, a, b in z = z0 + a*x + b*y

        nans and infs are filtered out

        input: 2d array containing z (x and y will be the pixel numbers)
        output: array [z0, a, b]

        wp@tl20190819
    """
    S = im.shape
    im = im.flatten()

    # vector [1, x_ij, y_ij] and filter nans and infs
    xv, yv = np.meshgrid(*map(range, S))
    v = [i.flatten()[np.isfinite(im)] for i in [np.ones(im.shape), xv, yv]]
    # construct matrix for the regression
    Q = np.array([[np.sum(i * j) for i in v] for j in v])
    if np.linalg.matrix_rank(Q) == Q.shape[1]:
        return np.dot(np.linalg.inv(Q), [np.sum(im[np.isfinite(im)] * i) for i in v])
    else:
        return np.array((np.nanmean(im), 0, 0))


def fitgauss(im, xy=None, ell=False, tilt=False, fwhm=None, fix=None):
    """ Fit gaussian function to image
        im:    2D array with image
        xy:    Initial guess for x, y, optional, default: pos of max in im
        ell:   Fit with ellipicity if True
        fwhm:  fwhm of the peak, used for boundary conditions
        fix:   dictionary describing which parameter to fix, to fix theta: fix={6: theta}
        q:  [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
        dq: errors (std) on q

        wp@tl2019
    """

    if not fwhm is None:
        fwhm = np.round(fwhm, 2)

    # handle input options
    if xy is None:
        # filter to throw away any background and approximate position of peak
        fm = (im - scipy.ndimage.gaussian_filter(im, 0.2))[1:-1, 1:-1]
        xy = [i + 1 for i in np.unravel_index(np.nanargmax(fm.T), np.shape(fm))]
    else:
        xy = [int(np.round(i)) for i in xy]
    if fix is None:
        fix = {}
    if ell is False:
        if 5 not in fix:
            fix[5] = 1
        if 6 not in fix:
            fix[6] = 0
    if tilt is False:
        if 7 not in fix:
            fix[7] = 0
        if 8 not in fix:
            fix[8] = 0

    xy = np.array(xy)
    for i in range(2):
        if i in fix:
            xy[i] = int(np.round(fix[i]))

    # size initial crop around peak
    if fwhm is None or not np.isfinite(fwhm):
        r = 10
    else:
        r = 2.5 * fwhm

    # find tilt parameters from area around initial crop
    if tilt:
        cc = np.round(((xy[0] - 2 * r, xy[0] + 2 * r + 1), (xy[1] - 2 * r, xy[1] + 2 * r + 1))).astype('int')
        km = crop(im, cc)
        K = [i / 2 for i in km.shape]
        km[int(np.ceil(K[0] - r)):int(np.floor(K[0] + r + 1)),
        int(np.ceil(K[1] - r)):int(np.floor(K[1] + r + 1))] = np.nan
        t = fit_tilted_plane(km)
    else:
        t = [0, 0, 0]

    # find other initial parameters from initial crop with tilt subtracted
    cc = np.round(((xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))).astype('int')
    jm = crop(im, cc)
    xv, yv = meshgrid(*map(np.arange, jm.shape[::-1]))

    if 6 in fix:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv, theta=fix[6])
    else:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv)
    p[0:2] += cc[:, 0] + 1

    for i in range(2):
        if i in fix:
            p[i] = xy[i]

    if fwhm is None:
        fwhm = p[2]
    else:
        p[2] = fwhm

    # just give up in some cases
    if not 1 < p[2] < 2 * fwhm or p[3] < 0.1:
        q = np.full(9, np.nan)
        dq = np.full(9, np.nan)
        return q, dq, (np.nan, np.nan, np.nan)

    s = fwhm / np.sqrt(2)  # new crop size

    cc = np.round(((p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))).astype('int')
    jm = crop(im, cc)
    S = np.shape(jm)

    bnds = [(0, S[0] - 1), (0, S[1] - 1), (fwhm / 2, fwhm * 2), (1e2, None), (0, None), (0.5, 2), (None, None),
            (None, None), (None, None)]
    xv, yv = meshgrid(*map(np.arange, S[::-1]))

    # move fixed x and/or y with the crop
    for i in range(2):
        if i in fix:
            fix[i] -= cc[i, 0]
            xy[i] = p[i]

    # find tilt from area around new crop
    cd = np.round(((p[0] - 2 * s, p[0] + 2 * s + 1), (p[1] - 2 * s, p[1] + 2 * s + 1))).astype('int')
    km = crop(im, cd)
    K = [i / 2 for i in km.shape]
    km[int(np.ceil(K[0] - s)):int(np.floor(K[0] + s + 1)), int(np.ceil(K[1] - s)):int(np.floor(K[1] + s + 1))] = np.nan
    t = fit_tilted_plane(km)

    # update parameters to new crop
    p[0:2] -= cc[:, 0]
    p = np.append(p, (t[1], t[2]))
    # p = np.append(p, (1, 0, t[1], t[2]))
    p[4] = t[0] + t[1] * (p[0] + s) + t[2] * (p[1] + s)

    # remove fixed parameters and bounds from lists of initial parameters and bounds
    p = unfixpar(p, fix)
    [bnds.pop(i) for i in sorted(list(fix), reverse=True)]

    # define function to remove fixed parameters from list, then define function to be minimized
    fp = fixpar(9, fix)
    g = lambda a: np.nansum((jm - gaussian9grid(fp(a), xv, yv)) ** 2)

    # make sure the initial parameters are within bounds
    for i, b in zip(p, bnds):
        i = errwrap(np.clip, i, i, b[0], b[1])

    nPar = len(p)

    # fit and find error predictions
    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x

    # Check boundary conditions, maybe try to fit again
    refitted = False
    for idx, (i, b) in enumerate(zip(q, bnds)):
        try:
            if not b[0] < i < b[1] and not refitted:
                r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e7}, bounds=bnds)
                q = r.x
                refitted = True
        except:
            pass

    dq = fminerr(lambda p: gaussian9grid(fp(p), xv, yv), q, jm)[1]

    # reinsert fixed parameters
    q = fp(q)
    for i in sorted(fix):
        if i > len(dq):
            dq = np.append(dq, 0)
        else:
            dq = np.insert(dq, i, 0)

    # de-degenerate parameters and recalculate position from crop to frame
    q[2] = np.abs(q[2])
    q[0:2] += cc[:, 0]
    q[5] = np.abs(q[5])
    # q[6] %= np.pi
    q[6] = (q[6] + np.pi / 2) % np.pi - np.pi / 2

    # Chi-squared, R-squared, signal to noise ratio
    chisq = r.fun / (S[0] * S[1] - nPar)
    R2 = 1 - r.fun / np.nansum((jm - np.nanmean(jm)) ** 2)
    sn = q[3] / np.sqrt(r.fun / (S[0] * S[1])) / 2 / np.pi / q[2] ** 2

    return q, dq, (chisq, R2, sn)


def fitgaussint(im, xy=None, theta=None, mesh=None):
    """ finds initial parameters for a 2d Gaussian fit
        q = (x, y, fwhm, area, offset, ellipticity, angle) if 2D
        q = (x, y, z, fwhm, fwhmz, area, offset) if 3D
        wp@tl20191010
    """

    dim = np.ndim(im)
    S = np.shape(im)
    q = np.full(7, 0).astype('float')

    if dim == 2:
        if mesh is None:
            x, y = np.meshgrid(range(S[1]), range(S[0]))
        else:
            x, y = mesh

        if theta is None:
            tries = 10
            e = []
            t = np.delete(np.linspace(0, np.pi, tries + 1), tries)
            for th in t:
                e.append(fitgaussint(im, xy, th, (x, y))[5])
            q[6] = (fitcosint(2 * t, e)[2] / 2 + np.pi / 2) % np.pi - np.pi / 2
        else:
            q[6] = theta

        # q[4] = np.nanmin(im)
        jm = im.flatten()
        jm = jm[np.isfinite(jm)]
        q[4] = np.mean(np.percentile(jm, 0.25))
        q[3] = np.nansum((im - q[4]))

        if xy is None:
            q[0] = np.nansum(x * (im - q[4])) / q[3]
            q[1] = np.nansum(y * (im - q[4])) / q[3]
        else:
            q[:2] = xy

        cos, sin = np.cos(q[6]), np.sin(q[6])
        x, y = cos * (x - q[0]) - (y - q[1]) * sin, cos * (y - q[1]) + (x - q[0]) * sin

        s2 = np.nansum((im - q[4]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[4])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[4])) ** 2) / s2)

        q[2] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[5] = np.sqrt(sx / sy)
    else:
        if mesh is None:
            x, y, z = np.meshgrid(range(S[0]), range(S[1]), range(S[2]))
        else:
            x, y, z = mesh
        q[6] = np.nanmin(im)
        q[5] = np.nansum((im - q[6]))

        if xy is None:
            q[0] = np.nansum(x * (im - q[6])) / q[5]
            q[1] = np.nansum(y * (im - q[6])) / q[5]
            q[2] = np.nansum(z * (im - q[6])) / q[5]
        else:
            q[:3] = xy

        x, y, z = x - q[0], y - q[1], z - q[2]

        s2 = np.nansum((im - q[6]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[6])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[6])) ** 2) / s2)
        sz = np.sqrt(np.nansum((z * (im - q[6])) ** 2) / s2)

        q[3] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[4] = sz * 4 * np.sqrt(np.log(2))
    return q


def fitcosint(theta, y):
    """ Finds parameters to y=a*cos(theta-psi)+b
        wp@tl20191010
    """
    b = np.trapz(y, theta) / np.mean(theta) / 2
    a = np.trapz(np.abs(y - b), theta) / 4

    t = np.sin(theta)
    s = np.cos(theta)

    T = np.sum(t)
    S = np.sum(s)
    A = np.sum(y * t)
    B = np.sum(y * s)
    C = np.sum(t ** 2)
    D = np.sum(t * s)
    E = np.sum(s ** 2)

    q = np.dot(np.linalg.inv(((C, D), (D, E))) / a, (A - b * T, B - b * S))

    psi = (np.arctan2(*q)) % (2 * np.pi)
    if q[1] < 0:
        a *= -1
        psi -= np.pi
    psi = (psi + np.pi) % (2 * np.pi) - np.pi

    return np.array((a, b, psi))


@jit(nopython=True, nogil=True)
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)


@jit(nopython=True, nogil=True)
def erf2(x):
    s = x.shape
    y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            y[i,j] = erf(x[i,j])
    return y


@jit(nopython=True, nogil=True)
def meshgrid(x, y):
    s = (len(y), len(x))
    xv = np.zeros(s)
    yv = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            xv[i,j] = x[j]
            yv[i,j] = y[i]
    return xv, yv


@jit(nopython=True, nogil=True)
def gaussian9grid(p,xv,yv):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
        xv, yv = meshgrid(np.arange(Y),np.arange(X))
            calculation of meshgrid is done outside, so it doesn't
            have to be done each time this function is run
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac/p[5]
    dy = efac*p[5]
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2*dx*(cos*(xv-p[0])-(yv-p[1])*sin)
    y = 2*dy*(cos*(yv-p[1])+(xv-p[0])*sin)
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]+p[7]*xv+p[8]*yv-p[7]*p[0]-p[8]*p[1]