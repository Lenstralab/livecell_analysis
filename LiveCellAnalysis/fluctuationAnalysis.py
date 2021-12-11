from __future__ import print_function
from scipy import fft
import numpy as np
import warnings
from parfor import chunks, parfor
from copy import deepcopy
from tllab_common.misc import objFromDict
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "Degrees of freedom")

def mcR_FFT(v, t=None, meth='pad0', mT=0):
    """ Computes, using fast Fourier Transform, the raw correlation
        functions (i.e. non-central unnormalized moments) of a multi-channel
            signal, i.e. R(tau) = < v1(t) v2(t+tau) >_t.
        Equivalent to (but faster than) mcR(t,v,meth=('multiTau',0,'wrap')).

        Parameters:
        t: Time vector. Not necessarily starting from 0, but with dt
            somewhat uniform.
        v: Multi-channel signal where each row is a channel.
        meth: pad0: pad signal with zeros to get linear (not circular) correlations
              trim: trim overhangs of signal
              wrap: wrap signal around to get linear correlations
        mT:   multitau algorithm, start averaging after 2^mT points

        Returns:
        tau: Vector of time lags (same unit as t)
        R: Raw cross-covariance function.
    """
    assert not (meth.lower() == 'wrap' and mT), 'combination wrap and multi-tau not implemented'
    M = v.shape[-1]
    if meth.lower() == 'wrap':
        L = M
    else:
        L = int(2 ** np.ceil(1 + np.log2(M)))
        v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, L - M)], 'constant')

    fv = fft.fft(v)
    dt = 1 if t is None else np.nanmean(np.diff(t))
    tau, R = [], []
    p = 2 ** mT
    q = p // 2
    s = 1 if mT == 0 else np.clip(int(np.log2(L) - mT), 0, 15)

    for i in range(s):
        j = 2 ** i
        r = fft.ifft(np.tile(fv, fv.shape[0]).reshape((fv.shape[0],) + fv.shape).conj() * fv).real / fv.shape[-1]
        if s == 1:  # no multiTau
            u = np.arange(L)
        elif i == 0:  # first part of corfun
            u = np.arange(p)
            r = r[..., :p]
        else:  # other parts of corfun
            u = np.arange(q, p) * j
            r = r[..., q:p]
        if i < s - 1:  # remove half of the higher frequencies
            fv = np.concatenate((fv[..., :fv.shape[-1] // 4], fv[..., -fv.shape[-1] // 4:]), -1)
        tau.append(u.astype(float))
        R.append(r / j ** 2)

    tau = np.concatenate(tau, -1)
    i = np.argsort(tau)
    i = i[tau[i] < M]
    tau = tau[i]
    R = np.concatenate(R, -1)
    R = R[..., i] * L

    if meth.lower() == 'trim':
        R /= (M - tau)
    else:
        R /= M
    if t is None:
        return R
    return R, tau*dt


def linefit(x, y, n):
    x = x[:n]
    y = y[:n]
    S = [np.sum(i) for i in (x, y, x**2, x*y)]
    D = n*S[2] - S[0]**2
    a = (n*S[3] - S[0]*S[1])/D
    b = (S[1]*S[2] - S[0]*S[3])/D
    R2 = 1 - np.sum((y-a*x-b)**2)/np.sum((y-np.mean(y))**2)*(n-1)/(n-3)
    return a, b, R2


def fit_duo_line(xl, xr, yl, yr, wl=None, wr=None):
    """ Fit yl = p0*xl + p2 and yr = p1*xr + p2, return p
    """
    xl, xr, yl, yr = [np.asarray(i) for i in (xl, xr, yl, yr)]
    wl = np.ones(xl.shape, float) if wl is None else np.asarray(wl, float)
    wr = np.ones(xr.shape, float) if wr is None else np.asarray(wr, float)
    wl[xl == 0] /= 2
    wr[xr == 0] /= 2
    wx2l, wx2r = np.sum(wl * xl ** 2), np.sum(wr * xr ** 2)
    wxl, wxr = np.sum(wl * xl), np.sum(wr * xr)
    wxyl, wxyr = np.sum(wl * xl * yl), np.sum(wr * xr * yr)
    ww, wywy = np.sum(wl) + np.sum(wr), np.sum(wl * yl) + np.sum(wr * yr)
    A = ((wx2r * ww - wxr ** 2, wxl * wxr, -wxl * wx2r),
         (wxl * wxr, wx2l * ww - wxl ** 2, -wxr * wx2l),
         (-wxl * wx2r, -wxr * wx2l, wx2l * wx2r))
    return np.dot(A, (wxyl, wxyr, wywy)) / (wx2l * wx2r * ww - wxl ** 2 * wx2r - wxr ** 2 * wx2l)


def get_acf0(tau, g):
    """ Fit a line through n points g[1:n] where n maximizes R2_adj """
    a, b, R2 = zip(*[linefit(tau[1:], g[1:], i) for i in range(4, 10)])
    if np.any(np.isfinite(R2) * (np.array(R2) > 0)):
        n = np.nanargmax(R2)
        a, b, R2 = a[n], b[n], R2[n]
        return b
    else:
        return g[1]


def get_mask(t, frameWindow):
    frameWindow = np.clip(frameWindow, 0, len(t))
    frameWindow.sort()
    mask = [np.zeros(frameWindow[0])]
    for i, d in enumerate(np.diff(frameWindow)):
        if i % 2:
            mask.append(np.zeros(d))
        else:
            mask.append(np.ones(d))
    if len(frameWindow) % 2:
        mask.append(np.ones(len(t) - frameWindow[-1]))
    else:
        mask.append(np.zeros(len(t) - frameWindow[-1]))
    return np.hstack(mask).astype(float)


class mcSig(objFromDict):
    """ Multi-channel signal object. """
    def __init__(self, t, v, mask=None, frameWindow=None, v_orig=None):
        """ Parameters
            t: Either a time vector (1-d array) or a time interval dt (float).
            v: Signal vector. Can be a 1-d array if there is only 1 channel. If
                multiple channels, each row is a channel.
            mask: array with zeros and ones, describing where the signal is
                known (1) or not (0). Either a 1-d array if all channels have
                the same mask, or a 2-d array with the same dimensions as v if
                channel have different masks.
            frameWindow: how to create mask, argument mask is ignored if frameWindow is not None
                array: [start, stop, start, stop, ...] of areas where mask = 1
        """
        super(mcSig, self).__init__()
        self.v=np.c_[v].T if v.ndim==1 else v
        if not v_orig is None:
            self.v_orig=np.c_[v_orig].T if v_orig.ndim==1 else v_orig
        self.t=t if type(t)==np.ndarray else np.r_[:self.v.shape[1]]*1.*t
        self.dt=np.mean(np.diff(self.t))
        self.tTot=self.t.shape[0]*self.dt

        if not frameWindow is None:
            self.mask = np.vstack([get_mask(self.t, frameWindow)]*self.v.shape[0])
        elif mask is None:
            self.mask = np.ones(self.v.shape, dtype=float)
        else:
            self.mask = np.vstack([mask]*self.v.shape[0]) if mask.ndim == 1 else mask
        self.t=self.t.astype(float); self.v=self.v.astype(float); self.mask=self.mask.astype(float)
        if not (self.t.shape[0]==self.v.shape[1] and self.v.shape==self.mask.shape):
            raise ValueError('Dimensions of arguments should match. Try "help(mcSig)" for more info.')


class mcSigSet(objFromDict):
    """ Set of multi-channel signals, and some methods for correlation.

        Example:
            ss = mcSigSet([mcSig])
            ss.alignSignals()
            ss.compAvgCF()
            ss.bootstrap()

        After this, ss will have at least the following attributes:
            sigs: The list of signals provided as a parameter.
            sigsAlign: The list of signals, aligned and time-extended. If all
                signals in sigs have the same time vector, then sigs==sigsAlign.
            t: Time vector of the aligned signals and of the average signal.
            v: Average multi-channel signal. Axes of v are: channel, time and
                mean/SEM/coverage. The coverage represents the number of signals
                involved in the computation of a given data point.
            tau: Vector of time lags (same unit as t).
            G: Average correlation functions
            N: Non-stationary component of the correlation functions, i.e.
                correlation functions of the average signal.
            P: Corrected "pseudo-stationary" correlation functions, i.e.
                difference between G and N.
            Gr, Gg, Nr, Ng, Pr, Pg: Same as above, but normalized to the red and green acf's.
    """

    def __init__(self, sigs, sigsName='sigs'):
        """ By default sigs (list of mcSig) are saved in the attribute .sigs. """
        super(mcSigSet, self).__init__(**{sigsName: sigs})

    def alignSignals(self, t0=None, verb=False):
        """ Make sure the time vectors of each signal are the same, optionally start signals at t0
            t0: array or list an individual frame number offset for each signal.
        """
        if not t0 is None:
            for s, t in zip(self.sigs, t0):
                s.t -= t
                s.mask[:, s.t<0] = 0

        ## If all time vectors are identical
        if np.all([len(s.t) == len(self.sigs[0].t) and np.all(s.t == self.sigs[0].t) for s in self.sigs]):
            sigsAlign = self.sigs
            t = sigsAlign[0].t
        ## If time vectors differ between sigs
        else:
            # Compare signals
            dt = np.mean([s.dt for s in self.sigs])
            if np.max([abs(1 - s.dt / dt) for s in self.sigs]) > .02:
                print("Warning: Time intervals are not homogenous between signals.")
            t = np.arange(np.min([s.t[0] for s in self.sigs]), np.max([s.t[-1] for s in self.sigs])+dt, dt)
            if verb:
                print("** Aligning signals...")
            sigsAlign = []
            for s in self.sigs:
                v = np.zeros((s.v.shape[0], t.shape[0]))
                v[:, np.argmin(abs(t - s.t[0])):][:, :s.v.shape[1]] = s.v
                if 'v_orig' in s:
                    v_orig = np.zeros((s.v_orig.shape[0], t.shape[0]))
                    v_orig[:, np.argmin(abs(t - s.t[0])):][:, :s.v_orig.shape[1]] = s.v_orig
                else:
                    v_orig = None
                mask = np.zeros((s.mask.shape[0], t.shape[0]))
                mask[:, np.argmin(abs(t - s.t[0])):][:, :s.mask.shape[1]] = s.mask
                sigsAlign.append(mcSig(t, v, mask, v_orig=v_orig))
        self.sigsAlign = sigsAlign
        self.t = t

    def compAvgCF_SC(self, methEnd='trim', mT=0, fitWindow=None, bootstrap=False):
        """ The Stefono interpretation of averaging and ccf normalization:
            Subtract and divide by a global mean.
            G: raw, N: steady state correction, P: G - N - linefit
            Will omit linefitting when fitWindow is None
        """
        tlen = [s.mask.sum(1) for s in self.sigsAlign]  # number unmasked in signal
        mean_global = wmean([wmean(s.v, s.mask, 1) for s in self.sigsAlign], tlen, 0)  # scalar
        mean_signal = wmean([s.v for s in self.sigsAlign], [s.mask for s in self.sigsAlign], 0)  # trace

        if not bootstrap:
            self.fun = 'SC'
            self.methEnd, self.mT, self.fitWindow = methEnd, mT, fitWindow

        for s in self.sigsAlign:
            if not bootstrap:
                s.weight = np.round(s.mask.shape[-1] * mcR_FFT(s.mask, meth='wrap' if methEnd == 'wrap' else 'pad0',
                                                               mT=mT))
                s.W = mcR_FFT(s.mask, meth=methEnd, mT=mT)
            s.M, s.tau = mcR_FFT(s.mask * (s.v.T - mean_global).T, s.t, methEnd, mT)
            s.M[s.weight == 0] = np.nan
            s.M /= s.W
            s.G = divide(s.M, mean_global)
            s.N = divide(mcR_FFT(s.mask * np.nan_to_num((mean_signal.T - mean_global).T), meth=methEnd, mT=mT),
                         mean_global)
            s.N[s.weight == 0] = np.nan
            s.N /= s.W
            s.P = s.G - s.N

        self.M, self.G, self.P = [wmean([s[r] for s in self.sigsAlign], [s.weight for s in self.sigsAlign], 0)
                                  for r in 'MGP']
        self.tau = self.sigsAlign[0].tau
        self.weight = np.nanmean([s.weight for s in self.sigsAlign], 0)
        if fitWindow is not None:
            fw = slice(*fitWindow)
            self.p = fit_duo_line(self.tau[fw], self.tau[fw], self.P[1, 0, fw],
                                  self.P[0, 1, fw], self.weight[1, 0, fw], self.weight[0, 1, fw])
            self.q = np.array([np.polyfit(self.tau[fw], self.P[i, i, fw], 1, w=self.weight[i, i, fw])
                               for i in range(2)])

        for s in self.sigsAlign:
            if fitWindow is not None:
                s.N[1, 0] += self.p[0] * s.tau + self.p[2]
                s.N[0, 1] += self.p[1] * s.tau + self.p[2]
                s.N[0, 0] += np.polyval(self.q[0], s.tau)
                s.N[1, 1] += np.polyval(self.q[1], s.tau)
                s.P = s.G - s.N
            s.acf0 = [get_acf0(s.tau, s.P[i, i]) for i in range(2)]

        self.N = wmean([s.N for s in self.sigsAlign], [s.weight for s in self.sigsAlign], 0)
        self.P = self.G - self.N
        self.acf0 = [get_acf0(self.tau, self.P[i, i]) for i in range(2)]
        self.Gr, self.Gg = [wmean([(s.G + 1) / (s.acf0[i] + 1) for s in self.sigsAlign],
                                  [s.weight for s in self.sigsAlign], 0) for i in range(2)]
        self.Pr, self.Pg = [wmean([(s.G - s.N + 1) / (s.acf0[i] + 1) for s in self.sigsAlign],
                                  [s.weight for s in self.sigsAlign], 0) for i in range(2)]
        self.Nr, self.Ng = self.Gr - self.Pr, self.Gg - self.Pg
        self.v = mean_signal

    def compAvgCF(self, methEnd='trim', mT=0, bootstrap=False):
        """ The Wim interpretation of averaging and normalization:
            Scale each signal individually.
            Also includes the old way of calculating the unnormalized ccf's for comparison.
        """
        # TODO: implement linefit
        if not bootstrap:
            self.fun = ''
            self.methEnd, self.mT = methEnd, mT

        tlen = [s.mask.sum(1) for s in self.sigsAlign]  # number unmasked in signal
        mean_global = wmean([wmean(s.v, s.mask, 1) for s in self.sigsAlign], tlen, 0)
        mean_signal = wmean([s.v for s in self.sigsAlign], [s.mask for s in self.sigsAlign], 0)
        mean_signal_scaled = wmean([(s.v.T/wmean(s.v, s.mask, 1)).T for s in self.sigsAlign],
                                   [s.mask for s in self.sigsAlign], 0)

        for s in self.sigsAlign:
            if not bootstrap:
                s.weight = np.round(s.mask.shape[-1] * mcR_FFT(s.mask, meth='wrap' if methEnd == 'wrap' else 'pad0',
                                                               mT=mT))
                s.M, s.tau = mcR_FFT(s.mask * s.v, s.t, methEnd, mT)
                s.W = mcR_FFT(s.mask, None, methEnd)
                s.M[s.weight == 0] = np.nan
                s.M /= s.W
                s.Gs = divide(s.M, wmean(s.v, s.mask, 1))
            s.Ns = mcR_FFT(np.nan_to_num(s.mask * mean_signal_scaled), meth=methEnd, mT=mT)
            s.Ns[s.weight == 0] = np.nan
            s.Ns /= s.W
            s.Ps = s.Gs - s.Ns

            s.G = divide(s.M, mean_global)
            s.N = divide(mcR_FFT(np.nan_to_num(s.mask * mean_signal), meth=methEnd, mT=mT), mean_global)
            s.N[s.weight == 0] = np.nan
            s.N /= s.W
            s.P = s.G - s.N
            if not bootstrap:
                s.acf0 = [get_acf0(s.tau, s.Ps[i, i]) for i in range(2)]
                s.Gr, s.Gg = [(s.Gs + 1) / (s.acf0[i] + 1) for i in range(2)]
                s.Pr, s.Pg = [(s.Ps + 1) / (s.acf0[i] + 1) for i in range(2)]
                s.Nr, s.Ng = s.Gr - s.Pr, s.Gg - s.Pg

        self.M, self.G, self.N, self.P = [wmean([s[r] for s in self.sigsAlign],
                                                [s.weight for s in self.sigsAlign], 0) for r in 'MGNP']
        self.Gs, self.Ns, self.Ps = [wmean([s[r] for s in self.sigsAlign],
                                                [s.weight for s in self.sigsAlign], 0) for r in ('Gs', 'Ns', 'Ps')]
        self.Gr, self.Gg = [wmean([(s.Gs + 1) / (s.acf0[i] + 1) for s in self.sigsAlign],
                                  [s.weight for s in self.sigsAlign], 0) for i in range(2)]
        self.Pr, self.Pg = [wmean([(s.Ps + 1) / (s.acf0[i] + 1) for s in self.sigsAlign],
                                  [s.weight for s in self.sigsAlign], 0) for i in range(2)]
        self.Nr, self.Ng = self.Gr - self.Pr, self.Gg - self.Pg
        self.tau = self.sigsAlign[0].tau
        self.v = mean_signal
        self.vs = mean_signal_scaled

    def bootstrap(self, nBs=None):
        """ Bootstrap by running compAvgCF nBs times in parallel. """
        vars = [var for var in ('M', 'G', 'P', 'N', 'Gr', 'Gg', 'Pr', 'Pg', 'Nr', 'Ng', 'v', 'Gs', 'Ps', 'Ns', 'p', 'q')
                if var in self]
        if nBs == 0:
            """ Quickly add dX entries so that we can plot things when debugging. """
            for i in vars:
                if i in self:
                    self['d{}'.format(i)] = 0.05 * self[i]
            return
        nBs = nBs or 10000
        N = len(self.sigsAlign)
        @parfor(chunks(np.random.randint(0, N, (nBs, N)), s=250, r=2), (deepcopy(self.sigsAlign), self.get('methEnd'),
                self.get('mT'), self.get('fitWindow'), self.fun, vars), desc='Bootstrapping')
        def par(II, sa, methEnd, mT, fitWindow, fun, vars):
            R = {var: [0, 0, 0] for var in vars}
            for ii in II:
                s = mcSigSet([deepcopy(sa[i]) for i in ii], 'sigsAlign')
                if fun == 'SC':
                    s.compAvgCF_SC(methEnd, mT, fitWindow, True)
                else:
                    s.compAvgCF(methEnd, mT, True)
                for i in vars:  # storing cumulative sums of var and var^2, to calculate std without storing all values
                    R[i][0] += np.isfinite(s[i])
                    R[i][1] += np.nan_to_num(s[i])
                    R[i][2] += np.nan_to_num(s[i]**2)
            return R

        for var in vars:  # collecting all results and calculating std
            l = [sum([r[var][j] for r in par]) for j in range(3)]
            self[f'd{var}'] = np.sqrt(l[2] / l[0] - (l[1] / l[0]) ** 2)

def wmean(a, da, axis=None):
    """ Weighted mean of a with weights da along axis, omitting nans. """
    t = np.nansum(da, axis).astype(float)
    if not np.isscalar(t):
        t[t == 0] = np.nan
    elif t == 0:
        t = np.nan
    return np.nansum(np.array(a)*da, axis)/t


def divide(a, b):
    """ Divide a 3x2x2xN or a 2x2xN matrix a by the inproduct of the 2 element b. """
    return (a.T / np.dot(b.reshape(-1, 1), b.reshape(1, -1))).T
