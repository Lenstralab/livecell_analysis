#!/usr/bin/python3.8

import numpy as np
from argparse import ArgumentParser

def bootstrap_hypothesis_testing(y, z, B=1000, mode='mean'):
    """
    Two-sample hypothesis testing through bootstrapping

    CALL ASL, t = bootstrap_hypothesis_testing(y, z, B, mode)

    INPUT
        y: input values from sample 1
        z: input values from sample 2
        B: number of bootstrapping iterations (def:1e3. It can be tuned to keep the coefficient of variation sigma/mu < 0.1)
        mode: 'F=G' or 'mean' for two different algorithms (def:'mean')

    OUTPUT
        ASL: achieved significance level (aka alpha/p-value)
        t: bootstrapping t parameters

    Reference: An introduction to the bootstrap - Efron & Tibshirani
    Chapter 16 for the algorithms

    SC @LenstraLab 13/10/2020 > Matlab
    WP                        > Python
    """

    np.seterr(divide='ignore')

    n = len(z)
    m = len(y)

    # pool together y and z into x
    x = np.hstack((z.flatten(), y.flatten()))

    if mode=='F=G': # Algorithm 16.1 page 221
        t0 = np.mean(z) - np.mean(y)
        t = []
        for i in range(B): # it could be done in parallel
            k = int(np.round(n**(2./3)))
            l = int(np.round(n**(2./3)))
            x_ = x[np.random.randint(0, n+m, k+l)]
            t.append(np.mean(x_[:k])-np.mean(x_[l:]))

    elif mode=='mean': # Algorithm 16.2 page 224
        t0 = (np.mean(z)-np.mean(y))/np.sqrt(np.var(z)/n+np.var(y)/m)
        z0 = z-np.mean(z)+np.mean(x)
        y0 = y-np.mean(y)+np.mean(x)
        t = []
        for i in range(B): # it could be done in parallel/without loop
            k = int(np.round(n**(2./3)))
            l = int(np.round(n**(2./3)))
            z_ = z0[np.random.randint(0, n, k)]
            y_ = y0[np.random.randint(0, m, l)]
            t.append((np.mean(z_)-np.mean(y_))/np.sqrt(np.var(z_)/k+np.var(y_)/l))
    else:
        raise(ValueError('Unrecognized mode, options: mean or F=G'))

    ASL = 2*min(np.sum(t<=t0), np.sum(t>t0)).astype(float)/B

    return ASL, t

def main():
    parser = ArgumentParser(description=bootstrap_hypothesis_testing.__doc__)
    parser.add_argument('y_file', help='text file with distribution(s) in column(s)')
    parser.add_argument('z_file', help='text file with distribution(s) in column(s)')
    parser.add_argument('-B', '--bootstraps', help='number of bootstraps', type=int, default=1000)
    parser.add_argument('-m', '--mode', help='mode: mean or F=G', default='mean')
    args = parser.parse_args()

    y = np.loadtxt(args.y_file, ndmin=2, dtype=float)
    z = np.loadtxt(args.z_file, ndmin=2, dtype=float)

    ASL, t = zip(*[bootstrap_hypothesis_testing(y[:,i], z[:,i], args.bootstraps, args.mode) for i in range(y.shape[1])])

    print('ASL: {}'.format(ASL))

if __name__=='__main__':
    main()
