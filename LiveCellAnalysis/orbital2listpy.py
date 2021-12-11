#!/usr/bin/python3.8

import os
import lfdfiles
import numpy as np
import io
from glob import iglob
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from tllab_common.misc import color
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['figure.max_open_warning'] = 250
A4 = (11.69, 8.27)

from scipy.optimize import minimize
from parfor import parfor
if __package__ is None or __package__ == '':  # usual case
    from listpyedit import listFile
else:
    from .listpyedit import listFile


class dict_lowercase(dict):
    def __init__(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k.lower()] = v

    def __getitem__(self, key):
        item = super().__getitem__(key.lower())
        if isinstance(item, dict):
            return dict_lowercase(item)
        else:
            return item

    def __contains__(self, key):
        return super().__contains__(key.lower())


def intensity_fun(sb, sp, tb, tp):
    S = 2 * np.array((sb**2 + sp**2, sb**2 + sp**2, tb**2 + tp**2))
    def fun(Xb, Xp, A):
        return A * np.exp(np.sum(-(Xb-Xp)**2 / S, 1))
    return fun


def cost_fun(I, Xb, sb, sp, tb, tp):
    f = intensity_fun(sb, sp, tb, tp)
    def fun(P):
        return np.sum((I - f(Xb, P[1:], P[0]))**2)
    return fun


def fit_orbit(jrn_file, pxsize, ex_lambda, em_lambda, Nchannels=2, NA=1.2):
    path = os.path.split(jrn_file)[0]
    sb = ex_lambda / 2 / NA
    sp = em_lambda / 2 / NA
    tb = 2 * ex_lambda / NA ** 2
    tp = 2 * em_lambda / NA ** 2
    Res = []
    with lfdfiles.LfdFile(jrn_file) as jrn:
        for f in jrn:
            Res.append([])
            fl = dict_lowercase(f)
            if 'Parameters for tracking' in fl:
                cexpt = fl['correlation expt'].split('\\')[-1]
                p = fl['Parameters for tracking']
                r = p['Radius'] * pxsize
                Nr = p['Rperiods']
                N = p['Points per orbit']
                Norbits = Nr * p['z-period']
                Npts = N * Norbits
                t = 2 * np.pi * np.arange(N) / N
                x, y = np.tile(r * np.cos(t), Nr), np.tile(r * np.sin(t), Nr)
                if p['Movement type'] == 3:
                    dz = p['Z-radius'] / 2
                    z = np.hstack((dz * np.ones(Npts // 2), -dz * np.ones(Npts // 2)))
                else:
                    z = np.zeros(Npts)
                X = np.vstack((x, y, z)).T

                bin_files = [os.path.join(path, '{}{}{}'.format(cexpt, (i + 1), str(fl['extension']).zfill(3)))
                             for i in range(Nchannels)]

                for i, bin_file in enumerate(bin_files):
                    if os.path.exists(bin_file + '.b64'):
                        with lfdfiles.SimfcsB64(bin_file + '.b64') as b:
                            bin_data = np.reshape(b.asarray(), (-1, Npts)).T
                    elif os.path.exists(bin_file + '.bin'):
                        with lfdfiles.SimfcsBin(bin_file + '.bin') as b:
                            bin_data = np.reshape(b.asarray(), (-1, Npts)).T
                    else:
                        raise Exception('Bin file does not exist for channel {}'.format(i))

                    @parfor(bin_data.T, (X, sb, sp, tb, tp), serial=10)
                    def Q(I, X, sb, sp, tb, tp):
                        cf = cost_fun(I, X, sb, sp, tb, tp)
                        r = minimize(cf, (1, 0, 0, 0))
                        return r.x
                    Res[-1].append(Q)
    return Res


def get_files(jrn_file, Nchannels=3):
    jrn_file = os.path.abspath(jrn_file)
    path = os.path.split(jrn_file)[0]
    files = []
    with lfdfiles.SimfcsJrn(jrn_file, lower=True) as jrn:
        for f in jrn:
            if 'parameters for tracking' in f:
                cexpt = f['correlation expt'].split('\\')[-1]
                txt_file = os.path.join(path, cexpt + '.txt')
                bin_files = [os.path.join(path, '{}{}{}'.format(cexpt, (i+1), str(f['extension']).zfill(3)))
                             for i in range(Nchannels)]
                Norbits = f['parameters for tracking']['rperiods'] * f['parameters for tracking']['z-period']
                Npts = f['parameters for tracking']['points per orbit'] * Norbits
                dt = f['parameters for tracking']['period'] * 1e-6 * Norbits
                files.append((jrn_file, txt_file, bin_files, Npts, dt))
    return files


def save_trks(file, trk_folder=None, includeZ=False, fw_start=50):
    jrn_file, txt_file, bin_files, Npts, dt = file
    trk_folder = trk_folder or os.path.split(jrn_file)[0].replace('data', 'analysis')
    if not os.path.exists(trk_folder):
        os.mkdir(trk_folder)

    # read txt file for xyz positions
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as tracking:
            header = tracking.readline().strip().split()
            data = tracking.read()
        # fix an issue where sometimes there isn't a space between the last 2 columns
        data = [line if line[238] == ' ' else line[:238] + ' ' + line[238:] for line in data.splitlines()]
        with io.StringIO() as s:
            s.write('\n'.join(data))
            s.seek(0)
            data = np.loadtxt(s).T

        # fix header
        if data.shape[0] - len(header) == 2:
            header.extend(('unknown', 't[s]'))
        elif data.shape[0] - len(header) == 3:
            header.extend(('unknown', 'frame', 't[s]'))
        else:
            raise Exception('unknown columns in txt file')
    else:
        raise Exception('txt file does not exist')

    # Save data from bin files to trk files and make make carpet plots
    trk_files = [os.path.join(trk_folder, os.path.split(bin_file)[1] + '.trk') for bin_file in bin_files]
    lf_dict = {'metadata': jrn_file}
    fig = plt.figure(figsize=A4)
    gs = GridSpec(len(bin_files), 1, figure=fig)
    for i, (bin_file, trk_file, c) in enumerate(zip(bin_files, trk_files, 'rgby')):
        if os.path.exists(bin_file + '.b64'):
            with lfdfiles.SimfcsB64(bin_file + '.b64') as b:
                bin_data = np.reshape(b.asarray(), (-1, Npts)).T
        elif os.path.exists(bin_file + '.bin'):
            with lfdfiles.SimfcsBin(bin_file + '.bin') as b:
                bin_data = np.reshape(b.asarray(), (-1, Npts)).T
        else:
            raise Exception('Bin file does not exist for channel {}'.format(i))

        fig.add_subplot(gs[i, 0])
        plt.imshow(bin_data)
        if i == len(bin_files)-1:
            plt.xlabel('frame')
        else:
            plt.gca().get_xaxis().set_visible(False)
        plt.ylabel('point in orbit')

        I = bin_data.sum(0)
        frame = np.arange(len(I))
        bin_frame = list(data[header.index('frame')].astype(int))
        hid = [header.index(i) for i in (('x_p', 'y_p', 'z_p') if includeZ else ('x_p', 'y_p'))]
        X = []
        for f in frame:
            try:
                fid = bin_frame.index(f + 1)
                X.append(data[hid, fid])
            except ValueError:
                X.append(np.full(len(hid), np.nan))
        X = np.vstack(X)
        trk = np.hstack((X, np.expand_dims(I, 1), np.expand_dims(frame * dt, 1)))
        np.savetxt(trk_file, trk, fmt='%12.5f')
        lf_dict['trk_' + c] = trk_file

    frameWindow = [max(frame[np.isfinite(trk[:, 0])].min(), fw_start), frame[np.isfinite(trk[:, 0])].max()]
    lf_dict['frameWindow'] = frameWindow
    with PdfPages(os.path.join(trk_folder, os.path.split(jrn_file)[1].replace('.jrn', '_carpets.pdf'))) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    return lf_dict


def make_list(jrn_files, listFileOut=None, trk_folder=None, Nchannels=3, includeZ=False, fw_start=50):
    if len(jrn_files) == 0:
        jrn_files = list(iglob('*.jrn'))
    if isinstance(jrn_files, str):
        jrn_files = [jrn_files]
    if listFileOut is None:
        listFileOut = os.path.abspath(datetime.now().strftime('%Y%m%d_%H%M%S') + '.list.py')
    if not listFileOut.endswith('.py'):
        if not listFileOut.endswith('.list'):
            listFileOut += '.list'
        listFileOut += '.py'

    lf = listFile()
    for jrn_file in tqdm(jrn_files, desc='Adding experiments'):
        if not jrn_file.endswith('.jrn'):
            jrn_file += '.jrn'
        fileses = get_files(jrn_file, Nchannels)
        for files in fileses:
            lf.append(save_trks(files, trk_folder, includeZ, fw_start))
    lf.save(listFileOut)
    print('Saved as {}'.format(listFileOut))


def main():
    doc = color('''Convert Orbital Tracker files into trk files to be used with the correlation pipeline. Multiple jrn
        files can be given, wildcards * can be used (glob). For each jrn file a txt file (for x, y, z positions) and bin or b64 files (1 per
        channel) need to be present in the same folder.''', 'g:b')

    parser = ArgumentParser(description=doc)
    parser.add_argument('jrn_files', help='jrn files to be included (default: *.jrn)', nargs='*')
    parser.add_argument('-o', '--output', help='filename for .list.py (default: date_time.list.py)',
                        type=str, default=None)
    parser.add_argument('-t', '--trk_folder',
        help='folder where trk (track) files are to be stored (default: folder of jrn with data replaced by analysis)',
                        type=str, default=None)
    parser.add_argument('-n', '--n_channels', help='number of channels in data (default: 3)', type=int, default=3)
    parser.add_argument('-f', '--f_window_start', help='start of frame windows to save in .list.py file (default: 50)',
                        type=int, default=50)
    parser.add_argument('-z', '--include_z', help='include a column for z in trk files',
                        action='store_true', default=False)
    args = parser.parse_args()
    make_list(args.jrn_files, args.output, args.trk_folder, args.n_channels, args.include_z, args.f_window_start)

if __name__ == '__main__':
    main()