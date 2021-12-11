from __future__ import print_function
import re, os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import scipy.signal
from scipy.cluster.vq import whiten, kmeans, vq
from tqdm.auto import tqdm, trange
from lfdfiles import SimfcsJrn
from tllab_common.wimread import imread as imr
from tllab_common.tiffwrite import IJTiffWriter
from tllab_common.misc import objFromDict
from parfor import parfor
import hidden_markov

if __package__ is None or __package__=='': #usual case
    from listpyedit import listFile
    import misc
else: #in case you do from another package: from LiveCellAnalysis import dataAnalysis_functions
    from .listpyedit import listFile
    from . import misc


class hmm(hidden_markov.hmm):
    def __init__(self, data, color='r', bg=(1000, 500), states=None, start_prob=None, trans_prob=None, em_prob=None,
                 n_iter=10):
        """ Class for binarizing data using a hidden Markov model. The probabilities are optimized using the Baum-
            Welch algorithm upon instance creation and the model can subsequently be used to binarize a trace using the
            Viterbi algorithm.
            Inputs:
                data: misc.objFromDict object with traces
                color: which color in data to use eg. 'r' or 'g'
                bg: (mean background, std background) of the signal, used for initial probability generation

            Example:
                h = hmm(data, 'g')
                bin_trace_g0 = h(data[0].g)
                bin_trace_g1 = h(data[1].g)
        """
        if not isinstance(data, (list, tuple)):
            data = [data]
        states = (0, 1) if states is None else states
        observations = np.unique(np.hstack([np.unique(d[color].astype(int)) for d in data]))
        start_prob = np.matrix((0.8, 0.2)) if start_prob is None else np.asmatrix(start_prob)
        trans_prob = np.matrix(((0.8, 0.2), (0.2, 0.8))) if trans_prob is None else np.asmatrix(trans_prob)
        if em_prob is None:
            g = scipy.special.erf((bg[0] - observations) / np.sqrt(2) / bg[1]) / 2 + 1 / 2
            em_prob = np.array((g, g.max() - g))
            em_prob = (em_prob.T / em_prob.sum(1)).T
            em_prob = np.matrix(em_prob)
        else:
            em_prob = np.asmatrix(em_prob)
        # for i in np.array(em_prob):
        #     plt.plot(observations, i)

        super().__init__(states, observations.tolist(), start_prob, trans_prob, em_prob)
        self.train_hmm([d[color].astype(int) for d in data], n_iter, [len(d[color]) for d in data])
        # self.train_hmm([d[color][d.mask>0].astype(int) for d in data], n_iter, [np.sum(d.mask) for d in data])

    def __call__(self, observations):
        return np.array(self.viterbi(observations.astype(int)))

    def train_hmm(self, observation_list, iterations, quantities):
        obs_size = len(observation_list)
        prob = float('inf')

        # Train the model 'iteration' number of times
        # store em_prob and trans_prob copies since you should use same values for one loop
        for i in range(iterations):
            @parfor(((q, obs) for q, obs in zip(quantities, observation_list)), (self,), length=obs_size,
                    desc=f'Baum-Welch, iteration {i}', bar=obs_size>1)
            def fun(j, obj):
                # re-assing values based on weight
                q, obs = j
                return (q * obj._train_emission(obs),
                        q * obj._train_transition(obs),
                        q * obj._train_start_prob(obs))

            emProbNew, transProbNew, startProbNew = [np.asmatrix(sum(i)) for i in zip(*fun)]

            # Normalizing
            em_norm = emProbNew.sum(axis=1)
            trans_norm = transProbNew.sum(axis=1)
            start_norm = startProbNew.sum(axis=1)

            # emProbNew = emProbNew / em_norm.transpose()
            emProbNew = emProbNew / em_norm
            startProbNew = startProbNew / start_norm.transpose()
            transProbNew = transProbNew / trans_norm.transpose()

            self.em_prob, self.trans_prob = emProbNew, transProbNew
            self.start_prob = startProbNew

            if prob - self.log_prob(observation_list, quantities) > 0.0000001:
                prob = self.log_prob(observation_list, quantities)
            else:
                return


def kmeans_split(data, n):
    w = whiten(data)
    codebook, _ = kmeans(w, n)
    codes, _ = vq(w, codebook)
    return sorted([data[codes==code] for code in range(n)], key=lambda x: np.mean(x))

def getMolNumber(data, n=3, channel=0):
    if isinstance(channel, str):
        color = channel[0]
    else:
        color = ('r', 'g')[channel]
    for d in data:
        t = d['trk_'+color][:, 3]
        i = d['trk_'+color][:, 2]
        b = np.full(len(t), np.nan)
        sr = scipy.signal.savgol_filter(i, 19, 3)
        for idx, i in enumerate(kmeans_split(sr, n)):
            for j in i:
                # b[sr == j] = i.mean()
                b[sr == j] = idx*1000
        d[f'trk_{color}_orig'] = d[f'trk_{color}'].copy()
        d[f'{color}_orig'] = d[color].copy()
        d['trk_'+color][:, 2] = b
        d[color] = b

def smoothData(data, channel, window_length, polyorder):
    if isinstance(channel, str):
        color = channel[0]
    else:
        color = ('r', 'g')[channel]
    for d in data:
        d[f'trk_{color}_orig'] = d[f'trk_{color}'].copy()
        d[f'{color}_orig'] = d[color].copy()
        d['trk_' + color][:, 2] = scipy.signal.savgol_filter(d['trk_'+color][:, 2], window_length, polyorder)
        d[color] = d['trk_' + color][:, 2]

##################################
### Numerical computations

### Compute all crosscorrelations G
def compG_multiTau(v, t, n=0, ctr=0):
    """v: data vector (channels=rows), t: time, n: bin every n steps.\n--> Matrix of G, time vector"""
    def compInd(v1,v2):
        if len(t)<2:
            return np.array([[], []]).T
        tau=[]; G=[]; t0=t*1.; i=0; dt=t0[1]-t0[0]
        while i<t0.shape[0]:
            tau.append(i*dt)
            G.append(np.mean(v1[:int(v1.shape[0]-i)]*v2[int(i):]))
            if i==n:
                i=i/2
                dt*=2
                t0,v1,v2=np.c_[t0,v1,v2][:int(t0.shape[0]/2)*2].T.reshape(3,-1,2).mean(2)
            i+=1
        return np.array([tau,G]).T
    if ctr: vCtr=((v.T-np.mean(v,1)).T);
    else: vCtr=v
    res=np.array([[ compInd(v1,v2) for v2 in vCtr] for v1 in vCtr])
    return ( res[:,:,:,1].T /(np.dot(np.mean(v,1).reshape(-1,1),np.mean(v,1).reshape(1,-1)))).T, res[0,0,:,0]

#################################
#### Read experimental data

# Read head of metadata file
def readMetadata(pathToFile):
    if os.path.splitext(pathToFile)[1] == '.czi':
        with imr(pathToFile) as im:
            return objFromDict(Interval_ms=im.timeinterval * 1000)
    if os.path.splitext(pathToFile)[1] == '.jrn':
        with SimfcsJrn(pathToFile, lower=True) as jrn:
            for f in jrn:
                if 'parameters for tracking' in f:
                    a = f['parameters for tracking']
                    dwell_time = a['dwell time'].split(' ')
                    dwell_time = float(dwell_time[0]) * {'us': 1e-3, 'ms': 1}[dwell_time[1]]
                    return objFromDict(Interval_ms=dwell_time*a['rperiods']*a['points per orbit'])
    with open(pathToFile, 'r') as f:
        content = f.read(-1)
    if len(content)>5:
        # fix missing }
        if content[-2:] == ',\n':
            content = content[:-2]
        elif content[-1] == ',':
            content = content[:-1]
        lines = content.replace('\\\"', '').splitlines() #2. but ignore \"
        lines = ''.join([re.sub('\"[^\"]*\"', '', l) for l in lines]) #1. exclude {} between ""
        content += '}' * (lines.count('{') - lines.count('}'))
        tmpMD = json.loads(content)
        if 'Summary' in tmpMD:
            return objFromDict(**tmpMD['Summary'])
    return objFromDict(**{})


def loadExpData(fn, nMultiTau=8, ignore_comments=False):
    if fn[-3:]=='.py': fn=fn[:-3]
    if fn[-5:]!='.list': fn=fn+'.list'
    fn=os.path.expanduser(fn)
    lf = listFile(fn + '.py')
    if not ignore_comments:
        lf = lf.on

    data = []
    for a in tqdm(lf, desc='Loading experimental data'):
        d = objFromDict(**{})
        #    d.path=procDataPath
        if 'trk_r' in a:  d.trk_r=np.loadtxt(a.trk_r)
        if 'trk_g' in a:  d.trk_g=np.loadtxt(a.trk_g)
        if 'trk_b' in a:  d.trk_b=np.loadtxt(a.trk_b)

        if 'detr' in a: # Columns of detr: frame, red mean, red sd, green mean, green sd, red correction raw, red correction polyfit, green correction raw, green correction polyfit
            d.detr=np.loadtxt(a.detr)
            rn=d.detr[:,2]/d.detr[0,2]; x=np.where(abs(np.diff(rn))<.1)[0]; pf=np.polyfit(x,np.log(rn[x]),8)
            rf=np.exp(np.sum([d.detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
            gn=d.detr[:,4]/d.detr[0,4]; x=np.where(abs(np.diff(gn))<.1)[0]; pf=np.polyfit(x,np.log(gn[x]),8)
            gf=np.exp(np.sum([d.detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
            d.detr=np.c_[d.detr,rn,rf,gn,gf]

        if 'frameWindow' in a:  d.frameWindow=a.frameWindow
        if 'actualDt' in a: d.actualDt=d.dt=a.actualDt
        else: d.dt=0
        if 'hrsTreat' in a: d.hrsTreat=a.hrsTreat

        if 'rawPath' in a: d.rawPath=a.rawPath
        if 'rawTrans' in a: d.rawTrans=a.rawTrans

        if 'fcs_rr' in a: d.fcs_rr=np.loadtxt(a.fcs_rr,skiprows=7)
        if 'fcs_gg' in a: d.fcs_gg=np.loadtxt(a.fcs_gg,skiprows=7)
        if 'fcs_rg' in a: d.fcs_rg=np.loadtxt(a.fcs_rg,skiprows=7)
        if 'fcs_gr' in a: d.fcs_gr=np.loadtxt(a.fcs_gr,skiprows=7)

        if 'trk_r' in a:  d.name=a.trk_r.replace('_red.txt','').replace('_green.txt','').replace('.txt','')
        if 'trk_g' in a:  d.name=a.trk_g.replace('_green.txt','').replace('_red.txt','').replace('.txt','')
        if 'trk_b' in a:  d.name=a.trk_b.replace('_blue.txt','').replace('_blue.txt','').replace('.txt','')

        if 'ctrlOffset' in a:  d.ctrlOffset=np.array(a.ctrlOffset)
        if 'transfLev' in a:  d.transfLev=np.array(a.transfLev)

        if 'maxProj' in a:
            d.maxProj=a.maxProj
            # if not os.path.exists(d.maxProj): print("!! Warning: file '%s' does not exist."%(a.maxProj))

        if 'metadata' in a:
            d.metadata=readMetadata(a.metadata)
            if d.dt==0: d.dt=d.metadata.Interval_ms/1000.
            #else: print "Using provided dt=%fs, not %fs from metadata."%(d.dt,d.metadata.Interval_ms/1000.)
        elif 'timeInterval' in a:
            d.dt=int(float(a.timeInterval))
        # else:
        #     print("!! Warning: No metadata and no dt provided. Using dt=1."); d.dt=1.

        if a.trk_r.endswith('trk'):  # orbital
            d.t = d.trk_r[:, -1] * d.dt
            d.r = d.trk_r[:, -2]
            d.g = d.trk_g[:, -2]
            if 'trk_b' in d:
                d.b = d.trk_b[:, -2]
        else:  # widefield
            d.t = d.trk_r[:, -2] * d.dt
            d.r = d.trk_r[:, -3]
            d.g = d.trk_g[:, -3]
            if 'trk_b' in d:
                d.b = d.trk_b[:, -3]
        if 'detr' in d: # Detrending from s.d. polyfit
            d.r=d.r/d.detr[:,6]
            d.g=d.g/d.detr[:,8]

        if not 'frameWindow' in d:
            d.frameWindow=[0,d.t.shape[0]]
        else:
            if d.frameWindow[0]<0:
                d.frameWindow[0] = 0
            if d.frameWindow[1] > d.t.shape[0]-1:
                d.frameWindow[1] = d.t.shape[0]-1

        if nMultiTau != 0:
            if d.frameWindow[1]-d.frameWindow[0]:
                d.fcsRecomp=True
                d.G, d.tau = compG_multiTau(np.c_[d.r,d.g][d.frameWindow[0]:d.frameWindow[1]].T, d.t[d.frameWindow[0]:d.frameWindow[1]], 0)
                # Write .fcs4 files
                np.savetxt(d.name+'.fcs4', np.c_[d.tau,d.G[0,0],d.G[1,1],d.G[0,1],d.G[1,0]],'%12.5e  ',
                           header='Tau (in s)     Grr            Ggg            Grg            Ggr', comments='#')
            else:
                d.fcsRecomp = False
                d.G = np.zeros((2,2,0))
                d.tau = np.zeros(0)
        else:
            d.fcsRecomp=False
            d.tau=d.fcs_rr[:,0]*d.dt/d.fcs_rr[0,0]
            d.G=np.array([[d.fcs_rr[:,1],d.fcs_rg[:,1]],[d.fcs_gr[:,1],d.fcs_gg[:,1]]])        
        data.append(d)
    return data


#################
### Displays

def showData(data, *args, **kwargs):
    if isinstance(data, objFromDict):
        data = [data]
    d = data[0]
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = GridSpec(2, 3, figure=fig)

    fig.add_subplot(gs[0,:2])
    plt.plot(d.t, d.g/d.g.max(), 'g')
    plt.plot(d.t, d.r/d.r.max(), 'r')
    plt.plot((0, d.t.max()), (0, 0), '--', color='gray')
    plt.xlim(0, d.t.max())
    plt.ylim(-0.5, 1.1)
    plt.xlabel('time (s)')
    plt.ylabel('fluorescence (AU)')

    fig.add_subplot(gs[0,2])
    colors = ['-or', '-og', '-ob', '-oy']
    for i in range(4):
        a = np.c_[d.tau, d.G[i%2, int(i%3!=0)]]
        plt.plot(a[:,0], a[:,1], colors[i])
    plt.legend(('G_rr(t)', 'G_gg(t)', 'G_rg(t)', 'G_gr(t)'))
    if len(d.tau):
        plt.plot((0, d.tau.max()), (0, 0), '--', color='gray')
        plt.xlim(0, d.tau.max())
    plt.xlabel('time lag (s)')
    plt.ylabel('G(t)')

    fig.add_subplot(gs[1,:2])
    plt.plot(d.t, np.sqrt((d.trk_r[:,0]-d.trk_g[:,0])**2+(d.trk_r[:,1]-d.trk_g[:,1])**2)+10, 'k')
    plt.plot(d.t[1:], np.sqrt(np.diff(d.trk_r[:,0])**2+np.diff(d.trk_r[:,1])**2)+5, 'r')
    plt.plot(d.t[1:], np.sqrt(np.diff(d.trk_g[:,0])**2+np.diff(d.trk_g[:,1])**2), 'g')
    plt.plot((0, d.t.max()), (0, 0), '--', color='gray')
    plt.plot((0, d.t.max()), (5, 5), '--', color='gray')
    plt.plot((0, d.t.max()), (10, 10), '--', color='gray')
    plt.xlim(0, d.t.max())
    plt.xlabel('time (s)')
    plt.ylabel('distance (pixels)')

    fig.add_subplot(gs[1,2])
    plt.plot(d.trk_r[:,0], d.trk_r[:,1], 'r')
    plt.plot(d.trk_g[:,0], d.trk_g[:,1], 'g')
    plt.gca().invert_yaxis()
    #fig.subtitle(d.name)
    plt.tight_layout()
    return fig


def showTracking(Data, channels=None, expPathOut=None, sideViews=None, zSlices=None, frameRange=None, transform=False,
                 drift=None):
    """ saves tiff with localisations
        data:       one item of data as loaded by LoadExp
        channels:   which channels to consider, [0] for 1 color,
                     [0,1] or [1,0] for 2 color,                   default: all channels
        pathOut:    in which path to save the tiff                 default: path of raw data
        cell:       cell number for inclusion in filename,         default: 0
        sideViews:  True/False: yes/no,                            default: True if raw is z-stack
        zSlices:    list: which zSlices to take from the stack,    default: all slices
        frameRange: (start, end): which frames (time) to include,  default: all frames

        wp@tl20200310
    """
    # add squares around the spots in seperate channel (from empty image)
    squareStamp = [
        np.r_[-5, -5, -5, -5, -5, -5, -5, -5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4],
        np.r_[-5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4, -5, -5, -5, -5, -5, -5, -5]]
    nbPixPerZ = 2  # nr of pixels per z stack
    pathIn = Data[0].rawPath

    expPathOut = expPathOut or pathIn

    DataXY = [[data['trk_' + c][:, :2] for c in 'rgb' if 'trk_' + c in data] for data in Data]
    D = [(data, dataXY) for data, dataXY in zip(Data, DataXY) if np.sum(dataXY)]
    if not D:
        print('Warning: no localisations found!')
        return
    Data, DataXY = [d for d in zip(*D)]

    Cell = [int(re.findall('(?<=cellnr_)\d+', data.name)[0]) for data in Data]

    Outfile = [expPathOut + "_cellnr_" + str(cell) + "_track" + "Side" * sideViews + ".tif" for cell in Cell]

    if os.path.exists(Data[0].maxProj):
        maxFile = Data[0].maxProj
        maxFileExists = True
    else:
        maxFile = np.zeros(0)
        maxFileExists = False

    with imr(pathIn, transform=transform) as raw, imr(maxFile, transform=transform) as mx:
        mx.masterch = raw.masterch
        mx.slavech  = raw.slavech
        mx.detector = raw.detector

        channels = channels or np.arange(raw.shape[2])
        nCh = min(len(channels), 3)
        if sideViews is None:
            sideViews = raw.zstack
        if zSlices is None:
            zSlices = np.arange(raw.shape[3])
        else:
            try:
                zSlices = np.arange(zSlices)
            except:
                pass

        frameRange = frameRange or (0, raw.shape[4])
        # nbPixPerZ = int(np.round(raw.deltaz/raw.pxsize))
        with IJTiffWriter(Outfile, (4, 1, (frameRange[1] - frameRange[0]))) as out:
            Box = []
            Loc_xy = []
            Width = []
            Height = []
            for dataXY in DataXY:
                box = np.hstack((np.floor(np.vstack(dataXY).min(0)),
                                 np.ceil(np.vstack(dataXY).max(0)))).astype('int') + [-20, -20, 20, 20]
                box = np.maximum(np.minimum(box * [1, 1, -1, -1], (box.reshape(2, 2).mean(0).repeat(2).reshape(2, 2).astype(int)
                                + [-50, 50]).T.flatten() * [1, 1, -1, -1]), -np.array((0, 0) + raw.shape[:2])) * [1, 1, -1, -1]
                loc_xy = [np.round(d - box[:2] + 0.5).astype('int') for d in dataXY]
                width = (box[2] - box[0]) + sideViews * len(zSlices) * nbPixPerZ
                height = (box[3] - box[1]) + sideViews * len(zSlices) * nbPixPerZ

                Box.append(box)
                Loc_xy.append(loc_xy)
                Width.append(width)
                Height.append(height)

            for t in trange(*frameRange, desc='Saving tracking tiffs'):
                for c in range(nCh):
                    if maxFileExists and not sideViews:
                        CroppedIm = mx(c, 0, t)
                    else:
                        CroppedIm = raw[c, zSlices, t].squeeze((2, 4))
                        if not drift is None:
                            CroppedIm = misc.translateFrame(CroppedIm, drift[t])
                    for outfile, box, width, height in zip(Outfile, Box, Width, Height):
                        frame = np.zeros((height, (nCh + (nCh > 1)) * width), 'uint16')
                        if sideViews:
                            # Make xz and yz projection images. Projects 11 pixels around spot
                            croppedIm = CroppedIm[box[1]:box[3], box[0]:box[2], ...]
                            xyIm = np.nanmax(croppedIm, 2)
                            xzIm = np.nanmax(croppedIm, 0).repeat(nbPixPerZ, 1).T
                            yzIm = np.nanmax(croppedIm, 1).repeat(nbPixPerZ, 1)

                            # Make blank square for right bottom corner
                            blankSq = np.ones((xzIm.shape[0], yzIm.shape[1])) * np.mean(xyIm)
                            im = np.vstack((np.hstack((xyIm, yzIm)), np.hstack((xzIm, blankSq))))
                        elif maxFileExists:
                            im = CroppedIm[box[1]:box[3], box[0]:box[2]]
                        else:
                            croppedIm = CroppedIm[box[1]:box[3], box[0]:box[2], ...]
                            im = np.nanmax(croppedIm, 3).squeeze((2, 3))

                        frame[:, c * width:(c + 1) * width] = im.astype('uint16')
                        if nCh > 1:
                            frame[:, -width:] = im.astype('uint16')
                        out.save(outfile, frame, c, 0, t)

                for outfile, loc_xy, width, height in zip(Outfile, Loc_xy, Width, Height):
                    frame = np.zeros((height, (nCh + (nCh > 1)) * width), 'uint16')
                    for c in range(nCh):
                        if nCh > 1:
                            frame[loc_xy[c][t, 1] + squareStamp[1], loc_xy[c][t, 0] + squareStamp[0] + nCh * width] = 65535
                        frame[loc_xy[c][t, 1] + squareStamp[1], loc_xy[c][t, 0] + squareStamp[0] + c * width] = 65535
                    out.save(outfile, frame, 3, 0, t)


macro1color="""open("__imPath__");
    run("Stack to Hyperstack...", "order=xyczt(default) channels=2 slices=1 frames=__frames__ display=Color");
    Stack.setDisplayMode("color");
    Stack.setChannel(2);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setDisplayMode("composite");
    run("Save");
    run("Quit");"""


macro2color ="""open("__imPath__");
    run("Make Composite");
    Stack.setDisplayMode("color");
    Stack.setChannel(2);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setDisplayMode("composite");
    saveAs("Tiff", "__imPath__");
    run("Quit");
    """
