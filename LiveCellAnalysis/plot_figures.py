import os
import numpy as np
import matplotlib.pyplot as plt
import copy as copy2
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram

if __package__ is None or __package__=='': #usual case
    import dataAnalysis_functions as daf
    from misc import line
else: #in case you do from another package:
    from . import dataAnalysis_functions as daf
    from .misc import line
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.max_open_warning'] = 250
A4 = (11.69, 8.27)
np.seterr(invalid='ignore')

# This script defines functions used in the pipeline_livecell_correlationfunctions to plot figures


##################
##################
### Misc functions


def write_hist(pdf, color, dA, sdThreshold):
    # color: g or r
    if color=='g':
        m = dA.mg
        sd = dA.sdg
        dataDigital = dA.trk_g[:,-1]
        bg = dA.bgg
    else:
        m = dA.mr
        sd = dA.sdr
        dataDigital = dA.trk_r[:,-1]
        bg = dA.bgr
    
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.hist(bg, np.clip(int(np.ceil(np.sqrt(len(bg)))), 1, 100), density=True, color=color)
    xl = plt.xlim()
    X = np.linspace(xl[0], xl[1], 1000)
    if sd > 0:
        plt.plot(X, ((1/(X*sd*(2*np.pi)**0.5)) * np.exp(-(((np.log(X)-m)**2)/(2*sd**2)))), '-k')
    else:
        plt.plot(X, np.zeros(X.shape))
    expVal = np.exp(m-sd**2)
    plt.axvline(expVal,linestyle = '--')
    plt.xlabel('background counts')
    plt.ylabel('frequency')
    plt.title(dA.name.split("/")[-1] + ' ' + color)

    fig.add_subplot(gs[1,0])
    plt.plot((dA.t.min(), dA.t.max()), np.zeros(2), '--k')
    sdt = plt.plot((dA.t.min(), dA.t.max()), sdThreshold*np.ones(2), '--'+color)
    t0 = plt.plot(dA.t, dA[color], '-'+color)
    s = dA[color].max()
    d = plt.plot(dA.t,s*(dataDigital-1)/10, '-c')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*max(np.r_[dA.r,dA.g]), color = 'gray')
    plt.xlim(dA.t.min(), dA.t.max())
    plt.legend((t0[0], sdt[0], d[0]), ('Trace #0', 'sdThreshold', 'Digital'), loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('peak intensity (counts)')
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

### Display results of the binary thresholding
def showBinaryCall(pdf, dA, dB):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.plot(dB.t, dB.g*.8+1.1, '-g')
    plt.plot(dB.t, dB.r*.8+.1,  '-r')
    plt.ylim(0,2)
    plt.xlabel('time (s)')
    plt.title(dA.name + ' binary thresholding')

    fig.add_subplot(gs[1,0])
    plt.plot(dA.t, dA.g/dA.g.max(), '-g')
    plt.plot(dA.t, dA.r/dA.r.max(), '-r')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0], color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('peak intensity (AU)')
    if not pdf is None:
        pdf.savefig()
    else:
        return fig

def showBackgroundTrace(pdf, dA, color, sdThreshold):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    if color == 'g':
        bg = dA.bgg
        m = dA.mg

    else:
        bg = dA.bgr
        m = dA.mr

    fig.add_subplot(gs[0,0])
    plt.plot(dA.t, dA[color], '-'+color)
    plt.plot(dA.t, np.r_[bg[:len(dA.t)]]-np.exp(m), '-b')
    plt.plot(dA.t, np.r_[bg[len(dA.t) : 2*len(dA.t)]]-np.exp(m), '-c')
    plt.plot(dA.t, np.r_[bg[2*len(dA.t) : 3*len(dA.t)]]-np.exp(m), '-m')
    plt.plot(dA.t, np.r_[bg[3*len(dA.t) : 4*len(dA.t)]]-np.exp(m), '-y')
    plt.plot((dA.t.min(), dA.t.max()), sdThreshold * np.ones(2), '--', color = 'gray')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*dA[color].max(), color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('intensity (AU)')

    fig.add_subplot(gs[1,0])
    plt.plot(dA.t, np.c_[np.r_[bg[:len(dA.t)]],np.r_[bg[len(dA.t) : 2*len(dA.t)]], np.r_[bg[2*len(dA.t) : 3*len(dA.t)]],np.r_[bg[3*len(dA.t) : 4*len(dA.t)]] ].mean(axis = 1)-m , '-b')
    plt.plot((dA.t.min(), dA.t.max()), sdThreshold * np.ones(2), '--', color = 'gray')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*max(bg), color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('mean background intensity (AU)')


    if not pdf is None:
        pdf.savefig()
    else:
        return fig

def showAvTrace(pdf, ss, names):
    # _average_trace.pdf
    nppage = len(ss.sigsAlign)+1 if pdf is None else 2
    fig = plt.figure(figsize=(16, 3*(len(ss.sigsAlign)+1))) if pdf is None else plt.figure(figsize=A4)
    gs = GridSpec(nppage, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    axr = fig.gca()
    axg = axr.twinx()
    axr.fill_between(ss.t, ss.v[0]-ss.dv[0], ss.v[0]+ss.dv[0], facecolor='r', edgecolor=None, alpha=0.5)
    axg.fill_between(ss.t, ss.v[1]-ss.dv[1], ss.v[1]+ss.dv[1], facecolor='g', edgecolor=None, alpha=0.5)
    axr.plot(ss.t, ss.v[0], '-r')
    axg.plot(ss.t, ss.v[1], '-g')
    plt.plot(np.r_[0,0], np.r_[ss.v.min(),ss.v.max()], color = 'black' )
    plt.xlim(ss.t.min(), ss.t.max())
    plt.xlabel('time (s)')
    axr.set_ylabel('average peak intensity (counts)')
    axr.tick_params(axis='y', labelcolor='red')
    axg.tick_params(axis='y', labelcolor='green')
    plt.title('average peak intensity')

    idx = 1 if len(ss.sigsAlign)%nppage else 0

    nameInd = 0
    for s in ss.sigsAlign:
        if not idx:
            pdf.savefig(fig)
            fig = plt.figure(figsize=A4)
            gs = GridSpec(2, 1, figure=fig)
        fig.add_subplot(gs[idx,0])
        axr = fig.gca()
        axg = axr.twinx()
        axr.plot(s.t, s.v[0], '-r')
        axg.plot(s.t, s.v[1], '-g')

        if 'v_orig' in s:
            axr.plot(s.t, s.v_orig[0], '-r', alpha=0.2)
            axg.plot(s.t, s.v_orig[1], '-g', alpha=0.2)

        axr.plot(s.t, (1-s.mask[0])*s.v[0].max(), '--', color='gray')
        axr.plot(np.r_[0,0], np.r_[getylim(s.v[0], s.t, (ss.t.min(), ss.t.max()))], '--k')
        plt.xlim(ss.t.min(), ss.t.max())
        plt.xlabel('time (s)')
        axr.set_ylabel('peak intensity (counts)')
        axr.tick_params(axis='y', labelcolor='red')
        axg.tick_params(axis='y', labelcolor='green')
        plt.title((1 - idx) * 'peak intensity for individual traces\n' + names[nameInd].split("/")[-1])
        idx = (idx + 1) % nppage
        nameInd +=1
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig
    
def getylim(y, x=None, xlim=None, margin=0.05):
    """ get limits for plots according to data
        copied from matplotlib.axes._base.autoscale_view

        y: the y data
        optional, for when xlim is set manually on the plot
            x: corresponding x data
            xlim: limits on the x-axis in the plot, example: xlim=(0, 100)
            margin: what fraction of white-space to have at all borders
        y and x can be lists or tuples of different data in the same plot

        wp@tl20191220
    """
    y = np.array(y).flatten()
    if not x is None and not xlim is None:
        x = np.array(x).flatten()
        y = y[(np.nanmin(xlim)<x)*(x<np.nanmax(xlim))*(np.abs(x)>0)]
        if not np.any(np.isfinite(y)):
            return 0, 1
        if len(y) == 0:
            return -margin, margin
    y0t, y1t = np.nanmin(y), np.nanmax(y)
    if (np.isfinite(y1t) and np.isfinite(y0t)):
        delta = (y1t - y0t) * margin
        if y0t == y1t:
            delta = 0.5
    else:  # If at least one bound isn't finite, set margin to zero
        delta = 0
    return y0t-delta, y1t+delta

def showCorrFun(pdf, ss, plot_individual=False):
    for i, ((g, n, p), t) in enumerate(zip((('G', 'N', 'P'), ('Gs', 'Ns', 'Ps'), ('Gr', 'Nr', 'Pr'),
                                            ('Gg', 'Ng', 'Pg')), ('G', 'Scaled', 'Normalized r', 'Normalized g'))):
        if g not in ss or n not in ss or p not in ss:
            continue
        alpha = 0.1
        linewidth = 1
        dr, dn, dp = 'd'+g, 'd'+n, 'd'+p

        fig = plt.figure(figsize=A4)
        gs = GridSpec(2, 3, figure=fig)
        
        xlim = ss.tau.max()/2

        ##
        fig.add_subplot(gs[0,0])
        plt.fill_between(ss.tau, ss[g][0,0]-ss[dr][0,0], ss[g][0,0]+ss[dr][0,0], facecolor='r', edgecolor=None, alpha=0.5)
        plt.fill_between(ss.tau, ss[n][0,0]-ss[dn][0,0], ss[n][0,0]+ss[dn][0,0], facecolor='k', edgecolor=None, alpha=0.5)
        lg = plt.plot(ss.tau, ss[g][0,0], '-r')
        ln = plt.plot(ss.tau, ss[n][0,0], '-k')
        if plot_individual:
            for x, y, z in zip(ss.rb.tau, ss.rb[g], ss.rb[n]):
                plt.plot(x, y[0, 0], '-r', alpha=alpha, linewidth=linewidth)
                plt.plot(x, z[0, 0], '-k', alpha=alpha, linewidth=linewidth)

        plt.xlim(0, xlim)
        plt.ylim(getylim((ss[g][0,0]-ss[dr][0,0], ss[g][0,0]+ss[dr][0,0],
                          ss[n][0,0]-ss[dn][0,0], ss[n][0,0]+ss[dn][0,0]), 4*(ss.tau,), (0, xlim)))
        plt.legend((lg[0], ln[0]), ('G (tau)', 'G ns (tau)'))
        plt.ylabel(r'$\mathrm{G}(\tau)$')
        plt.title('Non-stationary\nred auto-correlation')

        ##
        fig.add_subplot(gs[0,1])
        plt.fill_between(+ss.tau, ss[g][0,1]-ss[dr][0,1], ss[g][0,1]+ss[dr][0,1], facecolor='b', edgecolor=None, alpha=0.5)
        plt.fill_between(-ss.tau, ss[g][1,0]-ss[dr][1,0], ss[g][1,0]+ss[dr][1,0], facecolor='y', edgecolor=None, alpha=0.5)
        plt.fill_between(+ss.tau, ss[n][0,1]-ss[dn][0,1], ss[n][0,1]+ss[dn][0,1], facecolor='k', edgecolor=None, alpha=0.5)
        plt.fill_between(-ss.tau, ss[n][1,0]-ss[dn][1,0], ss[n][1,0]+ss[dn][1,0], facecolor='k', edgecolor=None, alpha=0.5)
        grg = plt.plot(+ss.tau, ss[g][0,1], '-b')
        ggr = plt.plot(-ss.tau, ss[g][1,0], '-y')
        grgns = plt.plot(+ss.tau, ss[n][0,1], '-k')
        ggrns = plt.plot(-ss.tau, ss[n][1,0], '-k')

        if plot_individual:
            for x, y, z in zip(ss.rb.tau, ss.rb[g], ss.rb[n]):
                plt.plot(x, y[0, 1], '-b', alpha=alpha, linewidth=linewidth)
                plt.plot(-x, y[1, 0], '-y', alpha=alpha, linewidth=linewidth)
                plt.plot(x, z[0, 1], '-k', alpha=alpha, linewidth=linewidth)
                plt.plot(-x, z[1, 0], '-k', alpha=alpha, linewidth=linewidth)

        plt.xlim(-xlim, xlim)
        plt.ylim(getylim((ss[g][0,1]-ss[dr][0,1], ss[g][0,1]+ss[dr][0,1],
                          ss[g][1,0]-ss[dr][1,0], ss[g][1,0]+ss[dr][1,0]),
                         (ss.tau, ss.tau, -ss.tau, -ss.tau), (-xlim, xlim)))
        plt.ylim(getylim((ss[g][0,1]-ss[dr][0,1], ss[g][0,1]+ss[dr][0,1],
                          ss[g][1,0]-ss[dr][1,0], ss[g][1,0]+ss[dr][1,0],
                          ss[n][0,1]-ss[dn][0,1], ss[n][0,1]+ss[dn][0,1],
                          ss[n][1,0]-ss[dn][1,0], ss[n][1,0]+ss[dn][1,0]),
                         (ss.tau, ss.tau, -ss.tau, -ss.tau, ss.tau, ss.tau, -ss.tau, -ss.tau), (-xlim, xlim)))
        plt.legend((grg[0], grgns[0], ggr[0], ggrns[0]), ('G[rg] (tau)', 'G[rg] ns (tau)', 'G[gr] (tau)', 'G[gr] ns (tau)'))
        plt.title('Non-stationary\ncross-correlations')

        ##
        fig.add_subplot(gs[0,2])
        plt.fill_between(ss.tau, ss[g][1,1]-ss[dr][1,1], ss[g][1,1]+ss[dr][1,1], facecolor='g', edgecolor=None, alpha=0.5)
        plt.fill_between(ss.tau, ss[n][1,1]-ss[dn][1,1], ss[n][1,1]+ss[dn][1,1], facecolor='k', edgecolor=None, alpha=0.5)
        lg = plt.plot(ss.tau, ss[g][1,1], '-g')
        ln = plt.plot(ss.tau, ss[n][1,1], '-k')

        if plot_individual:
            for x, y, z in zip(ss.rb.tau, ss.rb[g], ss.rb[n]):
                plt.plot(x, y[1, 1], '-g', alpha=alpha, linewidth=linewidth)
                plt.plot(x, z[1, 1], '-k', alpha=alpha, linewidth=linewidth)

        plt.xlim(0, xlim)
        plt.ylim(getylim((ss[g][1,1]-ss[dr][1,1], ss[g][1,1]+ss[dr][1,1],
                          ss[n][1,1]-ss[dn][1,1], ss[n][1,1]+ss[dn][1,1]),
                         4*(ss.tau,), (0, xlim)))
        plt.legend((lg[0], ln[0]), ('G (tau)', 'G ns (tau)'))
        plt.title('Non-stationary\ngreen auto-correlation')

        ##
        fig.add_subplot(gs[1,0])
        plt.fill_between(ss.tau, ss[p][0,0]-ss[dp][0,0], ss[p][0,0]+ss[dp][0,0], facecolor='r', edgecolor=None, alpha=0.5)
        plt.plot(ss.tau, ss[p][0,0], '-r')

        if plot_individual:
            for x, y in zip(ss.rb.tau, ss.rb[p]):
                plt.plot(x, y[0, 0], '-r', alpha=alpha, linewidth=linewidth)

        plt.xlim(0, xlim)
        plt.ylim(getylim((ss[p][0,0]-ss[dp][0,0], ss[p][0,0]+ss[dp][0,0]), 2*(ss.tau,), (0, xlim)))
        plt.xlabel('time delay (s)')
        plt.ylabel(r'$\mathrm{G}(\tau)$')
        plt.title('Pseudo-stationary red')

        ##
        fig.add_subplot(gs[1,1])
        plt.fill_between(+ss.tau, ss[p][0,1]-ss[dp][0,1], ss[p][0,1]+ss[dp][0,1], facecolor='b', edgecolor=None, alpha=0.5)
        plt.fill_between(-ss.tau, ss[p][1,0]-ss[dp][1,0], ss[p][1,0]+ss[dp][1,0], facecolor='y', edgecolor=None, alpha=0.5)
        plt.plot(+ss.tau,ss[p][0,1], '-b')
        plt.plot(-ss.tau,ss[p][1,0], '-y')

        if plot_individual:
            for x, y in zip(ss.rb.tau, ss.rb[p]):
                plt.plot(x, y[0, 1], '-b', alpha=alpha, linewidth=linewidth)
                plt.plot(-x, y[1, 0], '-y', alpha=alpha, linewidth=linewidth)

        plt.xlim(-xlim, xlim)
        plt.ylim(getylim((ss[p][0,1]-ss[dp][0,1], ss[p][0,1]+ss[dp][0,1],
                          ss[p][1,0]-ss[dp][1,0], ss[p][1,0]+ss[dp][1,0]),
                         (ss.tau, ss.tau, -ss.tau, -ss.tau), (-xlim, xlim)))
        plt.xlabel('time delay (s)')
        plt.title('Pseudo-stationary cross-correlations')

        ##
        fig.add_subplot(gs[1,2])
        plt.fill_between(ss.tau, ss[p][1,1]-ss[dp][1,1], ss[p][1,1]+ss[dp][1,1], facecolor='g', edgecolor=None, alpha=0.5)
        plt.plot(ss.tau, ss[p][1,1], '-g')

        if plot_individual:
            for x, y in zip(ss.rb.tau, ss.rb[p]):
                plt.plot(x, y[1, 1], '-g', alpha=alpha, linewidth=linewidth)

        plt.xlim(0, xlim)
        plt.ylim(getylim((ss[p][1,1]-ss[dp][1,1], ss[p][1,1]+ss[dp][1,1]), 2*(ss.tau,), (0, xlim)))
        plt.xlabel('time delay (s)')
        plt.title('Pseudo-stationary green')
        fig.suptitle(t, fontweight='bold')

        if not pdf is None:
            pdf.savefig(fig)
            plt.close(fig)


def showAutoCorr(pdf, color, t, tau, G, dG, fitp=None, dfitp=None, xmax=None):
    """ color should be "g" or "r" """
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
    fill = plt.fill_between(tau, G-dG, G+dG, facecolor=color, edgecolor=None, alpha=0.5)
    if xmax is None:
        corr = plt.plot(tau, G, '-'+color)
        xlim = tau.max() / 2
    else:
        corr = plt.plot(tau, G, 'o-'+color)
        xlim = xmax

    x = np.linspace(tau.min(), tau.max(), int(tau.max()))
    if not fitp is None:
        if dfitp is None:
            dfitp = [i / 100 for i in fitp]
        fit = plt.plot(x, line(x, fitp), '-k')
        plt.plot(x, line(x, np.hstack((0, fitp[1:]))), '-k')
        plt.legend((corr[0], fill, fit[0]),('auto correlation', 'error',
            u'fit:\namplitude = {:.2f} \261 {:.2f}\ndwell time = {:.2f} \261 {:.2f} s'.format(fitp[0], dfitp[0], fitp[2], dfitp[2])))
    else:
        plt.legend((corr[0], fill ), ('auto correlation, error in fit', 'error'))
        
    plt.plot((0, xlim), (0,0), '--k')
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')

    plt.xlim(0, xlim)
    if xmax is None:
        plt.ylim(-1, 1.5*np.max(G[0]))
    else:
        plt.ylim(-0.2, 1.5 * np.max(G[0]))
    plt.title('ACF {}, {}'.format(color, t))
    if not pdf is None:
        pdf.savefig(fig)
        plt.close(fig)


def showCrossCorr(pdf, t, tau, G, G2, dG, dG2, ylim=None, perr0=None, perr1=None, popt0=None, popt1=None, popt2=None, popt3=None, xlim=None):
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
    plt.fill_between(tau, G-dG, G+dG, facecolor='b', edgecolor=None, alpha=0.5)
    plt.fill_between(-tau[::-1], G2-dG2, G2+dG2, facecolor='y', edgecolor=None, alpha=0.5)

    grg = plt.plot(tau, G, '-b')
    ggr = plt.plot(-tau[::-1], G2, '-y')
    if ylim is None:
        ylim = getylim((G-dG, G+dG, G2-dG2, G2+dG2), (tau, tau, -tau[::-1], -tau[::-1]), xlim)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim)
    
    def gauss_function(x, a, x0, sigma, b):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+b
    
    if not perr0 is None:
        fit = gauss_function(np.hstack((-tau[::-2],tau)), popt0, popt1, popt2, popt3)
        fitp = plt.plot(np.hstack((-tau[::-2],tau)),fit, '-k' )
    plt.plot((xlim[0], xlim[1]), (0,0), '--k')
    plt.plot((0,0), (ylim), '--k')
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')
    if not perr0 is None:
        plt.legend((grg[0], ggr[0], fitp[0]), ('G[rg] (tau)',  'G[gr] (tau)',  u'Gauss fit: \nmean = {:.2f} \261 {:.2f}\namplitude = {:.2f} \261 {:.2f}'.format(popt1,perr1,popt0,perr0)))
    else:
        plt.legend((grg[0], ggr[0], ), ('G[rg] (tau)', 'G[gr] (tau), error in fit'))
    plt.title('Cross correlation, {}'.format(t))
    if not pdf is None:
        pdf.savefig(fig)
        plt.close(fig)


### Display area under trace for all traces (this function can be helpful to see which traces dominate your autocorrelation function).
def showAreaUnderTraces(dataA, retainedTraces, color):
    # color should be "g" or "r"
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
          
    meanR = []
    meanRh = []
    for i in retainedTraces:
        if color == "g":
            meanR.append(dataA[i].g.mean())
            meanRh.append(1./dataA[i].g.mean())
        elif color == "r":
            meanR.append(dataA[i].r.mean())
            meanRh.append(1./dataA[i].r.mean())
    plt.plot(retainedTraces, meanR, color+'.')
    
    arith = plt.plot((0, len(retainedTraces)),(np.sum(meanR)/len(meanR),np.sum(meanR)/len(meanR)), '--', color = 'blue')
    harm = plt.plot((0, len(retainedTraces)),(len(meanRh)/np.sum(meanRh),len(meanRh)/np.sum(meanRh)), '--', color = 'gray')
    
    plt.legend((arith[0],harm[0]),('arithmetic mean', 'harmonic mean'))
    plt.title('Area under trace '+color)
    plt.xlabel('Trace nr')
    plt.ylabel('Mean of trace')
    return (fig)

#### Histograms of burst duration, code not tested
#def showBurstDurationHistogram(binSize=30,maxT=1200):
#    dt=dataOrig[0].dt*1.
#    burstTime=[concatenate([diff((lambda a: a[:(a.shape[0]/2)*2].reshape(-1,2))(where(abs(diff(dB.r)))[0][dB.r[0]:]),1) for dB in dataB]), concatenate([diff((lambda a: a[:(a.shape[0]/2)*2].reshape(-1,2))(where(abs(diff(dB.g[0:])))[0][dB.g[0]:]),1) for dB in dataB])]
#    hr=histogram(burstTime[0],bins=r_[:maxT/dt:binSize/dt]-.5,density=1);
#    hg=histogram(burstTime[1],bins=r_[:maxT/dt:binSize/dt]-.5,density=1);
#    fig = plt.figure()
#    plt.bar(hr[1][:-1]*dt, hr[0], '-r')
#    plt.bar(hg[1][:-1]*dt, hg[0], '-g')
#    plt.title('Burst duration after binary thresholding')
#    plt.xlabel('Burst duration (s)')
#    plt.ylabel('Frequency')
#    plt.legend((hr[0], hg[0]), ('red','green'))
#    return(fig)

   
def showHeatMap(data, maxRed=None, maxGreen=None, trimdata=None, sortedIds=None):
    fig = plt.figure(figsize=(A4[0], A4[1]))
    if maxRed == 'None' or isinstance(maxRed, type(None)): maxRed = np.nanmax(np.hstack([d.r for d in data]))
    if maxGreen == 'None' or isinstance(maxGreen, type(None)): maxGreen = np.nanmax(np.hstack([d.g for d in data]))
    if isinstance(sortedIds, type(None)): sortedIds=range(len(data))
    nbPts = np.max([d.r.shape[0] for d in data])
#    heatMap = array([r_[c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if isinstance(trimdata, type(None)):
        heatMap = np.array([np.r_[np.c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],np.zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    else: heatMap = np.array([np.r_[np.c_[data[i].r/maxRed,data[i].g/maxGreen,trimdata[i].g],np.zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if len(heatMap) != 0: plt.imshow(heatMap)
    lab = np.arange(len(data[0].t), step=6)
    plt.xticks(lab, data[0].t[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    plt.xlabel('time (s)')
    plt.ylabel('experiment #')
    plt.tight_layout()
    return(plt)

def showHeatMapCF(data, maxRed=None, maxGreen=None, maxCF = None, sortedIds=None, Z = None, Normalize = False):
    fig = plt.figure(figsize=(A4[0], A4[1]))
    gs = GridSpec(1, 5, figure=fig)
    if isinstance(sortedIds, type(None)): sortedIds=range(len(data))

    fig.add_subplot(gs[0,0])
    maxRed = maxRed or np.nanmax(np.hstack([d.G[0,0] for d in data]))
    maxtaulen = np.max(np.hstack([len(d.tau) for d in data]))
    for m in range(len(data)):
        if len(data[m].tau) == maxtaulen:
            tau = data[m].tau
    nbPts = np.max([d.G[0,0].shape[0] for d in data])
    #heatMap = array([r_[c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if Normalize == False: heatMap = np.array([np.r_[np.c_[data[i].G[0,0]/maxRed],np.zeros((nbPts - data[i].G[0,0].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.c_[data[i].G[0,0]/data[i].G[0,0][0]],np.zeros((nbPts - data[i].G[0,0].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.ylabel('experiment #')
    lab = np.arange(len(tau), step=2)
    plt.xticks(lab, tau[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    plt.title('Heatmap ACF red')
    
    fig.add_subplot(gs[0,1])
    maxGreen = maxGreen or np.nanmax(np.hstack([d.G[1,1] for d in data]))
    nbPts = np.max([d.G[1,1].shape[0] for d in data])
    if Normalize == False: heatMap = np.array([np.r_[np.c_[data[i].G[1,1]/maxGreen],np.zeros((nbPts - data[i].G[1,1].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.c_[data[i].G[1,1]/data[i].G[1,1][0]],np.zeros((nbPts - data[i].G[1,1].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.xticks(lab, tau[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize =6)
    plt.title('Heatmap ACF green')
 
    fig.add_subplot(gs[0,2:4])
    xx = []
    for x in range(len(data)):
        xx.extend(data[x].G[0,1])
        xx.extend(data[x].G[1,0])
    maxCF = maxCF or np.nanmax(xx)
    nbPts1 = np.max([d.G[1, 0][1:].shape[0] for d in data])
    nbPts2 = np.max([d.G[0, 1].shape[0] for d in data])
    if Normalize == False: heatMap = np.array([np.r_[np.zeros((nbPts1 - data[i].G[1, 0][::-1][:-1].shape[0],1)),np.c_[(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))/maxCF)],np.zeros((nbPts2 - data[i].G[0, 1].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.zeros((nbPts1 - data[i].G[1, 0][::-1][:-1].shape[0],1)),np.c_[(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))/max(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))))],np.zeros((nbPts2 - data[i].G[0, 1].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.title('Heatmap crosscorrelation')
    lab2 = np.arange(np.hstack((-tau[::-1][:-1], tau)).shape[0], step=2)
    plt.xticks(lab2, np.hstack((-tau[::-1][:-1], tau))[lab2].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    
    if Z is not None:
        fig.add_subplot(gs[0,4])
        dendrogram(Z, orientation = "right")
        plt.xticks([])
    
    plt.tight_layout()
    return(plt)

# Function to align traces on the start of the first burst
def alignTraces(dataIn,startFrames):
    dataOut=copy2.deepcopy(dataIn)
    nbPts=np.max([d.r.shape[0] for d in dataIn])*2
    for i in np.r_[:len(dataIn)]:
        d=dataOut[i]; d.t=np.r_[:nbPts]*np.diff(d.t).mean() #d.t=r_[d.t,d.t+d.t[-1]+d.t[1]];
        d.r=np.roll(np.r_[d.r,np.zeros(nbPts-d.r.shape[0])],nbPts/2-startFrames[i])
        d.g=np.roll(np.r_[d.g,np.zeros(nbPts-d.g.shape[0])],nbPts/2-startFrames[i])
    return dataOut


def showCorrelFunAll(pdf, data, ChannelsToAnalyze, params):
    # _individual_correlation_functions.pdf
    for k in ('G', 'Gr', 'Gg', 'P', 'Pr', 'Pg'):
        if k not in data[0]:
            continue
        for channel in ChannelsToAnalyze:
            color = ["red", "green"]
            fig = plt.figure(figsize=A4)
            gs = GridSpec(4, 5, figure=fig)

            x, y = [], []
            for d in data:
                x.extend(d.tau[1:])
                y.extend(d[k][channel, channel, 1:])
            xlim = (0, params['IACxlim'][channel])
            ylim = np.clip(getylim(y, x, xlim), -0.2, np.inf)

            names = [os.path.basename(d.name).split("_trk")[0] for d in data]

            # adds enter in name if it is longer than 19 characters (to fit in plot frame)
            make_enter = 0
            for pname in range(len(names)):
                 for letter in range(len(names[pname])):
                     letter2 = letter % 20
                     if letter2 == 19:
                         make_enter += 1
                     if names[pname][letter] == "_" and make_enter > 0:
                         names[pname] = "\n".join([names[pname][:letter], names[pname][letter:]])
                         make_enter = 0

            for i, d in enumerate(data):
                idx = i % 20
                if idx == 0:
                    plt.suptitle('Autocorrelation functions color: {}, normalization: {}'.format(color[channel], k))
                    if i > 0:
                        if pdf is not None:
                            pdf.savefig(fig)
                            plt.close(fig)
                        fig = plt.figure(figsize=A4)
                        gs = GridSpec(4, 5, figure=fig)
                figall = fig.add_subplot(gs[idx // 5, idx % 5])
                plt.plot(d.tau, d[k][channel, channel], f'-{color[channel][0]}')
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.text(0.5, 0.98, names[i], fontsize=6, horizontalalignment='center', verticalalignment='top',
                         transform=plt.gca().transAxes)
                figall.set_xlabel('tau')
                figall.set_ylabel('G(tau)')
                for figall in fig.get_axes():
                    figall.label_outer()
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

        if len(ChannelsToAnalyze) == 2:
            fig = plt.figure(figsize=A4)
            gs = GridSpec(4, 5, figure=fig)

            x, y = [], []
            for d in data:
                for _ in range(2):
                    x.extend(d.tau)
                y.extend(d[k][0,1])
                y.extend(d[k][1,0])
            xlim = params['ICCxlim']
            ylim = getylim(y, x, xlim)

            for i, d in enumerate(data):
                idx = i % 20
                if idx == 0:
                    plt.suptitle('Crosscorrelation functions, normalization: {}'.format(k))
                    if i > 0:
                        if not pdf is None:
                            pdf.savefig(fig)
                            plt.close(fig)
                        fig = plt.figure(figsize=A4)
                        gs = GridSpec(4, 5, figure=fig)
                figall = fig.add_subplot(gs[idx//5,idx%5])
                plt.plot(d.tau, d[k][0, 1], '-b')
                plt.plot(-d.tau[::-1], d[k][1, 0][::-1], '-y')
                plt.plot((0,0), (ylim), '--k', linewidth=0.5)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.text(0.5, 0.98, names[i], fontsize=6, horizontalalignment='center', verticalalignment='top',
                         transform=plt.gca().transAxes)
                figall.set_xlabel('tau')
                figall.set_ylabel('G(tau)')
                for figall in fig.get_axes():
                    figall.label_outer()

            if not pdf is None:
                pdf.savefig(fig)
                plt.close(fig)


#function to calculate bootstrap errors for non-normal distributions
def CalcBootstrap(input, repeats, samplesize=None):
    if samplesize is None:
        samplesize = len(input)
    if len(input) == 0:
        return np.nan, np.nan
    else:
       bootmeans = []
       for i in range(repeats):
           bootsample = np.random.choice(input, size=samplesize, replace=True)
           bootmeans.append(np.mean(bootsample))
       bootstrapmean = np.mean(bootmeans)
       bootstrapSD = np.std(bootmeans)
       return bootstrapmean, bootstrapSD

def HistogramPlot(input, nbrbins, titletext, xaxlabel, outname):
    hist = np.histogram(input, bins=nbrbins, density=True)

    binedges = hist[1]
    plotbins = []
    for i in range(len(binedges)-1):
        plotbins.append((0.5*(binedges[i+1]+binedges[i])))

    binwidth = plotbins[1]-plotbins[0]
    fig = plt.figure(figsize=A4)
    plt.bar(plotbins, hist[0], width=0.8*binwidth)
    plt.title(titletext)
    plt.xlabel(xaxlabel)
    plt.ylabel('Frequency')
    stats = [np.nan, np.nan] if len(input) == 0 else CalcBootstrap(input, 1000)
    plt.text(0.9, 0.9, f'Mean: {stats[0]:.2f} +/- {stats[1]:.2f}',
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    fig.savefig(outname+'.pdf')
    plt.close()

def CumHistogramPlot(input, titletext, xaxlabel, outname):
    sortedvals = np.sort(input)
    xToPlot = []
    yToPlot = []
    if len(input) == 0:
        xToPlot = 0
        yToPlot = 0
    else:
        for i in range(len(sortedvals)):
            xToPlot.append(sortedvals[i])
            yToPlot.append(i+1)
    fig = plt.figure(figsize=A4)
    plt.plot(xToPlot, yToPlot)
    plt.title(titletext)
    plt.xlabel(xaxlabel)
    plt.ylabel('Frequency')
    stats = [np.nan, np.nan] if len(input) == 0 else CalcBootstrap(input, 1000)
    plt.text(0.9, 0.9, f'Mean: {stats[0]:.2f} +/- {stats[1]:.2f}',
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    fig.savefig(outname+'.pdf')
    plt.close()

def PlotIntensities(pdf, dataOrig, dataA, dataB, params, color, outfile):
    nonbgcorrints = []
    bgcorrintsOn = []
    bgcorrintsOff = []
    bgcorrintsAll = []
    for cell in params['retainedTraces']:
        if params['alignTracesCF'] == 0:
            start = dataB[cell].frameWindow[0]
            if np.sum(dataB[cell].r) == 0 and np.sum(dataB[cell].g) == 0: continue
        if params['alignTracesCF'] == 1:
            if params['color2align'] == 'red':
                if np.sum(dataB[cell].r) == 0: continue
                start = np.where(dataB[cell].r)[0][0]
            if params['color2align'] == 'green':
                if np.sum(dataB[cell].g) == 0: continue
                start = np.where(dataB[cell].g)[0][0]

        for val in range(start, dataB[cell].frameWindow[1]):
            nonbgcorrints.append(dataOrig[cell][color][val])
            bgcorrintsAll.append(dataA[cell][color][val])
            if dataB[cell][color][val] == 1:
                bgcorrintsOn.append(dataA[cell][color][val])
            elif dataB[cell][color][val] == 0:
                bgcorrintsOff.append(dataA[cell][color][val])

    np.save(outfile, bgcorrintsOn)
    histInts = np.histogram(nonbgcorrints, bins=100, density=True)
    histIntsbg = np.histogram(bgcorrintsAll, bins=100, density=True)
    histIntsbgOn = np.histogram(bgcorrintsOn, bins=histIntsbg[1], density=True)
    histIntsbgOff = np.histogram(bgcorrintsOff, bins=histIntsbg[1], density=True)
    binedges = histInts[1]
    binedgesbg = histIntsbg[1]
    plotbins = []
    plotbinsbg = []
    for i in range(len(binedges) - 1):
        plotbins.append(0.5 * (binedges[i + 1] + binedges[i]))
    for i in range(len(binedgesbg) - 1):
        plotbinsbg.append(0.5 * (binedgesbg[i + 1] + binedgesbg[i]))
    binwidth = plotbins[1] - plotbins[0]
    binwidthbg = plotbinsbg[1] - plotbinsbg[0]

    statsInts = CalcBootstrap(nonbgcorrints, 1000)
    statsIntsbg = CalcBootstrap(bgcorrintsAll, 1000)
    statsIntsbgOn = CalcBootstrap(bgcorrintsOn, 1000)
    statsIntsbgOff = CalcBootstrap(bgcorrintsOff, 1000)

    fig = plt.figure(figsize=A4)
    gs = GridSpec(1, 2, figure=fig)

    fig.add_subplot(gs[0,0])
    labelInts = 'Mean: '+str(round(statsInts[0],2))+'+/- '+str(round(statsInts[1], 2))
    plt.bar(np.asarray(plotbins), histInts[0], width=0.8 * binwidth, label = labelInts)
    plt.title('Histogram of non bg corrected intensity values ' + color)
    plt.xlabel('Intensity value (AU)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')

    fig.add_subplot(gs[0,1])
    labelIntsbg  = 'All; mean: '+str(round(statsIntsbg[0],2))+'+/- '+str(round(statsIntsbg[1], 2))
    labelIntsbgOn = 'On; mean: '+str(round(statsIntsbgOn[0],2))+'+/- '+str(round(statsIntsbgOn[1], 2))
    labelIntsbgOff = 'Off; mean: '+str(round(statsIntsbgOff[0],2))+'+/- '+str(round(statsIntsbgOff[1], 2))
    if len(bgcorrintsAll) != 0:
        plt.bar(np.asarray(plotbinsbg), histIntsbg[0], width=0.8 * binwidthbg, label = labelIntsbg, alpha = 0.3)
        plt.bar(np.asarray(plotbinsbg), histIntsbgOn[0]*(len(bgcorrintsOn)*1./len(bgcorrintsAll)), width=0.8 * binwidthbg, alpha = 0.6, label =labelIntsbgOn)
        plt.bar(np.asarray(plotbinsbg), histIntsbgOff[0]*(len(bgcorrintsOff)*1./len(bgcorrintsAll)), width=0.8 * binwidthbg, alpha = 0.6, label=labelIntsbgOff)

    plt.title('Histogram of bg corrected intensity values ' + color)
    plt.xlabel('Intensity value (AU)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')

    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

def PlotDistances(pdf, dataA, retainedTraces):
    distAll = []
    distOn = []
    for cell in retainedTraces:
        data = dataA[cell]
        minframe = dataA[cell].frameWindow[0]
        maxframe = dataA[cell].frameWindow[1]
        xyRed = data.trk_r[:,:2]
        xyGreen = data.trk_g[:,:2]
        data.spotFoundBoth = list(set(np.where(data.trk_r[:,-1]>0)[0])& set( np.where(data.trk_g[:,-1]>0)[0]))
        data.spotFoundBoth = [val for val in data.spotFoundBoth if val >= minframe and val <= maxframe]
        data.spotFoundBoth.sort()
        dist0 = [0] * len(xyRed)
        for a in range(minframe,maxframe+1):
            dist0[a] = (((xyRed[a,0]-xyGreen[a,0])**2 + (xyRed[a,1]-xyGreen[a,1])**2)**0.5)
        distAll.append(dist0)
        dist0 = [0] * len(xyRed)
        for a in data.spotFoundBoth:
            dist0[a] = (((xyRed[a,0]-xyGreen[a,0])**2 + (xyRed[a,1]-xyGreen[a,1])**2)**0.5)
        distOn.append(dist0)
    
    distAll = [l for lst in distAll for l in lst] #flatten multidimensional list to 1D
    distOn = [l for lst in distOn for l in lst]
    distAll = list(filter(lambda a: a != 0, distAll)) #remove zeroes
    distOn = list(filter(lambda a: a != 0, distOn))

    histdistAll = np.histogram(distAll, bins=100, density=True)
    histdistOn = np.histogram(distOn, bins=histdistAll[1], density=True)
    binedges = histdistAll[1]
    plotbins = []
    for i in range(len(binedges) - 1):
        plotbins.append(0.5 * (binedges[i + 1] + binedges[i]))
    binwidth = plotbins[1] - plotbins[0]
    
    statsdistAll = CalcBootstrap(distAll, 1000)
    statsdistOn = CalcBootstrap(distOn, 1000)
    
    fig = plt.figure(figsize=(5,6))
    gs = GridSpec(1, 1, figure=fig)
    
    fig.add_subplot(gs[0,0])
    labeldistAll  = 'All; mean: '+str(round(statsdistAll[0],2))+'+/- '+str(round(statsdistAll[1], 2))
    labeldistOn = 'On; mean: '+str(round(statsdistOn[0],2))+'+/- '+str(round(statsdistOn[1], 2))
    plt.bar(np.asarray(plotbins), histdistAll[0], width=0.8 * binwidth, label = labeldistAll, alpha = 0.3)
    if len(distAll):
        plt.bar(np.asarray(plotbins), histdistOn[0]*(len(distOn)*1./len(distAll)), width=0.8 * binwidth, alpha = 0.6, label =labeldistOn)
    
    plt.title('Histogram of distances between alleles ')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')
    
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

def corrACFAmplToIndPlot(dataA, dataB, col, params):
    amplAfterInd = []
    indTimes = []
    nrcells = len(dataA)
    for i in range(nrcells):
        # read data, digital data and framewindow
        dataCell = dataA[i]
        dataDig = dataB[i]
        fW = dataCell.frameWindow

        # get induction frame of this cell
        if col == 'red':
            bindata = dataDig.r
        elif col == 'green':
            bindata = dataDig.g
        if sum(bindata) != 0:
            indframe = np.where(bindata > 0)[0][0]
        else:
            indframe = fW[1]

        if indframe < fW[0]: indframe = fW[0]

        if indframe > fW[1] - 2: continue

        CFAfterInd = daf.compG_multiTau(np.c_[dataA[i].r, dataA[i].g][indframe:fW[1]].T, dataA[i].t[indframe:fW[1]], 8)[0][0]
        if col == 'red':
            ACFAfterInd = CFAfterInd[0]
        elif col == 'green':
            ACFAfterInd = CFAfterInd[1]

        ampl = ACFAfterInd[1]
        amplAfterInd.append(ampl)
        indTimes.append(indframe * dataCell.dt)

    fig = plt.figure()
    plt.scatter(indTimes, amplAfterInd)
    plt.xlabel('Induction time (s)')
    plt.ylabel('Amplitude ACF (AU)')
    plt.title('Correlation of ACF amplitude with induction time')
    fig.savefig(params['file'] + '_correlation_induction_time_ACF_amplitude_' + col + '.pdf')
    plt.close()
