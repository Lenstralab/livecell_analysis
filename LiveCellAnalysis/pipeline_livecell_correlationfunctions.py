#!/usr/local/bin/ipython3 -i

import sys
import os
import psutil
import numpy as np
import copy as copy2
import yaml
import shutil
from scipy.optimize import curve_fit
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tllab_common.wimread import imread as imr


if __package__ is None or __package__ == '': #usual case
    import misc
    import fluctuationAnalysis as FA
    import dataAnalysis_functions as daf
    import plot_figures
else: #in case you do from another package:
    from . import misc
    from . import fluctuationAnalysis as FA
    from . import dataAnalysis_functions as daf
    from . import plot_figures

### if you want to execute this pipeline from a different file than pipeline_livecell_correlationfunctions_parameters, first run: sys.argv = ['pipeline_livecell_correlationfunctions.py','yourfilename'], which should be in the same folder, and then run: execfile("pipeline_livecell_correlationfunctions.py")

A4 = (11.69, 8.27)

if not '__file__' in locals():  # when executed using execfile
    import inspect
    __file__ = inspect.getframeinfo(inspect.currentframe()).filename


# This script is meant to calculate autocorrelations on a dataset. The dataset is specified in a list.py file (and can be a combination of multiple experiments).

#### specify input list.py and outputfolder

def get_paths(params, parameter_file=None):
    if not parameter_file == '':
        parameter_file = os.path.abspath(parameter_file)
        if not os.path.isabs(params['outputfolder']):
            params['outputfolder'] = os.path.abspath(os.path.join(os.path.split(parameter_file)[0], params['outputfolder']))
        if not os.path.isabs(params['PyFile']):
            params['PyFile'] = os.path.abspath(os.path.join(os.path.split(parameter_file)[0], params['PyFile']))

    params['outputfolder'] = os.path.join(params['outputfolder'], '')
    if not os.path.exists(params['outputfolder']):
        os.makedirs(params['outputfolder'])

    params['file'] = os.path.join(params['outputfolder'], os.path.split(params['PyFile'])[1].replace('.list.py', ''))

    dateTimeObj = datetime.now()
    dateTime = dateTimeObj.strftime("%d-%b-%Y_%Hh%Mm%Ss")

    if os.path.exists(parameter_file):
        shutil.copyfile(parameter_file,
            os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions_parameters_runtime_" + dateTime + ".yml"))
    else:
        parameter_file = os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions_parameters_runtime_" + dateTime + ".yml")
        with open(parameter_file, 'w') as f:
            yaml.dump(params, f)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions.py"))


#################
#################
### MAIN FUNCTIONS

######################################
### Process analog data
######################################

def bg_sub_traces(dataOrig, params):
    # Copy to new variable dataA: Analog data, processed
    dataA = copy2.deepcopy(dataOrig)

    # Scale traces to correct for day-to-day effects if necessary
    if not params.get('scaleFactors') is None and len(params['scaleFactors']) == len(dataA):
        for i in range(len(dataA)): dataA[i].r = dataA[i].r * params['scaleFactors'][i];
        for i in range(len(dataA)): dataA[i].g = dataA[i].g * params['scaleFactors'][i];

    for channel in params['ChannelsToAnalyze']:
        color = ('Red', 'Green')[channel]

        bgWindowColor = 'bgWindow{}'.format(color)
        c = color.lower()[0]
        trk_c = 'trk_'+c
        bgc = 'bg'+c
        mc = 'm'+c
        sdc = 'sd'+c

        # background subtract traces
        for i, dA in enumerate(dataA):
            if channel in params.get('quantizeTrace', {}).keys():
                dA[bgc] = np.zeros(dA[c].shape)
                dA[mc] = 0
                dA[sdc] = 1
            else:
                name = dA.name.split(".txt")[0]

                # load digital data
                # use background for n spots at fixed distance from TS
                if os.path.exists('{}_bg1_{}.txt'.format(dA.name, color.lower())):
                    bg = []
                    j = 1
                    found = True
                    while found:
                        bgtrk = np.loadtxt('{}_bg{}_{}.txt'.format(dA.name, j, color.lower()))
                        bg.extend(bgtrk[:,2])
                        j += 1
                        found = os.path.exists('{}_bg{}_{}.txt'.format(dA.name, j, color.lower()))
                    dA[bgc] = bg
                # define background manually in parameter file
                elif i in params[bgWindowColor]:
                    bg = range(int(params[bgWindowColor][i][0]/dA.dt),
                               int(params[bgWindowColor][i][1]/dA.dt))
                    dA[bgc] = dA[c][bg]
                # define background as part of trace were no strong spots were found (high threshold)
                elif dA[trk_c].shape[1]>4: #only do background subtraction when there actually is a 5th column with (0, 0.5, 1)'s
                    bg = np.where(dA[trk_c][:,-1] < 1)
                    dA[bgc] = dA[c][bg]
                else: # no background subtraction, for example when using orbital tracking data
                    dA[bgc] = np.zeros(dA[c].shape)

                # log-normal fit
                if np.any(np.asarray(dA[bgc]) > 0):
                    m, sd = misc.distfit(np.log(dA[bgc]))
                else:
                    m, sd = 0, 0

                # replaced by fitting with the cdf in the lines above (wp)
                # x, y = misc.histoF(np.histogram(dA[bgc], bins=np.r_[0:10:.1]*np.median(dA[bgc]), density=True)).T
                # x -= np.diff(x)[0]/2
                # m, sd = optimize.fmin(lambda a: sum((y-((1/(x*a[1]*(2*np.pi)**0.5))
                #                             * np.exp(-(((np.log(x)-a[0])**2)/(2*a[1]**2)))))**2), np.r_[10000,5000], disp=0)

                dA[mc] = m
                dA[sdc] = sd
                expVal = np.exp(dA[mc])
                dA[c] -= expVal

                # use this code if you want to set all the data outside the defined frameWindow to 0
                #dA[c][:dA.frameWindow[0]]=0.; dA[c][dA.frameWindow[1]:]=0.

                if 0 in params['ChannelsToAnalyze']:
                    np.savetxt(name + "_bg_sub_red.txt", dA.r)
                else:
                    dA.r = dA.g
                if 1 in params['ChannelsToAnalyze']:
                    np.savetxt(name + "_bg_sub_green.txt", dA.g)
                else:
                    dA.g = dA.r

#            # write PDF of trace with histogram and digital data trace
#            histFig = write_hist(None, "r", dA, sdThresholdRed*dA.sdr)
#            histFig.savefig(name+"_bg_sub_trace.pdf")
#            np.savetxt(name+"_bg_sub.txt",dA.g)
#            plt.close()
    return dataA


def binary_call_traces(dataA, params):
    """#############################################################
    ### Threshold data based on sdThreshold to make binary data
    #############################################################"""
    dataB = copy2.deepcopy(dataA)
    if params.get('binaryCallMethod', '').lower() == 'markov_ensemble':
        print("Calculating binary data using a hidden Markov model")
        for channel in params['ChannelsToAnalyze']:
            color = 'rgb'[channel]
            mbg = np.mean([np.exp(dA['m'+color]) for dA in dataA])
            sdbg = np.sqrt(np.sum([np.exp(dA['m'+color]) ** 2 * dA['sd'+color]**2 for dA in dataA])) / len(dataA)
            h = daf.hmm(dataA, color, (mbg, params['sdThreshold' + {'r': 'Red', 'g': 'Green'}[color]] * sdbg))
            for dA, dB in zip(dataA, dataB):
                dB[color] = h(dA[color]) * FA.get_mask(dA.t, dA.frameWindow)
    elif params.get('binaryCallMethod', '').lower() == 'markov_individual':
        print("Calculating binary data using hidden Markov models")
        for channel in params['ChannelsToAnalyze']:
            color = 'rgb'[channel]
            for dA, dB in zip(dataA, dataB):
                h = daf.hmm(dA, color, (np.exp(dA['m'+color]),
                    params['sdThreshold' + {'r': 'Red', 'g': 'Green'}[color]] * np.exp(dA['m'+color]) * dA['sd'+color]))
                dB[color] = h(dA[color]) * FA.get_mask(dA.t, dA.frameWindow)
    else:
        print("Calculating binary data using thresholds")
        names = []
        threshCell = []
        for dA, dB in zip(dataA, dataB):
            if 0 in params['ChannelsToAnalyze']:
            # threshold with sdTreshold above SD-background of trace
                sdbg = ((np.exp(2*dA.mr+dA.sdr**2))*(np.exp(dA.sdr**2)-1))**0.5
                threshCell.append(sdbg * params['sdThresholdRed'])
                names.append(dA.name)
                dB.r = dA.r > (sdbg * params['sdThresholdRed'])  # red binary
                # set all data outside define frameWindow to 0
                dB.r[0:dB.frameWindow[0]] = 0; dB.r[dB.frameWindow[1]: len(dB.r)+1] = 0
            if 1 not in params['ChannelsToAnalyze']:
                    dB.g = dB.r
            if 1 in params['ChannelsToAnalyze']:
            # threshold with sdTreshold above SD-background of trace
                sdbg = ((np.exp(2 * dA.mg + dA.sdg ** 2)) * (np.exp(dA.sdg ** 2) - 1)) ** 0.5
                threshCell.append(sdbg * params['sdThresholdGreen'])
                dB.g = dA.g > (sdbg * params['sdThresholdGreen'])  # green binary
                # set all data outside define frameWindow to 0
                dB.g[0:dB.frameWindow[0]] = 0; dB.g[dB.frameWindow[1]: len(dB.g)+1] = 0
            if 0 not in params['ChannelsToAnalyze']:
                     dB.r = dB.g
        with open(params['file'] + '_thresholds_per_cell_with_cell_label.txt', 'w') as f:
            f.write('\n'.join([f'{name},{thres}' for name, thres in zip(names, threshCell)]))

        for channel in params['ChannelsToAnalyze']:
            color = 'rg'[channel]

            for cell in range(len(dataB)):
                for i in range(len(dataB[cell][color])-2):
                    #remove one-frame gaps:
                    if params['Remove1FrameGaps'] == 1:
                        if dataB[cell][color][i] == 1 and dataB[cell][color][i+1] == 0 and dataB[cell][color][i+2] == 1:
                            dataB[cell][color][i+1] = 1

            for cell in range(len(dataB)):
                for i in range(len(dataB[cell][color])-2):
                    #remove one-frame jumps:
                    if params['Remove1FrameJumps'] == 1:
                        if dataB[cell][color][i] == 0 and dataB[cell][color][i+1] == 1 and dataB[cell][color][i+2] == 0:
                            dataB[cell][color][i+1] = 0
    return dataB

###################################
### code to select for specific traces, for example to select for traces that have a burst, or make a mask based on the distance of two transcripion sites
###################################

def filter_traces(dataA, dataB, params):
    # filter traces for tracesToExclude and traces that do not have a first burst
    if params['alignTracesCF'] == 1 and params['color2align'] == "red":
        params['retainedTraces'] = [i for i in np.r_[:len(dataB)] if dataB[i].r.sum() > 0 and i not in params['tracesToExclude']]
    elif params['alignTracesCF'] == 1 and params['color2align'] == "green":
        params['retainedTraces'] = [i for i in np.r_[:len(dataB)] if dataB[i].g.sum() > 0 and i not in params['tracesToExclude']]
    elif params['alignTracesCF'] == 0:
        params['retainedTraces']=[i for i in np.r_[:len(dataA)] if i not in params['tracesToExclude'] ]

    ##### Make mask based on distance distance
    if params['selectOnDistance']:
        if len(params['ChannelsToAnalyze']) == 2 :
            for data in dataA:
                xyRed = data.trk_r[:, :2]
                xyGreen = data.trk_g[:, :2]
                data.spotFoundBoth = list(
                    set(np.where(data.trk_r[:, -1] > 0)[0]) & set(np.where(data.trk_g[:, -1] > 0)[0]))

                dist = [np.NaN] * len(xyRed)
                distinv = [0] * len(xyRed)
                dist0 = [0] * len(xyRed)
                for a in range(len(xyRed)):
                    #          for a in data.spotFoundBoth:
                    dist[a] = dist0[a] = (
                                ((xyRed[a, 0] - xyGreen[a, 0]) ** 2 + (xyRed[a, 1] - xyGreen[a, 1]) ** 2) ** 0.5)
                    distinv[a] = 1 / (
                    (((xyRed[a, 0] - xyGreen[a, 0]) ** 2 + (xyRed[a, 1] - xyGreen[a, 1]) ** 2) ** 0.5))
                data.xydist = dist
                data.xydistinv = distinv
                data.distmask = (np.r_[data.xydist] < 3) * 1.
                data.distmask[:data.frameWindow[0]] = 0
                data.distmask[data.frameWindow[1]:] = 0
            #    data.distmask = (np.r_[data.xydist] < 4)*1.

    toInclude = []
    for i in params['retainedTraces']:
        dA = dataA[i]
        dB = dataB[i]

        if params['alignTracesCF'] == 0:
            start = dA.frameWindow[0]
            if np.sum(dB.r) == 0 and np.sum(dB.g) == 0:
                print(misc.color(f'Rejected {os.path.basename(dA.rawPath)} because red and green = 0', 208))
                continue
        else:
            if params['color2align'] == 'red':
                if np.sum(dB.r) == 0:
                    print(misc.color(f'Rejected {os.path.basename(dA.rawPath)} because red = 0', 208))
                    continue
                start = np.where(dB.r)[0][0]
            if params['color2align'] == 'green':
                if np.sum(dB.g) == 0:
                    print(misc.color(f'Rejected {os.path.basename(dA.rawPath)} because green = 0', 208))
                    continue
                start = np.where(dB.g)[0][0]
        if dA.frameWindow[-1] - start < params['minLengthCF']:
            print(misc.color(f'Rejected {os.path.basename(dA.rawPath)} because {dA.frameWindow[-1] - start} <\
                             {params["minLengthCF"]}', 208))
            continue
        toInclude.append(i)

    params['retainedTraces'] = toInclude
    return dataA, params


def calc_timefraction_on(dataB, params):
    """#####################################################################
    #### calculate time of being in on state, per cell (histogram) and overall
    ######################################################################"""
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]

        framesOn = []
        framesOff = []
        fracOn = []

        for i in params['retainedTraces']:
            if params['alignTracesCF'] == 0:
                start = dataB[i].frameWindow[0]
            elif np.sum(dataB[i][params['color2align'][0]]) != 0:
                start = np.where(dataB[i][params['color2align'][0]])[0][0]
            else:
                start = 0
            digidata = dataB[i][color][start:dataB[i].frameWindow[-1]]

            if len(digidata):
                framesOn.append(np.sum(digidata))
                framesOff.append(len(digidata)-np.sum(digidata))
                fracOn.append(float(np.sum(digidata))/float(len(digidata)))

        totalOn = np.sum(framesOn)
        totalOff = np.sum(framesOff)

        print('Frames on: '+str(totalOn)+' total frames: '+str(totalOff+totalOn))
        np.savetxt(params['file'] + "_framesOn_framesOff_" + col + ".txt", [totalOn, totalOff])
        plot_figures.HistogramPlot(fracOn, 20, 'Histogram of fraction of frames in on-state per cell ' + col,
                            'Fraction frames in on-state', params['file'] + '_histogram_fraction_on_per_cell_' + col)


def make_plots_traces(dataOrig, dataA, dataB, params):
    """#####################################################################
    #### write PDFs of binary data, shows red and green binary data on top
    ######################################################################"""

    print("Plotting histograms for background subtraction")
    for i in range(len(dataA)):
        dA = dataA[i]
        # cellnr = int(dA.name.split("_trk_results")[0][-1])-1
        name = dA.name.split(".txt")[0]
        with PdfPages(name+"_bg_sub_trace.pdf") as pdfTrace:
            if 0 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2*dA.mr+dA.sdr**2))*(np.exp(dA.sdr**2)-1))**0.5
                plot_figures.write_hist(pdfTrace, "r", dA, params['sdThresholdRed']*sdbg)
                plt.close()
                if len(dA.bgr) == 4* len(dA.t):
                    plot_figures.showBackgroundTrace(pdfTrace, dA,"r", params['sdThresholdRed']*dA.sdr)
                    plt.close()
            if 1 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2 * dA.mg + dA.sdg ** 2)) * (np.exp(dA.sdg ** 2) - 1)) ** 0.5
                plot_figures.write_hist(pdfTrace, "g", dA, params['sdThresholdGreen']*sdbg)
                plt.close()
                if len(dA.bgg) == 4* len(dA.t):
                    plot_figures.showBackgroundTrace(pdfTrace, dA,"g", params['sdThresholdGreen']*dA.sdg)
                    plt.close()
            plot_figures.showBinaryCall(pdfTrace, dA, dataB[i])
            plt.close()


    #####################################################################
    #### write PDFs of individual auto- and cross correlations as linegraphs
    ######################################################################

    # print("Plotting individual autocorrelations")
    # with PdfPages(params['file']+"_individual_correlation_functions.pdf") as pdfTrace:
    #     plot_figures.showCorrelFunAll(pdfTrace, dataA, params['ChannelsToAnalyze'], params)

    ###########################################################################
    # plot PDFs of individual auto- and cross correlations as heatmaps
    ###########################################################################

    print("Plotting heatmap correlation functions")

    nbPts1 = min([d.G[1, 0].shape[0] for d in dataA]) # determine minimum number of point in crosscorrelation gr
    nbPts2 = min([d.G[0, 1].shape[0] for d in dataA]) # determine minimum number of point in crosscorrelation rg

    # order heatmap correlationfunctions
    if params['OrderCFHeatmap'] == 'maxCC':
        if params['maxCCMethod'] == 'max':
            maxVal = [np.where(np.hstack((dataA[i].G[1, 0][::-1][-nbPts1:-1], dataA[i].G[0, 1][:nbPts2])) == max(np.hstack((dataA[i].G[1, 0][::-1][-nbPts1:-1], dataA[i].G[0, 1][:nbPts2]))))[0][0] for i in params['retainedTraces']]
            sortedIds = [x for _, x in sorted(zip(maxVal,params['retainedTraces']))]
            sortedIds = np.asarray(sortedIds)

        elif params['maxCCMethod'] == 'gaussian':  ### first takes out data that has no positive peak, then sort remaining on peak by gaussian fitting
            def gauss_function(x,a,x0,sigma,b):
                return a*np.exp(-(x-x0)**2/(2*sigma**2))+b
        
            maxVal2 = []
            amp = []
            for i in range(len(dataA)):
                # fit data to gaussion function
                x = np.hstack((-dataA[i].tau[::-1][-nbPts1:-1], dataA[i].tau[:nbPts2]))   #the number of data
                y = np.hstack((dataA[i].G[1, 0][::-1][-nbPts1:-1], dataA[i].G[0, 1][:nbPts2]))
                n = nbPts1 + nbPts2 -1
                # calculate mean and standard deviaiton
                mean = sum(x*y)/n
                sigma = abs((sum(y*(x-mean)**2)/n))**0.5
                popt, pcov = curve_fit(gauss_function, x, y, sigma = None, p0 = [0.5, mean, sigma, 0] )
                maxVal2.append(popt[1])
                amp.append(popt[0])
            NoPeak = np.where(np.r_[amp] < 0)[0]
            Peak =  np.where(np.r_[amp] >= 0)[0]
            for x in params['retainedTraces']:
                if x in Peak:
                   sortedIds2 = [x for _, x in sorted(zip([maxVal2[x] for x in range(len(maxVal2)) if x in Peak],params['retainedTraces']))]
            sortedIds = np.r_[NoPeak, sortedIds2]
            sortedIds = np.asarray(sortedIds)

        heatmapCF = plot_figures.showHeatMapCF(dataA, 3, 3, 3, sortedIds= sortedIds, Z= None, Normalize = False)
        heatmapCF.savefig(params['file']+"_Heatmap_correlation_functions.pdf")
        plt.close()
        
    elif params['OrderCFHeatmap'] == 'ACred':
        maxVal = []
        for i in range(len(dataA)):
            if np.sum(dataA[i].G) == 0: continue
            if len(dataA[i].G[0, 0]) <= 1: continue
            maxVal.append(dataA[i].G[0, 0][1])
        sortedIds2 = np.flip(np.argsort(maxVal))
        sortedIds = []
        for x in sortedIds2:
            if x in params['retainedTraces']:
                sortedIds.append(x)

    elif params['OrderCFHeatmap'] == 'ACgreen':
        maxVal = []
        for i in range(len(dataA)):
            if np.sum(dataA[i].G) == 0: continue
            maxVal.append(dataA[i].G[1, 1][1])
        sortedIds2 = np.flip(np.argsort(maxVal))
        sortedIds = []
        for x in sortedIds2:
            if x in params['retainedTraces']:
                sortedIds.append(x)

    elif params.get('OrderCFHeatmap') is None:
        sortedIds = range(len(dataA))

    else:
        sortedIds = range(len(dataA))

    heatmapCF = plot_figures.showHeatMapCF(dataA, 3, 3, 3, sortedIds=sortedIds, Z=None, Normalize=False)
    heatmapCF.savefig(params['file'] + "_Heatmap_correlation_functions.pdf")
    plt.close()

    #####################################################################
    #### make figure heatmap
    ######################################################################
    
    print("Plotting trace heatmaps")

    dataAtrim=copy2.deepcopy(dataA)
    dataBtrim=copy2.deepcopy(dataB)
    outsideFW=copy2.deepcopy(dataA)
    for i in range(len(dataAtrim)):
        dA=dataAtrim[i]
        dA.g[:dA.frameWindow[0]]=0.; dA.g[dA.frameWindow[1]:]=0.
        dA.r[:dA.frameWindow[0]]=0.; dA.r[dA.frameWindow[1]:]=0.
        
        dB=dataBtrim[i]
        dB.g[:dB.frameWindow[0]]=0.; dB.g[dB.frameWindow[1]:]=0.
        dB.r[:dB.frameWindow[0]]=0.; dB.r[dB.frameWindow[1]:]=0.
        
        dFW=outsideFW[i]
        dFW.g[:dFW.frameWindow[0]]=1.; dFW.g[dFW.frameWindow[1]:]=0.3
        dFW.g[dFW.frameWindow[0]:dFW.frameWindow[1]]=0.

    if params['trimHeatmaps'] == 1:
        dataHeatmapAn = dataAtrim
        dataHeatmapDig = dataBtrim
    else:
        dataHeatmapAn = dataA
        dataHeatmapDig = dataB

    if params['OrderTraceHeatmap'] == "CF":
        sortedIdsTrace = sortedIds
    else:
        sortedIdsTrace = params['retainedTraces']

    # write heatmaps of analog and digital data
    if params['trimHeatmaps'] == 1:
        heatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed = params['heatMapScalingRed'], maxGreen = params['heatMapScalingGreen'],trimdata = outsideFW, sortedIds = sortedIdsTrace)
        heatmapA.savefig(fname = params['file']+"_Heatmap_analog.pdf")
        plt.close()
    else:
        heatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed = params['heatMapScalingRed'], maxGreen = params['heatMapScalingGreen'], sortedIds = sortedIdsTrace)
        heatmapA.savefig(fname = params['file']+"_Heatmap_analog.pdf")
        plt.close()

    if params['trimHeatmaps'] == 1:
        heatmapB = plot_figures.showHeatMap(dataHeatmapDig,trimdata = outsideFW, sortedIds = sortedIdsTrace)
        heatmapB.savefig(fname = params['file']+"_Heatmap_digital.pdf")
        plt.close()
    else:
        heatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds = sortedIdsTrace)
        heatmapB.savefig(fname = params['file']+"_Heatmap_digital.pdf")
        plt.close()

    if params['CalcIndTimes'] == 1:
        print("Plotting ordered trace heatmaps")
        for channel in params['ChannelsToAnalyze']:
            if channel == 0:
                col = "red"
            elif channel == 1:
                col = "green"

            indtimes = []
            names = []
            for cell in params['retainedTraces']:
                if col == "red": bindata = dataB[cell].r
                if col == "green": bindata = dataB[cell].g
                if sum(bindata) == 0: continue
                indframe = np.where(bindata > 0)[0][0]
                indtime = (dataB[cell].dt) * float(indframe) / 60
                indtimes.append(indtime)
                names.append(dataB[cell].name)
            np.savetxt(params['file'] + "_induction_times_" + col + ".txt", indtimes)
            np.savetxt(params['file'] + "_induction_times_" + col + "_with_cell_label.txt",
                       np.transpose([names, indtimes]), delimiter=",", fmt="%s")

            plot_figures.HistogramPlot(indtimes, 20, 'Histogram of induction times ' + col, 'Induction time (min)',
                          params['file'] + '_histogram_induction_times_' + col)
            plot_figures.CumHistogramPlot(indtimes, 'Cumulative distribution of induction times ' + col, 'Induction time (min)',
                             params['file'] + '_cumulative_distribution_induction_times_' + col)

            if params['SortHeatMapIndTimes'] == col:
                indtimessortedRetainedTraces = np.flip(np.argsort(indtimes))
                indtimessorted = [params['retainedTraces'][i] for i in indtimessortedRetainedTraces]
            else:
                indtimessorted = params['retainedTraces']

        # making heatmaps sorted by induction time
        if params['SortHeatMapIndTimes'] != None:
            if params['trimHeatmaps'] == 1:
                sortedheatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed=params['heatMapScalingRed'],
                                                          maxGreen=params['heatMapScalingGreen'], sortedIds=indtimessorted,
                                                          trimdata=outsideFW)
            else:
                sortedheatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed=params['heatMapScalingRed'],
                                                          maxGreen=params['heatMapScalingGreen'], sortedIds=indtimessorted)
            sortedheatmapA.savefig(fname=params['file'] + "_Heatmap_analog_sorted_by_induction.pdf")
            plt.close()

        if params['trimHeatmaps'] == 1:
            sortedheatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds=indtimessorted, trimdata=outsideFW)
        else:
            sortedheatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds=indtimessorted)
            
        sortedheatmapB.savefig(fname=params['file'] + "_Heatmap_digital_sorted_by_induction.pdf")
        plt.close()


    #####################################################################
    #### Plot area under traces, useful to see if some traces dominate
    ######################################################################
   
    print("Plotting area under traces")
    if 0 in params['ChannelsToAnalyze']:
        figAvgR = plot_figures.showAreaUnderTraces(dataA, params['retainedTraces'], "r")
        figAvgR.savefig(params['file']+"_AreaUnderTraces_red.pdf")
        plt.close()
    if 1 in params['ChannelsToAnalyze']:
        figAvgG = plot_figures.showAreaUnderTraces(dataA, params['retainedTraces'], "g")
        figAvgG.savefig(params['file']+"_AreaUnderTraces_green.pdf")
        plt.close()

    #####################################################################
    #### Making plot of all non-background corrected intensities
    ######################################################################

    print("Plotting intensity distribution")
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]
        figInt = plot_figures.PlotIntensities(None, dataOrig, dataA, dataB, params, color, params['file'] +
                                              '_intensities_frames_on.npy')
        figInt.savefig(params['file'] + '_histogram_intensity_values_' + col + '.pdf')
        plt.close()

    if len(params['ChannelsToAnalyze']) == 2:
        print("Plotting distance distribution")
        figDist = plot_figures.PlotDistances(None, dataA, params['retainedTraces'])
        figDist.savefig(params['file'] + '_histogram_distances' + '.pdf')
        plt.close()

#####################################################################
#### compute auto and cross correlation.
######################################################################

def calculateCF(dataA, dataB, params):
    #   compute auto and cross correlation. For full options check Fluctuation_analysis.py script
    #   mask defines area of trace that correlation functions are calculated for.
    #   methAvg can be 'arithm' or 'harmo' for arthimetic or harmonic avarege calculation of the correlation functions
    #   to correct for non-steady state effects during induction, you need to align the traces on the start of the first burst. No need for alignment if data was taken at 'steady state'.

    print('Calculating correlation functions, including {} traces'.format(len(params['retainedTraces'])))

    if params['alignTracesCF']:  # for aligning traces on first green burst, will complain if no first burst in a trace
        t0 = [dB.t[np.where(dB[params['color2align'].lower()[0]])[0][0]] for dB in [dataB[i] for i in params['retainedTraces']]]
        fname = params['file'] + "_correlation_functions_aligned_startBurst_" + params['color2align'] + ".pdf"
    else:  # for auto/cross correlation from first until last frameWindow, without alignment
        t0 = None
        fname = params['file']+"_correlation_functions.pdf"

    ss = []
    for dA in [dataA[i] for i in params['retainedTraces']]:
        if 'r_orig' in dA or 'g_orig' in dA:
            v_orig = np.vstack([dA[c + '_orig'] for c in 'rgb' if c + '_orig' in dA])
        else:
            v_orig = None
        if params.get('selectOnDistance', False):
            ss.append(FA.mcSig(dA.t, np.vstack((dA.r, dA.g)), mask=dA.distmask, v_orig=v_orig))
        else:
            ss.append(FA.mcSig(dA.t, np.vstack((dA.r, dA.g)), frameWindow=dA.frameWindow, v_orig=v_orig))

    ss = FA.mcSigSet(ss)
    ss.alignSignals(t0)
    if params.get('ccfFunction') is not None and params['ccfFunction'].lower() in ('linefit', 'sc'):
        ss.compAvgCF_SC(mT=params.get('mTauOrder'), fitWindow=params.get('fitWindow'))
    else:
        ss.compAvgCF(mT=params.get('mTauOrder'))
    ss.bootstrap(nBs=params.get('nBs'))

    with PdfPages(fname) as pdf:
        plot_figures.showCorrFun(pdf, ss)

    """ ---- Explanation of results from ss ---
        ss.G = correlation functions: ss.G[0,0] = green, ss.G[1,1] = red, ss.G[0,1] = red to green, ss.G[1,0] = green to red
        ss.P = corrected correlation functions. Ordering the same
        ss.N = correlation function of average trace. Ordering the same
        *r, *g: correlation functions normalized wrt. red or green signal
        d*: error on correlation function
    """

    # display average trace and individual background subtracted traces
    print("Plotting average trace")
    with PdfPages(params['file']+"_average_trace.pdf") as pdf:
        plot_figures.showAvTrace(pdf, ss, names=[dataA[i].name for i in params['retainedTraces']])
        plt.close()

    # write average trace to file
    if 0 in params['ChannelsToAnalyze']:
        np.savetxt(params['file']+'_average_trace_red.txt', np.vstack((ss.t, ss.v[0], ss.dv[0])).T)
    if 1 in params['ChannelsToAnalyze']:
        np.savetxt(params['file'] + '_average_trace_green.txt', np.vstack((ss.t, ss.v[1], ss.dv[1])).T)
    return ss

#####################################################################
#### fit and display autocorrelation functions
######################################################################

def make_plots_CF(ss, params):
    print("Plotting individual autocorrelations")
    with PdfPages(params['file']+"_individual_correlation_functions.pdf") as pdfTrace:
        plot_figures.showCorrelFunAll(pdfTrace, ss.sigsAlign, params['ChannelsToAnalyze'], params)

    print("Fitting and plotting correlation functions")
    ### calculate fit for (non-) corrected red autocorrelation. The autocorrelation can be shifted up or down before the fit by changing the y the shiftACFRed parameter.
    for channel in params['ChannelsToAnalyze']:
        for correctCF in (False, True):
            Color = ['Red', 'Green'][channel]
            color = Color.lower()
            col = color[0]
            fitFramesColor = 'fitFrames' + Color
            shiftACFColor = 'shiftACF' + Color

            n1 = params[fitFramesColor][0]  # start frame of data for fitting
            n2 = params[fitFramesColor][1]  # end frame of data for fitting

            if correctCF:
                G = (ss.P.copy(), ss.Pr.copy(), ss.Pg.copy())
                dG = (ss.dP.copy(), ss.dPr.copy(), ss.dPg.copy())
            else:
                G = (ss.G.copy(), ss.Gr.copy(), ss.Gg.copy())
                dG = (ss.dG.copy(), ss.dGr.copy(), ss.dGg.copy())
                
            G = [g[channel, channel] for g in G]
            dG = [g[channel, channel] for g in dG]
            G[0] += params[shiftACFColor]

            with PdfPages('{}_autocorrelation_fit_{}_{}corrected.pdf'.format(params['file'], color, (1-correctCF)*'un')) as pdf, \
                PdfPages('{}_autocorrelation_fit_{}_{}corrected_zoom.pdf'.format(params['file'], color, (1-correctCF)*'un')) as zoompdf:
                for i, (g, dg, t) in enumerate(zip(G, dG, ('G', 'Normalized r', 'Normalized g'))):
                    fitp, dfitp = misc.fit_line(ss.tau[n1:n2], g[n1:n2], dg[n1:n2])
                    plot_figures.showAutoCorr(pdf, col, t, ss.tau, g, dg, fitp, dfitp)
                    plot_figures.showAutoCorr(zoompdf, col, t, ss.tau, g, dg, fitp, dfitp, params['ACFxmax'])
                    if i==0:
                        np.savetxt('{}_autocorrelation_{}_{}corrected.txt'.format(params['file'], color,
                            (1-correctCF)*'un'), np.vstack((ss.tau, g, dg)).T, delimiter = "\t", fmt = "%1.5f")

    if len(params['ChannelsToAnalyze']) == 2:
        ### calculate fit for (non-)corrected cross-correlation
        for correctCF in (False, True):
            if correctCF:
                Gt = [ss.P.copy(), ss.Pr.copy(), ss.Pg.copy()]
                dGt = [ss.dP.copy(), ss.dPr.copy(), ss.dPg.copy()]
            else:
                Gt = [ss.G.copy(), ss.Gr.copy(), ss.Gg.copy()]
                dGt = [ss.dG.copy(), ss.dGr.copy(), ss.dGg.copy()]
            Gt[0] += params['shiftCC']
            G = [g[0, 1] for g in Gt]
            G2 = [g[1, 0, ::-1] for g in Gt]
            dG = [g[0, 1] for g in dGt]
            dG2 = [g[1, 0, ::-1] for g in dGt]

            tau = ss.tau
            tau2 = -tau[::-1]

            ml, nl = params.get('fitFramesCCLeft', [1, len(tau)])
            mr, nr = params.get('fitFramesCCRight', [0, len(tau)])
            nl = min(nl, len(tau))
            nr = min(nr, len(tau))
            ml = max(ml, 1)
            mr = max(mr, 0)
            # n = nl + nr - ml - mr + 2  # total length for fit

            def gauss_function(x, a, x0, sigma, b):
                ''' gaussian function '''
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

            with PdfPages('{}_crosscorrelation_fit_{}corrected.pdf'.format(params['file'], (1-correctCF)*'un')) as pdf:
                for i, (g, g2, dg, dg2, t) in enumerate(zip(G, G2, dG, dG2, ('G', 'Normalized r', 'Normalized g'))):
                    # get data for fit in array
                    x = np.hstack((tau2[-nl-1:-ml], tau[mr:nr+1]))  # the number of data
                    y = np.hstack((g2[-nl-1:-ml], g[mr:nr+1]))
                    yerr = np.hstack((dg2[-nl-1:-ml], dg[mr:nr+1]))
    
                    # calculate initial parameters for fit
                    y0 = np.nanmin(y)
                    mean = np.sum(x*(y-y0))/np.sum(y-y0)
                    sigma = np.sqrt(np.sum((y-y0)*(x-mean)**2)/np.sum(y-y0))
                    A = np.trapz(y-y0, x)/sigma/np.sqrt(2*np.pi)

                    if params.get('CCxlim') is None:
                        params['CCxlim'] = [-ss.tau.max()/1.5, ss.tau.max()/1.5]
                    try:  # fit data to gaussian function
                        popt, pcov = curve_fit(gauss_function, x, y, sigma=yerr, p0=[A, mean, sigma, y0])
                        perr = np.sqrt(np.diag(pcov))
                        plot_figures.showCrossCorr(pdf, t, ss.tau, g, g2, dg, dg2, None, perr[0], perr[1], *popt, xlim=params['CCxlim'])
                        with open('{}_crosscorrelation_{}corrected_fit_params.txt'.format(params['file'],
                                                                    (1-correctCF)*'un'), 'a' if i else 'w') as f:
                            np.savetxt(f, np.expand_dims(np.hstack((popt, perr)), 0), delimiter='\t', fmt='%1.5f')
                    except Exception:
                        plot_figures.showCrossCorr(pdf, t, ss.tau, g, g2, dg, dg2, xlim=params['CCxlim'])
                        print(misc.color('Error fitting cross correlation', 208))
                        with open('{}_crosscorrelation_{}corrected_fit_params.txt'.format(params['file'],
                                                                    (1-correctCF)*'un'), 'a' if i else 'w') as f:
                            np.savetxt(f, np.full((1, 8), np.nan), delimiter='\t', fmt='%1.5f')
            np.savetxt('{}_crosscorrelation_{}corrected.txt'.format(params['file'], (1 - correctCF) * 'un'),
                np.hstack([np.c_[np.r_[tau2[:-1], tau]]] + [np.c_[np.r_[g2[:-1], g], np.r_[dg2[:-1], dg]]
                    for g, g2, dg, dg2 in zip(G, G2, dG, dG2)]), delimiter='\t', fmt='%1.5f')


#####################################################################
#### calculate burst duration and time between bursts from thresholded data
######################################################################

def durations(lbled_trace):
    lbl = np.unique(lbled_trace)
    Durations = []
    minDurations = []  # start and/or end not in trace, so duration is 'at least'
    for l in lbl[lbl > 0]:
        if lbled_trace[0] != l and lbled_trace[-1] != l:  # start and end both in trace
            Durations.append(sum(lbled_trace == l))
        else:
            minDurations.append(sum(lbled_trace == l))
    return Durations, minDurations


def make_burst_histograms(dataB, params):
    # _BurstDuration+Freq_after_threshold_
    print("Plotting burst histograms")
    for channel in params['ChannelsToAnalyze']:
        Col = ('Red', 'Green')[channel]
        col = Col.lower()
        color = col[0]
        sdThreshold = params['sdThreshold' + Col]

        BurstDuration = []
        minBurstDuration = []
        TimeBetweenBursts = []
        minTimeBetweenBursts = []

        for cc in params['retainedTraces']:
            # load digital data trace
            datacell = dataB[cc][color][slice(*dataB[cc].frameWindow[:2])]
            burstDuration = durations(label(datacell)[0])
            BurstDuration.extend(burstDuration[0])
            minBurstDuration.extend(burstDuration[1])
            timeBetweenBursts = durations(label(1 - datacell)[0])
            TimeBetweenBursts.extend(timeBetweenBursts[0])
            minTimeBetweenBursts.extend(timeBetweenBursts[1])

        np.savetxt(f"{params['file']}_minTimeBetweenBursts_checked_threshold_{col}_{sdThreshold}.txt",
                   minTimeBetweenBursts, fmt='%u')
        np.savetxt(f"{params['file']}_TimeBetweenBursts_checked_threshold_{col}_{sdThreshold}.txt",
                   TimeBetweenBursts, fmt='%u')
        np.savetxt(f"{params['file']}_BurstDuration_checked_threshold_{col}_{sdThreshold}.txt", BurstDuration, fmt='%u')
        np.savetxt(f"{params['file']}_minBurstDuration_checked_threshold_{col}_{sdThreshold}.txt",
                   minBurstDuration, fmt='%u')

        # make histogram of burst duration and time between bursts
        binSize = params['binSizeHistogram'] # time interval
        maxXaxis = params['maxXaxisHistogram'] # range histogram
        dt=dataB[0].dt*1.
        if len(BurstDuration) != 0:
            bootstrDuration = plot_figures.CalcBootstrap(BurstDuration, 1000)
        else: bootstrDuration = [np.nan,np.nan]
        BurstDurationMean = bootstrDuration[0]
        BurstDurationMeanErr = bootstrDuration[1]
        if len(TimeBetweenBursts) != 0:
            bootstrTimeBetwBurst = plot_figures.CalcBootstrap(TimeBetweenBursts, 1000)
        else: bootstrTimeBetwBurst = [np.nan, np.nan]
        TimeBetweenBurstMean = bootstrTimeBetwBurst[0]
        TimeBetweenBurstMeanErr = bootstrTimeBetwBurst[1]
        hBD=np.histogram(BurstDuration,bins=np.r_[:maxXaxis/dt:binSize/dt]-.5, density = True)
        hBF = np.histogram(TimeBetweenBursts, bins=np.r_[:maxXaxis / dt:binSize / dt] - .5, density=True)
        if len(TimeBetweenBursts) == 0:
            hBF = [np.zeros(len(hBF[0])),hBF[1]]
        if len(BurstDuration) == 0:
            hBD = [np.zeros(len(hBD[0])),hBD[1]]

        bins = (hBD[1]+0.5)[:-1]*dt

        # define exponential function
        def expofunc(x, a, b):
            return np.zeros(x.shape) if b == 0 else a * np.exp(-x/b)

        # find where histogram burst duration is non-zeo, do not use first point for fitting
        lox = np.where(hBD[0] !=0)[0][1:]
        # fit histogram burst duration to exponential function
        if len(lox) == 0 or np.isnan(np.sum(hBD[0])):
            poptDur = [0,0]
            pcovDur = [[0,0],[0,0]]
        else:
            poptDur, pcovDur = curve_fit(expofunc, bins[lox], hBD[0][lox], p0 = [400, 30], bounds = [0,np.inf])
        print('Fit parameters burst duration: '+str(poptDur))
        perr = np.sqrt(np.diag(pcovDur))
        fit = expofunc(bins, *poptDur)

        # find where histogram burst frequency is non-zeo, do not use first point for fitting
        lox2 = np.where(hBF[0] !=0)[0][1:]
        # fit histogram burst frequency to exponential function
        if len(lox2) == 0 or np.isnan(np.sum(hBF[0])):
            poptFreq = [0,0]
            pcovFreq = [[0,0],[0,0]]
        else:
            poptFreq, pcovFreq = curve_fit(expofunc, bins[lox2], hBF[0][lox2], p0 = [400, 200], bounds = [0,np.inf])
        print('Fit parameters time between bursts: ' +str(poptFreq))
        perrFreq = np.sqrt(np.diag(pcovFreq))
        fit2 = expofunc(bins, *poptFreq)

        fig = plt.figure(figsize=A4)
        gs = GridSpec(2,1, figure = fig)
        
        fig.add_subplot(gs[0,0])
        plt.bar(hBD[1][:-1]*dt, hBD[0], color ='blue', width = dt-2)
        plt.plot(hBD[1][:-1]*dt,fit, color='black')
        plt.ylim(0,1.1)
        plt.title('Burst duration')
        plt.xlabel('Burst duration (s)')
        plt.ylabel('Frequency')
        plt.text(0.9, 0.9,
                 f'burst duration, mean: {BurstDurationMean * dt:.2f} +/- {BurstDurationMeanErr * dt:.2f} s\n' +
                 f'Exp fit burst duration, tau: {poptDur[1]:.2f} +/- {perr[1]:.2f} s',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

        fig.add_subplot(gs[1,0])
        plt.bar(hBF[1][:-1]*dt, hBF[0], color='gray', width = dt-2)
        plt.plot(hBF[1][:-1]*dt,fit2, color='black')
        plt.ylim(0,1.1)
        plt.title('Burst frequency')
        plt.xlabel('Time between bursts (s)')
        plt.ylabel('Frequency')
        plt.text(0.9, 0.9,
                 f'burst duration, mean: {TimeBetweenBurstMean * dt:.2f} +/- {TimeBetweenBurstMeanErr * dt:.2f} s\n' +
                 f'Exp fit burst duration, tau: {poptFreq[1]:.2f} +/- {perrFreq[1]:.2f} s',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

        plt.tight_layout()
        fig.savefig(params['file']+ '_BurstDuration+Freq_after_threshold_'+col+'.pdf')
        plt.close()


##############################################################################
### Calculating single-cell parameters #######################################
##############################################################################
def make_burst_histograms_singlecells(dataOrig,dataA,dataB, params):
    print("Plotting burst histograms per single cell")
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]

        BurstDuration = []
        TimeBetweenBursts = []

        for cc in params['retainedTraces']:
            # load digital data trace
            datacell = dataB[cc][color][slice(*dataB[cc].frameWindow[:2])]
            BurstDuration.append([cc, durations(label(datacell)[0])[0]])
            TimeBetweenBursts.append([cc, durations(label(1 - datacell)[0])[0]])

        ### burst durations, individual and per cell
        meanDurs = []
        allDurs = []
        for cc, durs in BurstDuration:
            if len(durs):
                meanDurs.append(np.mean(durs))
                allDurs.extend(durs)

        meanDurssec = [i * 15 for i in meanDurs]  # is 15 here a hardcoded time interval?
        np.savetxt(params['file']+"_mean_burst_duration_per_cell_"+col+".txt",meanDurssec)
        plot_figures.HistogramPlot(meanDurssec, 20, 'Histogram of average burst duration per cell '+col, 'Average burst duration (s)', params['file']+'_histogram_average_burst_duration_per_cell_'+col)
        
        allDurssec = [i * 15 for i in allDurs]
        np.savetxt(params['file']+"_burst_duration_"+col+".txt",allDurssec)
        plot_figures.HistogramPlot(allDurssec, 20, 'Burst durations '+col, 'Average burst duration (s)', params['file']+'_histogram_burst_duration_'+col)
        
        ### time between bursts
        meanTimeBetw = []
        allTimeBetw = []
        for cc, times in TimeBetweenBursts:
            if len(times):
                meanTimeBetw.append(np.mean(times))
                allTimeBetw.extend(times)
                
        meanTimeBetwsec = [i * 15 for i in meanTimeBetw]
        np.savetxt(params['file']+"_mean_time_between_bursts_per_cell_"+col+".txt",meanTimeBetwsec)
        plot_figures.HistogramPlot(meanTimeBetwsec, 20, 'Histogram of average time between bursts per cell '+col, 'Average time between bursts (s)', params['file']+ '_histogram_average_time_between_bursts_per_cell_'+col)

        allTimeBetwsec = [i * 15 for i in allTimeBetw]
        np.savetxt(params['file']+"_time_between_bursts_"+col+".txt",allTimeBetwsec)
        plot_figures.HistogramPlot(allTimeBetwsec, 20, 'Time between bursts '+col, 'Average time between bursts (s)', params['file']+ '_histogram_time_between_bursts_'+col)
        
        ##number of bursts per cell
        nbrbursts = []
        for i in range(len(BurstDuration)):
            nbrbursts.append(len(BurstDuration[i][1]))
        np.savetxt(params['file']+"_number_of_bursts_per_cell_"+col+".txt",nbrbursts)
        plot_figures.HistogramPlot(nbrbursts, 20, 'Histogram of number of bursts per cell '+col, 'Number of bursts', params['file']+ '_histogram_number_of_bursts_per_cell_'+col)
            
        ##intensity in on state per cell and individual frames
        avIntCell = []
        allInts = []
        for cc in params['retainedTraces']:
            if col == "red": datacellInts = dataOrig[cc].r[dataOrig[cc].frameWindow[0]:dataOrig[cc].frameWindow[1]]
            datacellDigi = dataB[cc].r[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            if col == "green": datacellInts = dataOrig[cc].g[dataOrig[cc].frameWindow[0]:dataOrig[cc].frameWindow[1]]
            datacellDigi = dataB[cc].g[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            onints = datacellInts * datacellDigi
            if sum(datacellDigi)!=0:
                avInt = np.sum(onints)/np.sum(datacellDigi)
                avIntCell.append(avInt)
            for i in range(len(onints)):
                if onints[i]!=0: allInts.append(onints[i])
        
        np.savetxt(params['file']+"_intensity_frames_on_per_cell_"+col+".txt",avIntCell)
        np.savetxt(params['file']+"_intensity_frames_on_"+col+".txt",allInts)
        
        plot_figures.HistogramPlot(avIntCell, 20, 'Histogram of average intensity of frames in on-state per cell '+col, 'Intensity (AU)', params['file']+ '_intensity_frames_on_per_cell_'+col)
        plot_figures.HistogramPlot(allInts, 20, 'Histogram of intensity of all frames in on-state '+col, 'Intensity (AU)', params['file']+ '_intensity_frames_on_'+col)

        ##Second value of ACF plot (measure of ACF amplitude)
        ACFampl = []
        for cell in params['retainedTraces']:
            if np.sum(dataA[cell].G) == 0: continue
            if col == 'red':
                if len(dataA[cell].G[0, 0]) <= 1: continue
                ACFampl.append(dataA[cell].G[0,0][1])
            if col == 'green':
                if len(dataA[cell].G[1, 1]) <= 1: continue
                ACFampl.append(dataA[cell].G[1,1][1])
        np.savetxt(params['file']+"_ACF_amplitudes_per_cell_"+col+".txt", ACFampl)
        plot_figures.HistogramPlot(ACFampl, 20, 'Histogram of ACF amplitude per cell '+col, 'ACF amplitude', params['file']+ '_ACF_amplitude_per_cell_'+col)

        ##Correlation of ACF with induction time
        plot_figures.corrACFAmplToIndPlot(dataA, dataB, col, params)


def smoothOrquantize(data, params):
    # Better not smooth and quantize the same trace!
    if params.get('quantizeTrace') is not None:
        for channel, q in params['quantizeTrace'].items():
            daf.getMolNumber(data, q, channel)
    if params.get('smoothTrace') is not None:
        for channel, (window_length, polyorder) in params['smoothTrace'].items():
            daf.smoothData(data, channel, window_length, polyorder)


def globalFrameWindow(data, params):
    # change frameWindow if necessary
    if not params.get('globalFrameWindow') is None:
        globalFrameWindow = params['globalFrameWindow']
        if globalFrameWindow[0] == misc.none():
            globalFrameWindow[0] = -np.inf
        if globalFrameWindow[1] == misc.none():
            globalFrameWindow[1] = np.inf
        for d in data:
            d.frameWindow = np.clip(d.frameWindow, *globalFrameWindow).astype(int).tolist()


def pipeline_correlation_functions(params):
    # pipeline for correlation functions script
    # params is either a dictionary containing the parameters for the pipeline or a string pointing to the yml file with the parameters

    if not isinstance(params, dict):
        parameter_file = params
        if parameter_file[-3:] == '.py':
            print('Converting py parameter file into yml format')
            misc.convertParamFile2YML(parameter_file)
            parameter_file = parameter_file[:-3]+'.yml'
        if os.path.isdir(parameter_file):
            parameter_file = os.path.join(parameter_file, 'parameters.yml')
        if not parameter_file[-4:] == '.yml':
            parameter_file += '.yml'
        f = os.path.split(__file__)
        params = misc.getParams(parameter_file,
                                os.path.join(os.path.dirname(f[0]), f[1].replace('.py', '_parameters_template.yml')),
                                ('PyFile', 'outputfolder'))
    else:
        parameter_file = ''

    get_paths(params, parameter_file)
    
    if params['processTraces'] == 1:
        # Load original raw data
        dataOrig = daf.loadExpData(params['PyFile'])
        globalFrameWindow(dataOrig, params)
        smoothOrquantize(dataOrig, params)
        dataA = bg_sub_traces(dataOrig, params)
        dataB = binary_call_traces(dataA, params)
        if params.get('filterTraces', True):
            dataA, params = filter_traces(dataA, dataB, params)
        else:
            params['retainedTraces'] = list(range(len(dataA)))

    calc_timefraction_on(dataB, params)

    if params['makePlots'] == 1:
        make_plots_traces(dataOrig, dataA, dataB, params)

    if params['calcCF'] == 1:
        ss = calculateCF(dataA, dataB, params)
        params['ss'] = ss
        for i, s in zip(params['retainedTraces'], ss.sigsAlign):
            s.name = dataOrig[i].name

    if params['makeCFplot'] == 1:
        make_plots_CF(ss, params)
    
    if params['makeHistogram'] == 1:
        make_burst_histograms(dataB, params)
    
    if params['SingleCellParams'] == 1:
        make_burst_histograms_singlecells(dataOrig, dataA, dataB, params)

    return params

def main():
    if len(sys.argv) < 2:
        parameter_files = [os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                      'pipeline_livecell_correlationfunctions_test.yml'))]
    else:
        parameter_files = sys.argv[1:]


    if len(parameter_files)==1:
        parameter_file = parameter_files[0]
        if not os.path.exists(parameter_file):
            raise FileNotFoundError('Could not find the parameter file.')
        print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
        params = pipeline_correlation_functions(parameter_file)
    else:
        for parameter_file in parameter_files:
            print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
            print('')
            try:
                pipeline_correlation_functions(parameter_file)
            except Exception:
                print(misc.color('Exception while working on: {}'.format(parameter_file), 'r:b'))

    # this only runs when this script is run from command-line with ./pipeline..., not when run from ipython
    # if we do not kill the java vm, (i)python will not exit after completion
    # be sure to call imr.kill_vm() at the end of your script/session, note that you cannot use imread afterwards
    if os.path.basename(__file__) in [os.path.basename(i) for i in psutil.Process(os.getpid()).cmdline()]:
        imr.kill_vm() #stop java used for imread, needed to let python exit
        print('Stopped the java vm used for imread.')

    print('------------------------------------------------')
    print(misc.color('Pipeline finished.', 'g:b'))

if __name__ == '__main__':
    main()
