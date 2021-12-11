import skimage
import skimage.segmentation
import skimage.feature
import scipy
import numpy as np
from tqdm.auto import tqdm

# ----------------------- Helper functions ---------------------------------

def collectr(cf):
    #""" Makes a 1d corrfun out of a 2d corrfun
    #    The result of the maximum of cf at each distance from the center
    #    wp@tl20191102
    #"""
    x, y = np.meshgrid(range(cf.shape[0]), range(cf.shape[1]))
    c = [(i-1)/2 for i in cf.shape]
    r = np.sqrt((x-c[0])**2 + (y-c[1])**2).flatten()
    idx = np.argsort(r)
    r = np.round(r[idx]).astype('int')
    cf = np.round(cf.flatten()[idx]).astype('int')
    rr = np.unique(r)
    cf = [np.max(cf[r==i]) for i in rr]
    return rr, cf

def maskpk(pk, mask):
    #    """ remove points in nx2 array which are located outside mask
    #        wp@tl20190709
    #        """
    pk = np.round(pk)
    idx = []
    for i in range(pk.shape[0]):
        if mask[pk[i, 0], pk[i, 1]]:
            idx.append(i)
    return pk[idx,:]

def disk(s):
    #    """ make a disk shaped structural element to be used with
    #        morphological functions
    #        wp@tl20190709
    #        """
    d = np.zeros((s, s))
    c = (s-1)/2.
    x, y = np.meshgrid(range(s), range(s))
    d2 = (x-c)**2 + (y-c)**2
    d[d2<s**2/4.] = 1
    return d

def fill_nan(im):
    #""" Assigns the value of the nearest finite valued pixel to infinite valued pixels
    #    wp@tl20190910
    #"""
    im = im.copy()
    b = np.where(~np.isfinite(im))
    if len(b[0]):
        a = np.where(np.isfinite(im))
        v = scipy.interpolate.griddata(a, im[a[0], a[1]], b, 'nearest')
        for i, j, w in zip(b[0], b[1], v):
            im[i, j] = w
    return im

def collectr(cf):
    #""" Makes a 1d corrfun out of a 2d corrfun
    #    The result of the maximum of cf at each distance from the center
    #    wp@tl20191102
    #"""
    x, y = np.meshgrid(range(cf.shape[0]), range(cf.shape[1]))
    c = [(i-1)/2 for i in cf.shape]
    r = np.sqrt((x-c[0])**2 + (y-c[1])**2).flatten()
    idx = np.argsort(r)
    r = np.round(r[idx]).astype('int')
    cf = np.round(cf.flatten()[idx]).astype('int')
    rr = np.unique(r)
    cf = [np.max(cf[r==i]) for i in rr]
    return rr, cf

def corrfft(im, jm):
    #'''
    #% usage: d, cfunc = corrfft(images)
    #%
    #% input:
    #%   im, jm: images to be correlated
    #%
    #% output:
    #%   d:      offset (x,y) in px
    #%   cfunc:  correlation function
    #'''

    im = im.astype(float)
    jm = jm.astype(float)

    im -= np.nanmean(im)
    im /= np.nanstd(im)
    jm -= np.nanmean(jm)
    jm /= np.nanstd(jm)

    im[np.isnan(im)] = 0
    jm[np.isnan(jm)] = 0

    nY = np.shape(im)[0]
    nX = np.shape(im)[1]

    cfunc = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im)*np.conj(np.fft.fft2(jm)))))
    y, x = np.unravel_index(np.nanargmax(cfunc), cfunc.shape)

    d = [x-np.floor((nX)/2), y-np.floor(nY/2)]

    #peak at x=nX-1 means xoffset=-1
    if d[0]>nX/2:
        d[0] -= nX
    if d[1]>nY/2:
        d[1] -= nY
    return d, cfunc

def otsu_local(im, res=16):
    # Executes otsu on blocks of an image, then interpolates to give a local threshold
    # im: array with the image
    # res: image will be cut in res x res blocks
    # Image size should be an integer multiple of res!
    #
    # wp@tl20191121
    if res==1: #global otsu
        d = im.flatten()
        d = d[np.isfinite(d)]
        if len(np.unique(d))>1:
            return skimage.filters.threshold_otsu(d)
        else:
            return d.flatten()[0]
    th = np.zeros((res, res))
    s = im.shape[0]//res
    for i in range(res):
        for j in range(res):
            d = im[i*s:(i+1)*s, j*s:(j+1)*s].flatten()
            d = d[np.isfinite(d)]
            if len(d)==0:
                th[i,j] = np.nan
            elif len(np.unique(d))>1:
                th[i,j] = skimage.filters.threshold_otsu(d)
            else:
                th[i,j] = d.flatten()[0]
    th = skimage.transform.resize(th, im.shape, anti_aliasing=False, mode='edge')
    return scipy.ndimage.gaussian_filter(th, im.shape[0]/8, mode='reflect')

def fill_holes(im):
    #""" Fill holes (value==0) in each label individually in a faster way
    #    wp@tl20200120
    #"""
    im = im.copy()
    z = skimage.morphology.dilation(im)*(im==0)
    holes_lbl = skimage.measure.label(im==0)
    lbl_edges = np.unique(np.hstack((holes_lbl[:, 0], holes_lbl[:, -1], holes_lbl[0,:], holes_lbl[-1,:])))
    lbl_h = list(np.unique(holes_lbl))
    for i in lbl_h:
        if i==0 or i in lbl_edges:
            continue
        lbl_edge = list(np.unique((holes_lbl==i)*z))
        if 0 in lbl_edge:
            lbl_edge.remove(0)
        if len(lbl_edge)==1:
            im += lbl_edge[0]*(holes_lbl==i)
    return im

# ----------------------- Finally the function itself ---------------------------------

def findcells(im, imnuc=None, cellcolormask=None, ccdist=None, threshold=None, thresholdnuc=None, thres=1, thresnuc=1,
              smooth=2.5, smoothnuc=2.5, minfeatsize=5, minfeatsizenuc=5, dilate=5, dilatenuc=5, removeborders=True):
    #""" segement cells and nuclei from an image (nxm array)
    #    wp@tl20190710
    #
    #    im: 2d numpy array with an image
    #
    #    optional:
    #    imnuc: 2d numpy array with an image containing the nuclei, for example with DAPI
    #    cellcolormask: use a cellmask from another frame to assign the same colors
    #                   to the cells found by findcells
    #    ccdist:        (approximate) minimum distance between the centers of cells, good values: yeast: 25, mamalian: 150
    #
    #    optional parameters to debug/finetune findcells:
    #    threshold:     value used to threshold the image, default: use Otsu
    #    thres:         divide the image in thres x thres blocks and use a different threshold in each of them,
    #                   the image size should be an integer multiple of thres
    #    smooth:        smooth the image before using it
    #    minfeatsize:   remove features smaller than minfeatsize
    #    dilate:        >0: make every cell dilate px larger, <0: make them smaller
    #    thresholdnuc, thresnuc, smoothnuc, minfeatsizenuc, dilatenuc: parameters applying to imnuc
    #    removeborders: remove any cells (and their nuclei) touching any borders (case with nan's unhandled atm), default: True
    #"""
    
    def make_mask(im, threshold, thres, smooth=2.5, minfeatsize=5, dilate=5):
        LB = fill_nan(im)
        if smooth:
            LA = scipy.ndimage.gaussian_filter(LB, smooth, mode='nearest')
        else:
            LA = LB.copy()
        if threshold is None or isinstance(threshold, str) and threshold.lower() == 'none':
            th = otsu_local(LA, thres)
        else:
            th = float(threshold)
        mask = LA>th
        if minfeatsize>0:
            mask = skimage.morphology.binary_opening(mask, disk(minfeatsize))
        if dilate==0:
            return mask, LA
        elif dilate>0:
            return skimage.morphology.binary_dilation(mask, disk(dilate)), LA
        else:
            return skimage.morphology.binary_erosion(mask, disk(-dilate)), LA
    
    im = im.astype('float')
    if not imnuc is None:
        imnuc = imnuc.astype('float')
        
    if ccdist is None: #try to determine good ccdist if not given
        if imnuc is None:
            cf = corrfft(im.copy(), im.copy())[1]
        else:
            cf = corrfft(imnuc.copy(), imnuc.copy())[1]
        r, cf = collectr(cf)
        ccdist = r[scipy.signal.find_peaks(-np.array(cf))[0][0]]
        ccdist = np.round(np.clip(ccdist, 10, np.sqrt(np.prod(im.shape))/5)).astype('int')
    
    if imnuc is None: #only one channel
        mask, LA = make_mask(im, threshold, thres, smooth, minfeatsize, dilate)
        pk = skimage.feature.peak_local_max(fill_nan(LA), footprint=disk(ccdist), exclude_border=False)
        pk = maskpk(pk, mask)
        pk = np.array(sorted([q.tolist() for q in pk])[::-1])
        markers = np.zeros(im.shape)
        for i in range(pk.shape[0]):
            if cellcolormask is None:
                markers[pk[i, 0], pk[i, 1]] = i+1
            else:
                markers[pk[i, 0], pk[i, 1]] = cellcolormask[pk[i, 0], pk[i, 1]]
        cells = skimage.segmentation.watershed(-LA, markers=markers, mask=mask)
        cells *= mask #in Python2, watershed sometimes colors outside the mask
        lbl = list(np.unique(cells))
        lbl.remove(0)
        cells = fill_holes(cells)
        nuclei = np.zeros(im.shape)        
        for i in tqdm(lbl, disable=len(lbl)<25, leave=False): #threshold each cell to find its nucleus
            cellmask = cells==i
            cmLA = cellmask*LA
            cell = cmLA.flatten()
            cell = cell[cell>0]
            cell = cell[cell>np.percentile(cell, 25)]
            pxval = np.unique(cell)
            if len(pxval)==0:
                th = 0
            elif len(pxval)==1:
                th = cell[0]
            else:
                th = skimage.filters.threshold_otsu(cell)
            nucleus = cmLA>th
            nuclei += float(i)*nucleus
    else:  # extra channel with nuclei
        nuclei, _ = findcells(imnuc, cellcolormask=cellcolormask, ccdist=ccdist, threshold=thresholdnuc, thres=thresnuc,
                              smooth=smoothnuc, minfeatsize=minfeatsizenuc, dilate=dilatenuc, removeborders=False)
        mask, LA = make_mask(im, threshold, thres, smooth, minfeatsize, dilate)
        cells = skimage.segmentation.watershed(-LA, markers=nuclei, mask=mask)
        cells *= mask  # in Python2, watershed sometimes colors outside the mask
        cells = np.dstack((cells, nuclei)).max(2)  # make sure each nucleus has a cell body
        cells = fill_holes(cells)
        
    if removeborders:
        for lbl in np.unique(np.hstack((cells[:, 0], cells[:, -1], cells[0,:], cells[-1,:]))):
            cells[cells==lbl] = 0
            nuclei[nuclei==lbl] = 0
    return cells, nuclei
