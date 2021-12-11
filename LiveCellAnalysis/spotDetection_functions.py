from scipy import signal, fftpack
import numpy as np

def GaussianMaskFit2(im,coo,s,optLoc=1,bgSub=2,winSize=13,convDelta=.01,nbIter=20):
    """Applies the algorithm from [Thompson et al. (2002) PNAS, 82:2775].
    Parameters:
    - im: a numpy array with the image
    - coo: approximate coordinates (in pixels) of the spot to localize and measure. Note, the coordinates are x,y!
    - s: width of the PSF in pixels
    - optLoc: If 1, applied the iterative localization refinement algorithm, starting with the coordinates provided in coo. If 0, only measures the spot intensity at the coordinates provided in coo.
    - bgSub: 0 -> no background subtraction. 1 -> constant background subtraction. 2 -> tilted plane background subtraction.
    - winSize: Size of the window (in pixels) around the position in coo, used for the iterative localization and for the background subtraction.
    - convDelta: cutoff to determine convergence, i.e. the distance (in pixels) between two iterations
    - nbIter: the maximal number of iterations.

    Returns
    - the intensity value of the spot.
    - the coordinates of the spot.

    If convergence is not found after nbIter iterations, return 0 for both intensity value and coordinates.
    """
    coo=np.array(coo)
    for i in range(nbIter):
        if not np.prod(coo-winSize/2.>=0)*np.prod(coo+winSize/2.<=im.shape[::-1]):
            break
        winOrig=(coo-int(winSize)//2).astype(int)
        i,j=np.meshgrid(winOrig[0]+np.r_[:winSize],winOrig[1]+np.r_[:winSize])
        N=np.exp(-(i-coo[0])**2/(2*s**2)-(j-coo[1])**2/(2*s**2))/(2*np.pi*s**2)
        S=im[:,winOrig[0]:winOrig[0]+winSize][winOrig[1]:winOrig[1]+winSize]*1.
        if bgSub==2:
            xy=np.r_[:2*winSize]%winSize-(winSize-1)/2.
            bgx=np.polyfit(xy,np.r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgx[0]).T
            bgy=np.polyfit(xy,np.r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgy[0]).T
            bg=np.mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
            bg=np.r_[bg,bgx[0],bgy[0]]
        if bgSub==1:
            bg=np.mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
        S=S.clip(0) # Prevent negative values !!!!
        if optLoc:
            SN=S*N; ncoo=np.r_[np.sum(i*SN),np.sum(j*SN)]/np.sum(SN)
            #ncoo=ncoo+ncoo-coo # Extrapolation
            if abs(coo-ncoo).max()<convDelta:
                return np.sum(SN)/np.sum(N**2),coo,bg
            else:
                coo=ncoo
        else:
            return np.sum(S*N)/np.sum(N**2),coo,bg
    if bgSub == 1: bg = 0
    elif bgSub == 2: bg = [0.,0.,0.]
    return 0.,np.r_[0.,0.], bg


sHS=fftpack.fftshift # Swap half-spaces. sHS(matrix[, axes]). axes=all by default
def hS(m,axes=None):
    if axes==None: axes=range(np.ndim(m))
    elif type(axes)==int: axes=[axes]
    elif axes==[]: return m
    return hS(m.swapaxes(0,axes[-1])[:m.shape[axes[-1]]/2].swapaxes(0,axes[-1]),axes[:-1])

def sHSM(m,axes=None):
    if axes==None: axes=range(np.ndim(m))
    elif type(axes)==int: axes=[axes]
    m=m.swapaxes(0,axes[0]); max=m[1]+m[-1]; m=(m+max/2)%max-max/2; m=m.swapaxes(0,axes[0])
    return sHS(m,axes)


def bpass(im,r1=1.,r2=1.7):
    ker1x=np.exp(-(sHS(sHSM(np.r_[:im.shape[1]]))/r1)**2/2); ker1x/=np.sum(ker1x); fker1x=fftpack.fft(ker1x)
    ker1y=np.exp(-(sHS(sHSM(np.r_[:im.shape[0]]))/r1)**2/2); ker1y/=np.sum(ker1y); fker1y=fftpack.fft(ker1y)
    ker2x=np.exp(-(sHS(sHSM(np.r_[:im.shape[1]]))/r2)**2/2); ker2x/=np.sum(ker2x); fker2x=fftpack.fft(ker2x)
    ker2y=np.exp(-(sHS(sHSM(np.r_[:im.shape[0]]))/r2)**2/2); ker2y/=np.sum(ker2y); fker2y=fftpack.fft(ker2y)
    fim=fftpack.fftn(im)
    return fftpack.ifftn((fim*fker1x).T*fker1y-(fim*fker2x).T*fker2y).real.T


def hotpxfilter(im):
    """ recognizes hot pixels by comparing the image with a median-filtered version,
        then replaces the hot pixels with the local medians
        
        wp@tl20190719
    """
    med = signal.medfilt2d(im)
    h = abs(im-med)
    hb = np.zeros(h.shape)
    hb[h>np.mean(h) + 15*np.std(h)] = 1
    jm = im.copy()
    jm[hb==1] = med[hb==1]
    return jm

