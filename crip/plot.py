'''
    Figure artist module of crip.

    https://github.com/z0gSh1u/crip
'''

import cv2
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches
import matplotlib.axes
from mpl_toolkits.axes_grid1 import ImageGrid
from ._typing import *
from .utils import cripAssert, is1D, isInt, cripWarning
from .physics import Spectrum, DiagEnergyRange, Atten
from .shared import resize

__all__ = ['smooth', 'window', 'average', 'addFont', 'fontdict', 'zoomIn', 'plotSpectrum', 'makeImageGrid', 'windowFullRange']


def smooth(data: NDArray, winSize: int = 5):
    '''
        Smooth an 1D array by moving averaging window. This name follows MATLAB.

        The implementation is from https://stackoverflow.com/questions/40443020
    '''
    cripAssert(is1D(data), '`data` should be 1D array.')
    cripAssert(isInt(winSize) and winSize % 2 == 1, '`winSize` should be odd integer.')

    out0 = np.convolve(data, np.ones(winSize, dtype=int), 'valid') / winSize
    r = np.arange(1, winSize - 1, 2)
    start = np.cumsum(data[:winSize - 1])[::2] / r
    stop = (np.cumsum(data[:-winSize:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def window(img: TwoOrThreeD, win: Or[Tuple[int], Tuple[float]], style: str = 'lr', normalize: Or[str, None] = None):
    '''
        Window `img` using `win` (ww, wl) with style 'wwwl' or (left, right) with style 'lr'.
        Set `normalize` to '0255' to convert to 8-bit image, or '01' to [0, 1] float image.
    '''
    cripAssert(len(win) == 2, '`win` should have length of 2.')
    cripAssert(style in ['wwwl', 'lr'], "`style` should be 'wwwl' or 'lr'")

    if style == 'wwwl':
        ww, wl = win
        l = wl - ww / 2
        r = l + ww
    elif style == 'lr':
        l, r = win

    res = img.copy()
    res[res > r] = r
    res[res < l] = l

    if normalize == '0255':
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif normalize == '01':
        res = (res - l) / (r - l)

    return res


def windowFullRange(img: TwoOrThreeD, normalize='01'):
    '''Window `img` using full dynamic range of pixel values.
    '''
    return window(img, (np.max(img), np.min(img)), 'lr', normalize)


def addFont(dir_: str):
    '''
        Add font files under `dir` to matplotlib.
    '''
    for file in font_manager.findSystemFonts(fontpaths=[dir_]):
        font_manager.fontManager.addfont(file)


def average(imgs: ThreeD, i: int, r: int):
    '''
        Average along `channel` dimension [i - r, i + r].
        Use for example, show CT slices smoother.
    '''
    return np.mean(imgs[i - r:i + r], axis=0)


def zoomIn(img, row, col, h, w):
    '''
        Crop a patch. (row, col) determines the left-top point. (h, w) gives height and width.
    '''
    return img[row:row + h, col:col + w]


def stddev(img, row, col, h, w):
    '''
        Compute the standard deviation of a image crop.
        (row, col) determines the left-top point. (h, w) gives height and width.
    '''
    return np.std(zoomIn(img, row, col, h, w))


def fontdict(family, weight, size):
    return {'family': family, 'weight': weight, 'size': size}


def makeImageGrid(subimages: List[TwoD],
                  colTitles: List[str],
                  rowTitles: List[str],
                  preproc=None,
                  fontdict=None,
                  crops=None,
                  cropLoc='bottom right',
                  cropSize=96 * 2,
                  arrows=None,
                  arrowLen=24):
    # determine the number of rows and columns
    ncols = len(colTitles)
    nrows = len(rowTitles)
    cripAssert(len(subimages) == ncols * nrows, 'Number of subimages should be equal to ncols * nrows.')
    cripAssert(crops is None or len(crops) == nrows, 'Number of crops should be equal to nrows.')
    cripAssert(arrows is None or len(arrows) == nrows, 'Number of arrows should be equal to nrows.')

    # create the figure
    fig = plt.figure(figsize=(ncols * 2, nrows * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0)

    # apply the preprocessor
    if preproc:
        subimages = list(map(lambda ix: preproc(*ix), list(enumerate(subimages))))

    def overlayPatch(img, patch, loc):
        if loc == 'bottom left':
            img[-patch.shape[0]:, :patch.shape[1]] = patch
            box = matplotlib.patches.Rectangle((0, img.shape[0] - patch.shape[0]),
                                               patch.shape[1],
                                               patch.shape[0],
                                               linewidth=1,
                                               edgecolor='yellow',
                                               facecolor='none')
            return box
        else:
            raise 'Invalid cropLoc, not in `bottom left`.'

    # display the subimages
    cur = 0
    for ax, img in zip(grid, subimages):
        # remove the ticks and spines
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        list(map(lambda x: x.set_visible(False), ax.spines.values()))

        # prepare the image crop
        box = None
        if crops is not None and crops[cur // ncols]:
            r, c, hw = crops[cur // ncols]
            patch = resize(zoomIn(img, r, c, hw, hw), (cropSize, cropSize))
            box = overlayPatch(img, patch, cropLoc)

        # display the image
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)

        # display the crop box
        box and ax.add_patch(box)
        if box and cur % ncols == 0:
            r, c, hw = crops[cur // ncols]
            box = matplotlib.patches.Rectangle((c, r), hw, hw, linewidth=0.8, edgecolor='yellow', facecolor='none')
            ax.add_patch(box)

        # display the arrow
        if arrows is not None and arrows[cur // ncols]:
            r, c = arrows[cur // ncols]
            ax.arrow(c + arrowLen, r - arrowLen, -arrowLen, +arrowLen, color='yellow', head_width=10)

        # display the column titles
        if cur < len(colTitles):
            ax.set_title(colTitles[cur], loc='center', fontdict=fontdict)
        cur += 1

    # display the row titles
    for i in range(nrows):
        grid[ncols * i].set_ylabel(rowTitles[i], fontdict=fontdict)

    return fig


def plotSpectrum(ax: matplotlib.axes.Axes, spec: Spectrum):
    '''
        Plot the spectrum in `ax`. Example
        ```
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plotSpectrum(ax, spec)
    '''
    energies = DiagEnergyRange
    omega = spec.omega

    ax.plot(energies, omega, 'k')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Omega')


def plotMu(ax: matplotlib.axes.Axes, atten: Atten, startEnergy: int = 1, logScale=True):
    '''
        Plot the LACs of `atten` from `startEnergy` keV in ax in `logScale` if true.
    '''
    x = list(DiagEnergyRange)[startEnergy:]
    ax.plot(x, atten.mu[startEnergy:])

    if logScale:
        ax.set_yscale('log')

    ax.xlabel('Energy (keV)')
    ax.ylabel('LAC (1/mm)')


def savefigTight(fig, path, dpi=200, pad=0.05):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad)


def meanstd(x):
    return np.mean(x), np.std(x)
