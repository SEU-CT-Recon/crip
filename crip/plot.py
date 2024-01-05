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

__all__ = ['smooth', 'window', 'average', 'addFont', 'fontdict', 'zoomIn', 'plotSpectrum', 'makeImageGrid']


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
                  preprocessor=None,
                  fontdict=None,
                  crops=None,
                  cropLocs='bottom right',
                  cropEdgeColor='yellow',
                  cropSize=64,
                  figsize=None,
                  vmin0vmax1=True):
    '''
        Make an Image Grid.
        `preprocessor(index, subimage)` is applied to each subimage, e.g, perform windowing.
        `figsize` accepts that for `plt.figure(figsize=...)`.
        `vmin0vmax1=True` guarantees correct windowing when windowed image is not compactly supported in [0, 1].
        Return the handle of plt.Figure.
    ```
               colTitles
            +-------------+
            |             |
    row     |  subImages  |  ...
    Titles  |       +-----|
            |       |crop |
            +-------------+
                  ...
    '''

    cripWarning(
        vmin0vmax1,
        'vmin0vmax1=False is not recommended because it might cause incorrect windowing. Make sure you know what you are doing.'
    )

    cripAssert(len(subimages) == ncols * nrows)
    cripAssert(crops is None or len(crops) == nrows)

    ncols = len(colTitles)
    nrows = len(rowTitles)

    if isinstance(cropLocs, str):
        cropLocs = [cropLocs] * nrows

    cripAssert(all(list(map(lambda x: x in ['bottom right', 'bottom left', 'top right', 'top left'], cropLocs))),
               'Invalid cropLocs, not in `bottom right`, `bottom left`, `top right`, `top left`.')

    fig = plt.figure(figsize=figsize or (ncols * 2, nrows * 2))

    if preprocessor:
        subimages = list(map(lambda ix: preprocessor(*ix), list(enumerate(subimages))))

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0)

    if crops is not None:
        rects = [
            matplotlib.patches.Rectangle((x, y), hw, hw, linewidth=1, edgecolor=cropEdgeColor, facecolor='none')
            for x, y, hw in crops
        ]
    else:
        rects = []

    i = 0
    for ax, im in zip(grid, subimages):
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        list(map(lambda x: x.set_visible(False), ax.spines.values()))

        if i % ncols == 0:
            ax.add_patch(rects[i // ncols])

        patch = zoomIn(im, *crops[i // ncols])
        patch = resize(patch, (cropSize, cropSize))
        cropLocs = cropLocs[i // ncols]

        if cropLocs == 'bottom right':
            im[-patch.shape[0]:, -patch.shape[1]:] = patch
            patchRect = matplotlib.patches.Rectangle((im.shape[1] - patch.shape[1], im.shape[0] - patch.shape[0]),
                                                     patch.shape[1],
                                                     patch.shape[0],
                                                     linewidth=1,
                                                     edgecolor=cropEdgeColor,
                                                     facecolor='none')
        elif cropLocs == 'top left':
            im[:patch.shape[0], :patch.shape[1]] = patch
            patchRect = matplotlib.patches.Rectangle((0, 0),
                                                     patch.shape[1],
                                                     patch.shape[0],
                                                     linewidth=1,
                                                     edgecolor=cropEdgeColor,
                                                     facecolor='none')
        elif cropLocs == 'bottom left':
            im[-patch.shape[0]:, :patch.shape[1]] = patch
            patchRect = matplotlib.patches.Rectangle((0, im.shape[0] - patch.shape[0]),
                                                     patch.shape[1],
                                                     patch.shape[0],
                                                     linewidth=1,
                                                     edgecolor=cropEdgeColor,
                                                     facecolor='none')
        elif cropLocs == 'top right':
            im[:patch.shape[0], -patch.shape[1]:] = patch
            patchRect = matplotlib.patches.Rectangle((im.shape[1] - patch.shape[1], 0),
                                                     patch.shape[1],
                                                     patch.shape[0],
                                                     linewidth=1,
                                                     edgecolor=cropEdgeColor,
                                                     facecolor='none')

        ax.add_patch(patchRect)

        if i < len(colTitles):
            ax.set_title(colTitles[i], loc='center', fontdict=fontdict)
        i += 1

        if vmin0vmax1:
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(im, cmap='gray')

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
