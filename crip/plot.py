'''
    Figure artist module of crip.

    https://github.com/z0gSh1u/crip
'''

import cv2
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
from ._typing import *
from .utils import cripAssert, is1D, isInt, cripWarning
from .physics import Spectrum, DiagEnergyRange
from .shared import resize

__all__ = ['smooth', 'window', 'average', 'addFont', 'fontdict', 'zoomIn', 'plotSpectrum', 'makeImageGrid']


def smooth(data: NDArray, winSize: int = 5):
    '''
        Smooth an 1D array by moving averaging window. This name follows MATLAB.

        The implementation is from: https://stackoverflow.com/questions/40443020
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


def zoomIn(img, x, y, hw):
    '''
        Crop a patch.
    '''
    return img[y:y + hw, x:x + hw]


def stddev(img, leftTop, h, w):
    y, x = leftTop
    crop = img[x:x + h, y:y + w]
    return np.std(crop)


def fontdict(family, weight, size):
    return {'family': family, 'weight': weight, 'size': size}


def plotSpectrum(fig, spec: Spectrum):
    energies = DiagEnergyRange
    omega = spec.omega

    fig.plot(energies, omega, 'k')
    fig.xlabel('Energy (keV)')
    fig.ylabel('Omega')


def makeImageGrid(subimages: List[TwoD],
                  colTitles: List[str],
                  rowTitles: List[str],
                  preprocessor=None,
                  fontdict=None,
                  crops=None,
                  cropLocs='bottom right',
                  cropEdgeColor='yellow',
                  cropSize=512 // 4,
                  figsize=None,
                  vmin0vmax1=True):

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

    fig = plt.figure(figsize=figsize or (ncols * 2, nrows * 2))

    if preprocessor:
        subimages = list(map(lambda ix: preprocessor(*ix), list(enumerate(subimages))))

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0)

    if crops is not None:
        rects = [
            patches.Rectangle((x, y), hw, hw, linewidth=1, edgecolor=cropEdgeColor, facecolor='none')
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
        cropLoc = cropLocs[i // ncols]

        if cropLoc == 'bottom right':
            im[-patch.shape[0]:, -patch.shape[1]:] = patch
            patchRect = patches.Rectangle((im.shape[1] - patch.shape[1], im.shape[0] - patch.shape[0]),
                                          patch.shape[1],
                                          patch.shape[0],
                                          linewidth=1,
                                          edgecolor=cropEdgeColor,
                                          facecolor='none')
        elif cropLoc == 'top left':
            im[:patch.shape[0], :patch.shape[1]] = patch
            patchRect = patches.Rectangle((0, 0),
                                          patch.shape[1],
                                          patch.shape[0],
                                          linewidth=1,
                                          edgecolor=cropEdgeColor,
                                          facecolor='none')
        elif cropLoc == 'bottom left':
            im[-patch.shape[0]:, :patch.shape[1]] = patch
            patchRect = patches.Rectangle((0, im.shape[0] - patch.shape[0]),
                                          patch.shape[1],
                                          patch.shape[0],
                                          linewidth=1,
                                          edgecolor=cropEdgeColor,
                                          facecolor='none')
        elif cropLoc == 'top right':
            im[:patch.shape[0], -patch.shape[1]:] = patch
            patchRect = patches.Rectangle((im.shape[1] - patch.shape[1], 0),
                                          patch.shape[1],
                                          patch.shape[0],
                                          linewidth=1,
                                          edgecolor=cropEdgeColor,
                                          facecolor='none')
        else:
            raise

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
