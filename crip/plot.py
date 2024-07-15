'''
    Figure artist module of crip.

    https://github.com/z0gSh1u/crip
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches
import matplotlib.axes
from mpl_toolkits.axes_grid1 import ImageGrid as MplImageGrid
from scipy.ndimage import uniform_filter

from ._typing import *
from .utils import *
from .physics import Spectrum, DiagEnergyRange, Atten
from .shared import resizeTo

VMIN0_VMAX1 = {'vmin': 0, 'vmax': 1}


def smooth1D(data: NDArray, winSize: int = 5) -> NDArray:
    ''' Smooth an 1D array using moving average window with length `winSize`.
        The implementation is from https://stackoverflow.com/questions/40443020
    '''
    cripAssert(is1D(data), '`data` should be 1D array.')
    cripAssert(isInt(winSize) and winSize % 2 == 1, '`winSize` should be odd positive integer.')

    out0 = np.convolve(data, np.ones(winSize, dtype=int), 'valid') / winSize
    r = np.arange(1, winSize - 1, 2)
    start = np.cumsum(data[:winSize - 1])[::2] / r
    stop = (np.cumsum(data[:-winSize:-1])[::2] / r)[::-1]

    return np.concatenate((start, out0, stop))


def smoothZ(img: ThreeD, ksize=3) -> ThreeD:
    ''' Smooth a 3D image using a uniform filter with `ksize` along Z dimension.
    '''
    cripAssert(is3D(img), '`img` should be 3D array.')

    kernel = (ksize, 1, 1)
    img = uniform_filter(img, kernel, mode='reflect')

    return img


def window(img: TwoOrThreeD,
           win: Tuple[float, float],
           style: str = 'lr',
           normalize: Or[str, None] = None) -> TwoOrThreeD:
    ''' Window `img` using `win` (WW, WL) with style `wwwl` or (left, right) with style `lr`.
        Set `normalize` to `0255` to convert to 8-bit image, or `01` to [0, 1] float image.
    '''
    cripAssert(len(win) == 2, '`win` should have length of 2.')
    cripAssert(style in ['wwwl', 'lr'], "`style` should be 'wwwl' or 'lr'")

    if style == 'wwwl':
        ww, wl = win
        l = wl - ww / 2
        r = l + ww
    elif style == 'lr':
        l, r = win

    res = asFloat(img.copy())
    res[res > r] = r
    res[res < l] = l

    if normalize == '0255':
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif normalize == '01':
        res = (res - l) / (r - l)

    return res


def windowFullRange(img: TwoOrThreeD, normalize='01') -> TwoOrThreeD:
    ''' Window `img` using its full dynamic range of pixel values.
    '''
    return window(img, (np.max(img), np.min(img)), 'lr', normalize)


def zoomIn(img: TwoD, row: int, col: int, h: int, w: int) -> TwoD:
    ''' Crop a patch. `(row, col)` determines the left-top point. `(h, w)` gives height and width.
    '''
    return img[row:row + h, col:col + w]


def stddev(img: TwoD) -> float:
    ''' Compute the standard deviation of a image crop.
        `(row, col)` determines the left-top point. (h, w) gives height and width.
    '''
    return np.std(img)


def meanstd(x: Any) -> Tuple[float, float]:
    ''' Compute the mean and standard deviation of `x`.
    '''
    return np.mean(x), np.std(x)


def fontdict(family, weight, size):
    return {'family': family, 'weight': weight, 'size': size}


class ImageGrid:
    subimgs: List[TwoD]
    nrow: int
    ncol: int
    fig: matplotlib.figure.Figure
    grid: MplImageGrid

    # titles
    rowTitles: List[str] = None
    colTitles: List[str] = None
    # preprocessor
    preprocessor: Callable = None
    # fontdict
    fontdict: Dict = None
    # crops
    crops: List[Tuple[int, int, int]] = None
    cropLoc: str = 'bottom left'
    cropSize: int = 96 * 2

    def __init__(self, subimgs: List[TwoD], nrow: int, ncol: int) -> None:
        ''' Initialize the ImageGrid with `subimgs` in `nrow` * `ncol` layout.
        '''
        self.subimgs = subimgs
        self.nrow = nrow
        self.ncol = ncol
        cripAssert(len(subimgs) == nrow * ncol, 'Number of subimages should be equal to `nrow * ncol`.')

    def setTitles(self, rowTitles: List[str], colTitles: List[str]):
        ''' Set the row and column titles.
        '''
        self.rowTitles = rowTitles
        self.colTitles = colTitles

    def setPreprocessor(self, fn: Callable):
        ''' Set the preprocessor for the subimages.
            A preprocessor is a function that takes the index of a subimage and the subimage and returns a new one.
        '''
        self.preprocessor = fn

    def setFontdict(self, fontdict: Dict):
        ''' Set the fontdict for the texts in the figure.
        '''
        self.fontdict = fontdict

    def setCrops(self, crops, cropLoc='bottom left', cropSize=96 * 2):
        self.crops = crops
        self.cropLoc = cropLoc
        self.cropSize = cropSize

    def _overlayPatch(self, img, patch, loc):
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
            cripAssert(False, 'Currently only loc at `bottom left` is supported.')

    def fig(self):
        ''' Execute the plot and return the figure.
        '''
        # preprocess the subimages
        if self.preprocessor is not None:
            self.subimages = list(map(lambda ix: self.preprocessor(*ix), list(enumerate(self.subimages))))

        # create the figure
        self.fig = plt.figure(figsize=(self.ncol * 2, self.nrow * 2))
        self.grid = MplImageGrid(self.fig, 111, nrows_ncols=(self.nrow, self.ncol), axes_pad=0)

        # display the subimages
        cur = 0
        for ax, img in zip(self.grid, self.subimages):
            # remove the ticks and spines
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            list(map(lambda x: x.set_visible(False), ax.spines.values()))

            # prepare the image crop
            box = None
            if self.crops is not None and self.crops[cur // self.ncol]:
                r, c, hw = self.crops[cur // self.ncol]
                patch = resizeTo(zoomIn(img, r, c, hw, hw), (self.cropSize, self.cropSize))
                box = self._overlayPatch(img, patch, self.cropLoc)

            # display the image
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            # display the crop box
            box and ax.add_patch(box)
            if box and cur % self.ncol == 0:
                r, c, hw = self.crops[cur // self.ncol]
                box = matplotlib.patches.Rectangle((c, r), hw, hw, linewidth=0.8, edgecolor='yellow', facecolor='none')
                ax.add_patch(box)

            # display the column titles
            if self.colTitles and cur < len(self.colTitles):
                ax.set_title(self.colTitles[cur], loc='center', fontdict=fontdict)
            cur += 1

        # display the row titles
        if self.rowTitles:
            for i in range(self.nrows):
                self.grid[self.ncols * i].set_ylabel(self.rowTitles[i], fontdict=fontdict)

        return self.fig


def plotSpectrum(ax: matplotlib.axes.Axes, spec: Spectrum):
    ''' Plot the spectrum using handler `ax`.
    '''
    energies = DiagEnergyRange
    omega = spec.omega

    ax.plot(energies, omega, 'k')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Omega (a.u.)')


def plotMu(ax: matplotlib.axes.Axes, atten: Atten, startEnergy: int = 1, logScale=True):
    ''' Plot the LACs of `atten` from `startEnergy` keV in ax in `logScale` if true.
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
