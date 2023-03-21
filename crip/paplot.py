'''
    Figure artist module of crip.
    The name paplot follows pyplot and means paper a lot.
    ! This module is still working in progress.

    https://github.com/z0gSh1u/crip
'''

import cv2
import numpy as np
from matplotlib import font_manager
from ._typing import *
from .utils import cripAssert, is1D, is2D, isInt

__all__ = ['smooth', 'window', 'average', 'addFont', 'fontdict', 'zoomIn', 'Helper', 'LineProfile']


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


class Helper():
    def __init__(self, plt) -> None:
        self.plt = plt
        plt.rc('axes', unicode_minus=False)
        plt.rc('grid', linestyle="--", color='#D3D3D3')

    def figure(self, figsize: Tuple[int, int]):
        self.plt.figure(figsize=figsize)

    def xyLabel(self, x: str, y: str):
        self.plt.xlabel(x)
        self.plt.ylabel(y)

    def legend(self, legend: List[str], loc='best', style='stack'):
        if style == 'stack':
            self.plt.legend(legend, loc=loc)
        elif style == 'expand':
            self.plt.legend(legend, loc=loc, mode='expand', ncol=len(legend))

    def grid(self, on=True):
        self.plt.grid(on)

    def xyLim(self, xlim, ylim):
        self.plt.xlim(*xlim)
        self.plt.ylim(*ylim)

    def tight(self):
        self.plt.tight_layout()

    def save(self, path: str, dpi=100, tight=True):
        if tight:
            self.tight()
        self.plt.savefig(path, dpi=dpi)

    def show(self):
        self.plt.show()

    def xyNBin(self, xbins, ybins):
        self.plt.locator_params(nbins=xbins, axis='x')
        self.plt.locator_params(nbins=ybins, axis='y')


class LineProfile(Helper):
    def __init__(self, plt, imgs: List[TwoOrThreeD], smoother: Or[Callable, None] = None) -> None:
        super().__init__(plt)

        self.imgs = np.array(imgs)
        if is2D(imgs[0]):
            self.imgs = self.imgs[np.newaxis, ...]
        if len(imgs) == 1:
            self.imgs = self.imgs[np.newaxis, ...]
        # Now we have B, C, H, W.
        self.slice = 0
        self.startPoint = None
        self.length = None
        self.direction = 'x'
        self.smoother = smoother

    def fetch(self):
        cripAssert(self.startPoint is not None, '`startPoint` is None.')
        cripAssert(self.length is not None, '`length` is None.')
        cripAssert(self.direction in ['x', 'y'], '`direction` should be `x` or `y`.')
        values = []

        for img in self.imgs:
            if self.direction == 'x':
                value = img[self.slice, self.startPoint[0], self.startPoint[1] + self.length]
            else:
                value = img[self.slice, self.startPoint[0] + self.length, self.startPoint[1]]
            if self.smoother is not None:
                value = self.smoother(value)

            values.append(value)

        return values


def fontdict(family, weight, size):
    return {'family': family, 'weight': weight, 'size': size}
