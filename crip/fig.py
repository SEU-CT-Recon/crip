'''
    Figure artist module of crip.

    https://github.com/z0gSh1u/crip
'''

import numpy as np
from matplotlib import font_manager


# https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
def smooth(a, WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def window(x, w, style='lr'):
    res = x.copy()
    if style == 'wwwl':
        ww, wl = w
        l = wl - ww / 2
        r = l + ww
    elif style == 'lr':
        l, r = w
    else:
        raise
    res[res > r] = r
    res[res < l] = l
    return res


def zoomIn(img, x, y, hw):
    return img[y:y + hw, x:x + hw]


def average(imgs, i, r):
    return np.mean(imgs[i - r:i + r], axis=0)


def addFont(path):
    font_files = font_manager.findSystemFonts(fontpaths=[path])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


def regionCrop(x, lt, rb):
    y = x.copy()
    y = y[lt[0]:rb[0], lt[1]:rb[1]]
    return np.array(y)
