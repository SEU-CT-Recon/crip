'''
    Shared module of crip.

    https://github.com/z0gSh1u/crip
'''

import numpy as np
import cv2  # TODO: remove this in the future.

from .utils import ConvertListNDArray, cripAssert, inArray, is2D, is2or3D, is3D, isInt, isType
from ._typing import *


@ConvertListNDArray
def rotate(img: TwoOrThreeD, deg: int):
    '''
        Rotate the image or each image in a volume by deg [DEG] (multiple of 90) clockwise.
    '''
    deg = int(deg % 360)
    cripAssert(deg % 90 == 0, 'deg should be multiple of 90.')

    k = deg // 90
    axes = (1, 2) if is3D(img) else (0, 1)

    return np.rot90(img, -k, axes)


def verticalFlip(img: TwoOrThreeD):
    '''
        Vertical flip one image, or each image in a volume.
    '''
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    return img[..., ::-1, :]


def horizontalFlip(img: TwoOrThreeD):
    '''
        Horizontal flip one image, or each image in a volume.
    '''
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    return img[..., ::-1]


def resize(img: TwoOrThreeD, dsize: Tuple[int] = None, scale: Tuple[Or[float, int]] = None, interp: str = 'bicubic'):
    '''
        Resize the image or each image in a volume to `dsize = (H, W)` (if dsize is not None) or scale 
        by `scale = (facH, facW)` using `interp` (bicubic, linear, nearest available).
    '''
    cripAssert(inArray(interp, ['bicubic', 'linear', 'nearest']), 'Invalid interp method.')

    if dsize is None:
        cripAssert(scale is not None, 'dsize and scale cannot be None at the same time.')
        cripAssert(len(scale) == 2, 'scale should have length 2.')
        fH, fW = scale
    else:
        cripAssert(scale is None, 'scale should not be set when dsize is set.')
        # OpenCV dsize is in (W, H) form, so we reverse it.
        dsize = dsize[::-1]

    interp_ = {'bicubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}

    if is3D(img):
        c, _, _ = img.shape
        res = [cv2.resize(img[i, ...], dsize, None, fW, fH, interpolation=interp_[interp]) for i in range(c)]
        return np.array(res)
    else:
        return cv2.resize(img, dsize, None, fW, fH, interpolation=interp_[interp])


def gaussianSmooth(img: TwoOrThreeD, sigma: Or[float, int, Tuple[Or[float, int]]], ksize: int = None):
    '''
        Perform Gaussian smooth with kernel size = ksize and Gaussian \sigma = sigma (int or tuple (x, y)).
        
        Leave `ksize = None` to auto determine to include the majority of Gaussian energy.
    '''
    cripAssert(isInt(ksize), 'ksize should be int.')

    if not isType(sigma, Tuple):
        sigma = (sigma, sigma)

    if is3D(img):
        c, _, _ = img.shape
        res = [cv2.GaussianBlur(img[i, ...], ksize, sigmaX=sigma[0], sigmaY=sigma[1]) for i in range(c)]
        return np.array(res)
    else:
        return cv2.GaussianBlur(img, ksize, sigmaX=sigma[0], sigmaY=sigma[1])


@ConvertListNDArray
def stackImages(imgList: ListNDArray, dtype=None):
    '''
        Stack seperate image into one numpy array. I.e., views * (h, w) -> (views, h, w).

        Convert dtype with `dtype != None`.
    '''
    if dtype is not None:
        imgList = imgList.astype(dtype)

    return imgList


def splitImages(imgs: ThreeD, dtype=None):
    '''
        Split stacked image into seperate numpy arrays. I.e., (views, h, w) -> views * (h, w).

        Convert dtype with `dtype != None`.
    '''
    cripAssert(is3D(imgs), 'imgs should be 3D.')

    if dtype is not None:
        imgs = imgs.astype(dtype)

    return list(imgs)


@ConvertListNDArray
def binning(img: TwoOrThreeD, rate: int = 1, axis='y', mode='sample'):
    '''
        Perform binning on certain `axis` with `rate` and `mode`.
    '''
    cripAssert(isInt(rate) and rate > 0, 'rate should be positive int.')
    cripAssert(inArray(axis, ['x', 'y', 'z']), 'Invalid axis.')
    cripAssert(inArray(mode, ['sample', 'min', 'max', 'sum', 'mean']))

    axis = ['z', 'y', 'x'].index(axis)

    if is2D(img):
        cripAssert(axis != 'z', 'img is 2D, while axis is z.')
        axis -= 1

    if mode == 'sample':  # sample per `rate` frames
        # TODO make it nicer.
        if axis == 0:
            return img[..., ::rate]
        if axis == 1:
            return img[..., ::rate, :]
        if axis == 2:
            return img[::rate, ...]

    else:
        reducer = eval(f'np.{mode}')  # min, max, sum, mean
        # Tricky. Is it right?
        return reducer(np.reshape(img, (-1, rate)), axis)


def unBinning():
    pass  # TODO
