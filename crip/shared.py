'''
    Shared module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from os import path
import cv2
import scipy.ndimage

from .utils import *
from ._typing import *
from .io import imreadTiff


@ConvertListNDArray
def rotate(img: TwoOrThreeD, deg: int) -> TwoOrThreeD:
    ''' Rotate each image by deg [DEG] (multiple of 90) clockwise.
    '''
    cripAssert(deg % 90 == 0, f'`deg` should be multiple of 90, but got {deg}.')

    k = int(deg % 360) // 90
    axes = (1, 2) if is3D(img) else (0, 1)

    return np.rot90(img, -k, axes)


@ConvertListNDArray
def verticalFlip(img: TwoOrThreeD, copy=False) -> TwoOrThreeD:
    ''' Vertical flip each image.
    '''
    cripAssert(is2or3D(img), f'img should be 2D or 3D, but got {img.ndim}-D.')

    if copy:
        return np.array(img[..., ::-1, :])
    else:
        return img[..., ::-1, :]


@ConvertListNDArray
def horizontalFlip(img: TwoOrThreeD, copy=False) -> TwoOrThreeD:
    ''' Horizontal flip each image.
    '''
    cripAssert(is2or3D(img), f'img should be 2D or 3D, but got {img.ndim}-D.')

    if copy:
        return np.array(img[..., ::-1])
    else:
        return img[..., ::-1]


@ConvertListNDArray
def stackFlip(img: ThreeD, copy=False) -> ThreeD:
    ''' Flip a stack w.r.t. z-axis, i.e., reverse slices.
    '''
    cripAssert(is3D(img), 'img should be 3D.')

    if copy:
        return np.array(np.flip(img, axis=0))
    else:
        return np.flip(img, axis=0)


@ConvertListNDArray
def resizeTo(img: TwoOrThreeD, dsize: Tuple[int], interp='bicubic') -> TwoOrThreeD:
    ''' Resize each image to `dsize=(H, W)` using `interp` [bicubic, linear, nearest].
    '''
    cripAssert(interp in ['bicubic', 'linear', 'nearest'], f'Invalid interp method: {interp}.')
    cripAssert(len(dsize) == 2, '`dsize` should be 2D.')

    dsize = dsize[::-1]  # OpenCV dsize is in (W, H) form, so we reverse it.
    interp_ = {'bicubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}

    if is3D(img):
        res = [cv2.resize(img[i, ...], dsize, None, interpolation=interp_[interp]) for i in range(img.shape[0])]
        return np.array(res)
    else:
        return cv2.resize(img, dsize, None, interpolation=interp_[interp])


@ConvertListNDArray
def resizeBy(img: TwoOrThreeD, factor: Or[Tuple[float], float], interp='bicubic') -> TwoOrThreeD:
    ''' Resize each slice in img by `factor=(fH, fW)` or `(f, f)` float using `interp` [bicubic, linear, nearest].
    '''
    cripAssert(interp in ['bicubic', 'linear', 'nearest'], f'Invalid interp method: {interp}.')
    if not isType(factor, Tuple):
        factor = (factor, factor)
    cripAssert(len(factor) == 2, '`factor` should be 2D.')

    interp_ = {'bicubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}
    fH, fW = factor

    if is3D(img):
        res = [cv2.resize(img[i, ...], None, None, fW, fH, interpolation=interp_[interp]) for i in range(img.shape[0])]
        return np.array(res)
    else:
        print(img, fW, fH, interp_[interp])
        return cv2.resize(img, None, None, fW, fH, interpolation=interp_[interp])


@ConvertListNDArray
def resize3D(img: ThreeD, factor: Tuple[int], order=3) -> ThreeD:
    ''' Resize 3D `img` by `factor=(fSlice, fH, fW)` using `order`-spline interpolation.
    '''
    cripAssert(len(factor) == 3, '`factor` should be 3D.')

    return scipy.ndimage.zoom(img, factor, None, order, mode='mirror')


@ConvertListNDArray
def gaussianSmooth(img: TwoOrThreeD,
                   sigma: Or[float, int, Tuple[Or[float, int]]],
                   ksize: Or[Tuple[int], int, None] = None) -> TwoOrThreeD:
    ''' Perform Gaussian smooth with kernel size `ksize` and Gaussian sigma = `sigma` [int or tuple (row, col)].
        Leave `ksize=None` to auto determine it to include the majority of kernel energy.
    '''
    if ksize is not None:
        if not isType(ksize, Sequence):
            cripAssert(isInt(ksize), 'ksize should be int or sequence of int.')
            ksize = (ksize, ksize)
        else:
            cripAssert(len(ksize) == 2, 'ksize should have length 2.')
            cripAssert(isInt(ksize[0]) and isInt(ksize[1]), 'ksize should be int or sequence of int.')

    if not isType(sigma, Sequence):
        sigma = (sigma, sigma)

    if is3D(img):
        res = [cv2.GaussianBlur(img[i, ...], ksize, sigmaX=sigma[1], sigmaY=sigma[0]) for i in range(img.shape[0])]
        return np.array(res)
    else:
        return cv2.GaussianBlur(img, ksize, sigmaX=sigma[1], sigmaY=sigma[0])


@ConvertListNDArray
def stackImages(imgList: ListNDArray, dtype=None) -> NDArray:
    ''' Stack seperate image into one numpy array. I.e., views * (h, w) -> (views, h, w).
        Convert dtype when `dtype` is not `None`.
    '''
    if dtype is not None:
        imgList = imgList.astype(dtype)

    return imgList


def splitImages(imgs: ThreeD, dtype=None) -> ListNDArray:
    ''' Split stacked image into seperate numpy arrays. I.e., (views, h, w) -> views * (h, w).
        Convert dtype when `dtype` is not `None`.
    '''
    cripAssert(is3D(imgs), 'imgs should be 3D.')

    if dtype is not None:
        imgs = imgs.astype(dtype)

    return list(imgs)


@ConvertListNDArray
def binning(img: TwoOrThreeD, rates: Tuple[int]) -> TwoOrThreeD:
    ''' Perform binning with `rates = (c, h, w) / (h, w)`.
    '''
    for rate in rates:
        cripAssert(isInt(rate) and rate > 0, 'rates should be positive int.')
    if is2D(img):
        cripAssert(len(rates) == 2, 'img is 2D, while rates is 3D.')
    if is3D(img):
        cripAssert(len(rates) == 3, 'img is 3D, while rates is 2D.')

    if rates[-1] != 1:
        img = img[..., ::rates[-1]]
    if rates[-2] != 1:
        img = img[..., ::rates[-2], :]
    if len(rates) == 3 and rates[0] != 1:
        img = img[::rates[0], ...]

    return img


@ConvertListNDArray
def transpose(vol: TwoOrThreeD, order: Tuple[int]) -> TwoOrThreeD:
    ''' Transpose vol with axes swapping `order`.
    '''
    if is2D(vol):
        cripAssert(len(order) == 2, 'order should have length 2 for 2D input.')
    if is3D(vol):
        cripAssert(len(order) == 3, 'order should have length 3 for 3D input.')

    return vol.transpose(order)


@ConvertListNDArray
def permute(vol: ThreeD, from_: str, to: str) -> ThreeD:
    ''' Permute axes (transpose) from `from_` to `to`, reslicing the reconstructed volume.

        Valid directions are: 'sagittal', 'coronal', 'transverse'.
    '''
    dirs = ['sagittal', 'coronal', 'transverse']

    cripAssert(from_ in dirs, f'Invalid direction string: {from_}')
    cripAssert(to in dirs, f'Invalid direction string: {to}')

    if from_ == to:
        return vol

    dirFrom = dirs.index(from_)
    dirTo = dirs.index(to)

    # TODO check this matrix
    orders = [
        # to sag       cor         tra      # from
        [(0, 1, 2), (1, 2, 0), (2, 1, 0)],  # sag
        [(1, 2, 0), (0, 1, 2), (1, 0, 2)],  # cor
        [(2, 1, 0), (1, 0, 2), (0, 1, 2)],  # tra
    ]
    order = orders[dirFrom][dirTo]

    return transpose(vol, order)


def shepplogan(size: int = 512) -> TwoD:
    ''' Generate a `size x size` Shepp-Logan phantom.
    '''
    cripAssert(size in [256, 512, 1024], 'Shepp-Logan size should be in [256, 512, 1024]')

    return imreadTiff(path.join(getAsset('shepplogan'), f'{size}.tif'))


def fitPolyV2D2(x1: NDArray, x2: NDArray, y: NDArray, bias: bool = True) -> NDArray:
    ''' Fit a degree-2 polynomial using pseudo-inverse to a pair of variables `x1, x2`.
        Output 6 coefficients `c[0~5]`, minimizing the error of `y` and
        `c[0] * x1**2 + c[1] * x2**2 + c[2] * x1 * x2 + c[3] * x1 + c[4] * x2 + c[5]`.
        If `bias` is False, `c[5]` will be 0.
    '''
    cripAssert(is1D(x1) and is1D(x2) and is1D(y), 'Inputs should be 1D sequence.')
    cripAssert(isOfSameShape(x1, x2) and isOfSameShape(x1, y), '`x1`, `x2` and `y` should have same shape.')

    x1sq = x1.T**2
    x2sq = x2.T**2
    x1x2 = (x1 * x2).T
    x1 = x1.T
    x2 = x2.T
    const = (np.ones if bias else np.zeros)((x1.T.shape[0]))
    A = np.array([x1sq, x2sq, x1x2, x1, x2, const]).T

    return np.linalg.pinv(A) @ y


def applyPolyV2D2(coeff: NDArray, x1: NDArray, x2: NDArray) -> NDArray:
    ''' Apply a degree-2 polynomial to a pair of variables `x1, x2`.
        `coeff` has 5 or 6 elements, expands to 
        `c[0] * x1**2 + c[1] * x2**2 + c[2] * x1 * x2 + c[3] * x1 + c[4] * x2 + (c[5] or 0)`.
    '''
    cripAssert(len(coeff) in [5, 6], '`coeff` should have length of 5 or 6.')

    if len(coeff) == 5:
        bias = 0
    else:
        bias = coeff[5]

    return coeff[0] * x1**2 + coeff[1] * x2**2 + coeff[2] * x1 * x2 + coeff[3] * x1 + coeff[4] * x2 + bias


def fitPolyV1D2(x1: NDArray, y: NDArray, bias: bool = True) -> NDArray:
    ''' Fit a degree-2 polynomial using pseudo-inverse to a variable `x1`.
        Output 3 coefficients `c[0~2]`, minimizing the error of `y` and
        `c[0] * x1**2 + c[1] * x1 + c[2]`.
        If `bias` is False, `c[2]` will be 0.
    '''
    x1sq = x1.T**2
    x1 = x1.T
    const = (np.ones if bias else np.zeros)((x1.T.shape[0]))
    A = np.array([x1sq, x1, const]).T

    return np.linalg.pinv(A) @ y


def applyPolyV1D2(coeff: NDArray, x1: NDArray) -> NDArray:
    ''' Apply a degree-2 polynomial to a variable `x1`.
        `coeff` has 2 or 3 elements, expands to 
        `c[0] * x1**2 + c[1] * x1 + (c[2] or 0)`.
    '''
    cripAssert(len(coeff) in [2, 3], 'coeff should have length 2 or 3.')

    if len(coeff) == 2:
        bias = 0
    else:
        bias = coeff[2]

    return coeff[0] * x1**2 + coeff[1] * x1 + bias
