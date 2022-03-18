'''
    Shared module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import numpy as np
import cv2 # TODO: remove this.

from .utils import ConvertListNDArray, cripAssert, is2or3D, is3D
from .typing import ListNDArray, Or, Proj, ReconVolume


def rotate(img, deg):
    '''
        Rotate img by deg [DEG] (multiple of 90) clockwise.
    '''
    deg = int(deg % 360)
    if deg == 0:
        return img
    degToCode = {
        '90': cv2.ROTATE_90_CLOCKWISE,
        '180': cv2.ROTATE_180,
        '270': cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(img, degToCode[str(deg)])


def resize(projection, dsize=None, fH=None, fW=None, interp='bicubic'):
    '''
        Resize the projection to `dsize` (H, W) (if dsize is not None) or scale by ratio `(fH, fW)`
        using `interp` (bicubic, linear, nearest available).
    '''
    if dsize is None:
        assert fH is not None or fW is not None
        fH = fH if fH else 1
        fW = fW if fW else 1
    else:
        assert fH is None and fW is None
    interp_ = {'bicubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}
    # OpenCV dsize is in (W, H) form, so we reverse it.
    return cv2.resize(projection.astype(np.float32), dsize[::-1], None, fW, fH, interpolation=interp_[interp])


def gaussianSmooth(projection, sigma, ksize=None):
    '''
        Perform Gaussian smooth with kernel size = ksize and Gaussian \sigma = sigma (int or tuple (x, y)).
        
        Leave `ksize = None` to auto determine to include the majority of Gaussian energy.
    '''
    if isinstance(sigma, int):
        sigma = (sigma, sigma)
    return cv2.GaussianBlur(projection.astype(np.float32), ksize, sigmaX=sigma[0], sigmaY=sigma[1])


def verticalFlip(img: Or[Proj, ReconVolume]):
    '''
        Vertical flip one image, or each image in a volume.
    '''
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    return img[..., ::-1, :]


def horizontalFlip(img: Or[Proj, ReconVolume]):
    '''
        Horizontal flip one image, or each image in a volume.
    '''
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    return img[..., ::-1]


@ConvertListNDArray
def stackImages(imgList: ListNDArray, dtype=None):
    '''
        Stack seperate image into one numpy array. I.e., views * (h, w) -> (views, h, w).

        Convert dtype with `dtype != None`.
    '''
    if dtype is not None:
        imgList = imgList.astype(dtype)

    return imgList


def splitImages(imgs: np.ndarray, dtype=None):
    '''
        Split stacked image into seperate numpy arrays. I.e., (views, h, w) -> views * (h, w).

        Convert dtype with `dtype != None`.
    '''
    cripAssert(is3D(imgs), 'imgs should be 3D.')

    if dtype is not None:
        imgs = imgs.astype(dtype)

    return list(imgs)
