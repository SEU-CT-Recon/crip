'''
    Postprocess module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np

from .shared import *
from ._typing import *
from .utils import *


def fovCropRadius(sid: float, sdd: float, detectorWidth: float, pixelSize: float, roundOff=True) -> Or[int, float]:
    ''' Compute the radius [pixel] of the valid FOV of the reconstruction.
        `sid` and `sdd` are Source-Isocenter-Distance and Source-Detector-Distance, respectively.
        `detectorWidth` is the width of the detector, i.e., `#elements * elementWidth`.
        `pixelSize` is the pixel size of the reconstructed image.
        Recommended length unit is [mm].
    '''
    halfDW = detectorWidth / 2
    L = np.sqrt(halfDW**2 + sdd**2)

    r1 = sid * np.arcsin(halfDW / L)  # table as arc
    r2 = sid / (sdd / detectorWidth) / 2  # table as plane
    r3 = sid / (L / halfDW)

    r = min(r1, r2, r3) / pixelSize

    return round(r) if roundOff else r


@ConvertListNDArray
def fovCrop(img: TwoOrThreeD, radius: int, fill: Or[int, float] = 0) -> TwoOrThreeD:
    ''' Crop a circle FOV on reconstructed image `img` with `radius` [pixel]
        and `fill` outside FOV.
    '''
    cripAssert(radius >= 1 and isInt(radius), 'Invalid radius. Radius should be positive int.')
    cripAssert(is2or3D(img), 'img should be 2D or 3D.')

    N, M = img.shape[-2:]
    x = np.array(range(N), dtype=DefaultFloatDType) - N / 2 - 0.5
    y = np.array(range(M), dtype=DefaultFloatDType) - M / 2 - 0.5
    xx, yy = np.meshgrid(x, y)
    outside = xx**2 + yy**2 > radius**2

    cropped = img.copy()
    if is3D(img):
        cropped[:, outside] = fill
    else:
        cropped[outside] = fill

    return cropped


@ConvertListNDArray
def muToHU(image: TwoOrThreeD, muWater: float, b=1000) -> TwoOrThreeD:
    ''' Convert mu to HU using `HU = (mu - muWater) / muWater * b`
    '''
    return (image - muWater) / muWater * b


@ConvertListNDArray
def huToMu(image: TwoOrThreeD, muWater: float, b=1000) -> TwoOrThreeD:
    ''' Convert HU to mu (invert `muToHU`).
    '''
    return image / b * muWater + muWater


@ConvertListNDArray
def huNoRescale(image: TwoOrThreeD, b: float = -1000, k: float = 1) -> TwoOrThreeD:
    ''' Invert the rescale-slope (y = kx + b) of HU value to get linear relationship between HU and mu.
    '''
    return (image - b) / k


@ConvertListNDArray
def postlogsToRaws(postlogs: TwoOrThreeD, flat: Or[TwoD, float]) -> TwoOrThreeD:
    ''' Invert `postlog` images to the original raw projections according to `flat` field.
    '''
    return np.exp(-postlogs) * flat
