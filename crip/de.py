'''
    Dual-Energy CT module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

__all__ = [
    'singleMatMuDecomp', 'calcAttenedSpec', 'calcPostLog', 'deDecompGetCoeff', 'deDecompProj', 'deDecompRecon',
    'genMaterialPhantom', 'deDecompReconVolCon', 'teDecompRecon', 'teDecompReconVolCon'
]

import numpy as np
from scipy.ndimage import uniform_filter

from .postprocess import gaussianSmooth
from .utils import ConvertListNDArray, cripAssert, cripWarning, is2D, isOfSameShape
from ._typing import *
from .physics import Atten, DiagEnergyRange, Spectrum, calcAttenedSpec, calcPostLog
from .shared import applyPolyV2D2, fitPolyV2D2


def singleMatMuDecomp(src: Atten, base1: Atten, base2: Atten, method='coeff', energyRange=DiagEnergyRange) -> NDArray:
    '''
        Decompose single material `src`'s attenuation onto `base1` and `base2`.

        Return decomposing coefficients along energies when `method = 'coeff'`, or proportion when `method = 'prop'`.
    '''
    cripAssert(method in ['coeff', 'prop'], 'Invalid method.')

    range_ = np.array(energyRange)
    srcMu = src.mu[range_]
    baseMu1 = base1.mu[range_]
    baseMu2 = base2.mu[range_]

    M = np.array([baseMu1, baseMu2], dtype=DefaultFloatDType).T

    if method == 'prop':
        M /= srcMu
        srcMu = np.ones_like(srcMu)

    res = np.linalg.pinv(M) @ (srcMu.T)

    return res


def deDecompGetCoeff(lowSpec: Spectrum, highSpec: Spectrum, base1: Atten, len1: Or[NDArray, Iterable], base2: Atten,
                     len2: Or[NDArray, Iterable]):
    '''
        Calculate the decomposing coefficient (Order 2 with bias term) of two spectra onto two material bases.
    '''
    lenCombo = []
    postlogLow = []
    postlogHigh = []
    for i in len1:
        for j in len2:
            lenCombo.append([i, j])
            postlogLow.append(calcPostLog(lowSpec, [base1, base2], [i, j]))
            postlogHigh.append(calcPostLog(highSpec, [base1, base2], [i, j]))

    beta, gamma = fitPolyV2D2(np.array(postlogLow), np.array(postlogHigh), np.array(lenCombo)).T

    return beta, gamma


@ConvertListNDArray
def deDecompProj(lowProj: TwoOrThreeD, highProj: TwoOrThreeD, coeff1: NDArray,
                 coeff2: NDArray) -> Tuple[TwoOrThreeD, TwoOrThreeD]:
    '''
        Perform dual-energy decompose in projection domain point-by-point using coeffs.

        Coefficients can be generated using @see `deDecompGetCoeff`.
    '''
    cripAssert(isOfSameShape(lowProj, highProj), 'Two projection sets should have same shape.')
    cripAssert(
        len(coeff1) == 6 and len(coeff2) == 6,
        'Decomposing coefficients should have length 6 (2 variable, order 2 with bias).')

    return applyPolyV2D2(coeff1, lowProj, highProj), applyPolyV2D2(coeff2, lowProj, highProj)


@ConvertListNDArray
def deDecompRecon(low: TwoOrThreeD,
                  high: TwoOrThreeD,
                  muBase1Low: float,
                  muBase1High: float,
                  muBase2Low: float,
                  muBase2High: float,
                  checkCond: bool = True):
    '''
        Perform dual-energy decompose in reconstruction domain. \\mu values can be calculated using @see `calcMu`.

        The values of input volumes should be \\mu value. The outputs are decomposing coefficients.
    '''
    cripAssert(isOfSameShape(low, high), 'Two volumes should have same shape.')
    COND_TOLERANCE = 1000

    A = np.array([[muBase1Low, muBase2Low], [muBase1High, muBase2High]])
    if checkCond:
        cripWarning(
            np.linalg.cond(A) <= COND_TOLERANCE, 'The material decomposition matrix possesses high condition number.')
    M = np.linalg.inv(A)

    def decompOne(low, high):
        c1 = M[0, 0] * low + M[0, 1] * high
        c2 = M[1, 0] * low + M[1, 1] * high
        return c1, c2

    if is2D(low):
        return decompOne(low, high)
    else:
        return np.array(list(map(lambda args: decompOne(*args), zip(low, high)))).transpose((1, 0, 2, 3))


def softThreshold(img: np.ndarray, l, h, mode='lower'):
    shape = img.shape
    img = img.flatten()

    lower = img < l
    upper = img >= h

    transitional = (img >= l) * (img < h)
    if mode == 'upper':
        transitional = transitional * (img - l) / (h - l)
        res = upper + transitional
    elif mode == 'lower':
        transitional = transitional * (h - img) / (h - l)
        res = lower + transitional

    return res.reshape(shape)


def genMaterialPhantom(img, zsmooth=3, sigma=1, l=80, h=300, base=1000):
    '''
        Generate the phantom of material bases (one is water) from SECT using soft-thresholding.
        zsmooth: smooth window in slice direction.
        sigma: Gaussian smooth sigma in single slice.
        [l, h] defines the fuzzy range of another material, e.g., bone.
        base: the reference HU of another material.
    '''
    assert np.min(img) < 0  # HU

    kernel = (zsmooth, 1, 1) if zsmooth is not None else (1, 1)
    img = uniform_filter(img, kernel, mode='reflect')
    img = gaussianSmooth(img, sigma)

    water = (img + 1000) / 1000 * softThreshold(img, l, h, 'lower')
    b2 = (img + 1000) / (base + 1000) * softThreshold(img, l, h, 'upper')

    return water, b2


def compose2(b1, b2, v1, v2):
    return b1 * v1 + b2 * v2


def compose3(b1, b2, b3, v1, v2, v3):
    return b1 * v1 + b2 * v2 + b3 * v3


@ConvertListNDArray
def deDecompReconVolCon(low: TwoOrThreeD, high: TwoOrThreeD, muBase1, muBase2, muBase3):
    '''Dual-Energy Triple-Material decomposition with Volume Conservation.
       muBase* = [low, high]
    '''
    cripAssert(isOfSameShape(low, high), 'Volumes should have same shape.')

    A = np.array([
        [muBase1[0], muBase2[0], muBase3[0]],
        [muBase1[1], muBase2[1], muBase3[1]],
        [1, 1, 1],
    ])
    M = np.linalg.pinv(A)

    def decompOne(low, high):
        c1 = M[0, 0] * low + M[0, 1] * high + M[0, 2]
        c2 = M[1, 0] * low + M[1, 1] * high + M[1, 2]
        c3 = M[2, 0] * low + M[2, 1] * high + M[2, 2]

        return c1, c2, c3

    if is2D(low):
        return decompOne(low, high)
    else:
        return np.array(list(map(lambda args: decompOne(*args), zip(low, high)))).transpose((1, 0, 2, 3))


@ConvertListNDArray
def teDecompRecon(low: TwoOrThreeD, mid: TwoOrThreeD, high: TwoOrThreeD, muBase1, muBase2, muBase3):
    '''Triple-Energy Triple-Material decomposition.
       muBase* = [low, mid, high]
    '''
    cripAssert(isOfSameShape(low, mid) and isOfSameShape(low, high), 'Volumes should have same shape.')

    A = np.array([
        [muBase1[0], muBase2[0], muBase3[0]],
        [muBase1[1], muBase2[1], muBase3[1]],
        [muBase1[2], muBase2[2], muBase3[2]],
    ])
    M = np.linalg.inv(A)

    def decompOne(low, mid, high):
        c1 = M[0, 0] * low + M[0, 1] * mid + M[0, 2] * high
        c2 = M[1, 0] * low + M[1, 1] * mid + M[1, 2] * high
        c3 = M[2, 0] * low + M[2, 1] * mid + M[2, 2] * high

        return c1, c2, c3

    if is2D(low):
        return decompOne(low, mid, high)
    else:
        return np.array(list(map(lambda args: decompOne(*args), zip(low, mid, high)))).transpose((1, 0, 2, 3))


@ConvertListNDArray
def teDecompReconVolCon(low: TwoOrThreeD, mid: TwoOrThreeD, high: TwoOrThreeD, muBase1, muBase2, muBase3):
    '''Triple-Energy Triple-Material decomposition with Volume Conservation.
       muBase* = [low, mid, high]
    '''
    cripAssert(isOfSameShape(low, mid) and isOfSameShape(low, high), 'Volumes should have same shape.')

    A = np.array([
        [muBase1[0], muBase2[0], muBase3[0]],
        [muBase1[1], muBase2[1], muBase3[1]],
        [muBase1[2], muBase2[2], muBase3[2]],
        [1, 1, 1],
    ])
    M = np.linalg.pinv(A)

    def decompOne(low, mid, high):
        c1 = M[0, 0] * low + M[0, 1] * mid + M[0, 2] * high + M[0, 3]
        c2 = M[1, 0] * low + M[1, 1] * mid + M[1, 2] * high + M[1, 3]
        c3 = M[2, 0] * low + M[2, 1] * mid + M[2, 2] * high + M[2, 3]

        return c1, c2, c3

    if is2D(low):
        return decompOne(low, mid, high)
    else:
        return np.array(list(map(lambda args: decompOne(*args), zip(low, mid, high)))).transpose((1, 0, 2, 3))
