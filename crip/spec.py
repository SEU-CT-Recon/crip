'''
    Spectrum CT module of crip.

    https://github.com/SEU-CT-Recon/crip
'''

import numpy as np
from scipy.ndimage import uniform_filter

from .postprocess import gaussianSmooth
from .utils import ConvertListNDArray, cripAssert, cripWarning, is2D, isOfSameShape
from ._typing import *
from .physics import Atten, Spectrum, computeAttenedSpectrum
from .shared import applyPolyV2D2, fitPolyV2D2


def deDecompProjCoeff(lowSpec: Spectrum, highSpec: Spectrum, base1: Atten, len1: Or[NDArray, Iterable], base2: Atten,
                      len2: Or[NDArray, Iterable]):
    ''' Compute the decomposing coefficient (Order 2 with bias term) of two spectra onto two material bases
        used in the projection domain.
    '''

    def computePostlog(spec, attens, Ls):
        flat = np.sum(spec.omega)
        attenedSpec = computeAttenedSpectrum(spec, attens, Ls)

        return -np.log(attenedSpec.omega / flat)

    lenCombo = []
    postlogLow = []
    postlogHigh = []
    for i in len1:
        for j in len2:
            lenCombo.append([i, j])
            postlogLow.append(computePostlog(lowSpec, [base1, base2], [i, j]))
            postlogHigh.append(computePostlog(highSpec, [base1, base2], [i, j]))

    beta, gamma = fitPolyV2D2(np.array(postlogLow), np.array(postlogHigh), np.array(lenCombo)).T

    return beta, gamma


@ConvertListNDArray
def deDecompProj(lowProj: TwoOrThreeD, highProj: TwoOrThreeD, coeff1: NDArray,
                 coeff2: NDArray) -> Tuple[TwoOrThreeD, TwoOrThreeD]:
    ''' Perform dual-energy decompose in projection domain point-by-point using coeffs.
        Coefficients can be generated using @see `deDecompProjCoeff`.
    '''
    cripAssert(isOfSameShape(lowProj, highProj), 'Two projection sets should have same shape.')
    cripAssert(
        len(coeff1) == 6 and len(coeff2) == 6,
        'Decomposing coefficients should have length 6 (2 variable, order 2 with bias).')

    return applyPolyV2D2(coeff1, lowProj, highProj), applyPolyV2D2(coeff2, lowProj, highProj)


@ConvertListNDArray
def deDecompRecon(low: TwoOrThreeD,
                  high: TwoOrThreeD,
                  muB1: List[float],
                  muB2: List[float],
                  checkCond: bool = True) -> TwoOrThreeD:
    ''' Perform Dual-Energy Two-Material decomposition in image domain.
        The outputs are decomposing coefficients. Used values can be LAC (mu), or HU+1000.
        `muB*` stores the value of base* in (low-energy, high-energy) order.
    '''
    cripAssert(isOfSameShape(low, high), 'Two volumes should have same shape.')
    COND_TOLERANCE = 1000

    A = np.array([
        [muB1[0], muB2[0]],
        [muB1[1], muB2[1]],
    ])
    checkCond and cripWarning(
        np.linalg.cond(A) <= COND_TOLERANCE, 'Material decomposition matrix possesses high condition number.')
    M = np.linalg.inv(A)

    def decomp1(low, high):
        c1 = M[0, 0] * low + M[0, 1] * high
        c2 = M[1, 0] * low + M[1, 1] * high
        return c1, c2

    if is2D(low):
        return decomp1(low, high)
    else:
        return np.array(list(map(lambda args: decomp1(*args), zip(low, high)))).transpose((1, 0, 2, 3))


@ConvertListNDArray
def teDecompRecon(low: TwoOrThreeD, mid: TwoOrThreeD, high: TwoOrThreeD, muB1: List[float], muB2: List[float],
                  muB3: List[float]) -> TwoOrThreeD:
    ''' Perform Triple-Energy Three-Material decomposition in image domain.
        The outputs are decomposing coefficients. Used values can be LAC (mu), or HU+1000.
        `muB*` stores the value of base* in low-energy, mid-energy, high-energy order.
    '''
    cripAssert(isOfSameShape(low, mid) and isOfSameShape(low, high), 'Volumes should have same shape.')

    A = np.array([
        [muB1[0], muB2[0], muB3[0]],
        [muB1[1], muB2[1], muB3[1]],
        [muB1[2], muB2[2], muB3[2]],
    ])
    M = np.linalg.inv(A)

    def decomp1(low, mid, high):
        c1 = M[0, 0] * low + M[0, 1] * mid + M[0, 2] * high
        c2 = M[1, 0] * low + M[1, 1] * mid + M[1, 2] * high
        c3 = M[2, 0] * low + M[2, 1] * mid + M[2, 2] * high

        return c1, c2, c3

    if is2D(low):
        return decomp1(low, mid, high)
    else:
        return np.array(list(map(lambda args: decomp1(*args), zip(low, mid, high)))).transpose((1, 0, 2, 3))


@ConvertListNDArray
def deDecompReconVolCon(low: TwoOrThreeD, high: TwoOrThreeD, muB1: List[float], muB2: List[float],
                        muB3: List[float]) -> TwoOrThreeD:
    ''' Perform Dual-Energy Three-Material decomposition with Volume Conservation constraint in image domain.
        `muB*` stores the value of base* in low-energy, high-energy order.
    '''
    pseudoMid = np.zeros_like(low)

    return teDecompRecon(low, pseudoMid, high, [*muB1, 1.0], [*muB2, 1.0], [*muB3, 1.0])


def genMaterialPhantom(img, zsmooth=3, sigma=1, l=80, h=300, base=1000):
    ''' Generate the phantom of material bases (one is water) from SECT using soft-thresholding.
        zsmooth: smooth window in slice direction.
        sigma: Gaussian smooth sigma in single slice.
        [l, h] defines the fuzzy range of another material, e.g., bone.
        base: the reference HU of another material.
    '''
    cripAssert(np.min(img)) < 0  # HU

    def softThreshold(img: NDArray, l, h, mode='lower'):
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

    kernel = (zsmooth, 1, 1) if zsmooth is not None else (1, 1)
    img = uniform_filter(img, kernel, mode='reflect')
    img = gaussianSmooth(img, sigma)

    water = (img + 1000) / 1000 * softThreshold(img, l, h, 'lower')
    b2 = (img + 1000) / (base + 1000) * softThreshold(img, l, h, 'upper')

    return water, b2


def compose2(b1: float, b2: float, v1: NDArray, v2: NDArray) -> NDArray:
    ''' Compose two vectors `(v1, v2)` by coeffs `(b1, b2)`.
    '''
    return b1 * v1 + b2 * v2


def compose3(b1: float, b2: float, b3: float, v1: NDArray, v2: NDArray, v3: NDArray) -> NDArray:
    ''' Compose three vectors `(v1, v2, v3)` by coeffs `(b1, b2, b3)`.
    '''
    return b1 * v1 + b2 * v2 + b3 * v3


def vmi2Mat(b1: TwoOrThreeD, b2, b1Mat: Atten, b2Mat: Atten, E: int):
    ''' Virtual Monoenergetic Imaging using two-material decomposition at energy `E` [keV].
    '''
    return b1 * b1Mat.mu[E] + b2 * b2Mat.mu[E]


def vmi3Mat(b1, b2, b3, b1Mat, b2Mat, b3Mat, E):
    ''' Virtual Monoenergetic Imaging using three-material decomposition.
    '''
    return b1 * b1Mat.mu[E] + b2 * b2Mat.mu[E] + b3 * b3Mat.mu[E]


def deSubtration(low: TwoOrThreeD, high: TwoOrThreeD, kLow: float, kHigh: float = 1) -> TwoOrThreeD:
    ''' Dual-Energy subtraction for e.g. bone removal in CT slices or X-ray DR projections.
    '''
    return kHigh * high - kLow * low


def vncBasis(contrastHULow: float, contrastHUHigh: float) -> NDArray:
    ''' Construct Virtual Non-Contrast basis on CT images with HU as value.
    '''
    pass
