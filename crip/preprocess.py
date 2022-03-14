'''
    Preprocess module of crip.

    https://github.com/z0gSh1u/crip
'''

import numpy as np
from .shared import *
from .typing import *
from .utils import *


@ConvertProjList
def averageProjections(projections: Or[ProjList, ProjStack]):
    """
        Average projections. For example, to calculate the flat field.
        Projections can be either `(views, H, W)` shaped numpy array, or
        `views * (H, W)` Python List.
    """
    cripAssert(is3D(projections), '`projections` should be 3D array.')
    projections = ensureFloatArray(projections)

    res = projections.sum(axis=0) / projections.shape[0]

    return res


@ConvertProjList
def flatDarkFieldCorrection(projections: Or[Proj, ProjList, ProjStack],
                            flat: Or[Proj, float],
                            coeff: float = 1,
                            dark: Or[Proj, float] = 0):
    """
        Perform flat field (air) and dark field correction to get post-log value.
        I.e., `- log [(X - D) / (C * F - D)]`. Multi projections accepted.
    """
    sampleProjection = projections if is2D(projections) else projections[0]

    if isType(flat, Proj):
        cripAssert(haveSameShape(sampleProjection, flat), "`projection` and `flat` should have same shape.")
    if isType(dark, Proj):
        cripAssert(haveSameShape(sampleProjection, dark), "`projection` and `dark` should have same shape.")

    res = -np.log((projections - dark) / (coeff * flat - dark))
    res[res == np.inf] = 0
    res[res == np.nan] = 0

    return res


def flatDarkFieldCorrectionStandalone(projection: Proj):
    """
        Perform flat field and dark field correction without actual field image. \\
        Air is estimated using the brightest pixel by default.
    """
    # We support 2D only in standalone version, since `flat` for each projection might differ.
    cripAssert(is2D(projection), '`projection` should be 2D.')

    return flatDarkFieldCorrection(projection, np.max(projection), 1, 0)


@ConvertProjList
def injectGaussianNoise(projections: Or[Proj, ProjList, ProjStack], sigma: float, mu: float = 0):
    """
        Inject Gaussian noise which obeys distribution `N(\mu, \sigma^2)`.
    """
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')

    def injectOne(img):
        noise = np.random.randn(*img.shape) * sigma + mu
        return img + noise

    if is3D(projections):
        res = np.zeros_like(projections)
        for c in projections.shape[0]:
            res[c, ...] = injectOne(projections[c, ...])
    else:
        res = injectOne(projections)

    return res


def injectPoissonNoise(projection: Proj):
    raise 'Unimplemented for now.'
    pass  # TODO


def limitedAngle(projections, srcDeg, dstDeg, startDeg=0):
    """
        Sample limited angle projections from `startDeg` to `startDeg + dstDeg`. \\
        The original total angle is `srcDeg`.
    """
    assert startDeg + dstDeg <= srcDeg
    nProjPerDeg = float(projections.shape[0]) / srcDeg
    startLoc = int(startDeg * nProjPerDeg)
    dstLen = int(dstDeg * nProjPerDeg)

    return np.array(projections[startLoc:startLoc + dstLen, :, :])


def limitedView(projections, ratio):
    """
        Sample projections uniformly with `::ratio` to get sparse views projections. \\
        The second returning is the number of remaining projections.
    """
    dstLen = projections.shape[0] / ratio
    assert dstLen == int(dstLen), "Cannot achieve uniform sampling."

    return np.array(projections[::ratio, :, :]), projections.shape[0] % ratio - 1


@ConvertProjList
def projectionsToSinograms(projections: Or[ProjList, ProjStack]):
    """
        Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.
    """
    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w), dtype=projections.dtype)
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]

    return sinograms


@ConvertProjList
def sinogramsToProjections(sinograms: Or[ProjList, ProjStack]):
    """
        Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.
    """
    (h, views, w) = sinograms.shape
    projections = np.zeros((views, h, w), dtype=sinograms.dtype)
    for i in range(views):
        projections[i, :, :] = sinograms[:, i, :]

    return projections


def padImage(proj, padding, mode='symmetric', smootherDecay=False):
    """
        Pad the image on four directions using symmetric `padding` (Up, Right, Down, Left) \\
        and descending cosine window decay. `mode` can be `symmetric` or `edge`.
    """
    h, w = proj.shape
    nPadU, nPadR, nPadD, nPadL = padding
    padH = h + nPadU + nPadD
    padW = w + nPadL + nPadR
    xPad = np.pad(proj, ((nPadU, nPadD), (nPadL, nPadR)), mode=mode)

    CommonCosineDecay = lambda ascend, dot: np.cos(
        np.linspace(-np.pi / 2, 0, dot) if ascend else np.linspace(0, np.pi / 2, dot))
    SmootherCosineDecay = lambda ascend, dot: 0.5 * np.cos(np.linspace(
        -np.pi, 0, dot)) + 0.5 if ascend else 0.5 * np.cos(np.linspace(0, np.pi, dot)) + 0.5

    decay = SmootherCosineDecay if smootherDecay else CommonCosineDecay

    def decayLR(xPad, w, nPadL, nPadR):
        xPad[:, 0:nPadL] *= decay(True, nPadL)[:]
        xPad[:, w - nPadR:w] *= decay(False, nPadR)[:]
        return xPad

    xPad = decayLR(xPad, padW, nPadL, nPadR)
    xPad = decayLR(xPad.T, padH, nPadU, nPadD)
    xPad = xPad.T

    return xPad


def padSinogram(sgm, padding, mode='symmetric', smootherDecay=False):
    """
        'Truncated Artifact Correction' specialized function.
        Extend the projection on horizontal directions using symmetric `padding` (single, or (Right, Left))
        and descending cosine window decay. `mode` can be `symmetric` or `edge`.
    """
    if type(padding) == int:
        padding = (padding, padding)
    l, r = padding

    return padImage(sgm, (0, r, 0, l), mode, smootherDecay)


def correctBeamHardeningPolynomial(postlog, coeffs, bias=True):
    pass