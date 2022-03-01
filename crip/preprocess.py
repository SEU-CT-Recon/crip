'''
    Preprocess module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''

import numpy as np
from .shared import *


def averageProjections(projections):
    """
        Average projections. For example, to calculate the flat field.
    """
    res = np.array(projections.astype(np.float32).sum() / projections.shape[0])

    return res


def flatDarkFieldCorrection(projection, flat, coeff=1, dark=None):
    """
        Perform flat field (air) and dark field correction to get post-log value.
    """
    assert np.array_equal(projection.shape, flat.shape), "projection and flat should have same shape."
    if dark is None:
        dark = 0
    res = -np.log((projection.astype(np.float32) - dark) / ((coeff * flat) - dark))
    res[res == np.inf] = 0
    res[res == np.nan] = 0

    return res


def flatDarkFieldCorrectionStandalone(projection):
    """
        Perform flat field and dark field correction without actual field image. \\
        Air is estimated using the brightest pixel.
    """
    return flatDarkFieldCorrection(projection, np.max(projection))


def gaussianNoiseInject(projection, sigma):
    """
        Inject Gaussian noise which obeys distribution N(0, sigma^2).
    """
    noiseMask = np.random.randn(*projection.shape) * sigma
    res = projection + noiseMask

    return res


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


def projectionsToSinograms(projections):
    """
        Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.
    """
    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w), dtype=np.float32)
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]

    return sinograms


def sinogramsToProjections(sinograms):
    """
        Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.
    """
    (h, views, w) = sinograms.shape
    projections = np.zeros((views, h, w), dtype=np.float32)
    for i in range(views):
        projections[i, :, :] = sinograms[:, i, :]

    return projections


def padProjection(proj, padding, mode='symmetric', smootherDecay=False):
    """
        Extend the projection on four directions using symmetric `padding` (Up, Right, Down, Left) \\
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
        Extend the projection on horizontal directions using symmetric `padding` (Right, Left)
        and descending cosine window decay. `mode` can be `symmetric` or `edge`.
    """
    CommonCosineDecay = lambda ascend, dot: np.cos(
        np.linspace(-np.pi / 2, 0, dot) if ascend else np.linspace(0, np.pi / 2, dot))
    SmootherCosineDecay = lambda ascend, dot: 0.5 * np.cos(np.linspace(
        -np.pi, 0, dot)) + 0.5 if ascend else 0.5 * np.cos(np.linspace(0, np.pi, dot)) + 0.5

    decay = SmootherCosineDecay if smootherDecay else CommonCosineDecay

    def decayLR(pad):
        pad[:, 0:padding] *= decay(True, padding)[:]
        pad[:, w - padding:w] *= decay(False, padding)[:]
        return pad

    pad = np.pad(sgm, ((0, 0), (padding, padding)), mode=mode)
    _, w = pad.shape

    return decayLR(pad)