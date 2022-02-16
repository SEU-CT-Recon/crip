import numpy as np
import cv2


def averageProjections(projections):
    """
        Average projections. For example, to calculate the flat field.
    """
    res = np.array(projections.asarray(np.float32).sum() / projections.shape[0])
    return res


def flatDarkFieldCorrection(projection, flat, coeff=1, dark=None):
    """
        Perform flat field (air) and dark field correction to get post-log value.
    """
    assert np.array_equal(projection.shape, flat.shape), "projection and flat should have same shape."
    if dark is None:
        dark = 0
    res = -np.log((projection.asarray(np.float32) - dark) / ((coeff * flat) - dark))
    res[res == np.inf] = 0
    res[res == np.nan] = 0
    return res


def flatDarkFieldCorrectionStandalone(projection):
    """
        Perform flat field and dark field correction without actual field image. \\
        Air is estimated using the brightest pixel.
    """
    return flatDarkFieldCorrection(projection, np.max(projection))


def gaussianSmooth(projection, ksize, sigma):
    """
        Perform Gaussian smooth with kernel size = ksize and Gaussian \sigma = sigma (int or tuple (x, y)).
    """
    if isinstance(sigma, int):
        sigma = (sigma, sigma)
    return cv2.GaussianBlur(projection.astype(np.float32), ksize, sigmaX=sigma[0], sigmaY=sigma[1])


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
    assert dstLen == int(dstLen), "cannot achieve uniform sampling"
    return np.array(projections[::ratio, :, :]), projections.shape[0] % ratio - 1


def projectionsToSinograms(projections):
    """
        Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.
    """
    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w))
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]


def sinogramsToProjections(sinograms):
    """
        Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.
    """
    (h, views, w) = sinograms.shape
    projections = np.zeros((views, h, w))
    for i in range(views):
        projections[i, :, :] = sinograms[:, i, :]



def padProj(proj, padding: int):
    def cosineDecay(offCenter, padding):
        ratio = offCenter / padding
        return np.cos(np.pi / 2 * ratio)

    pad = np.pad(proj, ((0, 0), (padding, padding)), mode='symmetric')
    h, w = pad.shape

    for r in range(h):
        for c in range(0, padding):
            pad[r, c] *= cosineDecay(padding - c, padding)
        for c in range(w - padding, w):
            pad[r, c] *= cosineDecay(c - w + padding, padding)

    return pad