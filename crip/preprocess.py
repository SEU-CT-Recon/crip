'''
    Preprocess module of crip.

    https://github.com/z0gSh1u/crip
'''

import numpy as np
from .shared import *
from ._typing import *
from .utils import *


@ConvertListNDArray
def averageProjections(projections: TwoOrThreeD):
    '''
        Average projections. For example, to calculate the flat field.
        Projections can be either `(views, H, W)` shaped numpy array, or
        `views * (H, W)` Python List.
    '''
    cripAssert(is3D(projections), '`projections` should be 3D array.')
    projections = asFloat(projections)

    res = projections.sum(axis=0) / projections.shape[0]

    return res


@ConvertListNDArray
def flatDarkFieldCorrection(projections: TwoOrThreeD,
                            flat: Or[TwoD, float],
                            coeff: float = 1,
                            dark: Or[TwoD, float] = 0):
    '''
        Perform flat field (air) and dark field correction to get post-log value.

        I.e., `- log [(X - D) / (C * F - D)]`. Multi projections accepted.
    '''
    sampleProjection = projections if is2D(projections) else projections[0]

    if isType(flat, TwoD):
        cripAssert(isOfSameShape(sampleProjection, flat), "`projection` and `flat` should have same shape.")
    if isType(dark, TwoD):
        cripAssert(isOfSameShape(sampleProjection, dark), "`projection` and `dark` should have same shape.")

    res = -np.log((projections - dark) / (coeff * flat - dark))
    res[res == np.inf] = 0
    res[res == np.nan] = 0

    return res


def flatDarkFieldCorrectionStandalone(projection: TwoD):
    '''
        Perform flat field and dark field correction without actual field image.

        Air is estimated using the brightest pixel by default.
    '''
    # We support 2D only in standalone version, since `flat` for each projection might differ.
    cripAssert(is2D(projection), '`projection` should be 2D.')

    return flatDarkFieldCorrection(projection, np.max(projection), 1, 0)


@ConvertListNDArray
def injectGaussianNoise(projections: TwoOrThreeD, sigma: float, mu: float = 0):
    '''
        Inject Gaussian noise which obeys distribution `N(\mu, \sigma^2)`.
    '''
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')
    cripAssert(sigma > 0, 'sigma should > 0.')

    injectOne = lambda img: (np.random.randn(*img.shape) * sigma + mu) + img

    if is3D(projections):
        res = np.zeros_like(projections)
        for c in range(projections.shape[0]):
            res[c, ...] = injectOne(projections[c, ...])
    else:
        res = injectOne(projections)

    return res


@ConvertListNDArray
def injectPoissonNoise(projections: TwoOrThreeD, nPhoton: int):
    '''
        Inject Poisson noise which obeys distribution `P(\lamda)`.
        `nPhoton` is the number of photon hitting per detector element.
    '''
    cripAssert(is2or3D(projections), '`projections` should be 2D or 3D.')

    def injectOne(img):
        I0 = np.max(img)
        proj = nPhoton * np.exp(-img / I0)
        proj = np.random.poisson(proj)
        proj = -np.log(img / nPhoton) * I0
        return proj

    if is3D(projections):
        res = np.zeros_like(projections)
        for i in range(projections.shape[0]):
            res[i, ...] = injectOne(projections[i, ...])
    else:
        res = injectOne(projections)

    return res


@ConvertListNDArray
def projectionsToSinograms(projections: ThreeD):
    '''
        Permute projections to sinograms by axes swapping `(views, h, w) -> (h, views, w)`.

        Note that the width direction is along detector channels of one line.
    '''
    cripAssert(is3D(projections), 'projections should be 3D.')

    (views, h, w) = projections.shape
    sinograms = np.zeros((h, views, w), dtype=projections.dtype)
    for i in range(views):
        sinograms[:, i, :] = projections[i, :, :]

    return sinograms


@ConvertListNDArray
def sinogramsToProjections(sinograms: ThreeD):
    '''
        Permute sinograms back to projections by axes swapping `(h, views, w) -> (views, h, w)`.

        Note that the width direction is along detector channels of one line.
    '''
    cripAssert(is3D(sinograms), 'projections should be 3D.')

    (h, views, w) = sinograms.shape
    projections = np.zeros((views, h, w), dtype=sinograms.dtype)
    for i in range(views):
        projections[i, :, :] = sinograms[:, i, :]

    return projections


@ConvertListNDArray
def padImage(img: TwoOrThreeD,
             padding: Tuple[int, int, int, int],
             mode: str = 'symmetric',
             decay: Or[str, None] = None):
    '''
        Pad the image on four directions using symmetric `padding` (Up, Right, Down, Left). \\
        `mode` determines the border value, can be `symmetric`, `edge`, `constant` (zero), `reflect`. \\
        `decay` can be None, `cosine`, `smoothCosine` to perform a decay on padded border.
    '''
    cripAssert(mode in ['symmetric', 'edge', 'constant', 'reflect'], f'Invalid mode: {mode}.')
    cripAssert(decay in [None, 'cosine', 'smoothCosine'], f'Invalid decay: {decay}.')

    decays = {
        'cosine':
        lambda ascend, dot: np.cos(np.linspace(-np.pi / 2, 0, dot) if ascend else np.linspace(0, np.pi / 2, dot)),
        'smoothCosine':
        lambda ascend, dot: 0.5 * np.cos(np.linspace(-np.pi, 0, dot)) + 0.5
        if ascend else 0.5 * np.cos(np.linspace(0, np.pi, dot)) + 0.5
    }

    h, w = getHW(img)
    nPadU, nPadR, nPadD, nPadL = padding
    padH = h + nPadU + nPadD
    padW = w + nPadL + nPadR

    def decayLR(xPad, w, nPadL, nPadR, decay):
        xPad[:, 0:nPadL] *= decay(True, nPadL)[:]
        xPad[:, w - nPadR:w] *= decay(False, nPadR)[:]
        return xPad

    def procOne(img):
        xPad = np.pad(img, ((nPadU, nPadD), (nPadL, nPadR)), mode=mode)
        if decay is not None:
            xPad = decayLR(xPad, padW, nPadL, nPadR, decays[decay])
            xPad = decayLR(xPad.T, padH, nPadU, nPadD, decays[decay])
            xPad = xPad.T
        return xPad

    if is3D(img):
        res = []
        c = img.shape[0]
        for i in range(c):
            res.append(procOne(img[i, ...]))
        return np.array(res)
    else:
        return procOne(img)


@ConvertListNDArray
def padSinogram(sgms: TwoOrThreeD, padding: Or[int, Tuple[int, int]], mode='symmetric', decay='smoothCosine'):
    '''
        Pad sinograms in width direction (same line detector elements) using `mode` and `decay`\\
        with `padding` (single int, or (right, left)).

        @see padImage for parameter details.
    '''
    if isType(padding, int):
        padding = (padding, padding)

    l, r = padding

    return padImage(sgms, (0, r, 0, l), mode, decay)


@ConvertListNDArray
def correctBeamHardeningPolynomial(postlog: TwoOrThreeD, coeffs: Or[Tuple, np.poly1d], bias=True):
    '''
        Apply the polynomial (\mu L vs. PostLog fit) on postlog to perform basic beam hardening correction.
        `coeffs` can be either `tuple` or `np.poly1d`. Set `bias=True` if your coeffs includes the bias term.
    '''
    cripWarning(isType(coeffs, np.poly1d) and bias is False, 'When using np.poly1d as coeffs, bias is always True.')
    if isType(coeffs, Tuple):
        if bias is False:
            coeffs = np.poly1d([*coeffs, 0])

    return coeffs(postlog)


def limitAngle(projections: ThreeD, total: float, start: float, dst: float):
    cripAssert(False, 'Unimplemented.')  # TODO


def limitView(projections: ThreeD, ratio: float):
    cripAssert(False, 'Unimplemented.')  # TODO
